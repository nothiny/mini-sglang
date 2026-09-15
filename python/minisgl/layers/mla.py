from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseOP
from .linear import LinearColParallelMerged, LinearOProj, LinearReplicated
from .norm import RMSNorm
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

# MLA attention reference math.
#
# q_nope: [T, H, qk_nope_head_dim]
# q_pe:   [T, H, qk_rope_head_dim]
# c_kv:   [S, kv_lora_rank]              (the cached latent, shared across heads)
# k_pe:   [S, qk_rope_head_dim]          (the cached decoupled RoPE key, shared across heads)
# w_uk:   [H, qk_nope_head_dim, kv_lora_rank]   (k_nope = w_uk @ c_kv)
# w_uv:   [H, v_head_dim, kv_lora_rank]         (v      = w_uv @ c_kv)
#
# These two functions are mathematically equivalent; the naive one materializes
# K/V, the absorbed one attends directly over the latent. They are the oracle for
# the fused kernels and the paged backends.


def _causal_scores(scores: torch.Tensor, causal: bool) -> torch.Tensor:
    # scores: [H, T, S]
    if not causal:
        return scores
    t, s = scores.shape[-2], scores.shape[-1]
    mask = torch.ones(t, s, dtype=torch.bool, device=scores.device).tril(diagonal=s - t)
    return scores.masked_fill(~mask, float("-inf"))


def _softmax(scores: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.softmax(scores.float(), dim=-1).to(dtype)


def mla_attention_naive(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    sm_scale: float,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Up-project the latent to K/V, then run ordinary attention."""

    num_heads = q_nope.shape[1]
    num_kv = c_kv.shape[0]
    k_nope = torch.einsum("hnr,sr->shn", w_uk, c_kv)
    v = torch.einsum("hvr,sr->shv", w_uv, c_kv)
    k = torch.cat([k_nope, k_pe[:, None, :].expand(num_kv, num_heads, -1)], dim=-1)
    q = torch.cat([q_nope, q_pe], dim=-1)
    scores = torch.einsum("thd,shd->hts", q, k) * sm_scale
    probs = _softmax(_causal_scores(scores, causal), q.dtype)
    return torch.einsum("hts,shv->thv", probs, v)


def mla_latent(
    q_abs: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    sm_scale: float,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Absorbed attention over the latent, before the ``w_uv`` output projection.

    ``q_abs`` is the already-absorbed query ([T, H, kv_lora_rank]). The result is
    ``z`` [T, H, kv_lora_rank]; this is exactly what FlashInfer's
    ``BatchMLAPagedAttentionWrapper.run`` returns.
    """

    num_heads = q_abs.shape[1]
    num_kv = c_kv.shape[0]
    q = torch.cat([q_abs, q_pe], dim=-1)
    k = torch.cat(
        [
            c_kv[:, None, :].expand(num_kv, num_heads, -1),
            k_pe[:, None, :].expand(num_kv, num_heads, -1),
        ],
        dim=-1,
    )
    scores = torch.einsum("thd,shd->hts", q, k) * sm_scale
    probs = _softmax(_causal_scores(scores, causal), q.dtype)
    return torch.einsum("hts,sr->thr", probs, c_kv)


def mla_attention_absorbed(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    sm_scale: float,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Fold ``w_uk`` into the query and ``w_uv`` into the output; attend on the latent."""

    q_abs = torch.einsum("hnr,thn->thr", w_uk, q_nope)
    z = mla_latent(q_abs, q_pe, c_kv, k_pe, sm_scale, causal=causal)
    return torch.einsum("hvr,thr->thv", w_uv, z)


class MLAttention(BaseOP):
    """DeepSeek MLA attention block.

    With ``context=None`` it is a single-shot full-context attention, which is
    what the M1 correctness tests exercise. The paged latent KV cache is layered
    on top of ``_attend`` in M2.
    """

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        assert config.is_mla, "MLAttention requires an MLA model config"
        tp_size = get_tp_info().size
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope = config.qk_nope_head_dim
        self.qk_rope = config.qk_rope_head_dim
        self.v_head = config.v_head_dim
        self.qk_head_dim = self.qk_nope + self.qk_rope
        self.num_heads = div_even(config.num_qo_heads, tp_size)
        self.sm_scale = self.qk_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = LinearColParallelMerged(
                config.hidden_size, [self.num_heads * self.qk_head_dim], has_bias=False
            )
        else:
            self.q_a_proj = LinearReplicated(config.hidden_size, self.q_lora_rank, has_bias=False)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = LinearColParallelMerged(
                self.q_lora_rank, [self.num_heads * self.qk_head_dim], has_bias=False
            )
        self.kv_a_proj_with_mqa = LinearReplicated(
            config.hidden_size, self.kv_lora_rank + self.qk_rope, has_bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = LinearColParallelMerged(
            self.kv_lora_rank,
            [self.num_heads * (self.qk_nope + self.v_head)],
            has_bias=False,
        )
        self.o_proj = LinearOProj(self.num_heads * self.v_head, config.hidden_size, has_bias=False)
        rope_scaling = (
            tuple(config.rotary_config.scaling.items()) if config.rotary_config.scaling else None
        )
        self.rotary = get_rope(
            head_dim=self.qk_rope,
            rotary_dim=self.qk_rope,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=rope_scaling,
            is_neox=False,
        )

    def _split_kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.kv_b_proj.weight.view(
            self.num_heads, self.qk_nope + self.v_head, self.kv_lora_rank
        )
        return weight[:, : self.qk_nope, :], weight[:, self.qk_nope :, :]

    def _attend(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        from minisgl.core import try_get_global_ctx

        w_uk, w_uv = self._split_kv_b()
        q_abs = torch.einsum("hnr,thn->thr", w_uk, q_nope)
        ctx = try_get_global_ctx()
        backend = ctx.attn_backend if ctx is not None else None
        if backend is not None and hasattr(backend, "forward_mla"):
            z = backend.forward_mla(q_abs, q_pe, c_kv, k_pe, self.layer_id, ctx.batch)
        else:
            z = mla_latent(q_abs, q_pe, c_kv, k_pe, self.sm_scale, causal=causal)
        return torch.einsum("hvr,thr->thv", w_uv, z)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        causal: bool = True,
    ) -> torch.Tensor:
        from minisgl.core import get_global_ctx

        if positions is None:
            positions = get_global_ctx().batch.positions
        num_tokens = x.shape[0]

        if self.q_lora_rank is None:
            q = self.q_proj.forward(x)
        else:
            q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x)))
        q = q.view(num_tokens, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q[..., : self.qk_nope], q[..., self.qk_nope :].contiguous()

        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv, k_pe = kv[..., : self.kv_lora_rank], kv[..., self.kv_lora_rank :]
        c_kv = self.kv_a_layernorm.forward(c_kv)
        k_pe = k_pe.unsqueeze(1).contiguous()
        q_pe, k_pe = self.rotary.forward(positions, q_pe, k_pe)
        k_pe = k_pe.squeeze(1)

        o = self._attend(q_nope, q_pe, c_kv, k_pe, causal=causal)
        return self.o_proj.forward(o.reshape(num_tokens, self.num_heads * self.v_head))


__all__ = ["MLAttention", "mla_attention_absorbed", "mla_attention_naive", "mla_latent"]
