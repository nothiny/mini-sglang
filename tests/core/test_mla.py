"""MLA correctness tests.

`test_mla_naive_equals_absorbed` and the rotary test are pure-torch. The
`MLAttention` vs `transformers` comparison needs a CUDA GPU (FlashInfer RoPE).
"""

from __future__ import annotations

import pytest
import torch

import minisgl.distributed.info as distributed_info
from minisgl.distributed import DistributedInfo
from minisgl.layers.mla import mla_attention_absorbed, mla_attention_naive

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(autouse=True)
def isolated_global_state(monkeypatch):
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 1))
    yield


@pytest.mark.parametrize(
    ("num_tokens", "num_kv", "causal"), [(8, 8, True), (1, 8, True), (8, 8, False)]
)
def test_mla_naive_equals_absorbed(num_tokens: int, num_kv: int, causal: bool) -> None:
    torch.manual_seed(0)
    num_heads, qk_nope, qk_rope, v_head, rank = 16, 128, 64, 128, 512
    sm_scale = (qk_nope + qk_rope) ** -0.5

    q_nope = torch.randn(num_tokens, num_heads, qk_nope)
    q_pe = torch.randn(num_tokens, num_heads, qk_rope)
    c_kv = torch.randn(num_kv, rank)
    k_pe = torch.randn(num_kv, qk_rope)
    w_uk = torch.randn(num_heads, qk_nope, rank) * 0.05
    w_uv = torch.randn(num_heads, v_head, rank) * 0.05

    naive = mla_attention_naive(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, sm_scale, causal=causal)
    absorbed = mla_attention_absorbed(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, sm_scale, causal=causal)
    assert naive.shape == (num_tokens, num_heads, v_head)
    assert torch.allclose(naive, absorbed, atol=1e-5, rtol=1e-4)


def test_mla_rope_is_interleaved() -> None:
    """FlashInfer's ``is_neox=False`` must match the interleaved pairing DeepSeek uses."""
    from minisgl.layers.rotary import RotaryEmbedding, set_rope_device

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the FlashInfer RoPE kernel")

    head_dim, max_pos, base = 64, 32, 10000.0
    set_rope_device(torch.device("cuda"))
    rope = RotaryEmbedding(head_dim, head_dim, max_pos, base, is_neox=False)
    rope._cos_sin_cache = rope._cos_sin_cache.cuda()

    seq, heads = 5, 3
    q = torch.randn(seq, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(seq, 1, head_dim, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(seq, dtype=torch.int32, device="cuda")
    q_out, k_out = rope.forward(positions, q.clone(), k.clone())

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq).float(), inv_freq)
    cos, sin = freqs.cos().cuda(), freqs.sin().cuda()

    def interleaved(x: torch.Tensor) -> torch.Tensor:
        pairs = x.float().reshape(*x.shape[:-1], head_dim // 2, 2)
        x1, x2 = pairs[..., 0], pairs[..., 1]
        c = cos[positions][:, None, :]
        s = sin[positions][:, None, :]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).reshape_as(x)

    assert torch.allclose(q_out.float(), interleaved(q), atol=5e-2, rtol=5e-2)
    assert torch.allclose(k_out.float(), interleaved(k), atol=5e-2, rtol=5e-2)


def test_mlattention_matches_hf_deepseek_attention() -> None:
    from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2Attention,
        DeepseekV2RotaryEmbedding,
    )

    from minisgl.layers.mla import MLAttention
    from minisgl.layers.rotary import set_rope_device
    from minisgl.models.config import ModelConfig, RotaryConfig

    torch.manual_seed(0)
    device = "cuda"
    num_heads, qk_nope, qk_rope, v_head, rank, hidden, seq = 8, 128, 64, 128, 512, 512, 6
    qk_head = qk_nope + qk_rope

    config = ModelConfig(
        num_layers=1,
        num_qo_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=qk_head,
        hidden_size=hidden,
        vocab_size=128,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(qk_head, qk_head, 64, 10000.0, None),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="deepseek_v2",
        architectures=["DeepseekV2ForCausalLM"],
        q_lora_rank=None,
        kv_lora_rank=rank,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=v_head,
    )
    hf_config = DeepseekV2Config(
        hidden_size=hidden,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=v_head,
        kv_lora_rank=rank,
        q_lora_rank=None,
        max_position_embeddings=64,
        rope_theta=10000.0,
        attention_bias=False,
    )
    hf_config._attn_implementation = "eager"
    hf_attn = DeepseekV2Attention(hf_config, layer_idx=0).to(device).to(torch.bfloat16).eval()

    set_rope_device(torch.device(device))
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            attention = MLAttention(config, layer_id=0)
    finally:
        torch.set_default_dtype(previous_dtype)

    with torch.no_grad():
        attention.q_proj.weight.copy_(hf_attn.q_proj.weight)
        attention.kv_a_proj_with_mqa.weight.copy_(hf_attn.kv_a_proj_with_mqa.weight)
        attention.kv_a_layernorm.weight.copy_(hf_attn.kv_a_layernorm.weight)
        attention.kv_b_proj.weight.copy_(hf_attn.kv_b_proj.weight)
        attention.o_proj.weight.copy_(hf_attn.o_proj.weight)

    hidden_states = torch.randn(seq, hidden, dtype=torch.bfloat16, device=device)
    positions = torch.arange(seq, dtype=torch.int32, device=device)
    batch = hidden_states.unsqueeze(0)
    rotary = DeepseekV2RotaryEmbedding(hf_config).to(device)
    position_embeddings = rotary(batch, positions.unsqueeze(0))

    with torch.no_grad():
        ours = attention.forward(hidden_states, positions, causal=False)
        reference, _ = hf_attn(batch, position_embeddings=position_embeddings)
        assert torch.allclose(ours, reference.view(seq, hidden), atol=2e-2, rtol=2e-2)

        causal_mask = torch.full(
            (1, 1, seq, seq),
            torch.finfo(torch.bfloat16).min,
            device=device,
            dtype=torch.bfloat16,
        ).triu(1)
        ours_causal = attention.forward(hidden_states, positions, causal=True)
        reference_causal, _ = hf_attn(
            batch, attention_mask=causal_mask, position_embeddings=position_embeddings
        )
        assert torch.allclose(ours_causal, reference_causal.view(seq, hidden), atol=2e-2, rtol=2e-2)
