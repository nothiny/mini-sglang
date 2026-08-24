"""Correctness and integration tests for fused MoE GPU kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from minisgl.layers.moe import MoELayer
from minisgl.moe import MoeBackendConfig
from minisgl.moe.fused import FusedMoe
from minisgl.moe.weights import ExpertResidentCache, quantize_expert_weight


def torch_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    router_probs = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(router_probs, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    output = torch.zeros_like(hidden_states)
    for token_idx in range(hidden_states.shape[0]):
        for slot_idx in range(top_k):
            expert_idx = int(topk_ids[token_idx, slot_idx])
            gate_up = F.linear(hidden_states[token_idx], w1[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            expert_output = F.linear(activated, w2[expert_idx])
            output[token_idx] += expert_output * topk_weights[token_idx, slot_idx]
    return output


@pytest.mark.parametrize(("num_tokens", "small_m_threshold"), [(4, 8), (32, 8)])
def test_fused_moe_matches_torch_reference(
    num_tokens: int,
    small_m_threshold: int,
) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_experts, top_k = 8, 2
    hidden_size, intermediate_size = 64, 64
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        / hidden_size**0.5
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        / intermediate_size**0.5
    )
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
    expected = torch_moe_reference(hidden_states, w1, w2, router_logits, top_k)

    backend = FusedMoe(
        MoeBackendConfig(
            enable_workspace_cache=True,
            small_m_threshold=small_m_threshold,
        )
    )
    actual = backend.forward(
        hidden_states.clone(),
        w1,
        w2,
        router_logits,
        top_k,
        True,
        "silu",
        False,
    )
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)


def test_direct_and_grouped_moe_paths_match() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    hidden = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(8, 128, 64, device=device, dtype=torch.bfloat16) / 8
    w2 = torch.randn(8, 64, 64, device=device, dtype=torch.bfloat16) / 8
    logits = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
    direct = FusedMoe(MoeBackendConfig(small_m_threshold=8)).forward(
        hidden.clone(), w1, w2, logits, 2, True, "silu", False
    )
    grouped = FusedMoe(MoeBackendConfig(small_m_threshold=0)).forward(
        hidden.clone(), w1, w2, logits, 2, True, "silu", False
    )
    torch.testing.assert_close(direct, grouped, atol=0.03, rtol=0.03)


@pytest.mark.parametrize("small_m_threshold", [0, 8])
def test_int8_expert_weights(small_m_threshold: int) -> None:
    torch.manual_seed(13)
    device = torch.device("cuda")
    hidden = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(8, 128, 64, device=device, dtype=torch.bfloat16) / 8
    w2 = torch.randn(8, 64, 64, device=device, dtype=torch.bfloat16) / 8
    logits = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
    expected = FusedMoe(MoeBackendConfig(small_m_threshold=small_m_threshold)).forward(
        hidden.clone(), w1, w2, logits, 2, True, "silu", False
    )
    w1_int8, w1_scale = quantize_expert_weight(w1)
    w2_int8, w2_scale = quantize_expert_weight(w2)
    actual = FusedMoe(MoeBackendConfig(small_m_threshold=small_m_threshold)).forward(
        hidden.clone(),
        w1_int8,
        w2_int8,
        logits,
        2,
        True,
        "silu",
        False,
        w1_scale,
        w2_scale,
    )
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)


def test_int8_experts_are_quantized_on_cpu_before_gpu_transfer() -> None:
    layer = MoELayer.__new__(MoELayer)
    layer._moe_config = MoeBackendConfig(expert_quantization="int8")
    layer.gate_up_proj = torch.randn(4, 128, 64, dtype=torch.bfloat16)
    layer.down_proj = torch.randn(4, 64, 64, dtype=torch.bfloat16)
    layer._gate_up_scale = None
    layer._down_scale = None
    layer._weights_prepared = False

    layer.prepare_weights(torch.device("cuda"))

    assert layer.gate_up_proj.dtype == torch.int8
    assert layer.gate_up_proj.device.type == "cuda"
    assert layer.down_proj.dtype == torch.int8
    assert layer.down_proj.device.type == "cuda"
    assert layer._gate_up_scale is not None
    assert layer._gate_up_scale.device.type == "cuda"
    assert layer._down_scale is not None
    assert layer._down_scale.device.type == "cuda"


def test_fused_moe_reuses_workspace() -> None:
    device = torch.device("cuda")
    hidden_states = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(8, 128, 64, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(8, 64, 64, device=device, dtype=torch.bfloat16)
    router_logits = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
    backend = FusedMoe(MoeBackendConfig(enable_workspace_cache=True))

    for _ in range(2):
        backend.forward(hidden_states.clone(), w1, w2, router_logits, 2, True, "silu", False)

    assert backend.workspace_cache.allocation_count == 1


def test_fused_moe_autotunes_once_per_shape() -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    hidden = torch.randn(1, 64, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(8, 128, 64, device=device, dtype=torch.bfloat16) / 8
    w2 = torch.randn(8, 64, 64, device=device, dtype=torch.bfloat16) / 8
    logits = torch.randn(1, 8, device=device, dtype=torch.bfloat16)
    expected = torch_moe_reference(hidden, w1, w2, logits, 2)
    backend = FusedMoe(MoeBackendConfig(autotune=True, small_m_threshold=0))

    for _ in range(2):
        actual = backend.forward(hidden.clone(), w1, w2, logits, 2, True, "silu", False)
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)

    assert len(backend._kernel_choices) == 1


def test_fused_moe_autotunes_prefill_grouped_config() -> None:
    torch.manual_seed(18)
    device = torch.device("cuda")
    hidden = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(8, 128, 64, device=device, dtype=torch.bfloat16) / 8
    w2 = torch.randn(8, 64, 64, device=device, dtype=torch.bfloat16) / 8
    logits = torch.randn(64, 8, device=device, dtype=torch.bfloat16)
    expected = FusedMoe(MoeBackendConfig(small_m_threshold=0)).forward(
        hidden.clone(), w1, w2, logits, 2, True, "silu", False
    )
    backend = FusedMoe(MoeBackendConfig(autotune=True, small_m_threshold=0))

    actual = backend.forward(hidden.clone(), w1, w2, logits, 2, True, "silu", False)

    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    assert list(backend._kernel_choices.values()) == [False]
    assert len(backend._grouped_config_choices) == 1


@pytest.mark.parametrize(
    ("quantized", "cache_size"),
    [(False, 4), (True, 4), (False, 2), (True, 2)],
)
def test_fused_moe_expert_offload_matches_resident_weights(
    quantized: bool,
    cache_size: int,
) -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    hidden = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(4, 128, 64, device=device, dtype=torch.bfloat16) / 8
    w2 = torch.randn(4, 64, 64, device=device, dtype=torch.bfloat16) / 8
    logits = torch.tensor(
        [[9, 8, 0, 0], [0, 0, 9, 8], [8, 9, 0, 0], [0, 0, 8, 9]],
        device=device,
        dtype=torch.bfloat16,
    )
    expected = FusedMoe(MoeBackendConfig(small_m_threshold=0)).forward(
        hidden.clone(), w1, w2, logits, 2, True, "silu", False
    )

    w1_scale = w2_scale = None
    if quantized:
        w1, w1_scale = quantize_expert_weight(w1)
        w2, w2_scale = quantize_expert_weight(w2)
    w1_host = w1.cpu().pin_memory()
    w2_host = w2.cpu().pin_memory()
    w1_scale_host = None if w1_scale is None else w1_scale.cpu().pin_memory()
    w2_scale_host = None if w2_scale is None else w2_scale.cpu().pin_memory()
    backend = FusedMoe(
        MoeBackendConfig(
            small_m_threshold=0,
            expert_quantization="int8" if quantized else "none",
            expert_offload=True,
            expert_cache_size=cache_size,
        )
    )

    for _ in range(2):
        actual = backend.forward(
            hidden.clone(),
            w1_host,
            w2_host,
            logits,
            2,
            True,
            "silu",
            False,
            w1_scale_host,
            w2_scale_host,
        )
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)

    cache = next(iter(backend._resident_caches.values()))
    if cache_size == 4:
        assert cache.misses == 4
        assert cache.hits == 4
    else:
        assert cache.misses == 8
        assert cache.hits == 0


def test_expert_resident_cache_lru_and_remapping() -> None:
    device = torch.device("cuda")
    w1 = torch.arange(4 * 8 * 4, dtype=torch.bfloat16).view(4, 8, 4).pin_memory()
    w2 = torch.arange(4 * 4 * 4, dtype=torch.bfloat16).view(4, 4, 4).pin_memory()
    cache = ExpertResidentCache(w1, w2, None, None, capacity=2, device=device)

    first = cache.resolve(torch.tensor([[0, 1]], dtype=torch.int32, device=device))
    assert first.expert_ids.tolist() == [[0, 1]]
    second = cache.resolve(torch.tensor([[1, 2]], dtype=torch.int32, device=device))
    assert second.expert_ids.tolist() == [[1, 0]]
    torch.testing.assert_close(second.w1[1].cpu(), w1[1])
    torch.testing.assert_close(second.w1[0].cpu(), w1[2])
    assert cache.hits == 1
    assert cache.misses == 3
