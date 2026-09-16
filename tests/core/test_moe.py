from __future__ import annotations

import pytest
import torch
from minisgl.layers.base import BaseOP
from minisgl.layers.moe import MoELayer, get_moe_expert_cache_bytes
from minisgl.moe import MoeBackendConfig
from minisgl.moe.expert_parallel import build_replicated_dispatch_inputs
from minisgl.moe.weights import quantize_expert_weight
from minisgl.server.args import parse_args


class _MoeRoot(BaseOP):
    def __init__(self, layer: MoELayer) -> None:
        self.layer = layer

    def forward(self) -> None:
        raise NotImplementedError


def test_moe_backend_config_validation() -> None:
    with pytest.raises(ValueError, match="small_m_threshold"):
        MoeBackendConfig(small_m_threshold=-1)
    with pytest.raises(ValueError, match="expert_cache_size"):
        MoeBackendConfig(expert_offload=True, expert_cache_size=0)
    with pytest.raises(ValueError, match="expert_parallel_rank"):
        MoeBackendConfig(expert_parallel_size=2, expert_parallel_rank=2)


def test_replicated_ep_dispatch_assigns_each_route_once() -> None:
    hidden = torch.arange(6 * 3, dtype=torch.float32).view(6, 3)
    topk_ids = torch.tensor([[0, 2], [1, 3], [2, 0], [3, 1], [0, 3], [2, 1]])
    weights = torch.ones_like(topk_ids, dtype=torch.float32)
    all_token_ids = []
    all_send_counts = []
    for rank in range(2):
        _, local_ids, _, token_ids, counts = build_replicated_dispatch_inputs(
            hidden,
            weights,
            topk_ids,
            rank=rank,
            world_size=2,
            num_experts=4,
        )
        assert bool(torch.all((0 <= local_ids) & (local_ids < 2)))
        all_token_ids.extend(token_ids.tolist())
        all_send_counts.append(counts.tolist())

    assert sorted(all_token_ids) == sorted(list(range(6)) * 2)
    assert [sum(counts) for counts in all_send_counts] == [6, 6]


def test_hot_expert_replica_keeps_routes_on_origin_rank() -> None:
    hidden = torch.arange(4 * 3, dtype=torch.float32).view(4, 3)
    topk_ids = torch.zeros((4, 2), dtype=torch.int32)
    weights = torch.ones_like(topk_ids, dtype=torch.float32)
    for rank in range(2):
        _, local_ids, _, _, counts = build_replicated_dispatch_inputs(
            hidden,
            weights,
            topk_ids,
            rank=rank,
            world_size=2,
            num_experts=4,
            replicated_experts=(0,),
        )
        assert counts.tolist() == ([4, 0] if rank == 0 else [0, 4])
        assert local_ids.tolist() == [2, 2, 2, 2]


def test_int8_quantization_is_per_output_channel() -> None:
    weight = torch.tensor(
        [[[1.0, -2.0], [100.0, -50.0]]],
        dtype=torch.float32,
    )
    quantized, scale = quantize_expert_weight(weight)
    reconstructed = quantized.float() * scale.unsqueeze(-1)
    torch.testing.assert_close(reconstructed, weight, atol=0.8, rtol=0.01)
    assert scale.shape == (1, 2)


def test_moe_cli_configuration() -> None:
    args, run_shell = parse_args(
        [
            "--model",
            "unused/model",
            "--dtype",
            "float16",
            "--moe-autotune",
            "--moe-small-m-threshold",
            "4",
            "--expert-parallel-size",
            "2",
            "--disable-moe-communication-overlap",
            "--moe-expert-parallel-dispatch",
            "static",
            "--moe-expert-placement",
            "round-robin",
            "--moe-replicated-experts",
            "0,3",
            "--moe-expert-quantization",
            "int8",
            "--moe-expert-offload",
            "--moe-expert-cache-size",
            "8",
            "--disable-moe-expert-offload-overlap",
        ]
    )

    assert not run_shell
    assert args.moe_autotune
    assert args.moe_small_m_threshold == 4
    assert args.expert_parallel_size == 2
    assert not args.moe_expert_parallel_overlap
    assert args.moe_expert_parallel_dispatch == "static"
    assert args.moe_expert_placement == "round-robin"
    assert args.moe_replicated_experts == (0, 3)
    assert args.moe_expert_quantization == "int8"
    assert args.moe_expert_offload
    assert args.moe_expert_cache_size == 8
    assert not args.moe_expert_offload_overlap


def test_expert_cache_memory_reservation_includes_weights_scales_and_mapping() -> None:
    layer = MoELayer.__new__(MoELayer)
    layer.gate_up_proj = torch.empty(4, 8, 4, dtype=torch.int8)
    layer.down_proj = torch.empty(4, 4, 4, dtype=torch.int8)
    layer._gate_up_scale = torch.empty(4, 8, dtype=torch.float32)
    layer._down_scale = torch.empty(4, 4, dtype=torch.float32)

    bytes_per_expert = sum(
        tensor[0].nbytes
        for tensor in (
            layer.gate_up_proj,
            layer.down_proj,
            layer._gate_up_scale,
            layer._down_scale,
        )
    )
    expected = 2 * bytes_per_expert + 4 * torch.int64.itemsize
    assert get_moe_expert_cache_bytes(_MoeRoot(layer), capacity=2) == expected


def test_moe_layer_loads_prequantized_weights_and_scales() -> None:
    layer = MoELayer.__new__(MoELayer)
    layer.gate_up_proj = torch.empty(2, 8, 4, device="meta")
    layer.down_proj = torch.empty(2, 4, 4, device="meta")
    layer._gate_up_scale = None
    layer._down_scale = None
    layer._weights_quantized = False
    state = {
        "experts.gate_up_proj": torch.ones(2, 8, 4, dtype=torch.float8_e4m3fn),
        "experts.down_proj": torch.ones(2, 4, 4, dtype=torch.float8_e4m3fn),
        "experts.gate_up_proj_scale": torch.ones(2, 8),
        "experts.down_proj_scale": torch.ones(2, 4),
    }

    layer.load_state_dict(state, prefix="experts")

    assert layer.gate_up_proj.dtype == torch.float8_e4m3fn
    assert layer.down_proj.dtype == torch.float8_e4m3fn
    assert layer._weights_quantized
    assert not state
