from __future__ import annotations

from types import SimpleNamespace

import torch
from minisgl.distributed import DistributedInfo
from minisgl.models import weight as weight_module
from minisgl.models.config import ModelConfig
from minisgl.models.weight import (
    _get_expert_stack_info,
    _local_expert_index,
    _local_expert_indices,
    _shard_tensor,
)
from minisgl.moe import MoeBackendConfig
from safetensors.torch import save_file


def test_expert_parallel_keeps_full_owned_expert_tensor() -> None:
    value = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    key = "model.layers.0.mlp.experts.3.gate_proj.weight"

    tp_shard = _shard_tensor(key, value, r=1, n=2, num_kv_heads=2)
    ep_shard = _shard_tensor(
        key,
        value,
        r=1,
        n=2,
        num_kv_heads=2,
        expert_parallel=True,
    )

    torch.testing.assert_close(tp_shard, value.chunk(2, dim=0)[1])
    torch.testing.assert_close(ep_shard, value)


def test_expert_checkpoint_name_maps_to_local_packed_weight() -> None:
    assert _get_expert_stack_info("model.layers.0.mlp.experts.3.down_proj.weight") == (
        "model.layers.0.mlp.experts.down_proj",
        3,
    )


def test_round_robin_expert_placement_maps_global_to_local() -> None:
    assert (
        _local_expert_index(5, ep_size=2, ep_rank=1, local_experts=4, placement="round-robin") == 2
    )
    assert (
        _local_expert_index(4, ep_size=2, ep_rank=1, local_experts=4, placement="round-robin")
        is None
    )

    assert _local_expert_indices(
        5,
        ep_size=2,
        ep_rank=1,
        local_experts=4,
        placement="round-robin",
        replicated_experts=(5,),
    ) == (2, 4)


def test_weight_loader_quantizes_packed_experts_before_state_dict(tmp_path, monkeypatch) -> None:
    tensors = {}
    for expert_id in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert_id}"
        tensors[f"{prefix}.gate_proj.weight"] = torch.randn(4, 4)
        tensors[f"{prefix}.up_proj.weight"] = torch.randn(4, 4)
        tensors[f"{prefix}.down_proj.weight"] = torch.randn(4, 4)
    save_file(tensors, tmp_path / "model.safetensors")

    model_config = SimpleNamespace(is_moe=True, num_experts=2, num_kv_heads=1)
    monkeypatch.setattr(weight_module, "download_hf_weight", lambda _: str(tmp_path))
    monkeypatch.setattr(weight_module, "cached_load_hf_config", lambda _: object())
    monkeypatch.setattr(ModelConfig, "from_hf", classmethod(lambda cls, _: model_config))
    monkeypatch.setattr(weight_module, "get_tp_info", lambda: DistributedInfo(0, 1))
    monkeypatch.setattr(
        weight_module,
        "get_moe_backend_config",
        lambda: MoeBackendConfig(expert_quantization="fp8"),
    )

    loaded = dict(weight_module.load_weight("unused", torch.device("cpu")))

    gate_up_key = "model.layers.0.mlp.experts.gate_up_proj"
    down_key = "model.layers.0.mlp.experts.down_proj"
    assert loaded[gate_up_key].shape == (2, 8, 4)
    assert loaded[down_key].shape == (2, 4, 4)
    assert loaded[gate_up_key].dtype == torch.float8_e4m3fn
    assert loaded[down_key].dtype == torch.float8_e4m3fn
    assert loaded[f"{gate_up_key}_scale"].shape == (2, 8)
    assert loaded[f"{down_key}_scale"].shape == (2, 4)
