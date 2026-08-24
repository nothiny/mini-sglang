from __future__ import annotations

import torch
from minisgl.models.weight import _get_expert_stack_info, _shard_tensor


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
