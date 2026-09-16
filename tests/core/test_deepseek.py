"""DeepSeek-V2 / mini-deepseek integration tests.

The config test is pure CPU. The end-to-end benchmark lives in
``benchmark/offline/bench_deepseek.py`` because loading the 30 GiB checkpoint
takes minutes.
"""

from __future__ import annotations

import os

import pytest

DEEPSEEK_V2_LITE = "/home/yzd/models/DeepSeek-V2-Lite"


@pytest.mark.skipif(
    not os.path.exists(DEEPSEEK_V2_LITE), reason="DeepSeek-V2-Lite is not available"
)
def test_deepseek_v2_lite_config_maps_to_mla_moe() -> None:
    from transformers import AutoConfig

    from minisgl.models.config import ModelConfig

    hf_config = AutoConfig.from_pretrained(DEEPSEEK_V2_LITE, trust_remote_code=False)
    config = ModelConfig.from_hf(hf_config)

    assert config.is_mla
    assert config.is_moe
    assert (config.kv_lora_rank, config.qk_nope_head_dim, config.qk_rope_head_dim) == (512, 128, 64)
    assert config.v_head_dim == 128
    assert (config.num_experts, config.n_shared_experts, config.num_experts_per_tok) == (64, 2, 6)
    assert config.moe_intermediate_size == 1408
    assert config.first_k_dense_replace == 1
    assert config.moe_layer_freq == 1
    assert not config.norm_topk_prob
