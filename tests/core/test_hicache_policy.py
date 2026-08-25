from __future__ import annotations

from minisgl.kvcache import CacheTier, TransferDirection
from minisgl.scheduler.cache import HiCacheCostModel


def _cost_model(**overrides) -> HiCacheCostModel:
    kwargs = {
        "policy": "cost",
        "recompute_us_per_token": 100.0,
        "host_bandwidth_gib_s": 100.0,
        "storage_bandwidth_gib_s": 0.01,
        "margin": 1.1,
    }
    kwargs.update(overrides)
    return HiCacheCostModel(**kwargs)


def test_always_policy_selects_longest_tier():
    model = _cost_model(policy="always")
    assert model.select_restore(
        cuda_len=16,
        host_len=64,
        storage_len=128,
        bytes_per_token=1024,
    ) == (CacheTier.STORAGE, 128)


def test_cost_policy_can_prefer_shorter_host_hit_over_slow_storage():
    model = _cost_model()
    assert model.select_restore(
        cuda_len=0,
        host_len=100,
        storage_len=200,
        bytes_per_token=1024,
    ) == (CacheTier.HOST, 100)


def test_cost_policy_skips_restore_when_recompute_is_cheaper():
    model = _cost_model(
        recompute_us_per_token=0.01,
        host_bandwidth_gib_s=0.01,
        storage_bandwidth_gib_s=0.01,
    )
    assert model.select_restore(
        cuda_len=8,
        host_len=128,
        storage_len=256,
        bytes_per_token=4096,
    ) == (CacheTier.GPU, 8)


def test_prefill_observations_update_recompute_prior():
    model = _cost_model(recompute_us_per_token=10.0)
    before = model.recompute_seconds_per_token
    model.observe_prefill(tokens=100, seconds=0.1)
    assert model.recompute_seconds_per_token > before
    assert model.prefill_observations == 1


def test_backup_admission_charges_only_future_restore_path():
    model = _cost_model(
        recompute_us_per_token=100.0,
        host_bandwidth_gib_s=100.0,
        storage_bandwidth_gib_s=0.005,
    )
    assert model.should_backup(tier=CacheTier.HOST, tokens=128, bytes_per_token=1024)
    assert not model.should_backup(tier=CacheTier.STORAGE, tokens=128, bytes_per_token=1024)


def test_storage_write_observation_does_not_poison_read_prior():
    model = _cost_model()
    model._observed_seconds_per_byte[TransferDirection.H2S] = 1.0
    assert model._estimate_transfer(TransferDirection.S2H, 1024) < 1.0
