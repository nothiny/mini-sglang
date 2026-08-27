from __future__ import annotations

import pytest

from minisgl.kvcache import CacheTier, TransferDirection, TransferTicket
from minisgl.kvcache.tiered_pool import TransferState
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
    before = model._estimate_recompute(100)
    model.observe_prefill(tokens=100, seconds=0.1)
    assert model._estimate_recompute(100) > before
    assert model.recompute_fixed_seconds > 0
    assert model.prefill_observations == 1


def test_affine_cost_model_learns_short_recompute_long_restore_crossover():
    model = _cost_model(
        recompute_us_per_token=20.0,
        host_bandwidth_gib_s=1 / (5e-6 * (1 << 10)),
    )
    bytes_per_token = 1024
    for tokens in (32, 512, 32, 512):
        model.observe_prefill(tokens=tokens, seconds=0.001 + tokens * 20e-6)
        num_bytes = tokens * bytes_per_token
        model.observe_transfer(
            _completed_transfer(
                TransferDirection.H2D,
                num_bytes,
                seconds=0.004 + tokens * 5e-6,
            )
        )

    assert model.select_restore(
        cuda_len=0,
        host_len=32,
        storage_len=0,
        bytes_per_token=bytes_per_token,
    ) == (CacheTier.GPU, 0)
    assert model.select_restore(
        cuda_len=0,
        host_len=512,
        storage_len=0,
        bytes_per_token=bytes_per_token,
    ) == (CacheTier.HOST, 512)
    snapshot = model.snapshot()
    assert float(snapshot["recompute_fixed_ms"]) == pytest.approx(1.0)
    assert float(snapshot["estimated_h2d_fixed_ms"]) == pytest.approx(4.0)


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


def _completed_transfer(
    direction: TransferDirection, num_bytes: int, *, seconds: float
) -> TransferTicket:
    source = CacheTier.HOST
    destination = CacheTier.GPU
    ticket = TransferTicket(
        direction=direction,
        source=source,
        destination=destination,
        pages=1,
        num_bytes=num_bytes,
        created_at=0.0,
        submitted_at=0.0,
        started_at=0.0,
        work_completed_at=seconds,
        completed_at=seconds,
        state=TransferState.COMPLETED,
    )
    return ticket
