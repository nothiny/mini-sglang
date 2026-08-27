from __future__ import annotations

import torch
from minisgl.kvcache import CacheTier, HiRadixTree


def _tree(page_size: int = 2) -> HiRadixTree:
    cpu = torch.device("cpu")
    return HiRadixTree(
        page_size=page_size,
        tier_devices={
            CacheTier.GPU: cpu,
            CacheTier.HOST: cpu,
            CacheTier.STORAGE: cpu,
        },
    )


def _indices(start: int, length: int) -> torch.Tensor:
    return torch.arange(start, start + length, dtype=torch.int32)


def test_tiers_share_topology_but_keep_independent_residency() -> None:
    tree = _tree()
    input_ids = _indices(10, 8)

    tree.view(CacheTier.GPU).insert_prefix(input_ids, _indices(0, 8))
    tree.view(CacheTier.HOST).insert_prefix(input_ids[:6], _indices(100, 6))
    tree.view(CacheTier.STORAGE).insert_prefix(input_ids, _indices(200, 8))

    match = tree.match_prefixes(input_ids, include_storage=True)
    gpu = match.cuda_handle
    assert match.host_handle is not None and match.storage_handle is not None
    host = match.host_handle
    storage = match.storage_handle
    assert gpu.cached_len == 8
    assert host.cached_len == 6
    assert storage.cached_len == 8
    assert gpu.get_matched_indices().tolist() == list(range(8))
    assert host.get_matched_indices().tolist() == list(range(100, 106))
    assert storage.get_matched_indices().tolist() == list(range(200, 208))

    evicted = tree.view(CacheTier.GPU).evict(2)
    assert evicted.tolist() == [6, 7]
    assert tree.view(CacheTier.GPU).match_prefix(input_ids).cuda_handle.cached_len == 6
    assert tree.view(CacheTier.HOST).match_prefix(input_ids).cuda_handle.cached_len == 6
    assert tree.view(CacheTier.STORAGE).match_prefix(input_ids).cuda_handle.cached_len == 8
    tree.check_integrity()


def test_split_preserves_every_tier_and_locked_handle() -> None:
    tree = _tree()
    input_ids = _indices(10, 8)
    for tier, start in (
        (CacheTier.GPU, 0),
        (CacheTier.HOST, 100),
        (CacheTier.STORAGE, 200),
    ):
        tree.view(tier).insert_prefix(input_ids, _indices(start, 8))

    original = tree.view(CacheTier.GPU).match_prefix(input_ids).cuda_handle
    tree.view(CacheTier.GPU).lock_handle(original)
    divergent = torch.tensor([10, 11, 12, 13, 90, 91], dtype=torch.int32)
    shorter = tree.view(CacheTier.HOST).match_prefix(divergent).cuda_handle

    assert shorter.cached_len == 4
    assert shorter.get_matched_indices().tolist() == [100, 101, 102, 103]
    assert original.get_matched_indices().tolist() == list(range(8))
    assert tree.view(CacheTier.GPU).size_info.protected_size == 8
    tree.view(CacheTier.GPU).lock_handle(original, unlock=True)
    assert tree.view(CacheTier.GPU).size_info.evictable_size == 8
    tree.check_integrity()


def test_tier_is_removed_only_after_its_own_eviction() -> None:
    tree = _tree()
    input_ids = _indices(10, 4)
    for tier, start in (
        (CacheTier.GPU, 0),
        (CacheTier.HOST, 100),
        (CacheTier.STORAGE, 200),
    ):
        tree.view(tier).insert_prefix(input_ids, _indices(start, 4))

    tree.view(CacheTier.GPU).evict(4)
    assert tree.view(CacheTier.GPU).match_prefix(input_ids).cuda_handle.cached_len == 0
    assert tree.view(CacheTier.HOST).match_prefix(input_ids).cuda_handle.cached_len == 4
    assert tree.view(CacheTier.STORAGE).match_prefix(input_ids).cuda_handle.cached_len == 4
    assert tree.root_node.children

    tree.view(CacheTier.HOST).evict(4)
    assert tree.view(CacheTier.STORAGE).match_prefix(input_ids).cuda_handle.cached_len == 4
    assert tree.root_node.children

    tree.view(CacheTier.STORAGE).evict(4)
    assert not tree.root_node.children
    tree.check_integrity()


def test_duplicate_insert_reports_existing_prefix_per_tier() -> None:
    tree = _tree()
    input_ids = _indices(10, 8)
    tree.view(CacheTier.GPU).insert_prefix(input_ids, _indices(0, 8))
    tree.view(CacheTier.HOST).insert_prefix(input_ids[:4], _indices(100, 4))

    inserted = tree.view(CacheTier.HOST).insert_prefix(input_ids, _indices(100, 8))
    assert inserted.cached_len == 4
    assert inserted.handle.get_matched_indices().tolist() == list(range(100, 108))
    assert tree.view(CacheTier.GPU).match_prefix(input_ids).cuda_handle.cached_len == 8
    tree.check_integrity()


def test_each_tier_maintains_an_independent_lru_order() -> None:
    tree = _tree()
    prefix_a = _indices(10, 4)
    prefix_b = _indices(20, 4)
    tree.view(CacheTier.GPU).insert_prefix(prefix_a, _indices(0, 4))
    tree.view(CacheTier.HOST).insert_prefix(prefix_a, _indices(100, 4))
    tree.view(CacheTier.GPU).insert_prefix(prefix_b, _indices(10, 4))
    tree.view(CacheTier.HOST).insert_prefix(prefix_b, _indices(110, 4))

    tree.view(CacheTier.GPU).match_prefix(prefix_a)
    tree.view(CacheTier.HOST).match_prefix(prefix_b)

    assert tree.view(CacheTier.GPU).evict(4).tolist() == list(range(10, 14))
    assert tree.view(CacheTier.HOST).evict(4).tolist() == list(range(100, 104))
    assert tree.view(CacheTier.GPU).match_prefix(prefix_a).cuda_handle.cached_len == 4
    assert tree.view(CacheTier.HOST).match_prefix(prefix_b).cuda_handle.cached_len == 4
    tree.check_integrity()
