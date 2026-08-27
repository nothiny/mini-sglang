from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TypeAlias

import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo
from .tiered_pool import CacheTier

KeyFn: TypeAlias = Callable[[torch.Tensor], Any]


class HiRadixTreeNode:
    """One token-tree node with independent residency in every cache tier."""

    counter = 0

    def __init__(self, key_fn: KeyFn, timestamp: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, HiRadixTreeNode] = {}
        self.parent: HiRadixTreeNode | None = None
        self.key = torch.empty(0, dtype=torch.int32)
        self.values: Dict[CacheTier, torch.Tensor | None] = dict.fromkeys(CacheTier)
        self.ref_counts: Dict[CacheTier, int] = dict.fromkeys(CacheTier, 0)
        initial_timestamp = timestamp or time.monotonic_ns()
        self.timestamps: Dict[CacheTier, int] = dict.fromkeys(CacheTier, initial_timestamp)
        self.uuid = HiRadixTreeNode.counter
        HiRadixTreeNode.counter += 1

    @property
    def length(self) -> int:
        return len(self.key)

    def is_root(self) -> bool:
        return self.parent is None

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        return int(fast_compare_key(self.key, input_ids))


@dataclass(frozen=True)
class HiRadixCacheHandle(BaseCacheHandle):
    node: HiRadixTreeNode
    tier: CacheTier

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        values: List[torch.Tensor] = []
        while not node.is_root():
            value = node.values[self.tier]
            if value is None:
                raise RuntimeError(f"Stale {self.tier.value} HiCache handle")
            values.append(value)
            assert node.parent is not None
            node = node.parent
        if not values:
            root_value = node.values[self.tier]
            assert root_value is not None
            return root_value
        values.reverse()
        return torch.cat(values)


class HiRadixTree:
    """Shared token topology with tier-local values, locks, LRU, and accounting."""

    def __init__(
        self,
        *,
        page_size: int,
        tier_devices: Dict[CacheTier, torch.device],
    ) -> None:
        if page_size < 1:
            raise ValueError("HiRadixTree page size must be positive")
        self.page_size = page_size
        self.key_fn = _get_key_fn(page_size)
        self.tier_devices = dict(tier_devices)
        if set(self.tier_devices) != set(CacheTier):
            raise ValueError("HiRadixTree requires a device for every cache tier")
        self.evictable_sizes: Dict[CacheTier, int] = dict.fromkeys(CacheTier, 0)
        self.protected_sizes: Dict[CacheTier, int] = dict.fromkeys(CacheTier, 0)
        self.root_node = HiRadixTreeNode(self.key_fn)
        for tier, device in self.tier_devices.items():
            self.root_node.values[tier] = torch.empty(0, dtype=torch.int32, device=device)
            self.root_node.ref_counts[tier] = 1
        self._views = {tier: HiRadixPrefixCache(self, tier) for tier in CacheTier}

    def view(self, tier: CacheTier) -> HiRadixPrefixCache:
        return self._views[tier]

    def match_prefixes(self, input_ids: torch.Tensor, *, include_storage: bool) -> MatchResult:
        """Match every configured residency tier during one topology walk."""
        tiers = [CacheTier.GPU, CacheTier.HOST]
        if include_storage:
            tiers.append(CacheTier.STORAGE)
        lengths = dict.fromkeys(tiers, 0)
        last_nodes = dict.fromkeys(tiers, self.root_node)
        active = dict.fromkeys(tiers, True)

        node = self.root_node
        topology_len = 0
        while topology_len < len(input_ids) and any(active.values()):
            child = node.children.get(self.key_fn(input_ids[topology_len:]))
            if child is None:
                break
            match_len = _align_down(child.get_match_len(input_ids[topology_len:]), self.page_size)
            if match_len == 0:
                break
            if match_len < child.length:
                child = self._split_node(child, match_len)
            node = child
            topology_len += node.length
            now = time.monotonic_ns()
            for tier in tiers:
                if not active[tier]:
                    continue
                if node.values[tier] is None:
                    active[tier] = False
                    continue
                lengths[tier] += node.length
                last_nodes[tier] = node
                node.timestamps[tier] = now

        storage_handle = (
            HiRadixCacheHandle(
                lengths[CacheTier.STORAGE],
                last_nodes[CacheTier.STORAGE],
                CacheTier.STORAGE,
            )
            if include_storage
            else None
        )
        return MatchResult(
            HiRadixCacheHandle(lengths[CacheTier.GPU], last_nodes[CacheTier.GPU], CacheTier.GPU),
            HiRadixCacheHandle(lengths[CacheTier.HOST], last_nodes[CacheTier.HOST], CacheTier.HOST),
            storage_handle,
        )

    def match_prefix(self, tier: CacheTier, input_ids: torch.Tensor) -> HiRadixCacheHandle:
        node = self.root_node
        last_resident = node
        prefix_len = 0
        while prefix_len < len(input_ids):
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                break
            match_len = _align_down(child.get_match_len(input_ids[prefix_len:]), self.page_size)
            if match_len == 0:
                break
            if match_len < child.length:
                child = self._split_node(child, match_len)
            node = child
            if node.values[tier] is None:
                break
            prefix_len += node.length
            node.timestamps[tier] = time.monotonic_ns()
            last_resident = node
            if match_len < child.length:
                break
        return HiRadixCacheHandle(prefix_len, last_resident, tier)

    def insert_prefix(
        self,
        tier: CacheTier,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> InsertResult:
        insert_len = _align_down(len(input_ids), self.page_size)
        input_ids = input_ids[:insert_len]
        indices = indices[:insert_len]
        if len(indices) != insert_len:
            raise ValueError("HiRadixTree key/value lengths differ")
        if insert_len == 0:
            return InsertResult(0, HiRadixCacheHandle(0, self.root_node, tier))

        node = self.root_node
        offset = 0
        existing_prefix = 0
        found_missing = False
        while offset < insert_len:
            child_key = self.key_fn(input_ids[offset:])
            child = node.children.get(child_key)
            if child is None:
                child = HiRadixTreeNode(self.key_fn)
                child.parent = node
                child.key = input_ids[offset:].clone()
                child.values[tier] = indices[offset:].clone()
                child.timestamps[tier] = time.monotonic_ns()
                node.children[child_key] = child
                self.evictable_sizes[tier] += child.length
                node = child
                offset = insert_len
                found_missing = True
                break

            match_len = _align_down(child.get_match_len(input_ids[offset:]), self.page_size)
            if match_len == 0:
                raise RuntimeError("HiRadixTree child key matched less than one page")
            if match_len < child.length:
                child = self._split_node(child, match_len)
            node = child

            value = node.values[tier]
            if value is None:
                node.values[tier] = indices[offset : offset + node.length].clone()
                node.timestamps[tier] = time.monotonic_ns()
                self.evictable_sizes[tier] += node.length
                found_missing = True
            else:
                if found_missing:
                    raise RuntimeError(
                        f"HiRadixTree {tier.value} residency contains an internal hole"
                    )
                existing_prefix += node.length
                node.timestamps[tier] = time.monotonic_ns()
            offset += node.length

        return InsertResult(existing_prefix, HiRadixCacheHandle(insert_len, node, tier))

    def lock_handle(self, handle: BaseCacheHandle, *, unlock: bool) -> None:
        if not isinstance(handle, HiRadixCacheHandle):
            raise TypeError("HiRadixTree received an incompatible cache handle")
        tier = handle.tier
        node = handle.node
        while not node.is_root():
            value = node.values[tier]
            if value is None:
                raise RuntimeError(f"Cannot lock non-resident {tier.value} HiCache node")
            ref_count = node.ref_counts[tier]
            if unlock:
                if ref_count <= 0:
                    raise RuntimeError(f"Cannot unlock an unlocked {tier.value} HiCache node")
                node.ref_counts[tier] = ref_count - 1
                if ref_count == 1:
                    self.protected_sizes[tier] -= node.length
                    self.evictable_sizes[tier] += node.length
            else:
                if ref_count == 0:
                    self.evictable_sizes[tier] -= node.length
                    self.protected_sizes[tier] += node.length
                node.ref_counts[tier] = ref_count + 1
            assert node.parent is not None
            node = node.parent

    def evict(self, tier: CacheTier, size: int) -> torch.Tensor:
        if size == 0:
            root_value = self.root_node.values[tier]
            assert root_value is not None
            return root_value
        if size > self.evictable_sizes[tier]:
            raise RuntimeError(
                f"Cannot evict {size} {tier.value} tokens, only "
                f"{self.evictable_sizes[tier]} are evictable"
            )

        leaves = self._collect_tier_leaves(tier)
        heap: List[tuple[int, int, HiRadixTreeNode]] = [
            (node.timestamps[tier], node.uuid, node) for node in leaves
        ]
        heapq.heapify(heap)
        evicted: List[torch.Tensor] = []
        evicted_size = 0
        while evicted_size < size:
            if not heap:
                raise RuntimeError(f"Cannot evict enough {tier.value} HiCache pages")
            _, _, node = heapq.heappop(heap)
            value = node.values[tier]
            if value is None or node.ref_counts[tier] != 0 or self._has_tier_child(node, tier):
                continue
            evicted.append(value)
            evicted_size += node.length
            self.evictable_sizes[tier] -= node.length
            node.values[tier] = None
            parent = node.parent
            assert parent is not None
            self._prune_empty_leaf(node)
            if (
                not parent.is_root()
                and parent.values[tier] is not None
                and parent.ref_counts[tier] == 0
                and not self._has_tier_child(parent, tier)
            ):
                heapq.heappush(heap, (parent.timestamps[tier], parent.uuid, parent))
        return torch.cat(evicted)

    def reset_tier(self, tier: CacheTier) -> None:
        stack = list(self.root_node.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            node.values[tier] = None
            node.ref_counts[tier] = 0
        self.evictable_sizes[tier] = 0
        self.protected_sizes[tier] = 0
        self._prune_tree()

    def check_integrity(self) -> None:
        evictable = dict.fromkeys(CacheTier, 0)
        protected = dict.fromkeys(CacheTier, 0)
        owned: Dict[CacheTier, List[int]] = {tier: [] for tier in CacheTier}
        nodes = [self.root_node]
        for tier in CacheTier:
            if self.root_node.ref_counts[tier] != 1:
                raise RuntimeError(f"HiRadixTree {tier.value} root lock is corrupted")

        while nodes:
            parent = nodes.pop()
            for child_key, child in parent.children.items():
                if child.parent is not parent:
                    raise RuntimeError("HiRadixTree parent link is corrupted")
                if self.key_fn(child.key) != child_key:
                    raise RuntimeError("HiRadixTree child key is corrupted")
                if child.length <= 0 or child.length % self.page_size != 0:
                    raise RuntimeError("HiRadixTree node is not page aligned")
                for tier in CacheTier:
                    value = child.values[tier]
                    ref_count = child.ref_counts[tier]
                    if ref_count < 0:
                        raise RuntimeError("HiRadixTree node has a negative reference count")
                    if value is None:
                        if ref_count != 0:
                            raise RuntimeError(f"Non-resident {tier.value} HiCache node is locked")
                        continue
                    if len(value) != child.length:
                        raise RuntimeError(f"HiRadixTree {tier.value} key/value lengths differ")
                    if not parent.is_root() and parent.values[tier] is None:
                        raise RuntimeError(
                            f"HiRadixTree {tier.value} residency contains an internal hole"
                        )
                    if ref_count == 0:
                        evictable[tier] += child.length
                    else:
                        protected[tier] += child.length
                    owned[tier].extend(value.detach().cpu().tolist())
                nodes.append(child)

        for tier in CacheTier:
            if (
                evictable[tier] != self.evictable_sizes[tier]
                or protected[tier] != self.protected_sizes[tier]
            ):
                raise RuntimeError(
                    f"HiRadixTree {tier.value} size accounting mismatch: expected "
                    f"({self.evictable_sizes[tier]}, {self.protected_sizes[tier]}), found "
                    f"({evictable[tier]}, {protected[tier]})"
                )
            if len(owned[tier]) != len(set(owned[tier])):
                raise RuntimeError(f"HiRadixTree {tier.value} contains duplicate physical indices")

    def _split_node(self, child: HiRadixTreeNode, position: int) -> HiRadixTreeNode:
        if not 0 < position < child.length:
            raise ValueError("HiRadixTree split position is outside the node")
        parent = child.parent
        assert parent is not None
        old_child_key = self.key_fn(child.key)
        new_node = HiRadixTreeNode(self.key_fn)
        new_node.parent = parent
        new_node.key = child.key[:position].clone()
        for tier in CacheTier:
            value = child.values[tier]
            if value is not None:
                new_node.values[tier] = value[:position].clone()
                child.values[tier] = value[position:].clone()
            new_node.ref_counts[tier] = child.ref_counts[tier]
            new_node.timestamps[tier] = child.timestamps[tier]

        child.key = child.key[position:].clone()
        child.parent = new_node
        new_node.children[self.key_fn(child.key)] = child
        parent.children[old_child_key] = new_node
        return new_node

    def _collect_tier_leaves(self, tier: CacheTier) -> List[HiRadixTreeNode]:
        leaves: List[HiRadixTreeNode] = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if (
                not node.is_root()
                and node.values[tier] is not None
                and node.ref_counts[tier] == 0
                and not self._has_tier_child(node, tier)
            ):
                leaves.append(node)
        return leaves

    @staticmethod
    def _has_tier_child(node: HiRadixTreeNode, tier: CacheTier) -> bool:
        return any(child.values[tier] is not None for child in node.children.values())

    def _prune_empty_leaf(self, node: HiRadixTreeNode) -> None:
        while (
            not node.is_root()
            and not node.children
            and all(node.values[tier] is None and node.ref_counts[tier] == 0 for tier in CacheTier)
        ):
            parent = node.parent
            assert parent is not None
            parent.children.pop(self.key_fn(node.key), None)
            node = parent

    def _prune_tree(self) -> None:
        def visit(node: HiRadixTreeNode) -> None:
            for child in list(node.children.values()):
                visit(child)
                self._prune_empty_leaf(child)

        visit(self.root_node)


class HiRadixPrefixCache(BasePrefixCache):
    """Tier-specific view over a shared :class:`HiRadixTree`."""

    def __init__(self, tree: HiRadixTree, tier: CacheTier) -> None:
        self.tree = tree
        self.tier = tier

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        if not isinstance(handle, HiRadixCacheHandle) or handle.tier != self.tier:
            raise TypeError(f"Expected a {self.tier.value} HiCache handle")
        self.tree.lock_handle(handle, unlock=unlock)

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(self.tree.match_prefix(self.tier, input_ids))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return self.tree.insert_prefix(self.tier, input_ids, indices)

    def evict(self, size: int) -> torch.Tensor:
        return self.tree.evict(self.tier, size)

    def reset(self) -> None:
        self.tree.reset_tier(self.tier)

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.tree.evictable_sizes[self.tier],
            protected_size=self.tree.protected_sizes[self.tier],
        )

    def check_integrity(self) -> None:
        self.tree.check_integrity()


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _get_key_fn(page_size: int) -> KeyFn:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())


__all__ = [
    "HiRadixCacheHandle",
    "HiRadixPrefixCache",
    "HiRadixTree",
    "HiRadixTreeNode",
]
