from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo
from .eviction import (
    BaseEvictionPolicy,
    EvictionCandidate,
    EvictionContext,
    LRUPolicy,
    estimate_recompute_cost,
)

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]
_FNV_OFFSET_BASIS = 14695981039346656037
_FNV_PRIME = 1099511628211
_HASH_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class _GhostEntry:
    recompute_cost: float
    evicted_at_ns: int


class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()
        self.prefix_hash = _FNV_OFFSET_BASIS
        self.depth = 0

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        self.depth = parent.depth + self.length
        self.prefix_hash = _extend_prefix_hash(parent.prefix_hash, self._key)
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    def __init__(
        self,
        device: torch.device,
        eviction_policy: BaseEvictionPolicy | None = None,
        *,
        total_tokens: int = 0,
        kv_bytes_per_token: int = 1,
        ghost_capacity: int = 4096,
    ) -> None:
        super().__init__()
        if total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if kv_bytes_per_token < 1:
            raise ValueError("kv_bytes_per_token must be at least 1")
        if ghost_capacity < 0:
            raise ValueError("ghost_capacity must be non-negative")
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.total_tokens = total_tokens
        self.kv_bytes_per_token = kv_bytes_per_token
        self.eviction_policy = eviction_policy or LRUPolicy()
        self.ghost_capacity = ghost_capacity
        self._ghost_cache: OrderedDict[int, _GhostEntry] = OrderedDict()
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        now_ns = time.monotonic_ns()
        self._report_ghost_hits(input_ids, now_ns)
        node, prefix_len = self._tree_walk(input_ids, now_ns=now_ns, record_access=True)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        now_ns = time.monotonic_ns()
        self._report_ghost_hits(input_ids, now_ns)
        node, prefix_len = self._tree_walk(input_ids, now_ns=now_ns, record_access=False)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn, now_ns)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            self.eviction_policy.on_insert(new_node.uuid, now_ns)
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        if size < 0:
            raise ValueError("Eviction size must be non-negative")
        if size == 0:
            return self.empty_tensor
        if size > self.evictable_size:
            raise RuntimeError(f"Cannot evict {size}, only {self.evictable_size} is evictable")

        leaf_nodes = self._collect_leaf_nodes_for_evict()
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0
        initial_free_tokens = max(self.total_tokens - self.size_info.total_size, 0)

        while evicted_size < size:
            if not leaf_nodes:
                raise RuntimeError(
                    f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
                )
            now_ns = time.monotonic_ns()
            candidates = [self._make_candidate(node) for node in leaf_nodes]
            total_tokens = self.total_tokens or self.size_info.total_size
            context = EvictionContext(
                now_ns=now_ns,
                requested_tokens=size - evicted_size,
                free_tokens=initial_free_tokens + evicted_size,
                total_tokens=total_tokens,
            )
            victim_id = self.eviction_policy.select_victim(candidates, context)
            node = next((node for node in leaf_nodes if node.uuid == victim_id), None)
            if node is None:
                raise RuntimeError(
                    f"Eviction policy '{self.eviction_policy.name}' selected "
                    f"ineligible node {victim_id}"
                )
            leaf_nodes.remove(node)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                raise RuntimeError(f"Radix node {node.uuid} is not eligible for eviction")
            recompute_cost = estimate_recompute_cost(node.depth - node.length, node.length)
            self.eviction_policy.on_evict(
                node.uuid,
                node.prefix_hash,
                recompute_cost,
                now_ns,
            )
            self._remember_ghost(node.prefix_hash, recompute_cost, now_ns)
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                leaf_nodes.append(parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        if self.root_node.ref_count != 1 or not self.root_node.is_root():
            raise RuntimeError("Radix root must be protected and parentless")

        evictable_size = 0
        protected_size = 0
        seen_ids = set()
        nodes = [self.root_node]
        while nodes:
            node = nodes.pop()
            if node.uuid in seen_ids:
                raise RuntimeError(f"Radix node {node.uuid} is reachable more than once")
            seen_ids.add(node.uuid)

            for child_key, child in node.children.items():
                if child._parent is not node:
                    raise RuntimeError(f"Radix node {child.uuid} has an invalid parent")
                if child_key != self.key_fn(child._key):
                    raise RuntimeError(f"Radix node {child.uuid} has an invalid child key")
                if child.length != len(child._key) or child.length != len(child._value):
                    raise RuntimeError(f"Radix node {child.uuid} has inconsistent lengths")
                if child.length == 0 or child.length % self.page_size != 0:
                    raise RuntimeError(f"Radix node {child.uuid} is not page-aligned")
                if child.depth != node.depth + child.length:
                    raise RuntimeError(f"Radix node {child.uuid} has an invalid depth")
                expected_hash = _extend_prefix_hash(node.prefix_hash, child._key)
                if child.prefix_hash != expected_hash:
                    raise RuntimeError(f"Radix node {child.uuid} has an invalid prefix hash")
                if child.ref_count < 0:
                    raise RuntimeError(f"Radix node {child.uuid} has a negative ref_count")
                if child.ref_count == 0:
                    evictable_size += child.length
                else:
                    protected_size += child.length
                nodes.append(child)

        if evictable_size != self.evictable_size:
            raise RuntimeError(
                f"Radix evictable size mismatch: {evictable_size} != {self.evictable_size}"
            )
        if protected_size != self.protected_size:
            raise RuntimeError(
                f"Radix protected size mismatch: {protected_size} != {self.protected_size}"
            )

    def _collect_leaf_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leaf_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0 and not node.is_root():
                    leaf_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leaf_nodes

    def _tree_walk(
        self,
        input_ids: torch.Tensor,
        *,
        now_ns: int,
        record_access: bool,
    ) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                old_node_id = node.uuid
                node = node.split_at(match_len)
                self.eviction_policy.on_split(old_node_id, node.uuid, old_node_id)
                if record_access:
                    node.timestamp = now_ns
                    self.eviction_policy.on_access(node.uuid, now_ns)
                return node, prefix_len

            # update timestamp for accessed node
            if record_access:
                node.timestamp = now_ns
                self.eviction_policy.on_access(node.uuid, now_ns)

        return node, prefix_len

    def _make_candidate(self, node: RadixTreeNode) -> EvictionCandidate:
        return EvictionCandidate(
            node_id=node.uuid,
            segment_tokens=node.length,
            prefix_depth=node.depth - node.length,
            kv_bytes=node.length * self.kv_bytes_per_token,
            ref_count=node.ref_count,
        )

    def _remember_ghost(
        self,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        if not self.eviction_policy.uses_ghost_feedback or self.ghost_capacity == 0:
            return
        self._ghost_cache[prefix_hash] = _GhostEntry(recompute_cost, now_ns)
        self._ghost_cache.move_to_end(prefix_hash)
        while len(self._ghost_cache) > self.ghost_capacity:
            self._ghost_cache.popitem(last=False)

    def _report_ghost_hits(self, input_ids: torch.Tensor, now_ns: int) -> None:
        if not self.eviction_policy.uses_ghost_feedback or not self._ghost_cache:
            return
        prefix_hash = _FNV_OFFSET_BASIS
        for prefix_len, token in enumerate(input_ids.tolist(), start=1):
            prefix_hash = _extend_hash(prefix_hash, int(token))
            if prefix_len % self.page_size != 0:
                continue
            ghost = self._ghost_cache.pop(prefix_hash, None)
            if ghost is not None:
                self.eviction_policy.on_ghost_hit(
                    prefix_hash,
                    ghost.recompute_cost,
                    now_ns,
                )


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())


def _extend_hash(prefix_hash: int, token: int) -> int:
    return ((prefix_hash ^ (token & _HASH_MASK)) * _FNV_PRIME) & _HASH_MASK


def _extend_prefix_hash(prefix_hash: int, tokens: torch.Tensor) -> int:
    result = prefix_hash
    for token in tokens.tolist():
        result = _extend_hash(result, int(token))
    return result
