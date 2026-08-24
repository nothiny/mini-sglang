from typing import Iterator

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.moe import get_moe_backend_config
from minisgl.moe.weights import quantize_expert_weight
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        moe_config = get_moe_backend_config()
        self.tp_size = tp_size = tp_info.size
        self.ep_size = moe_config.expert_parallel_size
        self.ep_rank = moe_config.expert_parallel_rank
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self._moe_config = moe_config
        self._gate_up_scale: torch.Tensor | None = None
        self._down_scale: torch.Tensor | None = None
        self._weights_prepared = False
        if self.ep_size > 1:
            local_num_experts = div_even(num_experts, self.ep_size)
            intermediate_size_per_partition = intermediate_size
        else:
            local_num_experts = num_experts
            intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        self.gate_up_proj = torch.empty(
            local_num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
        )
        self.down_proj = torch.empty(
            local_num_experts,
            hidden_size,
            intermediate_size_per_partition,
        )

    def prepare_weights(self, device: torch.device) -> None:
        if self._weights_prepared:
            return
        if self._moe_config.expert_quantization == "int8":
            self.gate_up_proj, self._gate_up_scale = quantize_expert_weight(self.gate_up_proj)
            self.down_proj, self._down_scale = quantize_expert_weight(self.down_proj)
        if self._moe_config.expert_offload:
            self.gate_up_proj = self.gate_up_proj.cpu().pin_memory()
            self.down_proj = self.down_proj.cpu().pin_memory()
            if self._gate_up_scale is not None:
                self._gate_up_scale = self._gate_up_scale.cpu().pin_memory()
            if self._down_scale is not None:
                self._down_scale = self._down_scale.cpu().pin_memory()
        else:
            self.gate_up_proj = self.gate_up_proj.to(device)
            self.down_proj = self.down_proj.to(device)
            if self._gate_up_scale is not None:
                self._gate_up_scale = self._gate_up_scale.to(device)
            if self._down_scale is not None:
                self._down_scale = self._down_scale.to(device)
        self._weights_prepared = True

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        ctx = get_global_ctx()
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            w1_scale=self._gate_up_scale,
            w2_scale=self._down_scale,
        )
        if self.tp_size > 1 and self.ep_size == 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states


def _iter_moe_layers(root: BaseOP) -> Iterator[MoELayer]:
    seen: set[int] = set()

    def visit(value: object) -> Iterator[MoELayer]:
        object_id = id(value)
        if object_id in seen:
            return
        seen.add(object_id)
        if isinstance(value, MoELayer):
            yield value
            return
        if isinstance(value, BaseOP):
            for child in vars(value).values():
                yield from visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from visit(child)

    yield from visit(root)


def prepare_moe_weights(root: BaseOP, device: torch.device) -> None:
    """Apply quantization/offload after checkpoint tensors have been loaded."""

    for layer in _iter_moe_layers(root):
        layer.prepare_weights(device)


def get_moe_expert_cache_bytes(root: BaseOP, capacity: int) -> int:
    """Return the exact persistent tensor bytes reserved by GPU expert caches."""

    total = 0
    for layer in _iter_moe_layers(root):
        local_capacity = min(capacity, layer.gate_up_proj.shape[0])
        tensors = (
            layer.gate_up_proj,
            layer.down_proj,
            layer._gate_up_scale,
            layer._down_scale,
        )
        total += sum(local_capacity * tensor[0].nbytes for tensor in tensors if tensor is not None)
        total += layer.gate_up_proj.shape[0] * torch.int64.itemsize
    return total


__all__ = ["MoELayer", "get_moe_expert_cache_bytes", "prepare_moe_weights"]
