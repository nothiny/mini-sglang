from __future__ import annotations

from typing import Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseMoeBackend
from .config import (
    MoeBackendConfig,
    get_moe_backend_config,
    set_moe_backend_config,
)

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    def __call__(self, config: MoeBackendConfig | None = None) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend(config: MoeBackendConfig | None = None) -> BaseMoeBackend:
    from .fused import FusedMoe

    return FusedMoe(config=config)


def create_moe_backend(
    backend: str,
    config: MoeBackendConfig | None = None,
) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend](config)


__all__ = [
    "BaseMoeBackend",
    "MoeBackendConfig",
    "create_moe_backend",
    "get_moe_backend_config",
    "set_moe_backend_config",
    "SUPPORTED_MOE_BACKENDS",
]
