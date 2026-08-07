# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional RL-framework adapters."""

from collections.abc import Callable, Mapping
from types import MappingProxyType

from rlxfer.errors import MissingDependencyError

from .base import ExperienceAdapter, IncompatibleExperienceError
from .miles import MilesAdapter
from .nemo_rl import NemoRLAdapter
from .prime_rl import PrimeRLAdapter
from .slime import SlimeAdapter

AdapterFactory = Callable[[], ExperienceAdapter]


class AdapterRegistry:
    """Instance-scoped framework-adapter registry."""

    def __init__(self, factories: Mapping[str, AdapterFactory] | None = None) -> None:
        self._factories = dict(factories or {})

    def register(self, name: str, factory: AdapterFactory) -> None:
        """Register one factory without modifying framework or transport code."""
        if not name or name in self._factories:
            raise ValueError(f"adapter {name!r} is already registered or invalid")
        self._factories[name] = factory

    def create(self, name: str) -> ExperienceAdapter:
        """Create an adapter by registered framework name."""
        try:
            factory = self._factories[name]
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(
                f"unknown adapter {name!r}; registered adapters: {available}"
            ) from error
        return factory()


def default_adapter_registry() -> AdapterRegistry:
    """Return a fresh registry containing the four built-in adapters."""
    factories: Mapping[str, AdapterFactory] = MappingProxyType(
        {
            "miles": MilesAdapter,
            "nemo_rl": NemoRLAdapter,
            "prime_rl": PrimeRLAdapter,
            "slime": SlimeAdapter,
        }
    )
    return AdapterRegistry(factories)


def create_adapter(name: str, registry: AdapterRegistry | None = None) -> ExperienceAdapter:
    """Create a built-in or caller-registered framework adapter."""
    return (registry or default_adapter_registry()).create(name)


__all__ = [
    "AdapterFactory",
    "AdapterRegistry",
    "ExperienceAdapter",
    "IncompatibleExperienceError",
    "MilesAdapter",
    "MissingDependencyError",
    "NemoRLAdapter",
    "PrimeRLAdapter",
    "SlimeAdapter",
    "create_adapter",
    "default_adapter_registry",
]
