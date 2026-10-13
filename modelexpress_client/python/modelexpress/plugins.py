# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover trusted external loading strategies in a private module namespace."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
import threading
import uuid
from pathlib import Path
from types import ModuleType

import torch.nn as nn

from . import envs
from .load_strategy import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    StrategyFailed,
    StrategyRecoveryError,
)

logger = logging.getLogger("modelexpress.plugins")

plugins: dict[str, type[LoadStrategy]] = {}
_loaded_directories: dict[Path, ModuleType] = {}
_lock = threading.RLock()


def register_plugin(name: str, strategy_class: type[LoadStrategy]) -> None:
    """Register a concrete, no-argument strategy class under its declared name."""
    with _lock:
        if name in plugins:
            raise ValueError(f"Plugin {name!r} is already registered")
        if (
            not inspect.isclass(strategy_class)
            or not issubclass(strategy_class, LoadStrategy)
            or inspect.isabstract(strategy_class)
        ):
            raise TypeError(f"Plugin {name!r} must return a concrete LoadStrategy class")
        if not name or getattr(strategy_class, "name", None) != name:
            raise ValueError(f"Plugin {name!r} must match the strategy class name")
        try:
            inspect.signature(strategy_class).bind()
        except (ValueError, TypeError) as exc:
            raise TypeError(f"Plugin {name!r} must support no-argument construction") from exc
        plugins[name] = strategy_class


def validate_plugin_names(builtin_names: set[str]) -> None:
    """Reject external names reserved by the loading chain."""
    with _lock:
        conflicts = builtin_names.intersection(plugins)
        if conflicts:
            raise ValueError(f"Plugins conflict with built-in strategies: {sorted(conflicts)}")


def initialize_plugins() -> None:
    """Import and validate each directory once per worker."""
    with _lock:
        directory = Path(envs.MX_PLUGIN_DIR).resolve()
        if directory in _loaded_directories or not directory.exists():
            return
        if not directory.is_dir():
            raise ValueError(f"Plugin directory is not a directory: {directory}")
        namespace = f"_mx_plugins_{uuid.uuid4().hex}"
        package = ModuleType(namespace)
        package.__path__ = [str(directory)]
        package.__package__ = namespace
        sys.modules[namespace] = package
        previous = dict(plugins)
        try:
            for path in sorted(directory.glob("mx_plugin_*.py")):
                name = path.stem.removeprefix("mx_plugin_")
                module_name = f"{namespace}.{path.stem}"
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load plugin entry {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                entry = getattr(module, "register_plugin", None)
                if not callable(entry):
                    raise TypeError(f"Plugin entry {path} must export register_plugin()")
                try:
                    inspect.signature(entry).bind()
                except (ValueError, TypeError) as exc:
                    raise TypeError(f"Plugin entry {path} must accept no arguments") from exc
                register_plugin(name, entry())
            _loaded_directories[directory] = package
            for name in sorted(plugins.keys() - previous.keys()):
                logger.info(
                    "Loaded loading strategy plugin %r from %s",
                    name,
                    directory / f"mx_plugin_{name}.py",
                )
        except BaseException:
            plugins.clear()
            plugins.update(previous)
            for module_name in list(sys.modules):
                if module_name == namespace or module_name.startswith(namespace + "."):
                    del sys.modules[module_name]
            raise


class _PluginStrategy(LoadStrategy):
    def __init__(self, strategy: LoadStrategy):
        self._strategy = strategy

    @property
    def name(self) -> str:
        return self._strategy.name

    def is_available(self, ctx: LoadContext) -> bool:
        return self._strategy.is_available(ctx)

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        try:
            loaded = self._strategy.load(result, ctx)
            if (
                not isinstance(loaded, LoadResult)
                or loaded.value is None
                or (loaded.model is not None and not isinstance(loaded.model, nn.Module))
                or not isinstance(loaded.publishable, bool)
                or not isinstance(loaded.metadata, dict)
            ):
                raise TypeError(f"Plugin {self.name!r} must return a valid LoadResult")
            return loaded
        except (StrategyFailed, StrategyRecoveryError):
            raise
        except Exception as exc:
            raise StrategyFailed(
                f"Plugin {self.name!r} failed: {exc}", mutated=True
            ) from exc

    def rollback(self, ctx: LoadContext) -> None:
        self._strategy.rollback(ctx)


def get_plugin(name: str) -> LoadStrategy:
    """Return a fresh selected strategy with conservative failure recovery."""
    with _lock:
        if name not in plugins:
            raise ValueError(f"Selected loading plugin {name!r} is not registered")
        return _PluginStrategy(plugins[name]())
