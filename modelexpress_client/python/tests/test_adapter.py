# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the engine adapter capability contract."""

from types import SimpleNamespace

import pytest

from modelexpress.adapter import EngineAdapter, UnsupportedCapability
from modelexpress.load_strategy.base import LoadStrategy
from modelexpress.load_strategy.context import LoadResult


class _DiscoverTensorsStrategy(LoadStrategy):
    name = "discover_tensors"
    requires = (EngineAdapter.discover_tensors,)

    def load(self, result, ctx):
        return result


class _BaseAdapter(EngineAdapter):
    pass


class _DiscoverTensorsAdapter(EngineAdapter):
    def discover_tensors(self, result: LoadResult):
        return {}


def test_inherited_gated_capability_raises_unsupported_capability():
    adapter = EngineAdapter()

    with pytest.raises(UnsupportedCapability, match="discover_tensors"):
        adapter.discover_tensors(LoadResult(value=None))


def test_strategy_unavailable_when_required_capability_is_inherited_default():
    strategy = _DiscoverTensorsStrategy()
    ctx = SimpleNamespace(adapter=_BaseAdapter())

    assert strategy.is_available(ctx) is False


def test_strategy_available_when_required_capability_is_overridden():
    strategy = _DiscoverTensorsStrategy()
    ctx = SimpleNamespace(adapter=_DiscoverTensorsAdapter())

    assert strategy.is_available(ctx) is True
