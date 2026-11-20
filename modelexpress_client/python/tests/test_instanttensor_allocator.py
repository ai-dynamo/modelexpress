# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InstantTensor allocator scope and iterator cleanup, without CUDA allocation."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress.adapter import StrategyFailed
from modelexpress.engines.vllm.adapter import (
    VllmAdapter,
    _instanttensor_allocator_context,
)
from modelexpress.load_strategy.context import LoadResult
from modelexpress.load_strategy.instant_tensor_strategy import InstantTensorStrategy


@pytest.fixture
def allocator(monkeypatch):
    state = SimpleNamespace(
        original="garbage_collection_threshold:0.8,max_split_size_mb:20,expandable_segments:False",
        current=None,
        changes=[],
    )
    state.current = state.original
    state.getter = MagicMock(side_effect=lambda: state.current)
    state.snapshot = MagicMock(side_effect=AssertionError("C getter is available"))

    def set_settings(settings):
        state.changes.append(settings)
        state.current = settings

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        torch._C, "_accelerator_getAllocatorSettings", state.getter, raising=False,
    )
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", set_settings)
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", state.snapshot)
    return state


def _relaxed(settings):
    return settings.replace("max_split_size_mb:20", "max_split_size_mb:21")


def _native_adapter(monkeypatch, native_iterator):
    load_config = SimpleNamespace(
        load_format="modelexpress",
        model_loader_extra_config={"enable_multithread_load": True},
    )
    adapter = object.__new__(VllmAdapter)
    adapter.load_config = load_config
    adapter.model_config = SimpleNamespace(model="/models/primary", revision="main")
    loader = MagicMock()
    loader.get_all_weights.side_effect = native_iterator
    factory = MagicMock(return_value=loader)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.default_loader",
        SimpleNamespace(DefaultModelLoader=factory),
    )
    return adapter, factory, loader


def test_scope_preserves_runtime_settings_and_ignores_environment(allocator, monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
    stats = MagicMock(return_value={"max_split_size": 512 * 1024 * 1024})
    monkeypatch.setattr(torch.cuda, "memory_stats", stats)

    with _instanttensor_allocator_context():
        assert allocator.current == _relaxed(allocator.original)
        with _instanttensor_allocator_context():
            assert allocator.current == _relaxed(allocator.original)
        assert allocator.current == _relaxed(allocator.original)

    assert allocator.current == allocator.original
    assert allocator.changes == [_relaxed(allocator.original), allocator.original]
    allocator.snapshot.assert_not_called()
    stats.assert_not_called()


def test_scope_restores_original_settings_on_exception(allocator):
    with pytest.raises(RuntimeError, match="consumer failed"):
        with _instanttensor_allocator_context():
            assert allocator.current == _relaxed(allocator.original)
            raise RuntimeError("consumer failed")

    assert allocator.current == allocator.original
    assert allocator.changes == [_relaxed(allocator.original), allocator.original]


@pytest.mark.parametrize("settings", ["", "max_split_size_mb:21", "max_split_size_mb:512"])
def test_other_runtime_limits_are_unchanged(allocator, settings):
    allocator.current = settings
    with _instanttensor_allocator_context():
        assert allocator.current == settings
    assert allocator.changes == []


@pytest.mark.parametrize("initialized,backend", [(False, "native"), (True, "cudaMallocAsync")])
def test_non_native_or_uninitialized_cuda_is_untouched(
    allocator, monkeypatch, initialized, backend,
):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: backend)
    with _instanttensor_allocator_context():
        assert allocator.current == allocator.original
    assert allocator.changes == []
    allocator.getter.assert_not_called()
    allocator.snapshot.assert_not_called()


def test_snapshot_fallback_uses_runtime_string(allocator, monkeypatch):
    monkeypatch.delattr(torch._C, "_accelerator_getAllocatorSettings")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    allocator.snapshot.side_effect = None
    allocator.snapshot.return_value = {
        "allocator_settings": {"PYTORCH_CUDA_ALLOC_CONF": allocator.original},
    }

    with _instanttensor_allocator_context():
        assert allocator.current == _relaxed(allocator.original)

    assert allocator.current == allocator.original
    allocator.snapshot.assert_called_once_with()
    allocator.getter.assert_not_called()


def test_snapshot_without_settings_leaves_allocator_unchanged(allocator, monkeypatch):
    monkeypatch.delattr(torch._C, "_accelerator_getAllocatorSettings")
    allocator.snapshot.side_effect = None
    allocator.snapshot.return_value = {}
    with _instanttensor_allocator_context():
        pass
    assert allocator.changes == []


def test_iterator_scope_is_lazy_and_preserves_native_sources(allocator, monkeypatch):
    primary, secondary = object(), object()
    model = SimpleNamespace(secondary_weights=["native-secondary-source"])

    def native_iterator(model_config, loaded_model):
        assert allocator.current == _relaxed(allocator.original)
        yield "primary.weight", primary
        yield "secondary.weight", secondary

    adapter, factory, loader = _native_adapter(monkeypatch, native_iterator)
    original_extra = adapter.load_config.model_loader_extra_config
    weights = adapter.build_instanttensor_weight_iter(model=model)
    assert allocator.changes == []
    loader.get_all_weights.assert_not_called()
    assert list(weights) == [("primary.weight", primary), ("secondary.weight", secondary)]

    loader.get_all_weights.assert_called_once_with(adapter.model_config, model)
    copied_config = factory.call_args.args[0]
    assert copied_config is not adapter.load_config
    assert copied_config.load_format == "instanttensor"
    assert copied_config.model_loader_extra_config is original_extra
    assert adapter.load_config.load_format == "modelexpress"
    assert adapter.load_config.model_loader_extra_config == {"enable_multithread_load": True}
    assert allocator.current == allocator.original


@pytest.mark.parametrize("started", [False, True])
def test_explicit_iterator_close_restores_scope(allocator, monkeypatch, started):
    finalized = []

    def native_iterator(*args):
        try:
            yield "first", object()
            yield "second", object()
        finally:
            finalized.append(allocator.current)

    adapter, _, _ = _native_adapter(monkeypatch, native_iterator)
    weights = adapter.build_instanttensor_weight_iter(model=object())
    if started:
        next(weights)
        assert allocator.current == _relaxed(allocator.original)
    weights.close()

    assert allocator.current == allocator.original
    assert finalized == ([_relaxed(allocator.original)] if started else [])
    assert allocator.changes == (
        [_relaxed(allocator.original), allocator.original] if started else []
    )


@pytest.mark.parametrize(
    "outcome", ["success", "early_return", "consumer_error", "iterator_error"],
)
def test_strategy_closes_iterator_on_completion_or_failure(
    allocator, monkeypatch, outcome,
):
    events = []
    iterator_refs = []

    def native_iterator(*args):
        try:
            yield "first", object()
            if outcome == "iterator_error":
                raise RuntimeError("iterator failed")
            yield "second", object()
        finally:
            events.append(("native_closed", allocator.current))

    def consume(weights):
        iterator_refs.append(weights)
        assert allocator.current == allocator.original
        next(weights)
        assert allocator.current == _relaxed(allocator.original)
        if outcome == "consumer_error":
            raise RuntimeError("consumer failed")
        if outcome != "early_return":
            list(weights)

    model = SimpleNamespace(load_weights=consume)
    adapter, _, _ = _native_adapter(monkeypatch, native_iterator)

    result = LoadResult(value=model, model=model)
    ctx = SimpleNamespace(adapter=adapter, global_rank=0)
    if outcome.endswith("error"):
        with pytest.raises(StrategyFailed, match="failed") as exc:
            InstantTensorStrategy().load(result, ctx)
        assert exc.value.mutated is True
    else:
        assert InstantTensorStrategy().load(result, ctx) is result

    assert allocator.current == allocator.original
    assert allocator.changes == [_relaxed(allocator.original), allocator.original]
    assert events[0] == ("native_closed", _relaxed(allocator.original))
    assert list(iterator_refs[0]) == []
