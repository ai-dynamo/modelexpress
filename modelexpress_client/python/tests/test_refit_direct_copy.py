# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
import torch
from modelexpress_rl.inference.engines.vllm import direct_copy
from modelexpress_rl.inference.plan import (
    PreparedDirectGroupTensors,
    PreparedStreamingTensors,
)
from torch import nn
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode


def prepare(model, batches=None, names=None, batch_names=None):
    expected = frozenset(dict(model.named_parameters())) if names is None else names
    values = {
        name: torch.full_like(value, 3) for name, value in model.named_parameters()
    }
    calls = []

    def iterator():
        calls.append("open")
        try:
            yield values
        finally:
            calls.append("close")

    source = PreparedStreamingTensors(batches or iterator, expected, {})
    result = direct_copy.prepare_direct_copy(
        model, version_id="v:1", source=source, batch_names=batch_names or (expected,)
    )
    return result, source, calls


def test_three_updates_preserve_values_aliases_objects_and_storage():
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    model[1].weight = model[0].weight
    parameter = model[0].weight
    pointer = parameter.data_ptr()
    for _ in range(3):
        prepared, source, calls = prepare(model)
        assert isinstance(prepared, PreparedDirectGroupTensors)
        assert calls == []
        direct_copy.install_direct_copy(prepared, model=model)
        assert torch.equal(parameter, torch.full_like(parameter, 3))
        assert model[0].weight is model[1].weight is parameter
        assert parameter.data_ptr() == pointer
        assert source.metrics["direct_copy_targets"] == 1
        assert calls == ["open", "close"]
        assert prepared.ownership.iterator is None


@pytest.mark.parametrize(
    "kind", ["subclass", "buffer", "helper", "hook", "tensor_attribute"]
)
def test_unsupported_whole_model_declines_before_iterator(kind):
    class OtherLinear(nn.Linear):
        pass

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    if kind == "subclass":
        model[1] = OtherLinear(2, 2)
    elif kind == "buffer":
        model[1].register_buffer("derived", torch.zeros(2))
    elif kind == "helper":
        model[1].helper = object()
    elif kind == "hook":
        model[1].register_forward_hook(lambda *args: None)
    else:
        model[1].weight.arbitrary = 1
    prepared, source, calls = prepare(model)
    assert prepared is source and calls == []


def test_distinct_objects_with_overlapping_storage_decline():
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    model[1].weight = nn.Parameter(model[0].weight.detach())
    prepared, source, calls = prepare(model)
    assert prepared is source and not calls


@pytest.mark.parametrize("component", ["a.b", ""])
def test_raw_registry_name_collision_declines_without_iteration(component):
    model = nn.Module()
    model.a = nn.Sequential(nn.Linear(2, 2, bias=False))
    model._modules[component] = nn.Linear(2, 2, bias=False)
    prepared, source, calls = prepare(model)
    assert prepared is source and calls == []


@pytest.mark.parametrize("destination", [False, True])
def test_tensor_subclass_rejected_without_dispatch(destination):
    callbacks = []

    class Parameter(nn.Parameter):
        @classmethod
        def __torch_function__(cls, *args, **kwargs):
            callbacks.append("parameter")
            raise AssertionError("unexpected parameter callback")

    class Tensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, *args, **kwargs):
            callbacks.append("tensor")
            raise AssertionError("unexpected tensor callback")

    model = nn.Linear(2, 2, bias=False)
    names = frozenset({"weight"})
    if destination:
        model._parameters["weight"] = Parameter(torch.zeros(2, 2))
    tensor = torch.Tensor._make_subclass(Tensor, torch.zeros(2, 2))
    source = PreparedStreamingTensors(lambda: iter(({"weight": tensor},)), names, {})
    prepared = direct_copy.prepare_direct_copy(
        model, version_id="v:1", source=source, batch_names=(names,)
    )
    if destination:
        assert prepared is source
    else:
        with pytest.raises(ValueError, match="tensor subclass"):
            direct_copy.install_direct_copy(prepared, model=model)
    assert callbacks == []


@pytest.mark.parametrize("kind", ["factory", "next"])
def test_source_failure_retains_transaction_even_if_cuda_drain_succeeds(kind):
    model = nn.Linear(2, 2, bias=False)
    primary = RuntimeError("source read status is unknown")
    closed = []

    class Source:
        def __next__(self):
            raise primary

        def close(self):
            closed.append(True)

    iterator = Source()

    def batches():
        if kind == "factory":
            raise primary
        return iterator

    prepared, _, _ = prepare(model, batches=batches)
    with pytest.raises(RuntimeError) as caught:
        direct_copy.install_direct_copy(prepared, model=model)
    assert caught.value is primary
    assert prepared.ownership.source_failed and prepared.ownership.release_blocked
    assert not closed
    assert prepared.ownership.iterator is (iterator if kind == "next" else None)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "split_group"])
def test_batch_coverage_and_group_closure_decline_before_iterator(kind):
    model = nn.Linear(2, 2)
    batch_names = {
        "missing": (frozenset({"weight"}),),
        "duplicate": (frozenset({"weight", "bias"}), frozenset({"weight"})),
        "split_group": (frozenset({"weight"}), frozenset({"bias"})),
    }[kind]
    prepared, source, calls = prepare(model, batch_names=batch_names)
    assert prepared is source and calls == []


@pytest.mark.parametrize("kind", ["function", "dispatch"])
def test_active_torch_modes_decline_without_invoking_mode(kind):
    model = nn.Linear(2, 2)
    prepared, source, calls = prepare(model)
    invoked = []

    class Function(TorchFunctionMode):
        def __torch_function__(self, *args, **kwargs):
            invoked.append("function")
            raise AssertionError("unexpected callback")

    class Dispatch(TorchDispatchMode):
        def __torch_dispatch__(self, *args, **kwargs):
            invoked.append("dispatch")
            raise AssertionError("unexpected callback")

    with Function() if kind == "function" else Dispatch():
        result = direct_copy.prepare_direct_copy(
            model,
            version_id="v:1",
            source=source,
            batch_names=prepared.plan.batch_names,
        )
    assert result is source and not invoked and not calls


def test_live_binding_drift_and_forged_snapshot_reject_before_iterator():
    model = nn.Linear(2, 2)
    prepared, _, calls = prepare(model)
    forged = replace(
        prepared, plan=replace(prepared.plan, snapshot="not an admission verdict")
    )
    with pytest.raises(ValueError, match="guard changed"):
        direct_copy.install_direct_copy(forged, model=model)
    model.weight = nn.Parameter(model.weight.detach().clone())
    with pytest.raises(ValueError, match="guard changed"):
        direct_copy.install_direct_copy(prepared, model=model)
    assert calls == []


def test_source_alias_rejected_before_any_copy():
    model = nn.Linear(2, 2, bias=False)
    before = model.weight.detach().clone()

    def iterator():
        yield {"weight": model.weight.detach()}

    prepared, _, _ = prepare(model, batches=iterator)
    with pytest.raises(ValueError, match="aliases live storage"):
        direct_copy.install_direct_copy(prepared, model=model)
    assert torch.equal(model.weight, before)


class _LegacyRuntimeError(RuntimeError):
    @property
    def add_note(self):
        raise AttributeError("add_note")


@pytest.mark.parametrize("error_type", [RuntimeError, _LegacyRuntimeError])
@pytest.mark.parametrize("cleanup", ["success", "drain", "close"])
def test_primary_failure_and_cleanup_ownership(monkeypatch, cleanup, error_type):
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    events = []
    primary = error_type("copy failed after first write")
    cleanup_error = RuntimeError("cleanup failed")

    class Iterator:
        step = 0

        def __next__(self):
            self.step += 1
            if self.step == 1:
                return {"0.weight": torch.full_like(model[0].weight, 4)}
            return {"1.weight": torch.full_like(model[1].weight, 4)}

        def close(self):
            events.append("close")
            if cleanup == "close":
                raise cleanup_error

    def drain(plan):
        events.append("drain")
        if cleanup == "drain" and events.count("drain") == 2:
            raise cleanup_error

    iterator = Iterator()
    names = (frozenset({"0.weight"}), frozenset({"1.weight"}))
    prepared, _, _ = prepare(model, batches=lambda: iterator, batch_names=names)
    monkeypatch.setattr(direct_copy, "_drain", drain)
    native_copy = direct_copy._COPY

    def copy(destination, source):
        if destination is model[1].weight:
            raise primary
        return native_copy(destination, source)

    monkeypatch.setattr(direct_copy, "_COPY", copy)
    with pytest.raises(RuntimeError) as error:
        direct_copy.install_direct_copy(prepared, model=model)
    assert error.value is primary
    assert primary.__cause__ is (cleanup_error if cleanup != "success" else None)
    assert torch.equal(model[0].weight, torch.full_like(model[0].weight, 4))
    assert events == (
        ["drain", "drain"] if cleanup == "drain" else ["drain", "drain", "close"]
    )
    assert prepared.ownership.release_blocked is (cleanup != "success")
    assert (prepared.ownership.iterator is iterator) is (cleanup != "success")
    assert prepared.ownership.drain_failed is (cleanup == "drain")
    assert prepared.ownership.close_failed is (cleanup == "close")
