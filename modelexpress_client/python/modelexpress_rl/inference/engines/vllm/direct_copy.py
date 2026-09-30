# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private copy-only admission and shared batch-installation helpers.

Default admission accepts only exact plain Torch containers/Linear/LayerNorm
without buffers, hooks or tensor overrides. GLM supplies separate admission
and reuses the batch-copy and cleanup helpers.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import nn
from torch.nn.modules import module as module_runtime
from torch.overrides import _get_current_function_mode_stack
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from ...plan import (
    PreparedDirectGroupTensors,
    PreparedStreamingTensors,
    _DirectCopyOwnership,
)


class _UnsupportedDirectPlan(ValueError):
    pass


_CLASSES = (nn.Module, nn.Sequential, nn.ModuleList, nn.Linear, nn.LayerNorm)
_CLASS_BINDINGS = tuple((cls, tuple(vars(cls).items())) for cls in _CLASSES)
_PARAMETER_BINDINGS = tuple(vars(nn.Parameter).items())
_TENSOR_BINDINGS = tuple(vars(torch.Tensor).items())
_COPY = torch.Tensor.copy_
_NO_GRAD = torch.no_grad
_CUDA_SYNCHRONIZE = torch.cuda.synchronize
_BASE_FIELDS = frozenset(vars(nn.Module()))
_EXTRA_FIELDS = {
    nn.Module: frozenset(),
    nn.Sequential: frozenset(),
    nn.ModuleList: frozenset(),
    nn.Linear: frozenset({"in_features", "out_features"}),
    nn.LayerNorm: frozenset({"normalized_shape", "eps", "elementwise_affine"}),
}
_GLOBAL_HOOK_FIELDS = tuple(
    name
    for name, value in vars(module_runtime).items()
    if name.startswith("_global_")
    and "hook" in name
    and type(value) in (dict, OrderedDict)
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _UnsupportedDirectPlan(message)


def _primitive(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is tuple:
        return tuple(_primitive(item) for item in value)
    raise _UnsupportedDirectPlan("unsupported module configuration value")


def _dispatch_guard() -> None:
    _require(not _get_current_function_mode_stack(), "TorchFunctionMode is active")
    _require(not _get_current_dispatch_mode_stack(), "TorchDispatchMode is active")
    _require(torch.no_grad is _NO_GRAD, "no_grad binding changed")
    _require(
        torch.cuda.synchronize is _CUDA_SYNCHRONIZE,
        "CUDA synchronization binding changed",
    )
    for cls, bindings in (
        *_CLASS_BINDINGS,
        (nn.Parameter, _PARAMETER_BINDINGS),
        (torch.Tensor, _TENSOR_BINDINGS),
    ):
        current = vars(cls)
        _require(
            len(current) == len(bindings)
            and all(current.get(key) is value for key, value in bindings),
            "Torch class binding changed",
        )
    for name in _GLOBAL_HOOK_FIELDS:
        value = vars(module_runtime).get(name)
        _require(
            type(value) in (dict, OrderedDict) and not value,
            "global module hook registry changed",
        )


def _geometry(tensor, *, destination: bool):
    _require(
        type(tensor) is (nn.Parameter if destination else torch.Tensor),
        "unsupported tensor subclass",
    )
    _require(not vars(tensor), "tensor has custom attributes")
    _require(
        tensor.layout is torch.strided and tensor.device.type in ("cpu", "cuda"),
        "unsupported tensor layout/device",
    )
    shape, stride = tuple(tensor.shape), tuple(tensor.stride())
    _require(
        tensor.numel() > 0 and tensor.is_contiguous(), "empty/noncontiguous tensor"
    )
    _require(
        tensor.dtype in (torch.float32, torch.float16, torch.bfloat16),
        "unsupported dtype",
    )
    storage = tensor.untyped_storage()
    offset, nbytes = tensor.storage_offset(), tensor.numel() * tensor.element_size()
    base, capacity = storage.data_ptr(), storage.nbytes()
    start = base + offset * tensor.element_size()
    _require(
        offset >= 0
        and tensor.data_ptr() == start
        and start + nbytes <= base + capacity,
        "tensor storage bounds differ",
    )
    if destination:
        _require(
            offset == 0 and nbytes == capacity,
            "destination has padding or nonzero offset",
        )
    return (
        id(tensor),
        shape,
        stride,
        tensor.dtype,
        tensor.device,
        offset,
        base,
        capacity,
        start,
        nbytes,
    )


@dataclass(frozen=True)
class _DirectDestination:
    name: str
    tensor: nn.Parameter
    geometry: tuple


@dataclass(frozen=True)
class _DirectInstallPlan:
    model: nn.Module
    version_id: str
    destinations: tuple[_DirectDestination, ...]
    groups: tuple[frozenset[str], ...]
    batch_names: tuple[frozenset[str], ...]
    snapshot: str


def _inspect_model(model):
    _dispatch_guard()
    destinations, objects, owners, snapshot = [], {}, [], []

    def walk(module, prefix, ancestors):
        cls = type(module)
        _require(
            any(cls is allowed for allowed in _CLASSES),
            "unsupported owner or helper class",
        )
        _require(id(module) not in ancestors, "module cycle")
        state = object.__getattribute__(module, "__dict__")
        _require(
            type(state) is dict and all(type(key) is str for key in state),
            "invalid module fields",
        )
        _require(
            set(state) == _BASE_FIELDS | _EXTRA_FIELDS[cls], "unknown module state"
        )
        parameters, children = state["_parameters"], state["_modules"]
        _require(
            type(parameters) is dict and type(children) is dict,
            "custom module registry",
        )
        _require(
            all(type(key) is str for key in (*parameters, *children)),
            "custom registry key",
        )
        _require(
            all(key and "." not in key for key in (*parameters, *children)),
            "invalid qualified-name component",
        )
        _require(
            not parameters.keys() & children.keys(), "parameter/child name collision"
        )
        _require(
            type(state["_buffers"]) is dict and not state["_buffers"],
            "buffers require an engine contract",
        )
        _require(
            type(state["_non_persistent_buffers_set"]) is set
            and not state["_non_persistent_buffers_set"],
            "buffer state",
        )
        for key in _BASE_FIELDS - {
            "_parameters",
            "_modules",
            "_buffers",
            "_non_persistent_buffers_set",
            "training",
            "_is_full_backward_hook",
        }:
            _require(
                type(state[key]) in (dict, OrderedDict) and not state[key],
                "module hook is installed",
            )
        _require(
            type(state["training"]) is bool and state["_is_full_backward_hook"] is None,
            "module execution state",
        )
        config = tuple(
            (key, _primitive(state[key])) for key in sorted(_EXTRA_FIELDS[cls])
        )
        owned, parameter_ids = set(), []
        for leaf, parameter in parameters.items():
            name = prefix + leaf
            parameter_ids.append((leaf, id(parameter)))
            if parameter is None:
                continue
            geometry = _geometry(parameter, destination=True)
            canonical = objects.get(id(parameter))
            if canonical is None:
                canonical = name
                objects[id(parameter)] = canonical
                destinations.append(_DirectDestination(name, parameter, geometry))
            owned.add(canonical)
        if owned:
            owners.append(owned)
        snapshot.append(
            (
                prefix,
                id(module),
                cls,
                state["training"],
                config,
                tuple(parameter_ids),
                tuple((key, id(child)) for key, child in children.items()),
            )
        )
        for leaf, child in children.items():
            _require(child is not None, "empty child registry entry")
            walk(child, prefix + leaf + ".", ancestors | {id(module)})

    walk(model, "", set())
    _require(destinations, "empty direct model")
    _require(
        len({destination.name for destination in destinations}) == len(destinations),
        "canonical destination name collision",
    )
    ranges = sorted(
        (str(d.geometry[4]), d.geometry[8], d.geometry[8] + d.geometry[9], d.name)
        for d in destinations
    )
    for left, right in pairwise(ranges):
        _require(
            left[0] != right[0] or left[2] <= right[1],
            "distinct targets overlap physically",
        )
    groups = []
    for owned in owners:
        merged = set(owned)
        remainder = []
        for group in groups:
            if merged & group:
                merged.update(group)
            else:
                remainder.append(group)
        groups = [*remainder, merged]
    frozen_groups = tuple(
        sorted((frozenset(group) for group in groups), key=lambda group: sorted(group))
    )
    signature = repr(
        (
            tuple(snapshot),
            tuple((d.name, d.geometry) for d in destinations),
            frozen_groups,
        )
    )
    return tuple(destinations), frozen_groups, signature


def prepare_direct_copy(
    model,
    *,
    version_id: str,
    source: PreparedStreamingTensors,
    batch_names: tuple[frozenset[str], ...],
):
    """Explicit private admission; decline preserves the original unconsumed source."""
    try:
        _require(
            type(source) is PreparedStreamingTensors, "unsupported source artifact"
        )
        _require(type(version_id) is str and bool(version_id), "invalid version")
        _require(type(batch_names) is tuple and bool(batch_names), "missing batch plan")
        _require(
            all(
                type(names) is frozenset
                and names
                and all(type(name) is str for name in names)
                for names in batch_names
            ),
            "invalid batch names",
        )
        destinations, groups, snapshot = _inspect_model(model)
        expected = frozenset(d.name for d in destinations)
        _require(
            type(source.parameter_names) is frozenset
            and all(type(name) is str for name in source.parameter_names)
            and source.parameter_names == expected,
            "whole-model source coverage differs",
        )
        seen = set()
        for names in batch_names:
            _require(
                not seen & names and names <= expected, "duplicate/unknown batch names"
            )
            _require(
                all(not names & group or group <= names for group in groups),
                "batch splits a dependency group",
            )
            seen.update(names)
        _require(seen == expected, "incomplete batch plan")
        plan = _DirectInstallPlan(
            model, version_id, destinations, groups, batch_names, snapshot
        )
        return PreparedDirectGroupTensors(
            version_id=version_id, plan=plan, source=source
        )
    except _UnsupportedDirectPlan:
        return source


def _validate_plan(prepared: PreparedDirectGroupTensors, model):
    _require(
        type(prepared) is PreparedDirectGroupTensors
        and type(prepared.plan) is _DirectInstallPlan,
        "invalid direct artifact",
    )
    _require(
        type(prepared.ownership) is _DirectCopyOwnership, "invalid direct ownership"
    )
    plan = prepared.plan
    _require(
        type(plan.snapshot) is str
        and type(plan.destinations) is tuple
        and all(
            type(d) is _DirectDestination and type(d.name) is str
            for d in plan.destinations
        ),
        "invalid direct plan fields",
    )
    _require(
        type(plan.version_id) is str
        and type(prepared.version_id) is str
        and plan.model is model
        and plan.version_id == prepared.version_id,
        "direct model/version differs",
    )
    rebuilt = prepare_direct_copy(
        model,
        version_id=prepared.version_id,
        source=prepared.source,
        batch_names=plan.batch_names,
    )
    _require(
        type(rebuilt) is PreparedDirectGroupTensors
        and rebuilt.plan.snapshot == plan.snapshot,
        "direct admission guard changed",
    )
    _require(
        tuple((d.name, id(d.tensor)) for d in plan.destinations)
        == tuple((d.name, id(d.tensor)) for d in rebuilt.plan.destinations),
        "direct destination plan differs",
    )
    _require(
        type(plan.groups) is tuple
        and all(
            type(group) is frozenset and all(type(name) is str for name in group)
            for group in plan.groups
        )
        and plan.groups == rebuilt.plan.groups,
        "direct dependency groups differ",
    )
    return rebuilt.plan


def _drain(plan: _DirectInstallPlan) -> None:
    for device in {destination.geometry[4] for destination in plan.destinations}:
        if device.type == "cuda":
            _CUDA_SYNCHRONIZE(device)


def install_direct_copy(prepared: PreparedDirectGroupTensors, *, model) -> None:
    """Copy full admitted groups without attaching receive tensors or entering reload."""
    plan = _validate_plan(prepared, model)
    _install_copy_batches(
        prepared, plan=plan, validate_batch=lambda: _validate_plan(prepared, model)
    )


def _install_copy_batches(prepared, *, plan, validate_batch, finish=None) -> None:
    """Shared lifetime protocol; engine-specific admission remains with the caller."""
    _require(
        prepared.ownership.iterator is None and not prepared.ownership.release_blocked,
        "direct transaction cannot be replayed while resources are retained",
    )
    destinations = {destination.name: destination for destination in plan.destinations}
    ranges = [
        (d.geometry[4], d.geometry[8], d.geometry[8] + d.geometry[9])
        for d in plan.destinations
    ]
    iterator = None
    started = time.perf_counter()
    copied = 0
    commit_s = 0.0
    primary_error = None
    sentinel = object()

    def drain():
        try:
            _drain(plan)
        except BaseException:
            prepared.ownership.drain_failed = True
            raise

    def next_batch():
        try:
            return next(iterator, sentinel)
        except BaseException:
            # CUDA synchronization cannot prove failed RDMA READs are drained.
            prepared.ownership.source_failed = True
            raise

    try:
        try:
            iterator = prepared.source.batches()
        except BaseException:
            prepared.ownership.source_failed = True
            raise
        prepared.ownership.iterator = iterator
        with _NO_GRAD():
            for names in plan.batch_names:
                tensors = next_batch()
                _require(tensors is not sentinel, "missing received batch")
                commit_started = time.perf_counter()
                validate_batch()
                _require(
                    type(tensors) is dict
                    and all(type(key) is str for key in tensors)
                    and frozenset(tensors) == names,
                    "received batch differs from admitted group plan",
                )
                for name, tensor in tensors.items():
                    geometry = _geometry(tensor, destination=False)
                    target = destinations[name].geometry
                    _require(
                        geometry[1] == target[1] and geometry[3:5] == target[3:5],
                        "received tensor geometry differs",
                    )
                    _require(
                        all(
                            geometry[4] != device
                            or geometry[8] + geometry[9] <= start
                            or geometry[8] >= end
                            for device, start, end in ranges
                        ),
                        "receive tensor aliases live storage",
                    )
                for name, tensor in tensors.items():
                    _COPY(destinations[name].tensor, tensor)
                    copied += 1
                drain()
                commit_s += time.perf_counter() - commit_started
                del tensors
            _require(next_batch() is sentinel, "extra received batch")
            if finish is not None:
                finish()
    except BaseException as error:
        primary_error = error
        raise
    finally:
        # Generator.close skips its normal post-yield reuse synchronization.
        if not prepared.ownership.drain_failed:
            try:
                drain()
                if iterator is not None and not prepared.ownership.source_failed:
                    try:
                        iterator.close()
                    except BaseException:
                        prepared.ownership.close_failed = True
                        raise
                if not prepared.ownership.source_failed:
                    prepared.ownership.iterator = None
            except BaseException as cleanup_error:
                if primary_error is not None:
                    add_note = getattr(primary_error, "add_note", None)
                    if add_note is not None:
                        add_note(
                            "Direct-copy cleanup is unproven; resources and lease must be retained."
                        )
                    raise primary_error from cleanup_error
                raise
        prepared.source.transfer_metrics["direct_copy_targets"] = copied
        prepared.source.transfer_metrics["install_commit_s"] = commit_s
        prepared.source.transfer_metrics["direct_copy_install_s"] = (
            time.perf_counter() - started
        )
