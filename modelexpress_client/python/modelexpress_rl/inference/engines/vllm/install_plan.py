# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-lifetime engine requirements, source bindings, and update execution state."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from collections.abc import Callable

import torch
from modelexpress_rl.inference.plan import SourceReplayBatch
from modelexpress_rl.refit.reshard.types import CaptureResult, IncompleteRefit
from typing import TYPE_CHECKING
from weakref import ref
from uuid import UUID

from modelexpress_rl.refit.reshard.types import _connected_groups

if TYPE_CHECKING:
    from .installer import _ParameterAliases


@dataclass(frozen=True, eq=False)
class _EngineRuntimeBindings:
    modules: tuple
    layouts: dict
    bare_tensors: tuple
    overlap_groups: tuple
    overlap_tensors: dict
    aliases: _ParameterAliases
    load_aliases: _ParameterAliases


@dataclass(frozen=True)
class _InputSlot:
    name: str
    leaf: str
    identity: int
    is_buffer: bool


@dataclass(frozen=True)
class _InputOwner:
    path: str
    layer_ref: ref
    slots: tuple[_InputSlot, ...]


@dataclass(frozen=True, eq=False)
class _LayerInstallPlan:
    path: str
    layer_ref: ref
    # Full load-layout name, local slot, canonical (possibly tied) name.
    parameters: tuple[tuple[str, str, str], ...]

    @property
    def layer(self):
        return self.layer_ref()


@dataclass(frozen=True, eq=False)
class _InstallTargets:
    by_name: dict[str, _LayerInstallPlan]
    owners: tuple[_LayerInstallPlan, ...]


@dataclass(frozen=True, eq=False)
class EngineInstallPlan:
    """Fixed engine slots, alias relationships and runtime restoration bindings."""

    runtime: _EngineRuntimeBindings
    inputs: tuple[_InputOwner, ...]

    def bind_targets(self, buffer_names: frozenset[str]) -> _InstallTargets:
        canonical, by_name, owners = {}, {}, []
        for owner in self.inputs:
            parameters = []
            for slot in owner.slots:
                if slot.is_buffer and slot.name not in buffer_names:
                    continue
                first = canonical.setdefault(slot.identity, slot.name)
                parameters.append((slot.name, slot.leaf, first))
            if not parameters:
                continue
            layer = _LayerInstallPlan(owner.path, owner.layer_ref, tuple(parameters))
            owners.append(layer)
            for name, _, first in parameters:
                if name == first:
                    by_name[name] = layer
        return _InstallTargets(by_name, tuple(owners))


@dataclass(frozen=True, eq=False)
class _ReplaySchedule:
    total_copies: int
    groups: dict
    ready: dict

    @classmethod
    def compile(cls, targets, overlap_groups, last_writes, total_copies):
        last = {
            owner: max(last_writes[name] for _, _, name in owner.parameters)
            for owner in targets.owners
        }
        owners_by_name = {
            name: owner for owner in targets.owners for name, _, _ in owner.parameters
        }
        canonical_slots = {}
        for owner in targets.owners:
            for name, _, canonical in owner.parameters:
                canonical_slots.setdefault(canonical, []).append(name)
        links = (
            (owners_by_name[name] for name in names if name in owners_by_name)
            for names in (*overlap_groups, *canonical_slots.values())
        )
        groups, ready = {}, {}
        for group in _connected_groups(targets.owners, links):
            ready.setdefault(max(last[owner] for owner in group), []).append(group)
            for owner in group:
                groups[owner] = group
        return cls(total_copies, groups, ready)


@dataclass(frozen=True, eq=False)
class SourceInstallPlan:
    """Own a captured replay identity, engine bindings, and completion schedule.

    Retain the isolated capture cache, load-time targets and persistent-input facts
    for one compatible capture. Create independent execution state for each weight
    update. Recapture replaces this plan and rejects batches carrying its old identity.
    """

    engine: EngineInstallPlan
    targets: _InstallTargets
    schedule: _ReplaySchedule
    replay_id: UUID
    buffer_names: frozenset[str]
    persistent_names: frozenset[str]
    cache_key: tuple
    capture_result: tuple

    @classmethod
    def compile(
        cls,
        *,
        engine: EngineInstallPlan,
        capture: CaptureResult,
        buffer_names: frozenset[str],
        persistent_names: frozenset[str],
        cache_key: tuple,
        capture_result: tuple,
    ) -> SourceInstallPlan | None:
        if capture.unsupported or capture.unattributed:
            return None
        targets = engine.bind_targets(buffer_names)
        last_writes = {
            item.param_name: index for index, item in enumerate(capture.copies)
        }
        if not targets.by_name.keys() <= last_writes.keys():
            return None
        return cls(
            engine,
            targets,
            _ReplaySchedule.compile(
                targets, engine.runtime.overlap_groups, last_writes, len(capture.copies)
            ),
            capture.replay_id,
            buffer_names,
            persistent_names,
            cache_key,
            copy.deepcopy(capture_result),
        )

    def new_execution(
        self,
        model: torch.nn.Module,
        restore_aliases: Callable,
        *,
        bounded: bool = False,
    ) -> _InstallExecution:
        return _InstallExecution(self, model, restore_aliases, bounded)


@dataclass(eq=False)
class _InstallExecution:
    """Apply ordered source writes and finalize ready owner groups.

    Own mutable progress and borrowed engine inputs for one update. Validate replay
    identity and write order, materialize tied owners before copying, and process and
    restore each group after its last dependency. Completion requires all recorded
    writes and all active owners to finish.
    """

    _captured: SourceInstallPlan
    _model: torch.nn.Module
    _restore_parameter_aliases: Callable
    _bounded: bool = False
    _pending: dict = field(default_factory=dict, init=False)
    _replay_started: dict = field(default_factory=dict, init=False)
    _replay_next: int = field(default=0, init=False)
    tied_input_peak_bytes: int = field(default=0, init=False)

    def __post_init__(self):
        self._pending = {
            name: ref(getattr(entry.layer, leaf))
            for entry in self._captured.targets.owners
            for name, leaf, _ in entry.parameters
        }

    def _validate_source_owner(self, plan) -> None:
        layer = plan.layer
        if self._model.get_submodule(plan.path) is not layer:
            raise IncompleteRefit(
                f"a post-load hook replaced module {plan.path!r} "
                "before its parameters were committed"
            )
        current = {
            name for name, value in layer._parameters.items() if value is not None
        }
        current.update(
            name
            for name, value in layer._buffers.items()
            if value is not None
            and (f"{plan.path}.{name}" if plan.path else name)
            in self._captured.buffer_names
        )
        required = {leaf for _name, leaf, _canonical in plan.parameters}
        missing = sorted(current - required)
        if required - current:
            raise IncompleteRefit("installed canonical parameter disappeared")
        if missing:
            raise IncompleteRefit(
                f"source replay owner {plan.path!r} gained unplanned parameters={missing}"
            )

    @torch.no_grad()
    def apply(self, batch: SourceReplayBatch) -> None:
        if batch.replay_id != self._captured.replay_id:
            raise IncompleteRefit(
                "captured replay plan was invalidated before installation"
            )
        if not batch.writes:
            raise IncompleteRefit("empty source replay batch")
        for index, recorded, source in batch.writes:
            if (
                index != self._replay_next
                or index >= self._captured.schedule.total_copies
            ):
                raise IncompleteRefit("source replay is not in recorded order")
            owner = self._captured.targets.by_name[recorded.param_name]
            if owner not in self._replay_started:
                for pending_owner in self._captured.schedule.groups[owner]:
                    if pending_owner not in self._replay_started:
                        self._begin_source_owner(pending_owner)
            self._copy_source_write(owner, recorded, source)
            self._replay_next += 1
            for ready in self._captured.schedule.ready.get(index, ()):
                self._finish_source_group(ready)

    def _isolate_native_tied_inputs(self, owners):
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )

        native = {
            owner
            for owner in owners
            if not self._replay_started[owner][0]
            and isinstance(
                getattr(owner.layer, "quant_method", None), QuantizeMethodBase
            )
        }
        if not native:
            return
        slots = {}
        for owner in owners:
            for _, leaf, canonical in owner.parameters:
                slots.setdefault(canonical, {}).setdefault(owner, []).append(leaf)
        allocated = 0
        for consumers in slots.values():
            if len(consumers) < 2:
                continue
            # The last native consumer can use the original only when every
            # other consumer has its own input, including non-native restores.
            last = next(reversed(consumers)) if consumers.keys() <= native else None
            for owner, leaves in consumers.items():
                if owner not in native or owner is last:
                    continue
                original = getattr(owner.layer, leaves[0])
                isolated, nbytes = _clone_native_input(original)
                allocated += nbytes
                for leaf in leaves:
                    setattr(owner.layer, leaf, isolated)
        self.tied_input_peak_bytes = max(self.tied_input_peak_bytes, allocated)

    def _finish_source_group(self, owners):
        self._isolate_native_tied_inputs(owners)
        for owner in owners:
            self._process_owner(owner, self._replay_started[owner][0])
        for owner in owners:
            self._restore_owner(owner, self._replay_started[owner][0])
        for owner in owners:
            aliases = (
                self._captured.engine.runtime.load_aliases
                if self._captured.engine.runtime.load_aliases.by_owner is None
                else self._captured.engine.runtime.load_aliases.by_owner.get(
                    owner.layer
                )
            )
            if aliases is not None:
                self._restore_parameter_aliases(aliases)
            self._replay_started.pop(owner)

    def _copy_source_write(self, owner, recorded, source):
        if self._model.get_submodule(owner.path) is not owner.layer:
            raise IncompleteRefit(f"replay owner {owner.path!r} was replaced")
        leaf = recorded.param_name.rpartition(".")[2]
        target = getattr(owner.layer, leaf)
        if target is not self._replay_started[owner][1][leaf]:
            raise IncompleteRefit(
                f"pending parameter slot {recorded.param_name!r} was replaced"
            )
        destination = _receive_destination(target, bounded=self._bounded)
        if target.dtype != recorded.dest_dtype:
            raise IncompleteRefit("recorded destination has no compatible live storage")
        destination = destination.as_strided(
            recorded.dest_shape,
            recorded.dest_stride,
            target.storage_offset() + recorded.dest_offset,
        )
        value = source
        for op_name, args, frozen_kwargs in recorded.op_chain:
            value = getattr(value, op_name)(*args, **dict(frozen_kwargs))
        if tuple(value.shape) != tuple(recorded.dest_shape):
            raise IncompleteRefit("recorded source shape differs from its destination")
        if type(value) is not torch.Tensor:
            raise IncompleteRefit(
                "weight installation requires an ordinary source tensor"
            )
        torch.Tensor.copy_(destination, value)

    def _begin_source_owner(self, owner):
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
        from vllm.model_executor.model_loader.reload.meta import materialize_layer

        self._validate_source_owner(owner)
        layer = owner.layer
        for name, leaf, _ in owner.parameters:
            if getattr(layer, leaf) is not self._pending[name]():
                raise IncompleteRefit(
                    f"pending parameter slot {name!r} was replaced before its copy"
                )
        info = LAYERWISE_INFO.get(layer)
        if getattr(info, "loaded_weights", ()):
            raise IncompleteRefit("engine reload has pending checkpoint-loader calls")
        persistent = all(
            canonical in self._captured.persistent_names
            and not getattr(layer, leaf).is_meta
            for _, leaf, canonical in owner.parameters
        )
        if info is not None and not persistent:
            materialize_layer(layer, info)
            aliases = (
                self._captured.engine.runtime.load_aliases
                if self._captured.engine.runtime.load_aliases.by_owner is None
                else self._captured.engine.runtime.load_aliases.by_owner.get(layer)
            )
            if aliases is not None:
                self._restore_parameter_aliases(aliases)
                for tied_owner in self._captured.schedule.groups[owner]:
                    for name, leaf, _ in tied_owner.parameters:
                        if any(
                            tied_owner.layer is module and leaf == slot
                            for group in aliases.ties
                            for module, slot in group
                        ):
                            self._pending[name] = ref(getattr(tied_owner.layer, leaf))
        # Reconnect compatible overlapping load-time views to the graph's common
        # backing. Their recorded copies then have the same alias semantics as
        # native loading; later full-owner restoration cannot undo another write.
        for name, leaf, _ in owner.parameters:
            original = self._captured.engine.runtime.overlap_tensors.get(name)
            if original is not None:
                target = getattr(layer, leaf)
                if (
                    target.shape != original.shape
                    or target.dtype != original.dtype
                    or target.stride() != original.stride()
                ):
                    raise IncompleteRefit(
                        f"overlapping runtime destination {name!r} has a different load-time layout"
                    )
                target.data = original.data
        targets = {leaf: getattr(layer, leaf) for _, leaf, _ in owner.parameters}
        for name, leaf, canonical in owner.parameters:
            if name == canonical and canonical not in self._captured.persistent_names:
                targets[leaf].data.zero_()
        self._replay_started[owner] = (persistent, targets)

    def _process_owner(self, owner, persistent):
        if persistent:
            return
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )

        layer = owner.layer
        if hasattr(layer, "_already_called_process_weights_after_loading"):
            delattr(layer, "_already_called_process_weights_after_loading")
        quant_method = getattr(layer, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(layer)
            update_tp = getattr(layer, "update_param_tp_status", None)
            if update_tp is not None:
                update_tp()

    @staticmethod
    def _restore_owner(owner, persistent):
        if persistent:
            return
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
        )

        layer = owner.layer
        info = LAYERWISE_INFO.get(layer)
        if info is not None and info.kernel_tensors is not None:
            _copy_and_restore_kernel_tensors(layer, info)
        if info is not None and not _is_deferred_attention_layer(layer):
            info.reset()

    def finish(self):
        if self._bounded and self._replay_next == 0:
            raise IncompleteRefit(
                "streaming transfer ended before every parameter was installed"
            )
        if (
            self._replay_next != self._captured.schedule.total_copies
            or self._replay_started
        ):
            raise IncompleteRefit(
                "source replay ended before every recorded write was installed"
            )


def _is_deferred_attention_layer(layer):
    from vllm.model_executor.layers import attention

    predicate = getattr(attention, "is_deferred_attention_layer", None)
    if predicate is not None:
        return predicate(layer)
    return isinstance(layer, (attention.Attention, attention.MLAAttention))


def _clone_native_input(tensor):
    span = (
        1
        + sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        if tensor.numel()
        else 0
    )
    storage = tensor.data.as_strided((span,), (1,)).clone()
    isolated = torch.Tensor._make_subclass(
        type(tensor),
        storage.as_strided(tensor.shape, tensor.stride()),
        tensor.requires_grad,
    )
    isolated.__dict__.update(tensor.__dict__)
    return isolated, storage.numel() * storage.element_size()


def _check_install_dispatch(*, bounded: bool) -> None:
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._device import DeviceContext
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    if bounded and (
        any(
            type(mode) is not DeviceContext
            for mode in _get_current_function_mode_stack()
        )
        or _get_current_dispatch_mode() is not None
    ):
        raise IncompleteRefit(
            "weight installation does not support active tensor dispatch modes"
        )


def _receive_destination(target: torch.Tensor, *, bounded: bool) -> torch.Tensor:
    _check_install_dispatch(bounded=bounded)
    destination = target.data
    if type(destination) is not torch.Tensor:
        raise IncompleteRefit(
            "weight installation requires ordinary source and destination tensor views"
        )
    if (
        destination.is_meta
        or destination.data_ptr() != target.data_ptr()
        or destination.shape != target.shape
        or destination.stride() != target.stride()
        or destination.dtype != target.dtype
        or destination.device != target.device
        or destination.storage_offset() != target.storage_offset()
        or destination.untyped_storage().data_ptr()
        != target.untyped_storage().data_ptr()
    ):
        raise IncompleteRefit("load-time parameter does not expose its own storage")
    return destination
