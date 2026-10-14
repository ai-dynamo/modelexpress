# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-lifetime engine requirements, source bindings, and update execution state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from weakref import ref

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
    order: int
    path: str
    layer_ref: ref
    slots: tuple[_InputSlot, ...]


@dataclass(frozen=True, eq=False)
class _LayerInstallPlan:
    order: int
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
    buffer_names: frozenset[str]
    persistent_names: frozenset[str]


@dataclass(frozen=True, eq=False)
class EngineInstallPlan:
    """Fixed engine slots, alias relationships and runtime restoration bindings."""

    runtime: _EngineRuntimeBindings
    inputs: tuple[_InputOwner, ...]
    input_layout: dict
    persistent_names: frozenset[str]

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
            layer = _LayerInstallPlan(
                owner.order, owner.path, owner.layer_ref, tuple(parameters)
            )
            owners.append(layer)
            for name, _, first in parameters:
                if name == first:
                    by_name[name] = layer
        return _InstallTargets(
            by_name,
            tuple(owners),
            buffer_names,
            self.persistent_names & by_name.keys(),
        )


@dataclass(frozen=True, eq=False)
class _ReplaySchedule:
    last_writes: dict[str, int]
    total_copies: int
    groups: dict
    ready: dict

    @classmethod
    def compile(cls, targets, overlap_groups, last_writes, total_copies):
        last = {
            owner: max(last_writes[name] for _, _, name in owner.parameters)
            for owner in targets.owners
        }
        links = {owner: set() for owner in last}
        owners_by_name = {
            name: owner for owner in targets.owners for name, _, _ in owner.parameters
        }
        canonical_slots = {}
        for owner in targets.owners:
            for name, _, canonical in owner.parameters:
                canonical_slots.setdefault(canonical, []).append(name)
        for names in (*overlap_groups, *canonical_slots.values()):
            owners = tuple(
                dict.fromkeys(
                    owners_by_name[name] for name in names if name in owners_by_name
                )
            )
            for owner in owners[1:]:
                links[owners[0]].add(owner)
                links[owner].add(owners[0])
        groups, ready = {}, {}
        remaining = set(links)
        while remaining:
            first = remaining.pop()
            component, pending = {first}, [first]
            while pending:
                owner = pending.pop()
                for neighbor in links[owner] - component:
                    remaining.discard(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
            group = tuple(sorted(component, key=lambda owner: owner.order))
            ready.setdefault(max(last[owner] for owner in group), []).append(group)
            for owner in group:
                groups[owner] = group
        return cls(dict(last_writes), total_copies, groups, ready)


@dataclass(frozen=True, eq=False)
class SourceInstallPlan:
    """Captured engine writes and their client-lifetime completion schedule."""

    engine: EngineInstallPlan
    generation: int
    targets: _InstallTargets
    schedule: _ReplaySchedule


@dataclass(eq=False)
class _InstallExecution:
    """Mutable progress and borrowed references owned by one weight update."""

    engine: EngineInstallPlan
    source_plan: SourceInstallPlan | None
    targets: _InstallTargets
    installed: dict = field(default_factory=dict)
    bounded: bool = False
    pending: dict = field(default_factory=dict)
    replay_started: dict = field(default_factory=dict)
    replay_next: int = 0
    replay_metadata: tuple | None = None
    replay_initialized: set = field(default_factory=set)
    schedule: _ReplaySchedule | None = None

    @property
    def aliases(self):
        return self.engine.runtime.load_aliases

    @property
    def owner_aliases(self):
        return self.aliases.by_owner

    @property
    def by_name(self):
        return self.targets.by_name

    @property
    def owners(self):
        return self.targets.owners

    @property
    def replay_total(self):
        return self.schedule.total_copies if self.schedule is not None else 0

    @property
    def replay_groups(self):
        return self.schedule.groups

    @property
    def replay_ready(self):
        return self.schedule.ready
