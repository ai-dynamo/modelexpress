# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-independent checkpoint dependency selection and source staging.

Dependency groups must come from an audited engine adapter, not name heuristics.
The engine adapter supplies the bindings and executes their refresh operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CheckpointChanges:
    """Conservative changed-name union relative to an exact checkpoint version."""

    base_version: str
    target_version: str
    names: frozenset[str]


@dataclass(frozen=True)
class DependencyGroup:
    """Complete checkpoint inputs and refresh operations for one engine group.

    Overlapping source sets express dependencies such as tied storage or a
    projection consumed by an attention finalizer. Refresh identifiers are
    opaque to common code; they do not authorize invoking arbitrary callbacks.
    """

    name: str
    sources: frozenset[str]
    refresh: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CheckpointSelection:
    sources: frozenset[str] = frozenset()
    groups: frozenset[str] = frozenset()
    refresh: frozenset[str] = frozenset()
    fallback_reason: str | None = None


def select_checkpoint_sources(
    changes: CheckpointChanges | None,
    *,
    serving_version: str,
    target_version: str,
    checkpoint_names: frozenset[str],
    groups: tuple[DependencyGroup, ...],
) -> CheckpointSelection:
    """Close audited dependency groups, or request full reload before mutation.

    An empty, verified delta is distinct from unavailable change metadata.
    Unmapped sources are unsupported even if they resemble a known parameter.
    """
    if changes is None:
        return CheckpointSelection(fallback_reason="change metadata unavailable")
    if (
        not serving_version
        or not target_version
        or changes.base_version != serving_version
        or changes.target_version != target_version
    ):
        return CheckpointSelection(fallback_reason="engine version mismatch")
    if not changes.names <= checkpoint_names:
        return CheckpointSelection(fallback_reason="unknown changed tensor")
    group_names = [group.name for group in groups]
    if len(set(group_names)) != len(group_names) or any(
        not group.name or not group.sources or not group.sources <= checkpoint_names
        for group in groups
    ):
        return CheckpointSelection(fallback_reason="invalid dependency catalog")
    supported = frozenset().union(*(group.sources for group in groups))
    if not changes.names <= supported:
        return CheckpointSelection(fallback_reason="unsupported changed tensor")

    selected = set(changes.names)
    selected_groups: set[str] = set()
    refresh: set[str] = set()
    while True:
        previous = len(selected_groups)
        for group in groups:
            if group.sources.intersection(selected):
                selected.update(group.sources)
                selected_groups.add(group.name)
                refresh.update(group.refresh)
        if len(selected_groups) == previous:
            break
    return CheckpointSelection(
        frozenset(selected), frozenset(selected_groups), frozenset(refresh)
    )


def read_selected_tensors(
    selection: CheckpointSelection,
    *,
    checkpoint: Path,
    weight_map: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Stage owned CPU tensors from a locked, verified canonical checkpoint.

    This reads complete source tensors, not TP slices. It never loads unrelated
    shard payloads or mutates engine state. All tensors are staged before return;
    a read failure cannot expose a partial result to an installer.
    """
    from safetensors import safe_open

    if selection.fallback_reason is not None:
        raise ValueError(f"full reload required: {selection.fallback_reason}")
    root = checkpoint.resolve(strict=True)
    by_shard: dict[Path, list[str]] = {}
    for name in sorted(selection.sources):
        filename = weight_map.get(name)
        if (
            not isinstance(filename, str)
            or not filename.endswith(".safetensors")
            or Path(filename).name != filename
        ):
            raise ValueError(f"invalid shard mapping for {name!r}")
        path = (root / filename).resolve(strict=True)
        if path.parent != root:
            raise ValueError(f"shard escapes checkpoint for {name!r}")
        by_shard.setdefault(path, []).append(name)

    staged = {}
    for path, names in by_shard.items():
        with safe_open(path, framework="pt", device="cpu") as reader:
            if not set(names) <= set(reader.keys()):
                raise ValueError(f"selected tensors missing from {path.name!r}")
            for name in names:
                staged[name] = reader.get_tensor(name).clone()
    return staged
