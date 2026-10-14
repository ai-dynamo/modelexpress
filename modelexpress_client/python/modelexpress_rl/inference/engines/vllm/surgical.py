# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the checkpoint tensors a delta install must reload.

vLLM layerwise reload processes a module only once every one of its load-time
tensors has arrived, so a delta installs whole destination modules: a changed
expert reloads its fused MoE module, a changed ``up_proj`` also loads
``gate_proj``. Untouched modules keep their existing kernel tensors.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from modelexpress_rl.inference.receiver import DeltaChange

_EXPERT = re.compile(r"\.experts\.\d+\.[^.]+$")
_DEFAULT_PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
}


def _module_key(name: str, fused: Mapping[str, str]) -> str:
    module = _EXPERT.sub(".experts", name.rpartition(".")[0])
    parent, _, leaf = module.rpartition(".")
    leaf = fused.get(leaf, leaf)
    return f"{parent}.{leaf}" if parent else leaf


def module_groups(
    changed: Iterable[str],
    checkpoint_names: Iterable[str],
    packed_modules: Mapping[str, Sequence[str]],
) -> set[str]:
    """Return every checkpoint tensor of each module a changed tensor loads into."""
    fused = {
        part: parent
        for mapping in (_DEFAULT_PACKED, packed_modules)
        for parent, parts in mapping.items()
        for part in parts
    }
    by_module: dict[str, list[str]] = defaultdict(list)
    for name in checkpoint_names:
        by_module[_module_key(name, fused)].append(name)
    changed = set(changed)
    missing = changed - {name for names in by_module.values() for name in names}
    if missing:
        raise ValueError(
            f"changed tensors not in the checkpoint: {sorted(missing)[:5]}"
        )
    return {
        name
        for key in {_module_key(name, fused) for name in changed}
        for name in by_module[key]
    }


def changed_since(
    lineage: Sequence[DeltaChange],
    *,
    live_version: str | None,
    target_version: str,
) -> frozenset[str] | None:
    """Names that differ between the live engine and the target, if known."""
    if live_version is None or not lineage or lineage[-1].version != target_version:
        return None
    for position, change in enumerate(lineage):
        if change.base_version == live_version:
            return frozenset().union(*(c.tensor_names for c in lineage[position:]))
    return None


__all__ = ["changed_since", "module_groups"]
