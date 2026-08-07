# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audited native-version boundaries for framework adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from rlxfer.errors import CompatibilityError


@dataclass(frozen=True, slots=True)
class AdapterSupport:
    """One adapter's version and verified native API range."""

    adapter_version: str
    native_prefixes: tuple[str, ...]
    verified_revision: str


SUPPORT: Mapping[str, AdapterSupport] = MappingProxyType(
    {
        "miles": AdapterSupport("0.1.0", ("0.2",), "319716c"),
        "nemo_rl": AdapterSupport("0.1.0", ("0.5", "0.6", "0.7"), "daf46ff/81aa43d"),
        "prime_rl": AdapterSupport("0.1.0", ("0.5",), "2873bf2"),
        "slime": AdapterSupport("0.1.0", ("0.3",), "a6272da"),
    }
)


def verify_framework_version(framework: str, detected: str) -> None:
    """Reject a known version outside the audited native API range."""

    if detected in {"unknown", "unavailable"}:
        return
    support = SUPPORT[framework]
    revisions = tuple(support.verified_revision.split("/"))
    if not detected.startswith((*support.native_prefixes, *revisions)):
        expected = ", ".join(f"{prefix}.x" for prefix in support.native_prefixes)
        raise CompatibilityError(
            f"{framework} {detected} is outside adapter {support.adapter_version}'s "
            f"audited versions ({expected}); install a verified framework revision or "
            "add a version-specific compatibility implementation"
        )
