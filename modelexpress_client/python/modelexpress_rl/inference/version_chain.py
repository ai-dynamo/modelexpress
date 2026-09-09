# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared immutable-version chain resolution for generator refit."""

from __future__ import annotations

from collections.abc import Callable

from ..control import WeightVersion
from ..train import WeightPayloadFormat


def resolve_replay_chain(
    *,
    target_version_id: str,
    fetch_ready_version: Callable[[str], WeightVersion],
    max_chain_length: int,
    stop_before_version_id: str | None = None,
) -> tuple[WeightVersion, ...]:
    """Resolve a canonical base-to-target chain before payload preparation."""
    reverse_chain: list[WeightVersion] = []
    seen: set[str] = set()
    revision_id = target_version_id
    layout_signature: str | None = None
    while True:
        if (
            reverse_chain
            and stop_before_version_id is not None
            and revision_id == stop_before_version_id
        ):
            break
        if revision_id in seen:
            raise RuntimeError(
                f"target {target_version_id!r}: cycle detected at revision "
                f"{revision_id!r}"
            )
        if len(reverse_chain) >= max_chain_length:
            raise RuntimeError(
                f"target {target_version_id!r}: replay exceeds maximum chain "
                f"length {max_chain_length} before revision {revision_id!r}"
            )
        seen.add(revision_id)
        try:
            version = fetch_ready_version(revision_id)
        except RuntimeError as error:
            if stop_before_version_id is not None and revision_id != target_version_id:
                raise RuntimeError(
                    f"target {target_version_id!r}: chain does not match serving "
                    f"version {stop_before_version_id!r}; {error}"
                ) from error
            raise
        if version.object_storage is None:
            raise RuntimeError(
                f"target {target_version_id!r}: no legal source for revision "
                f"{revision_id!r}; object-storage source is missing"
            )
        if version.layout_signature:
            if layout_signature is None:
                layout_signature = version.layout_signature
            elif version.layout_signature != layout_signature:
                raise RuntimeError(
                    f"target {target_version_id!r}: layout format mismatch at "
                    f"revision {revision_id!r}"
                )
        reverse_chain.append(version)
        if version.version_id == stop_before_version_id:
            break
        if version.payload_format is WeightPayloadFormat.FULL_HF_CHECKPOINT:
            if version.base_version_id is not None:
                raise RuntimeError(
                    f"target {target_version_id!r}: FULL_HF_CHECKPOINT revision "
                    f"{revision_id!r} must not have base_version_id"
                )
            break
        if version.payload_format is not WeightPayloadFormat.XOR_DELTA:
            raise RuntimeError(
                f"target {target_version_id!r}: revision {revision_id!r} has "
                f"unsupported replay format {version.payload_format.value}"
            )
        if version.base_version_id is None:
            raise RuntimeError(
                f"target {target_version_id!r}: XOR_DELTA revision "
                f"{revision_id!r} is missing base_version_id"
            )
        revision_id = version.base_version_id

    return tuple(reversed(reverse_chain))


__all__ = ["resolve_replay_chain"]
