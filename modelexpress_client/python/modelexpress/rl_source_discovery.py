# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source discovery helpers for framework-agnostic RL weight transfer."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence

from modelexpress import p2p_pb2
from modelexpress.client import MxClientBase
from modelexpress.rl_metadata import (
    RlSourceCandidate,
    RlSourceRole,
    select_rl_source_candidates,
)
from modelexpress.rl_transfer_identity import candidates_for_base_identity

DEFAULT_RECEIVE_ROLES = (RlSourceRole.INFERENCE_REPLICA, RlSourceRole.TRAINER)


async def wait_for_source(
    *,
    mx_client: MxClientBase,
    base_identity: "p2p_pb2.SourceIdentity",
    worker_id: str,
    timeout_seconds: float,
    model_version: int | None,
    receiver_rank: int,
    roles: Sequence[RlSourceRole] = DEFAULT_RECEIVE_ROLES,
    same_rank_only: bool = False,
    source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
    require_complete_version: bool = True,
) -> RlSourceCandidate:
    """Poll MX metadata until the best matching source is visible."""
    return (
        await wait_for_sources(
            mx_client=mx_client,
            base_identity=base_identity,
            worker_id=worker_id,
            timeout_seconds=timeout_seconds,
            model_version=model_version,
            receiver_rank=receiver_rank,
            roles=roles,
            same_rank_only=same_rank_only,
            source_ranks_by_role=source_ranks_by_role,
            require_complete_version=require_complete_version,
        )
    )[0]


async def wait_for_sources(
    *,
    mx_client: MxClientBase,
    base_identity: "p2p_pb2.SourceIdentity",
    worker_id: str,
    timeout_seconds: float,
    model_version: int | None,
    receiver_rank: int,
    roles: Sequence[RlSourceRole] = DEFAULT_RECEIVE_ROLES,
    same_rank_only: bool = False,
    source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
    require_complete_version: bool = True,
) -> list[RlSourceCandidate]:
    """Poll MX metadata until matching requested/latest sources are visible."""
    deadline = time.monotonic() + timeout_seconds
    last_error: RuntimeError | None = None
    while True:
        try:
            return select_sources(
                mx_client=mx_client,
                base_identity=base_identity,
                worker_id=worker_id,
                model_version=model_version,
                receiver_rank=receiver_rank,
                roles=roles,
                same_rank_only=same_rank_only,
                source_ranks_by_role=source_ranks_by_role,
                require_complete_version=require_complete_version,
            )
        except RuntimeError as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise last_error
            await asyncio.sleep(0.25)


def select_source(
    *,
    mx_client: MxClientBase,
    base_identity: "p2p_pb2.SourceIdentity",
    worker_id: str,
    model_version: int | None,
    receiver_rank: int,
    roles: Sequence[RlSourceRole] = DEFAULT_RECEIVE_ROLES,
    same_rank_only: bool = False,
    source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
    require_complete_version: bool = True,
) -> RlSourceCandidate:
    """Select the best source for a receiver from MX metadata."""
    return select_sources(
        mx_client=mx_client,
        base_identity=base_identity,
        worker_id=worker_id,
        model_version=model_version,
        receiver_rank=receiver_rank,
        roles=roles,
        same_rank_only=same_rank_only,
        source_ranks_by_role=source_ranks_by_role,
        require_complete_version=require_complete_version,
    )[0]


def select_sources(
    *,
    mx_client: MxClientBase,
    base_identity: "p2p_pb2.SourceIdentity",
    worker_id: str,
    model_version: int | None,
    receiver_rank: int,
    roles: Sequence[RlSourceRole] = DEFAULT_RECEIVE_ROLES,
    same_rank_only: bool = False,
    source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
    require_complete_version: bool = True,
) -> list[RlSourceCandidate]:
    """Select source candidates for a requested or latest model version."""
    response = mx_client.list_sources(
        identity=None,
        status_filter=p2p_pb2.SOURCE_STATUS_READY,
    )
    candidates = candidates_for_base_identity(response, base_identity)
    candidates = [
        candidate
        for candidate in candidates
        if candidate.model_name == base_identity.model_name
        and candidate.worker_id != worker_id
    ]
    selected = select_rl_source_candidates(
        candidates,
        receiver_rank=receiver_rank,
        model_version=model_version,
        roles=roles,
        same_rank_only=same_rank_only,
        source_ranks_by_role=source_ranks_by_role,
        require_complete_version=require_complete_version,
    )
    if not selected:
        raise RuntimeError(
            f"No ModelExpress RL source found for model={base_identity.model_name!r} "
            f"version={version_label(model_version)}"
        )
    return selected


def version_label(model_version: int | None) -> str:
    return "latest" if model_version is None else str(model_version)
