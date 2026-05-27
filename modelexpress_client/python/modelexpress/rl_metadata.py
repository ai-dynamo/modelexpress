# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL weight-sync metadata helpers.

These helpers keep the first RL-specific contract inside the Python
client and encode it through SourceIdentity.extra_parameters so existing
servers and metadata backends remain compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from . import p2p_pb2

RL_SCHEMA_VERSION = "1"

RL_SCHEMA_VERSION_KEY = "mx_rl_schema_version"
RL_ROLE_KEY = "mx_rl_role"
RL_MODEL_VERSION_KEY = "mx_rl_model_version"
RL_WORLD_SIZE_KEY = "mx_rl_world_size"
RL_RETAIN_LATEST_K_KEY = "mx_rl_retain_latest_k"
RL_SHAPE_REGISTRY_KEY = "mx_rl_shape_registry_json_hex"
_RL_EXTRA_PARAMETER_KEYS = {
    RL_SCHEMA_VERSION_KEY,
    RL_ROLE_KEY,
    RL_MODEL_VERSION_KEY,
    RL_WORLD_SIZE_KEY,
    RL_RETAIN_LATEST_K_KEY,
    RL_SHAPE_REGISTRY_KEY,
}
_READY_OR_LEGACY_SOURCE_STATUSES = frozenset(
    (p2p_pb2.SOURCE_STATUS_UNKNOWN, p2p_pb2.SOURCE_STATUS_READY)
)


class RlSourceRole(str, Enum):
    """Role a source plays in an RL weight-sync update graph."""

    TRAINER = "trainer"
    INFERENCE_REPLICA = "inference_replica"


@dataclass(frozen=True)
class RlSourceMetadata:
    """Framework-agnostic metadata for one RL model-version source group.

    This describes the source group identity. Per-worker rank stays in
    WorkerMetadata/SourceInstanceRef so all ranks for one model version
    remain grouped under the same mx_source_id.
    """

    model_version: int
    role: RlSourceRole
    world_size: int
    retain_latest_k: int = 1
    shape_registry: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        role = self.role
        if not isinstance(role, RlSourceRole):
            role = RlSourceRole(str(role))
            object.__setattr__(self, "role", role)

        if self.model_version < 0:
            raise ValueError("model_version must be non-negative")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.retain_latest_k <= 0:
            raise ValueError("retain_latest_k must be positive")

    def to_extra_parameters(self) -> dict[str, str]:
        """Serialize as SourceIdentity.extra_parameters entries."""
        extra = {
            RL_SCHEMA_VERSION_KEY: RL_SCHEMA_VERSION,
            RL_ROLE_KEY: self.role.value,
            RL_MODEL_VERSION_KEY: str(self.model_version),
            RL_WORLD_SIZE_KEY: str(self.world_size),
            RL_RETAIN_LATEST_K_KEY: str(self.retain_latest_k),
        }
        if self.shape_registry:
            extra[RL_SHAPE_REGISTRY_KEY] = _encode_shape_registry(self.shape_registry)
        return extra

    @classmethod
    def from_extra_parameters(cls, extra_parameters: Mapping[str, str]) -> "RlSourceMetadata":
        """Deserialize RL metadata from SourceIdentity.extra_parameters."""
        schema_version = extra_parameters.get(RL_SCHEMA_VERSION_KEY, "")
        if schema_version != RL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported RL metadata schema version {schema_version!r}; "
                f"expected {RL_SCHEMA_VERSION!r}"
            )

        shape_registry = {}
        shape_registry_hex = extra_parameters.get(RL_SHAPE_REGISTRY_KEY, "")
        if shape_registry_hex:
            shape_registry = _decode_shape_registry(shape_registry_hex)

        try:
            return cls(
                model_version=int(extra_parameters[RL_MODEL_VERSION_KEY]),
                role=RlSourceRole(extra_parameters[RL_ROLE_KEY]),
                world_size=int(extra_parameters[RL_WORLD_SIZE_KEY]),
                retain_latest_k=int(extra_parameters.get(RL_RETAIN_LATEST_K_KEY, "1")),
                shape_registry=shape_registry,
            )
        except KeyError as exc:
            raise ValueError(f"missing RL metadata field {exc.args[0]!r}") from exc


@dataclass(frozen=True)
class RlSourceCandidate:
    """A source worker annotated with the RL identity metadata used to query it."""

    mx_source_id: str
    worker_id: str
    model_name: str
    worker_rank: int
    metadata: RlSourceMetadata
    status: int = p2p_pb2.SOURCE_STATUS_UNKNOWN
    updated_at: int = 0

    @classmethod
    def from_ref(
        cls,
        ref: "p2p_pb2.SourceInstanceRef",
        metadata: RlSourceMetadata,
    ) -> "RlSourceCandidate":
        return cls(
            mx_source_id=ref.mx_source_id,
            worker_id=ref.worker_id,
            model_name=ref.model_name,
            worker_rank=ref.worker_rank,
            metadata=metadata,
            status=int(ref.status),
            updated_at=int(ref.updated_at),
        )


def with_rl_source_metadata(
    identity: "p2p_pb2.SourceIdentity",
    metadata: RlSourceMetadata,
) -> "p2p_pb2.SourceIdentity":
    """Return a copy of identity with RL metadata merged into extra_parameters."""
    result = p2p_pb2.SourceIdentity()
    result.CopyFrom(identity)
    for key in _RL_EXTRA_PARAMETER_KEYS:
        result.extra_parameters.pop(key, None)
    result.extra_parameters.update(metadata.to_extra_parameters())
    return result


def get_rl_source_metadata(identity: "p2p_pb2.SourceIdentity") -> RlSourceMetadata:
    """Parse required RL metadata from a SourceIdentity."""
    return RlSourceMetadata.from_extra_parameters(identity.extra_parameters)


def try_get_rl_source_metadata(
    identity: "p2p_pb2.SourceIdentity",
) -> RlSourceMetadata | None:
    """Parse RL metadata when present, returning None for non-RL identities."""
    if RL_SCHEMA_VERSION_KEY not in identity.extra_parameters:
        return None
    return get_rl_source_metadata(identity)


def build_rl_query_identities(
    base_identity: "p2p_pb2.SourceIdentity",
    *,
    model_version: int,
    world_size: int,
    roles: Sequence[RlSourceRole] = (
        RlSourceRole.INFERENCE_REPLICA,
        RlSourceRole.TRAINER,
    ),
    retain_latest_k: int = 1,
    shape_registry: Mapping[str, Any] | None = None,
) -> list["p2p_pb2.SourceIdentity"]:
    """Build role-specific identities for source discovery.

    Callers can query each returned identity via list_sources and annotate
    the response refs with the same metadata using candidates_from_response.
    """
    identities = []
    for role in roles:
        metadata = RlSourceMetadata(
            model_version=model_version,
            role=role,
            world_size=world_size,
            retain_latest_k=retain_latest_k,
            shape_registry=shape_registry or {},
        )
        identities.append(with_rl_source_metadata(base_identity, metadata))
    return identities


def candidates_from_response(
    response: "p2p_pb2.ListSourcesResponse",
    metadata: RlSourceMetadata | None = None,
) -> list[RlSourceCandidate]:
    """Annotate ListSources refs with RL metadata.

    Older servers do not include SourceIdentity in SourceInstanceRef, so
    callers can pass the metadata used for an exact identity query. Newer
    servers include ref.identity, allowing callers to list broadly and
    discover version/role metadata from the response itself.
    """
    candidates = []
    for ref in response.instances:
        candidate_metadata = metadata
        if candidate_metadata is None:
            if not ref.HasField("identity"):
                continue
            candidate_metadata = try_get_rl_source_metadata(ref.identity)
            if candidate_metadata is None:
                continue
        candidates.append(RlSourceCandidate.from_ref(ref, candidate_metadata))
    return candidates


def source_candidate_is_ready(candidate: RlSourceCandidate) -> bool:
    """Return true for READY refs and legacy refs without discovery status."""
    return int(candidate.status) in _READY_OR_LEGACY_SOURCE_STATUSES


def latest_model_version(candidates: Iterable[RlSourceCandidate]) -> int | None:
    """Return the latest model version among candidates, or None if empty."""
    latest: int | None = None
    for candidate in candidates:
        version = candidate.metadata.model_version
        latest = version if latest is None else max(latest, version)
    return latest


def retained_model_versions(candidates: Iterable[RlSourceCandidate]) -> set[int]:
    """Return model versions inside the latest advertised retention window.

    The latest visible version is the best source of the current retention
    policy. If multiple source groups advertise that version, keep the widest
    window so one role cannot accidentally hide another role's retained source.
    """
    candidate_list = list(candidates)
    latest = latest_model_version(candidate_list)
    if latest is None:
        return set()

    retain_latest_k = max(
        candidate.metadata.retain_latest_k
        for candidate in candidate_list
        if candidate.metadata.model_version == latest
    )
    oldest_retained = max(0, latest - retain_latest_k + 1)
    return {
        candidate.metadata.model_version
        for candidate in candidate_list
        if oldest_retained <= candidate.metadata.model_version <= latest
    }


def complete_source_groups(
    candidates: Iterable[RlSourceCandidate],
) -> set[tuple[int, RlSourceRole]]:
    """Return version/role groups with all advertised ranks READY.

    The caller should pass candidates already filtered to one base identity.
    Legacy refs without discovery status are treated as READY because older
    servers only applied status through the ListSources filter.
    """
    groups: dict[tuple[int, RlSourceRole], tuple[set[int], int]] = {}
    for candidate in candidates:
        if not source_candidate_is_ready(candidate):
            continue
        key = (candidate.metadata.model_version, candidate.metadata.role)
        ranks, required_world_size = groups.setdefault(
            key,
            (set(), candidate.metadata.world_size),
        )
        ranks.add(candidate.worker_rank)
        if candidate.metadata.world_size > required_world_size:
            groups[key] = (ranks, candidate.metadata.world_size)

    return {
        group
        for group, (ranks, required_world_size) in groups.items()
        if len(ranks) >= required_world_size
    }


def complete_model_versions(candidates: Iterable[RlSourceCandidate]) -> set[int]:
    """Return model versions where at least one role group is complete."""
    return {
        version
        for version, _role in complete_source_groups(candidates)
    }


def latest_complete_model_version(candidates: Iterable[RlSourceCandidate]) -> int | None:
    """Return the latest complete model version among candidates."""
    versions = complete_model_versions(candidates)
    return max(versions) if versions else None


def select_rl_source_candidates(
    candidates: Iterable[RlSourceCandidate],
    *,
    receiver_rank: int,
    model_version: int | None = None,
    roles: Sequence[RlSourceRole] = (
        RlSourceRole.INFERENCE_REPLICA,
        RlSourceRole.TRAINER,
    ),
    same_rank_only: bool = True,
    require_complete_version: bool = True,
    source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
) -> list[RlSourceCandidate]:
    """Filter and order RL source candidates for a receiver rank.

    The default policy matches the first CE-parity milestone: prefer
    complete source groups, prefer same-rank sources, prefer inference
    replicas over trainer sources, honor the advertised retention window, and
    use the latest visible model version when a specific version is not
    requested.
    """
    role_priority = {role: index for index, role in enumerate(roles)}
    source_rank_filter = None
    if source_ranks_by_role is not None:
        source_rank_filter = {
            (role if isinstance(role, RlSourceRole) else RlSourceRole(str(role))): {
                int(rank) for rank in ranks
            }
            for role, ranks in source_ranks_by_role.items()
        }

    role_filtered = [
        candidate
        for candidate in candidates
        if source_candidate_is_ready(candidate)
        and candidate.metadata.role in role_priority
        and (
            source_rank_filter is None
            or candidate.worker_rank in source_rank_filter.get(candidate.metadata.role, set())
        )
    ]
    if not role_filtered:
        return []

    complete_groups = (
        complete_source_groups(role_filtered)
        if require_complete_version
        else None
    )
    version_pool = [
        candidate
        for candidate in role_filtered
        if complete_groups is None
        or (candidate.metadata.model_version, candidate.metadata.role) in complete_groups
    ]
    if not version_pool:
        return []

    retained_versions = retained_model_versions(version_pool)
    version_pool = [
        candidate
        for candidate in version_pool
        if candidate.metadata.model_version in retained_versions
    ]
    if not version_pool:
        return []

    target_version = (
        model_version
        if model_version is not None
        else latest_model_version(version_pool)
    )
    if target_version is None:
        return []

    filtered = [
        candidate
        for candidate in version_pool
        if candidate.metadata.model_version == target_version
        and (not same_rank_only or candidate.worker_rank == receiver_rank)
    ]

    return sorted(
        filtered,
        key=lambda candidate: (
            role_priority[candidate.metadata.role],
            candidate.worker_rank != receiver_rank,
            candidate.worker_rank,
            candidate.worker_id,
        ),
    )


def _encode_shape_registry(shape_registry: Mapping[str, Any]) -> str:
    payload = json.dumps(shape_registry, separators=(",", ":"), sort_keys=True)
    return payload.encode("utf-8").hex()


def _decode_shape_registry(shape_registry_hex: str) -> dict[str, Any]:
    try:
        payload = bytes.fromhex(shape_registry_hex).decode("utf-8")
        value = json.loads(payload)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid RL shape registry encoding") from exc
    if not isinstance(value, dict):
        raise ValueError("RL shape registry must decode to a JSON object")
    return value
