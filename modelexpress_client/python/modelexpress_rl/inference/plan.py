# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed composition contracts for one generator weight update."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from modelexpress import p2p_pb2

from ..control import TrainerMesh, WeightVersion
from ..version import TrainerMeshRef
from ..object_storage import ObjectStorageSource
from ..train.manifest import ShardChecksumKey
from ..train import WeightPayloadFormat
from .adapter import ReplicaGroup, TrainerSourceShard

if TYPE_CHECKING:
    from .nixl_staged_transfer import _ResolvedSources
    from .receiver import PreparedCheckpoint


@dataclass(frozen=True)
class StreamingSettings:
    """Fixed receive capacity and placement for one bounded runtime."""

    max_staging_bytes: int
    staging_device: str = "cuda"
    staging_buffers: int = 1


class WeightSource(str, Enum):
    """Location from which a generator obtains weight bytes."""

    TRAINER = "TRAINER"
    GENERATOR = "GENERATOR"
    OBJECT_STORAGE = "OBJECT_STORAGE"


def parse_weight_source_order(value: str) -> tuple[WeightSource, ...]:
    """Parse a comma-separated weight-source fallback order."""
    names = tuple(item.strip().upper() for item in value.split(","))
    if not names or any(not name for name in names):
        raise ValueError("MX_GENERATOR_SOURCE_ORDER must be a comma-separated list")
    try:
        sources = tuple(WeightSource(name) for name in names)
    except ValueError as exc:
        choices = ", ".join(source.value for source in WeightSource)
        raise ValueError(
            f"MX_GENERATOR_SOURCE_ORDER entries must be one of: {choices}"
        ) from exc
    if len(set(sources)) != len(sources):
        raise ValueError("MX_GENERATOR_SOURCE_ORDER must not contain duplicates")
    return sources


@dataclass(frozen=True)
class MethodCapabilities:
    """Combinations accepted and produced by an update method."""

    payload_formats: frozenset[WeightPayloadFormat]
    sources: frozenset[WeightSource]
    artifact_type: type[PreparedArtifact]
    requires_base_version: bool = False


@dataclass(frozen=True)
class EngineCapabilities:
    """Prepared artifacts that one engine installer can commit."""

    artifact_types: frozenset[type[PreparedArtifact]]


@dataclass(frozen=True)
class GeneratorPeerUpdateSource:
    """A generator peer that already serves the requested version."""

    worker: p2p_pb2.WorkerMetadata
    mx_source_id: str
    worker_id: str
    kind = WeightSource.GENERATOR
    payload_format = WeightPayloadFormat.FULL_TENSOR


@dataclass(frozen=True)
class TrainerSourceSnapshot:
    """Selected trainer shards and the mesh identity they belong to."""

    mesh_id: str
    mesh_generation: int
    shards: tuple[TrainerSourceShard, ...]
    replica_groups: tuple[ReplicaGroup, ...] = ()
    resolved_metadata: _ResolvedSources | None = field(
        default=None, compare=False, repr=False
    )
    kind = WeightSource.TRAINER
    payload_format = WeightPayloadFormat.FULL_TENSOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "shards", tuple(self.shards))
        object.__setattr__(self, "replica_groups", tuple(self.replica_groups))

    @property
    def physical_fingerprint(self) -> tuple:
        return (
            self.mesh_id,
            self.mesh_generation,
            tuple(
                (shard.source_slot_id, shard.worker_id)
                for shard in self.shards
            ),
        )


@dataclass(frozen=True)
class ObjectStorageUpdateSource:
    """Canonical payload published in object storage."""

    storage: ObjectStorageSource
    payload_format: WeightPayloadFormat
    kind = WeightSource.OBJECT_STORAGE


@dataclass(frozen=True)
class ResolvedTrainerSource:
    """A stable trainer selection and checksums owned by this leased round."""

    snapshot: TrainerSourceSnapshot
    checksums: Mapping[ShardChecksumKey, str] = field(default_factory=dict)
    prepare_groups: Callable[[frozenset[str]], None] | None = field(
        default=None, compare=False, repr=False
    )
    kind = WeightSource.TRAINER
    payload_format = WeightPayloadFormat.FULL_TENSOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "checksums", MappingProxyType(dict(self.checksums)))


ResolvedSource = (
    GeneratorPeerUpdateSource | ResolvedTrainerSource | ObjectStorageUpdateSource
)


class StagedEngineTensors(Protocol):
    """Tensor staging payload required by engine installers."""

    tensors: dict[str, Any]
    metrics: dict[str, Any]


class PreparedArtifact(ABC):
    """Typed method output handed to an engine installer."""

    @property
    @abstractmethod
    def metrics(self) -> dict[str, float]:
        """Return metrics recorded while preparing or applying this artifact."""


@dataclass(frozen=True)
class PreparedEngineTensors(PreparedArtifact):
    staged: StagedEngineTensors

    @property
    def metrics(self) -> dict[str, float]:
        return dict(getattr(self.staged, "metrics", {}))


@dataclass
class _StreamingOwnership:
    """Retain receive resources until both GPU work and transport are drained."""

    iterator: Iterator[dict[str, Any]] | None = None
    drain_failed: bool = False
    close_failed: bool = False
    source_failed: bool = False

    @property
    def release_blocked(self) -> bool:
        return (
            self.iterator is not None
            or self.drain_failed
            or self.close_failed
            or self.source_failed
        )


@dataclass(frozen=True)
class PreparedStreamingTensors(PreparedArtifact):
    """Deferred bounded transfer; payload is read only during installation."""

    batches: Callable[[], Iterator[dict[str, Any]]]
    parameter_names: frozenset[str]
    transfer_metrics: dict[str, float]
    ownership: _StreamingOwnership = field(default_factory=_StreamingOwnership)

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.transfer_metrics)


@dataclass(frozen=True)
class PreparedRuntimeTensors(PreparedArtifact):
    """A peer read prepared to write directly into live runtime tensors."""

    staged: StagedEngineTensors

    @property
    def metrics(self) -> dict[str, float]:
        return dict(getattr(self.staged, "metrics", {}))


@dataclass(frozen=True)
class PreparedCheckpointArtifact(PreparedArtifact):
    checkpoint: PreparedCheckpoint

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.checkpoint.metrics)


class SourceResolver(ABC):
    """Resolve one class of source without transferring weight bytes."""

    needs_mesh: bool = False

    @property
    @abstractmethod
    def kind(self) -> WeightSource:
        """Return the source kind produced by this resolver."""

    @abstractmethod
    def supports(self, version: WeightVersion) -> bool:
        """Return whether this resolver can represent the version's source."""

    @abstractmethod
    def payload_format(self, version: WeightVersion) -> WeightPayloadFormat:
        """Return the payload representation produced for this version."""

    def prepare_inputs(self, **kwargs: Any) -> Any:
        """Resolve source-specific method inputs, or None when none are needed."""
        return None

    @abstractmethod
    def candidates(self, version: WeightVersion) -> Iterator[ResolvedSource]:
        """Yield verified candidates in fallback order."""


class UpdateMethod(ABC):
    """Stage an update and protect method state while the engine applies it."""

    @property
    @abstractmethod
    def capabilities(self) -> MethodCapabilities:
        """Return the combinations implemented by this method."""

    @property
    def prepared_mesh(self) -> TrainerMeshRef | None:
        """Return the mesh whose reusable plans are cached."""
        return None

    def prepare(
        self,
        preparation: Any = None,
    ) -> None:
        """Prepare reusable resources without staging a weight version."""

    @abstractmethod
    def stage(
        self,
        *,
        version: WeightVersion,
        source: ResolvedSource,
    ) -> PreparedArtifact:
        """Stage one update without changing live weights.

        Recover method-owned state before raising a retryable error. Raise
        StrategyRecoveryError if cleanup cannot establish a safe retry.
        """

    @abstractmethod
    def release(self, prepared: PreparedArtifact) -> None:
        """Release method-owned staging for one prepared update."""

    def installation_context(self, prepared: PreparedArtifact):
        """Run method-specific work at the caller's safe point.

        Implementations may transfer into live storage. The framework must
        already have paused engine execution.
        """
        del prepared
        return nullcontext()

    def mutated_during_installation_context(
        self,
        prepared: PreparedArtifact,
    ) -> bool:
        """Return whether a failed context entry may have changed live storage."""
        del prepared
        return False

    def stage_chain(
        self,
        chain: tuple[tuple[WeightVersion, ResolvedSource], ...],
    ) -> PreparedArtifact:
        """Stage an ordered version chain as one installable artifact."""
        if len(chain) != 1:
            raise RuntimeError(
                f"{type(self).__name__} does not support version-chain replay"
            )
        version, source = chain[0]
        return self.stage(version=version, source=source)

    def installation_failed(self, prepared: PreparedArtifact) -> None:
        """Fence method-owned state after an engine installation failure."""

    def validate_close(self) -> None:
        """Reject teardown while transport or installation may still use resources."""

    def close(self) -> None:
        """Release method-owned process resources."""


class EngineInstaller(ABC):
    """Commit a prepared artifact into one live inference engine."""

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Return prepared artifact kinds accepted by this installer."""

    @abstractmethod
    def install(self, prepared: PreparedArtifact) -> Any:
        """Install a prepared artifact at the caller's safe point."""


@dataclass(frozen=True)
class WeightUpdatePlan:
    """Immutable source, method, and installer choice for one update."""

    version: WeightVersion
    source: ResolvedSource
    method: UpdateMethod
    installer: EngineInstaller


class WeightUpdatePlanner:
    """Select legal source, method, and installer compositions."""

    def __init__(
        self,
        *,
        resolvers: tuple[SourceResolver, ...],
        methods: tuple[UpdateMethod, ...],
        installer: EngineInstaller,
        max_transfer_attempts: int,
    ) -> None:
        self._resolvers = resolvers
        self._methods = methods
        self._installer = installer
        self._max_transfer_attempts = max_transfer_attempts

    @property
    def source_order(self) -> tuple[WeightSource, ...]:
        """Return source kinds in the order configured for fallback."""
        return tuple(resolver.kind for resolver in self._resolvers)

    def _source_resolvers(
        self, source_kind: WeightSource | None, version: WeightVersion | None = None,
    ) -> Iterator[SourceResolver]:
        return (
            resolver for resolver in self._resolvers
            if (source_kind is None or resolver.kind is source_kind)
            and (version is None or resolver.supports(version))
        )

    def _matching_methods(
        self,
        source_kind: WeightSource,
        payload_format: WeightPayloadFormat | None = None,
        version: WeightVersion | None = None,
    ) -> Iterator[UpdateMethod]:
        for method in self._methods:
            capabilities = method.capabilities
            if (
                source_kind in capabilities.sources
                and capabilities.artifact_type in self._installer.capabilities.artifact_types
                and (payload_format is None or payload_format in capabilities.payload_formats)
                and (version is None or not capabilities.requires_base_version
                     or version.base_version_id is not None)
            ):
                yield method

    def prepare(
        self,
        mesh: TrainerMesh | None,
        *,
        source_kind: WeightSource,
        validate_mesh: Callable[[], object] | None = None,
    ) -> None:
        """Prepare compatible methods for one source without version leases."""
        for resolver in self._source_resolvers(source_kind):
            if resolver.needs_mesh and mesh is None:
                continue
            methods = [
                method for method in self._matching_methods(resolver.kind)
                if not resolver.needs_mesh or method.prepared_mesh != mesh.ref
            ]
            if not methods:
                continue
            if resolver.needs_mesh:
                if validate_mesh is None:
                    raise ValueError("mesh preparation requires mesh validation")
                inputs = resolver.prepare_inputs(
                    mesh=mesh, validate_mesh=validate_mesh,
                )
            else:
                inputs = resolver.prepare_inputs()
            for method in methods:
                method.prepare(inputs)

    def plans(
        self,
        version: WeightVersion,
        *,
        source_kind: WeightSource | None = None,
    ) -> Iterator[WeightUpdatePlan]:
        for resolver in self._source_resolvers(source_kind, version):
            resolved_plans = []
            for attempt, source in enumerate(resolver.candidates(version)):
                if attempt >= self._max_transfer_attempts:
                    break
                method = next(
                    self._matching_methods(source.kind, source.payload_format, version), None,
                )
                if method is not None:
                    plan = WeightUpdatePlan(
                        version=version,
                        source=source,
                        method=method,
                        installer=self._installer,
                    )
                    resolved_plans.append(plan)
                    yield plan
            # Trainer and object-storage transfers may fail transiently even
            # when discovery returns only one candidate. Generator discovery
            # already returns independent peers, so retrying the same peer adds
            # no fallback value.
            if resolver.kind is not WeightSource.GENERATOR and len(resolved_plans) == 1:
                for _ in range(1, self._max_transfer_attempts):
                    yield resolved_plans[0]

    def validate(
        self,
        version: WeightVersion,
        *,
        source_kind: WeightSource | None = None,
    ) -> None:
        """Reject unsupported static combinations before source I/O or leases."""
        for resolver in self._source_resolvers(source_kind, version):
            if next(self._matching_methods(
                resolver.kind, resolver.payload_format(version), version,
            ), None) is not None:
                return
        raise RuntimeError(
            "no legal source, update method, and engine installer composition for "
            f"payload={version.payload_format.value}"
        )


__all__ = [
    "EngineCapabilities",
    "EngineInstaller",
    "GeneratorPeerUpdateSource",
    "MethodCapabilities",
    "ObjectStorageUpdateSource",
    "PreparedArtifact",
    "PreparedCheckpointArtifact",
    "PreparedEngineTensors",
    "PreparedRuntimeTensors",
    "ResolvedSource",
    "StagedEngineTensors",
    "TrainerSourceSnapshot",
    "ResolvedTrainerSource",
    "UpdateMethod",
    "WeightSource",
    "SourceResolver",
    "WeightUpdatePlan",
    "WeightUpdatePlanner",
]
