# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed composition contracts for one generator weight update."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from modelexpress import p2p_pb2

from ..control import WeightVersion
from ..object_storage import ObjectStorageSource
from ..train import WeightPayloadFormat
from .adapter import GeneratorTransferInputs

if TYPE_CHECKING:
    from .receiver import PreparedCheckpoint


class WeightSource(str, Enum):
    """Location from which a generator obtains weight bytes."""

    TRAINER = "TRAINER"
    GENERATOR = "GENERATOR"
    OBJECT_STORAGE = "OBJECT_STORAGE"


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
    kind = WeightSource.GENERATOR
    payload_format = WeightPayloadFormat.FULL_TENSOR


@dataclass(frozen=True)
class TrainerUpdateSource:
    """Trainer manifests for one complete full-tensor update."""

    inputs: GeneratorTransferInputs
    kind = WeightSource.TRAINER

    @property
    def payload_format(self) -> WeightPayloadFormat:
        return self.inputs.payload_format


@dataclass(frozen=True)
class ObjectStorageUpdateSource:
    """Canonical payload published in object storage."""

    storage: ObjectStorageSource
    payload_format: WeightPayloadFormat
    kind = WeightSource.OBJECT_STORAGE


ResolvedSource = (
    GeneratorPeerUpdateSource | TrainerUpdateSource | ObjectStorageUpdateSource
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
        """Return metrics recorded while preparing this artifact."""


@dataclass(frozen=True)
class PreparedEngineTensors(PreparedArtifact):
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

    @abstractmethod
    def candidates(self, version: WeightVersion) -> Iterator[ResolvedSource]:
        """Yield verified candidates in fallback order."""


class UpdateMethod(ABC):
    """Prepare weights from a resolved source without mutating the live engine."""

    @property
    @abstractmethod
    def capabilities(self) -> MethodCapabilities:
        """Return the combinations implemented by this method."""

    def supports(
        self,
        *,
        source: ResolvedSource,
    ) -> bool:
        capabilities = self.capabilities
        return (
            source.payload_format in capabilities.payload_formats
            and source.kind in capabilities.sources
        )

    @abstractmethod
    def prepare(
        self,
        *,
        version: WeightVersion,
        source: ResolvedSource,
    ) -> PreparedArtifact:
        """Transfer and verify one update without changing live weights."""

    @abstractmethod
    def release(self, prepared: PreparedArtifact) -> None:
        """Release method-owned staging for one prepared update."""

    def installation_context(self, prepared: PreparedArtifact):
        """Protect method-owned state while the installer reads it."""
        del prepared
        return nullcontext()

    def prepare_chain(
        self,
        chain: tuple[tuple[WeightVersion, ResolvedSource], ...],
    ) -> PreparedArtifact:
        """Prepare an ordered version chain as one installable artifact."""
        if len(chain) != 1:
            raise RuntimeError(
                f"{type(self).__name__} does not support version-chain replay"
            )
        version, source = chain[0]
        return self.prepare(version=version, source=source)

    def installation_failed(self, prepared: PreparedArtifact) -> None:
        """Fence method-owned state after an engine installation failure."""

    def publish_applied(self, *, version_id: str, prepared: PreparedArtifact) -> None:
        """Optionally advertise an installed update as a generator source."""

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

    def plans(self, version: WeightVersion):
        for resolver in self._resolvers:
            if not resolver.supports(version):
                continue
            resolved_plans = []
            for attempt, source in enumerate(resolver.candidates(version)):
                if attempt >= self._max_transfer_attempts:
                    break
                for method in self._methods:
                    if not method.supports(source=source):
                        continue
                    if (
                        method.capabilities.requires_base_version
                        and version.base_version_id is None
                    ):
                        continue
                    if (
                        method.capabilities.artifact_type
                        not in self._installer.capabilities.artifact_types
                    ):
                        continue
                    plan = WeightUpdatePlan(
                        version=version,
                        source=source,
                        method=method,
                        installer=self._installer,
                    )
                    resolved_plans.append(plan)
                    yield plan
                    break
            # Trainer and object-storage transfers may fail transiently even
            # when discovery returns only one candidate. Generator discovery
            # already returns independent peers, so retrying the same peer adds
            # no fallback value.
            if resolver.kind is not WeightSource.GENERATOR and len(resolved_plans) == 1:
                for _ in range(1, self._max_transfer_attempts):
                    yield resolved_plans[0]

    def validate(self, version: WeightVersion) -> None:
        """Reject unsupported static combinations before source I/O or leases."""
        candidates = []
        for resolver in self._resolvers:
            if not resolver.supports(version):
                continue
            candidates.append((resolver.kind, resolver.payload_format(version)))
        for method in self._methods:
            capabilities = method.capabilities
            if any(
                payload_format in capabilities.payload_formats
                and kind in capabilities.sources
                and capabilities.artifact_type
                in self._installer.capabilities.artifact_types
                and (
                    not capabilities.requires_base_version
                    or version.base_version_id is not None
                )
                for kind, payload_format in candidates
            ):
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
    "ResolvedSource",
    "StagedEngineTensors",
    "TrainerUpdateSource",
    "UpdateMethod",
    "WeightSource",
    "SourceResolver",
    "WeightUpdatePlan",
    "WeightUpdatePlanner",
]
