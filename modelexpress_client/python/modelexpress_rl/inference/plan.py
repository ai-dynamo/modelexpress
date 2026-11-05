# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight-update composition and prepared-artifact lifetimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from modelexpress import p2p_pb2
from modelexpress.adapter import StrategyRecoveryError
from modelexpress_rl.refit.reshard.types import IncompleteRefit
from modelexpress_rl.refit.timing import refit_span

from ..control import WeightVersion
from ..object_storage import ObjectStorageSource
from ..train.manifest import ShardChecksumKey
from ..train import WeightPayloadFormat
from ..version import MeshIdentity
from .adapter import TrainerSourceShard

if TYPE_CHECKING:
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
    streamed_checkpoints: bool = False


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

    mesh_identity: MeshIdentity
    shards: tuple[TrainerSourceShard, ...]
    kind = WeightSource.TRAINER
    payload_format = WeightPayloadFormat.FULL_TENSOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "shards", tuple(self.shards))

    @property
    def physical_fingerprint(self) -> tuple:
        """Server-pinned worker/shard digests require a new mesh generation when
        physical addresses or registrations change."""
        return (
            self.mesh_identity,
            tuple((shard.source_slot_id, shard.worker_id) for shard in self.shards),
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
    kind = WeightSource.TRAINER
    payload_format = WeightPayloadFormat.FULL_TENSOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "checksums", MappingProxyType(dict(self.checksums)))


ResolvedSource = (
    GeneratorPeerUpdateSource | ResolvedTrainerSource | ObjectStorageUpdateSource
)


class StagedEngineTensors(Protocol):
    """Tensor staging payload required by engine installers."""

    tensors: dict[str, Any] | SourceReplayBatch
    metrics: dict[str, Any]


@dataclass(frozen=True)
class SourceReplayBatch:
    """Independent source views plus their globally ordered destination writes.

    The stream owns source views until engine reads finish. The captured-plan
    identity binds these writes to the installer's complete replay schedule.
    """

    writes: tuple
    replay_id: UUID

    def values(self):
        return (source for _, _, source in self.writes)


class PreparedArtifact(ABC):
    """Carry a prepared payload to an engine installer."""

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
class PreparedStreamingTensors(PreparedArtifact):
    """Own deferred batch consumption, synchronization, and cleanup.

    Open the private producer lazily inside consume(), and retain each borrowed batch
    through its installation callback and GPU fence. Retain the batch or iterator
    when cleanup cannot be proven; the owning method uses this state to protect
    its resources and version leases. The installer supplies the fence and native
    reload window without managing iterator advancement or individual batch release.
    """

    _batch_factory: Callable[[], Iterator[dict[str, Any] | SourceReplayBatch]]
    replay_id: UUID
    transfer_metrics: dict[str, float]
    _iterator: Iterator | None = field(default=None, init=False, repr=False)
    _batch: Any = field(default=None, init=False, repr=False)
    _uncertain: bool = field(default=False, init=False, repr=False)
    _drain_failed: bool = field(default=False, init=False, repr=False)
    _drain: Callable[[], None] | None = field(default=None, init=False, repr=False)

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.transfer_metrics)

    @property
    def release_blocked(self) -> bool:
        return self._drain is not None or self._iterator is not None or self._uncertain

    def apply(self, install: Callable[[Any], None]) -> None:
        """Retain and fence each borrowed batch before advancing its source."""
        if self._drain is None:
            raise RuntimeError("stream consumption requires the installation context")
        if self._uncertain:
            raise IncompleteRefit("streaming cleanup is unproven; reset the process")
        if self._iterator is None:
            try:
                self._iterator = iter(self._batch_factory())
            except BaseException:
                self._uncertain = True
                raise
        commit_s = 0.0
        while True:
            self._release_batch()
            try:
                self._batch = next(self._iterator)
            except StopIteration:
                self.transfer_metrics["install_commit_s"] = commit_s
                return
            except BaseException:
                # A GPU fence cannot establish completion of a failed RDMA read.
                self._uncertain = True
                raise
            started = time.perf_counter()
            install(self._batch)
            self._release_batch()
            commit_s += time.perf_counter() - started

    def _release_batch(self) -> None:
        if self._batch is not None:
            self._synchronize()
            self._batch = None

    def _synchronize(self) -> None:
        try:
            self._drain()
        except BaseException:
            self._drain_failed = self._uncertain = True
            raise

    @contextmanager
    def consume(self, drain: Callable[[], None]):
        """Protect the entire reload, opening the source only when first read."""
        if self.release_blocked:
            raise IncompleteRefit(
                "streaming transaction still retains transfer resources"
            )
        self._drain = drain
        primary_error = None
        try:
            yield self
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                if not self._drain_failed:
                    started = time.perf_counter()
                    with refit_span("post_install"):
                        self._synchronize()
                    self._batch = None
                    self.transfer_metrics["post_install_sync_s"] = (
                        time.perf_counter() - started
                    )
                    if self._iterator is not None and not self._uncertain:
                        try:
                            close = getattr(self._iterator, "close", None)
                            if close is not None:
                                close()
                        except BaseException:
                            self._uncertain = True
                            raise
                        self._iterator = None
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                raise primary_error from cleanup_error
            finally:
                self._drain = None


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
    """Own update preparation, resource release safety, and recovery."""

    @property
    @abstractmethod
    def capabilities(self) -> MethodCapabilities:
        """Return the combinations implemented by this method."""

    @abstractmethod
    def prepare(
        self,
        *,
        version: WeightVersion,
        source: ResolvedSource,
    ) -> PreparedArtifact:
        """Prepare and verify one update without changing live weights."""

    @abstractmethod
    def release(self, prepared: PreparedArtifact) -> None:
        """Release method-owned staging for one prepared update."""

    def can_release(self, prepared: PreparedArtifact) -> bool:
        """Return whether update resources and version leases can be released."""
        return True

    def validate_release(self, prepared: PreparedArtifact) -> None:
        """Reject cleanup while the method still needs its resources protected."""
        if not self.can_release(prepared):
            raise RuntimeError(
                "update cleanup is unproven; retain resources and reset the process"
            )

    def prepared_metrics(self, prepared: PreparedArtifact) -> dict[str, float]:
        """Describe an update without exposing its payload to the session."""
        return prepared.metrics

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

    def recover_preparation(self, error: BaseException) -> None:
        """Recover a failed attempt before the session selects another source."""
        if isinstance(error, StrategyRecoveryError):
            raise error
        try:
            self.preparation_failed()
        except Exception as recovery_error:
            raise StrategyRecoveryError(
                f"{type(self).__name__} could not recover after preparation failed: {error}"
            ) from recovery_error

    def preparation_failed(self) -> None:
        """Restore method-owned state after preparation fails."""

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
]
