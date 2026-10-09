# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mooncake implementation of the artifact transport interface."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from importlib.machinery import PathFinder
from pathlib import Path

from .. import p2p_pb2
from .artifact_transfer import ArtifactBundle, ArtifactTransfer
from .artifact_transport import (
    ArtifactCacheMiss,
    ArtifactCacheStale,
    ArtifactInstallDisposition,
    ArtifactInstallState,
    ArtifactInstallStatus,
    ArtifactTransport,
    ArtifactTransportContext,
    ArtifactTransportUnavailable,
    PublicationHandle,
    StagedArtifact,
)
from .mooncake_artifact_store import (
    MooncakeArtifactCacheMiss,
    MooncakeArtifactCacheStale,
    MooncakeArtifactCacheUnavailable,
    compute_artifact_cache_key,
    install_from_mooncake,
    publish_to_mooncake,
)


@dataclass(frozen=True)
class _PendingPublication:
    stale_fingerprint: str | None = None


_pending_publications: dict[str, _PendingPublication] = {}
_pending_publications_lock = threading.Lock()


@dataclass
class MooncakePublicationHandle(PublicationHandle):
    """Handle for a completed Mooncake publication.

    Mooncake artifact objects are intentionally not removed when the client
    stops, so stopping this handle only marks the publication as released to
    the common scheduler.
    """

    identifier: str
    transport: str = "mooncake"

    def stop(self) -> None:
        return None


class MooncakeArtifactTransport(ArtifactTransport):
    """Fetch and publish artifacts through the shared Mooncake store."""

    name = "mooncake"
    # Mooncake publishes only after this process records a confirmed miss.
    publish_requires_install_state = True
    # Avoid repeatedly preparing and putting the same complete object from one
    # service process. Report the failure and let a later service start retry.
    retry_publish_on_failure = False

    def state_after_cache_miss(self) -> ArtifactInstallState:
        """Share a confirmed miss for the lifetime of the querying process."""
        pid = os.getpid()
        return ArtifactInstallState(
            ArtifactInstallStatus.MISS,
            transport=self.name,
            owner_pid=pid,
            owner_starttime=_process_starttime(pid),
        )

    def resolve_install_state(
        self, state: ArtifactInstallState
    ) -> ArtifactInstallDisposition:
        # ATTEMPTED is intentionally a one-attempt-per-marker tombstone for
        # operational failures (for example setup, registration, or RDMA
        # errors).  Those failures do not prove that the remote artifact is
        # absent, so later workers skip both another fetch and publication.
        # Only a confirmed MISS owned by a live process is shared below.
        if state.status is not ArtifactInstallStatus.MISS:
            return ArtifactInstallDisposition.SKIP
        if state.transport != self.name or not _state_owner_is_alive(state):
            return ArtifactInstallDisposition.FETCH
        return ArtifactInstallDisposition.CACHED_MISS

    def should_publish(self, state: ArtifactInstallState | None) -> bool:
        return (
            state is not None
            and state.status is ArtifactInstallStatus.MISS
            and state.transport == self.name
            and _state_owner_is_current_process(state)
        )

    @staticmethod
    def is_available() -> bool:
        """Check for ``mooncake.store`` without loading its native library.

        Importing ``mooncake.store`` loads the transfer-engine shared library,
        whose process-global RDMA configuration is initialized only once.  The
        actual import must therefore remain inside ``_store_session()``, after
        the ModelExpress ``MX_MC_*`` settings have been promoted to ``MC_*``.
        """
        package = PathFinder.find_spec("mooncake")
        if package is None or package.submodule_search_locations is None:
            return False
        return (
            PathFinder.find_spec(
                "mooncake.store", package.submodule_search_locations
            )
            is not None
        )

    def fetch(
        self,
        artifact: ArtifactTransfer,
        identity: p2p_pb2.SourceIdentity,
        context: ArtifactTransportContext,
    ) -> StagedArtifact:
        cache_key = compute_artifact_cache_key(
            artifact,
            identity,
            node_rank=context.node_rank,
            accelerator=context.accelerator,
        )
        _clear_pending_publication(cache_key)
        try:
            header = install_from_mooncake(
                artifact,
                identity,
                node_rank=context.node_rank,
                accelerator=context.accelerator,
            )
        except MooncakeArtifactCacheStale as exc:
            _set_pending_publication(
                cache_key,
                _PendingPublication(stale_fingerprint=exc.fingerprint),
            )
            raise ArtifactCacheStale(str(exc)) from exc
        except MooncakeArtifactCacheMiss as exc:
            _set_pending_publication(cache_key, _PendingPublication())
            raise ArtifactCacheMiss(str(exc)) from exc
        except MooncakeArtifactCacheUnavailable as exc:
            raise ArtifactTransportUnavailable(str(exc)) from exc
        except Exception as exc:
            raise ArtifactTransportUnavailable(
                f"Mooncake artifact fetch failed for {artifact.name}: {exc}"
            ) from exc
        return StagedArtifact(header=header, transport=self.name)

    def publish(
        self,
        artifact: ArtifactTransfer,
        identity: p2p_pb2.SourceIdentity,
        bundle: ArtifactBundle,
        context: ArtifactTransportContext,
    ) -> PublicationHandle:
        cache_key = compute_artifact_cache_key(
            artifact,
            identity,
            node_rank=context.node_rank,
            accelerator=context.accelerator,
        )
        pending = _get_pending_publication(cache_key)
        try:
            published_key = publish_to_mooncake(
                artifact,
                identity,
                bundle,
                node_rank=context.node_rank,
                accelerator=context.accelerator,
                repair_from_fingerprint=(
                    pending.stale_fingerprint if pending is not None else None
                ),
            )
        except MooncakeArtifactCacheUnavailable as exc:
            raise ArtifactTransportUnavailable(str(exc)) from exc
        except Exception as exc:
            raise ArtifactTransportUnavailable(
                f"Mooncake artifact publish failed for {artifact.name}: {exc}"
            ) from exc
        finally:
            _clear_pending_publication(cache_key)
        return MooncakePublicationHandle(identifier=published_key)


def _set_pending_publication(key: str, pending: _PendingPublication) -> None:
    with _pending_publications_lock:
        _pending_publications[key] = pending


def _get_pending_publication(key: str) -> _PendingPublication | None:
    with _pending_publications_lock:
        return _pending_publications.get(key)


def _clear_pending_publication(key: str) -> None:
    with _pending_publications_lock:
        _pending_publications.pop(key, None)


def _state_owner_is_alive(state: ArtifactInstallState) -> bool:
    pid = state.owner_pid
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    actual_starttime = _process_starttime(pid)
    if state.owner_starttime is None or actual_starttime is None:
        return True
    return actual_starttime == state.owner_starttime


def _state_owner_is_current_process(state: ArtifactInstallState) -> bool:
    """Return whether this process recorded the shared Mooncake miss."""
    pid = os.getpid()
    if state.owner_pid != pid:
        return False
    actual_starttime = _process_starttime(pid)
    if state.owner_starttime is None or actual_starttime is None:
        return True
    return actual_starttime == state.owner_starttime


def _process_starttime(pid: int) -> str | None:
    """Return Linux /proc process starttime (field 22)."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, IndexError):
        return None
