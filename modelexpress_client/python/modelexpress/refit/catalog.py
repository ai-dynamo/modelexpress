# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed boundary over the six public revision-catalog RPCs.

The catalog is the authority for revision identity, lineage, and lifecycle;
this client only translates between the domain records in
:mod:`modelexpress.refit.manifest` and the generated protobuf messages. It
holds no lifecycle policy, no retry state machine, and no knowledge of the
server's metadata backend. Page tokens are opaque strings that are echoed
back verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import grpc

from modelexpress import revision_pb2, revision_pb2_grpc

from .manifest import (
    PublicationMode,
    ReceiverRevisionState,
    ReceiverStateRecord,
    RecoveryCandidate,
    RevisionManifest,
    RevisionRecord,
    RevisionSummary,
)


class CatalogProtocolError(RuntimeError):
    """A catalog call returned OK with a response the contract cannot describe."""


@dataclass(frozen=True)
class PublishedRevision:
    """Publication outcome. ``created`` is ``False`` for a byte-identical retry."""

    revision: RevisionRecord
    created: bool


@dataclass(frozen=True)
class RevisionPage:
    revisions: tuple[RevisionSummary, ...] = ()
    next_page_token: str | None = None


@dataclass(frozen=True)
class RecoveryCandidatePage:
    candidates: tuple[RecoveryCandidate, ...] = ()
    next_page_token: str | None = None


@runtime_checkable
class RevisionCatalog(Protocol):
    """The exact public catalog surface available to publisher and receiver."""

    def publish_revision(
        self,
        manifest: RevisionManifest,
        *,
        publisher_id: str,
        publication_mode: PublicationMode | None = None,
    ) -> PublishedRevision: ...

    def get_revision(self, model_id: str, version: str) -> RevisionRecord: ...

    def list_ready_revisions(
        self,
        model_id: str,
        *,
        page_token: str | None = None,
        limit: int = 0,
    ) -> RevisionPage: ...

    def get_recovery_candidates(
        self,
        model_id: str,
        *,
        target_version: str,
        installed_version: str | None = None,
        max_delta_replay_length: int | None = None,
        page_token: str | None = None,
        limit: int = 0,
    ) -> RecoveryCandidatePage: ...

    def update_receiver_state(
        self,
        model_id: str,
        version: str,
        receiver_id: str,
        state: ReceiverRevisionState,
        *,
        installed_version: str | None = None,
        detail: str = "",
    ) -> ReceiverStateRecord: ...

    def commit_version(self, model_id: str, version: str) -> RevisionRecord: ...


def _present(**fields) -> dict:
    """Drop ``None`` entries; ``0`` stays, because it is a meaningful bound."""
    return {name: value for name, value in fields.items() if value is not None}


def _require(response, field: str):
    if not response.HasField(field):
        raise CatalogProtocolError(
            f"{type(response).__name__} succeeded without a {field!r} field"
        )
    return getattr(response, field)


def _page_token(token: str) -> str | None:
    """An empty wire token means there is no next page."""
    return token or None


class GrpcRevisionCatalog:
    """Concrete :class:`RevisionCatalog` over the generated tonic/gRPC service."""

    def __init__(self, endpoint: str | None = None, *, stub=None) -> None:
        if (endpoint is None) == (stub is None):
            raise ValueError("GrpcRevisionCatalog needs exactly one of endpoint or stub")
        self._channel = None
        if stub is None:
            target = endpoint.removeprefix("http://").removeprefix("https://")
            self._channel = grpc.insecure_channel(target)
            stub = revision_pb2_grpc.RevisionCatalogServiceStub(self._channel)
        self._stub = stub

    def __enter__(self) -> GrpcRevisionCatalog:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def publish_revision(
        self,
        manifest: RevisionManifest,
        *,
        publisher_id: str,
        publication_mode: PublicationMode | None = None,
    ) -> PublishedRevision:
        response = self._stub.PublishRevision(
            revision_pb2.PublishRevisionRequest(
                manifest=manifest.to_proto(),
                publisher_id=publisher_id,
                **_present(publication_mode=publication_mode),
            )
        )
        return PublishedRevision(
            revision=RevisionRecord.from_proto(_require(response, "revision")),
            created=response.created,
        )

    def get_revision(self, model_id: str, version: str) -> RevisionRecord:
        response = self._stub.GetRevision(
            revision_pb2.GetRevisionRequest(model_id=model_id, version=version)
        )
        return RevisionRecord.from_proto(_require(response, "revision"))

    def list_ready_revisions(
        self,
        model_id: str,
        *,
        page_token: str | None = None,
        limit: int = 0,
    ) -> RevisionPage:
        response = self._stub.ListReadyRevisions(
            revision_pb2.ListReadyRevisionsRequest(
                model_id=model_id, limit=limit, **_present(page_token=page_token)
            )
        )
        return RevisionPage(
            revisions=tuple(
                RevisionSummary.from_proto(summary) for summary in response.revisions
            ),
            next_page_token=_page_token(response.next_page_token),
        )

    def get_recovery_candidates(
        self,
        model_id: str,
        *,
        target_version: str,
        installed_version: str | None = None,
        max_delta_replay_length: int | None = None,
        page_token: str | None = None,
        limit: int = 0,
    ) -> RecoveryCandidatePage:
        response = self._stub.GetRecoveryCandidates(
            revision_pb2.GetRecoveryCandidatesRequest(
                model_id=model_id,
                target_version=target_version,
                limit=limit,
                **_present(
                    installed_version=installed_version,
                    max_delta_replay_length=max_delta_replay_length,
                    page_token=page_token,
                ),
            )
        )
        return RecoveryCandidatePage(
            candidates=tuple(
                RecoveryCandidate.from_proto(candidate) for candidate in response.candidates
            ),
            next_page_token=_page_token(response.next_page_token),
        )

    def update_receiver_state(
        self,
        model_id: str,
        version: str,
        receiver_id: str,
        state: ReceiverRevisionState,
        *,
        installed_version: str | None = None,
        detail: str = "",
    ) -> ReceiverStateRecord:
        response = self._stub.UpdateReceiverState(
            revision_pb2.UpdateReceiverStateRequest(
                model_id=model_id,
                version=version,
                receiver_id=receiver_id,
                state=state,
                detail=detail,
                **_present(installed_version=installed_version),
            )
        )
        return ReceiverStateRecord.from_proto(_require(response, "receiver"))

    def commit_version(self, model_id: str, version: str) -> RevisionRecord:
        response = self._stub.CommitVersion(
            revision_pb2.CommitVersionRequest(model_id=model_id, version=version)
        )
        return RevisionRecord.from_proto(_require(response, "revision"))
