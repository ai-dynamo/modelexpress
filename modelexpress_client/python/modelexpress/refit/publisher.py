# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Miles-owned canonical delta publisher."""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

import torch

from .api import PublishResult, PublisherConfig, PublisherStatus
from .catalog import GrpcRevisionCatalog
from .manifest import RevisionManifest, RevisionState
from .s3 import S3Uploader
from .source.canonical import CanonicalDeltaEncoder, canonical_json, load_hf_snapshot


def _key(model_id: str, version: str, filename: str) -> str:
    return (
        f"models/{quote(model_id, safe='')}/revisions/"
        f"{quote(version, safe='')}/canonical/{filename}"
    )


def _object(stored, size: int) -> dict[str, object]:
    result = {
        "bucket": stored.bucket,
        "checksum": stored.checksum,
        "key": stored.key,
        "size": size,
    }
    if stored.object_version is not None:
        result["object_version"] = stored.object_version
    return result


class Publisher:
    def __init__(
        self,
        *,
        launch_checkpoint: str | Path,
        bucket_bytes: int = 256 * 1024 * 1024,
        catalog=None,
        s3_client=None,
        sleep=time.sleep,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.launch_checkpoint = Path(launch_checkpoint)
        self.bucket_bytes = bucket_bytes
        self.catalog = catalog
        self.s3_client = s3_client
        self.sleep = sleep
        self.poll_interval_seconds = poll_interval_seconds
        self.uploader = None

    def initialize(self, config: PublisherConfig) -> None:
        self.config = config
        (
            self.snapshot,
            self.metadata,
            self.format_digest,
            self.target_digest,
        ) = load_hf_snapshot(self.launch_checkpoint)
        self.current_version = "0"
        self.state = None
        self.distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.rank = torch.distributed.get_rank() if self.distributed else 0

        if self.rank == 0:
            if self.catalog is None:
                self.catalog = GrpcRevisionCatalog(config.catalog_endpoint)
            self.uploader = S3Uploader(config.s3, client=self.s3_client)

    def publish_version(
        self,
        version: str,
        *,
        base_version: str | None = None,
        gather_hf_buckets=None,
    ) -> PublishResult:
        if version == "0":
            if base_version is not None or self.current_version != "0":
                raise RuntimeError("version 0 is the launch revision")
            if self.rank == 0:
                manifest = RevisionManifest(
                    model_id=self.config.model_id,
                    target_version="0",
                    target_digest=self.target_digest,
                    format_digest=self.format_digest,
                )
                self.catalog.publish_revision(manifest)
                self._wait_for_commit(manifest)
            self._barrier()
            self.state = RevisionState.COMMITTED
            return PublishResult(self.config.model_id, "0", self.state)

        if base_version != self.current_version:
            raise RuntimeError(
                f"base {base_version!r} does not match current version "
                f"{self.current_version!r}"
            )

        base_digest = self.target_digest
        uploaded = []
        encoder = None
        if self.rank == 0:
            base = self.catalog.get_revision(self.config.model_id, base_version)
            if (
                base.state is not RevisionState.COMMITTED
                or base.manifest.target_digest != base_digest
                or base.manifest.format_digest != self.format_digest
            ):
                raise RuntimeError("catalog base does not match the current snapshot")
            encoder = CanonicalDeltaEncoder(
                self.config.model_id,
                base_version,
                version,
                self.snapshot,
                self.metadata,
                self.format_digest,
                base_digest,
                self.bucket_bytes,
            )

        def encode_bucket(bucket, _pbar=None):
            if self.rank != 0:
                return
            encoded = encoder.encode_bucket(bucket)
            if encoded is None:
                return
            ordinal, data, decoded_size, tensors = encoded
            stored = self.uploader.put(
                _key(
                    self.config.model_id,
                    version,
                    f"bucket-{ordinal:08d}.mxcd",
                ),
                data,
            )
            uploaded.append(
                {
                    "decoded_size": decoded_size,
                    "object": _object(stored, len(data)),
                    "ordinal": ordinal,
                    "tensors": list(tensors),
                }
            )

        gather_hf_buckets(encode_bucket)

        if self.rank == 0:
            target_digest, coverage = encoder.finish()
            root = canonical_json(
                {
                    "base_digest": base_digest,
                    "base_version": base_version,
                    "buckets": uploaded,
                    "encoding": {"compression": "zstd", "delta": "xor"},
                    "format_digest": self.format_digest,
                    "model_id": self.config.model_id,
                    "schema": "mx.canonical.delta.v0",
                    "target_digest": target_digest,
                    "target_version": version,
                    "tensors": coverage,
                }
            )
            payload = self.uploader.put(
                _key(self.config.model_id, version, "root.json"), root
            )
            manifest = RevisionManifest(
                model_id=self.config.model_id,
                target_version=version,
                base_version=base_version,
                base_digest=base_digest,
                target_digest=target_digest,
                format_digest=self.format_digest,
                payload=payload,
            )
            self.catalog.publish_revision(manifest)
            self._wait_for_commit(manifest)
            self.target_digest = target_digest

        self._barrier()
        self.current_version = version
        self.state = RevisionState.COMMITTED
        return PublishResult(self.config.model_id, version, self.state)

    def status(self) -> PublisherStatus:
        return PublisherStatus(
            model_id=self.config.model_id,
            current_version=self.current_version,
            state=self.state,
        )

    def deregister(self) -> None:
        if self.uploader is not None:
            self.uploader.close()
        close = getattr(self.catalog, "close", None)
        if close is not None:
            close()

    def _wait_for_commit(self, manifest: RevisionManifest) -> None:
        while (
            self.catalog.get_revision(manifest.model_id, manifest.target_version).state
            is not RevisionState.COMMITTED
        ):
            self.sleep(self.poll_interval_seconds)

    def _barrier(self) -> None:
        if self.distributed:
            torch.distributed.barrier()
