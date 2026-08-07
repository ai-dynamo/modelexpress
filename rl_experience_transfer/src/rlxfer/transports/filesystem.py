# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent local transport with atomic filesystem queue transitions."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rlxfer.errors import BackpressureError, ClosedError, DeliveryError
from rlxfer.serialization import (
    BufferSegment,
    SerializationLimits,
    SerializedExperience,
    validate_transfer_limits,
)
from rlxfer.transport import (
    DeliveryReceipt,
    HealthStatus,
    ReceiptResult,
    ReceiptState,
    TransportCapabilities,
    TransportDelivery,
)


class FileSystemTransport:
    """Multiprocess, persistent, at-least-once transport for one local host."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_queue: int = 128,
        lease_timeout: float = 300.0,
        poll_interval: float = 0.01,
        limits: SerializationLimits | None = None,
    ) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be positive")
        if lease_timeout <= 0 or poll_interval <= 0:
            raise ValueError("lease_timeout and poll_interval must be positive")
        self._root = Path(path).expanduser().resolve()
        self._max_queue = max_queue
        self._lease_timeout = lease_timeout
        self._poll_interval = poll_interval
        self._limits = limits or SerializationLimits()
        self._closed = False
        for name in ("pending", "inflight", "receipts", "idempotency", "tmp"):
            (self._root / name).mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock_path = self._root / ".lock"
        self._lock_path.touch(exist_ok=True)
        self._recover_stale()

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> FileSystemTransport:
        return cls(**dict(options))

    @property
    def capabilities(self) -> TransportCapabilities:
        return TransportCapabilities(
            name="filesystem",
            remote=False,
            asynchronous=True,
            acknowledgements=True,
            persistence=True,
            max_transfer_size=(
                self._limits.max_metadata_bytes + self._limits.max_total_tensor_bytes
            ),
            delivery_guarantee="at-least-once",
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._lock_path.open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedError("filesystem transport is closed")

    def _depth(self) -> int:
        return sum(1 for _ in (self._root / "pending").iterdir()) + sum(
            1 for _ in (self._root / "inflight").iterdir()
        )

    @staticmethod
    def _index_name(idempotency_key: str) -> str:
        import hashlib

        return hashlib.sha256(idempotency_key.encode()).hexdigest() + ".json"

    @staticmethod
    def _record_id(value: object) -> str:
        if not isinstance(value, str):
            raise DeliveryError("filesystem record ID must be a UUID hex string")
        try:
            parsed = uuid.UUID(hex=value)
        except ValueError as error:
            raise DeliveryError("filesystem record ID must be a UUID hex string") from error
        if parsed.hex != value:
            raise DeliveryError("filesystem record ID must be a canonical UUID hex string")
        return value

    @staticmethod
    def _write_private(path: Path, data: bytes) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write_json(self, target: Path, value: Mapping[str, Any]) -> None:
        temporary = self._root / "tmp" / f"{uuid.uuid4().hex}.json"
        try:
            self._write_private(
                temporary,
                json.dumps(value, separators=(",", ":"), sort_keys=True).encode(),
            )
            os.replace(temporary, target)
            self._fsync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_receipt(self, record_id: str, result: ReceiptResult) -> None:
        self._write_json(
            self._root / "receipts" / f"{record_id}.json",
            {
                "state": result.state.value,
                "reason": result.reason,
                "attempts": result.attempts,
            },
        )

    def _read_manifest(self, directory: Path) -> dict[str, Any]:
        path = directory / "manifest.json"
        if path.stat().st_size > self._limits.max_metadata_bytes:
            raise DeliveryError("filesystem manifest exceeds metadata size limit")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise DeliveryError("filesystem manifest must be a JSON object")
        return value

    @staticmethod
    def _read_buffer(path: Path, expected_size: int) -> bytes:
        if path.stat().st_size != expected_size:
            raise DeliveryError("filesystem buffer size disagrees with its manifest")
        return path.read_bytes()

    @staticmethod
    def _declared_size(item: object) -> int:
        if not isinstance(item, dict):
            return -1
        value = item.get("nbytes")
        return value if isinstance(value, int) and not isinstance(value, bool) else -1

    def _recover_stale(self) -> None:
        now = time.time()
        with self._locked():
            for directory in (self._root / "inflight").iterdir():
                if now - directory.stat().st_mtime <= self._lease_timeout:
                    continue
                manifest = self._read_manifest(directory)
                record_id = self._record_id(manifest.get("record_id"))
                if (self._root / "receipts" / f"{record_id}.json").exists():
                    shutil.rmtree(directory)
                    continue
                destination = self._root / "pending" / record_id
                if destination.exists():
                    shutil.rmtree(directory)
                else:
                    os.replace(directory, destination)
            self._repair_indexes_locked()

    def _repair_indexes_locked(self) -> None:
        active: dict[str, dict[str, Any]] = {}
        for queue_name in ("pending", "inflight"):
            for directory in (self._root / queue_name).iterdir():
                manifest = self._read_manifest(directory)
                record_id = self._record_id(manifest.get("record_id"))
                if queue_name == "pending" and directory.name != record_id:
                    raise DeliveryError("pending directory and manifest record IDs differ")
                idempotency_key = manifest.get("idempotency_key")
                if not isinstance(idempotency_key, str) or not idempotency_key:
                    raise DeliveryError("filesystem manifest idempotency key is invalid")
                previous = active.setdefault(record_id, manifest)
                if previous is not manifest:
                    raise DeliveryError("duplicate active filesystem record ID")
                index_path = self._root / "idempotency" / self._index_name(idempotency_key)
                if not index_path.exists():
                    self._write_json(
                        index_path,
                        {
                            "record_id": record_id,
                            "experience_id": str(manifest["experience_id"]),
                            "published_at": float(manifest["published_at"]),
                        },
                    )
                    continue
                index = json.loads(index_path.read_text(encoding="utf-8"))
                if self._record_id(index.get("record_id")) != record_id:
                    raise DeliveryError("idempotency index points to another active record")

        receipt_ids = {path.stem for path in (self._root / "receipts").glob("*.json")}
        for index_path in (self._root / "idempotency").glob("*.json"):
            index = json.loads(index_path.read_text(encoding="utf-8"))
            record_id = self._record_id(index.get("record_id"))
            if record_id not in active and record_id not in receipt_ids:
                index_path.unlink()
                self._fsync_directory(index_path.parent)

    def publish(
        self,
        payload: SerializedExperience,
        *,
        experience_id: str,
        idempotency_key: str,
        timeout: float | None = None,
        max_retries: int = 3,
    ) -> DeliveryReceipt:
        self._ensure_open()
        if not experience_id or not idempotency_key:
            raise ValueError("experience_id and idempotency_key are required")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        validate_transfer_limits(
            metadata_bytes=len(payload.metadata),
            tensor_sizes=(segment.nbytes for segment in payload.buffers),
            limits=self._limits,
        )
        deadline = None if timeout is None else time.monotonic() + timeout
        index_path = self._root / "idempotency" / self._index_name(idempotency_key)
        record_id = uuid.uuid4().hex
        published_at = time.time()
        build = self._root / "tmp" / record_id
        buffer_directory = build / "buffers"
        buffer_directory.mkdir(mode=0o700, parents=True)
        catalogs: list[dict[str, Any]] = []
        try:
            self._write_private(build / "metadata.json", payload.metadata)
            for index, segment in enumerate(payload.buffers):
                filename = f"{index:06d}.bin"
                data = segment.materialize()
                self._write_private(buffer_directory / filename, data)
                catalog = segment.catalog_entry()
                catalog["filename"] = filename
                catalogs.append(catalog)
            manifest: dict[str, Any] = {
                "record_id": record_id,
                "experience_id": experience_id,
                "idempotency_key": idempotency_key,
                "published_at": published_at,
                "attempt": 1,
                "max_retries": max_retries,
                "buffers": catalogs,
            }
            self._write_private(
                build / "manifest.json",
                json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode(),
            )
            self._fsync_directory(buffer_directory)
            self._fsync_directory(build)
            while True:
                with self._locked():
                    self._ensure_open()
                    if index_path.exists():
                        index = json.loads(index_path.read_text(encoding="utf-8"))
                        indexed_record_id = self._record_id(index.get("record_id"))
                        if index.get("experience_id") != experience_id:
                            raise DeliveryError("idempotency key reused for another experience")
                        shutil.rmtree(build)
                        return self._receipt(
                            indexed_record_id,
                            experience_id,
                            idempotency_key,
                            float(index["published_at"]),
                        )
                    if self._depth() < self._max_queue:
                        pending = self._root / "pending" / record_id
                        os.replace(build, pending)
                        self._fsync_directory(pending.parent)
                        try:
                            self._write_json(
                                index_path,
                                {
                                    "record_id": record_id,
                                    "experience_id": experience_id,
                                    "published_at": published_at,
                                },
                            )
                        except BaseException:
                            shutil.rmtree(pending)
                            self._fsync_directory(pending.parent)
                            raise
                        break
                if deadline is not None and time.monotonic() >= deadline:
                    raise BackpressureError("filesystem queue is full")
                time.sleep(self._sleep_duration(deadline))
        except BaseException:
            if build.exists():
                shutil.rmtree(build)
            raise
        return self._receipt(record_id, experience_id, idempotency_key, published_at)

    def _receipt(
        self,
        record_id: str,
        experience_id: str,
        idempotency_key: str,
        published_at: float,
    ) -> DeliveryReceipt:
        return DeliveryReceipt(
            receipt_id=record_id,
            experience_id=experience_id,
            idempotency_key=idempotency_key,
            accepted_at=published_at,
            _wait=lambda timeout: self._wait_receipt(record_id, timeout),
        )

    def _wait_receipt(self, record_id: str, timeout: float | None) -> ReceiptResult:
        deadline = None if timeout is None else time.monotonic() + timeout
        path = self._root / "receipts" / f"{record_id}.json"
        while not path.exists():
            if deadline is not None and time.monotonic() >= deadline:
                return ReceiptResult(ReceiptState.EXPIRED, "receipt wait timed out")
            time.sleep(self._sleep_duration(deadline))
        value = json.loads(path.read_text(encoding="utf-8"))
        return ReceiptResult(
            ReceiptState(value["state"]), value.get("reason"), int(value["attempts"])
        )

    def _sleep_duration(self, deadline: float | None) -> float:
        if deadline is None:
            return self._poll_interval
        return max(0.0, min(self._poll_interval, deadline - time.monotonic()))

    def receive(self, timeout: float | None = None) -> TransportDelivery | None:
        self._ensure_open()
        self._recover_stale()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            for pending in sorted((self._root / "pending").iterdir()):
                token = uuid.uuid4().hex
                inflight = self._root / "inflight" / token
                with self._locked():
                    try:
                        os.replace(pending, inflight)
                        os.utime(inflight)
                    except FileNotFoundError:
                        continue
                try:
                    manifest = self._read_manifest(inflight)
                    if self._record_id(manifest.get("record_id")) != pending.name:
                        raise DeliveryError("filesystem manifest record ID is invalid")
                    manifest_buffers = manifest.get("buffers")
                    if not isinstance(manifest_buffers, list):
                        raise DeliveryError("filesystem buffer catalog must be a list")
                    metadata_path = inflight / "metadata.json"
                    validate_transfer_limits(
                        metadata_bytes=metadata_path.stat().st_size,
                        tensor_sizes=(self._declared_size(item) for item in manifest_buffers),
                        limits=self._limits,
                    )
                    buffers = tuple(
                        BufferSegment.from_catalog_entry(
                            item,
                            data=self._read_buffer(
                                inflight / "buffers" / f"{index:06d}.bin",
                                int(item["nbytes"]),
                            ),
                        )
                        for index, item in enumerate(manifest_buffers)
                        if isinstance(item, dict)
                        if item.get("filename") == f"{index:06d}.bin"
                    )
                    if len(buffers) != len(manifest_buffers):
                        raise DeliveryError("filesystem buffer filename is invalid")
                    metadata = metadata_path.read_bytes()
                except Exception as error:
                    with self._locked():
                        if inflight.exists():
                            self._write_receipt(
                                pending.name,
                                ReceiptResult(
                                    ReceiptState.REJECTED,
                                    "filesystem delivery is corrupt",
                                ),
                            )
                            shutil.rmtree(inflight)
                    raise DeliveryError("filesystem delivery is corrupt") from error
                return TransportDelivery(
                    token=token,
                    experience_id=str(manifest["experience_id"]),
                    idempotency_key=str(manifest["idempotency_key"]),
                    payload=SerializedExperience(
                        metadata=metadata,
                        buffers=buffers,
                    ),
                    attempt=int(manifest["attempt"]),
                    published_at=float(manifest["published_at"]),
                    max_retries=int(manifest["max_retries"]),
                )
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(self._sleep_duration(deadline))
            self._ensure_open()

    def _settle(self, token: str, state: ReceiptState, reason: str | None, retry: bool) -> None:
        directory = self._root / "inflight" / token
        with self._locked():
            if not directory.exists():
                raise DeliveryError("unknown or already-settled delivery token")
            manifest = self._read_manifest(directory)
            attempt = int(manifest["attempt"])
            if retry and attempt <= int(manifest["max_retries"]):
                manifest["attempt"] = attempt + 1
                manifest["last_error"] = reason
                self._write_json(directory / "manifest.json", manifest)
                os.replace(
                    directory,
                    self._root / "pending" / str(manifest["record_id"]),
                )
                self._fsync_directory(self._root / "pending")
                return
            self._write_receipt(str(manifest["record_id"]), ReceiptResult(state, reason, attempt))
            shutil.rmtree(directory)
            self._fsync_directory(directory.parent)

    def ack(self, token: str) -> None:
        self._settle(token, ReceiptState.ACKED, None, False)

    def nack(self, token: str, reason: str, *, retry: bool = True) -> None:
        self._settle(token, ReceiptState.NACKED, reason, retry)

    def reject(self, token: str, reason: str) -> None:
        self._settle(token, ReceiptState.REJECTED, reason, False)

    def cancel(self, receipt_id: str, reason: str = "cancelled") -> None:
        """Atomically cancel a pending record without touching inflight data."""

        receipt_id = self._record_id(receipt_id)
        pending = self._root / "pending" / receipt_id
        receipt = self._root / "receipts" / f"{receipt_id}.json"
        with self._locked():
            if receipt.exists():
                return
            if not pending.exists():
                for inflight in (self._root / "inflight").iterdir():
                    try:
                        record_id = str(self._read_manifest(inflight)["record_id"])
                    except Exception:
                        continue
                    if record_id == receipt_id:
                        raise DeliveryError("cannot cancel an inflight delivery")
                raise DeliveryError("unknown delivery receipt")
            manifest = self._read_manifest(pending)
            self._write_receipt(
                receipt_id,
                ReceiptResult(ReceiptState.CANCELLED, reason, int(manifest["attempt"])),
            )
            shutil.rmtree(pending)
            self._fsync_directory(pending.parent)

    def health(self) -> HealthStatus:
        with self._locked():
            return HealthStatus(
                not self._closed,
                "closed" if self._closed else "ok",
                self._depth(),
            )

    def close(self, timeout: float | None = None) -> None:
        del timeout
        self._closed = True
