# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-test-only create-once filesystem transport."""

from __future__ import annotations

import errno
import os
import stat
import threading
import uuid
from pathlib import Path, PurePosixPath

from ..codec import crc32c_hex
from ..manifest import DeltaLocation, FilesystemLocation
from .base import (
    CanonicalTransportIdentity,
    ImmutableObjectConflict,
    ObjectVerificationError,
    StoredObject,
    TransportClosedError,
    validate_checksum,
    validate_maximum_size,
    validate_relative_key,
    verify_payload,
)


class FilesystemCanonicalTransport:
    """Atomic immutable files for deterministic single-host tests."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._root_fd = os.open(
            self._root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        self._closed = False
        self._lock = threading.RLock()

    @property
    def identity(self) -> CanonicalTransportIdentity:
        return CanonicalTransportIdentity("filesystem", str(self._root))

    def publish(self, key: str, data: bytes, checksum: str) -> StoredObject:
        with self._lock:
            self._ensure_open()
            validate_checksum(checksum)
            if crc32c_hex(data) != checksum:
                raise ObjectVerificationError(
                    "payload checksum does not match bytes before publish"
                )
            relative = validate_relative_key(key)
            destination = self._root.joinpath(*relative.parts)
            stored = StoredObject(
                location=DeltaLocation(
                    filesystem=FilesystemLocation(path=str(destination))
                ),
                checksum=checksum,
                size=len(data),
            )
            parent_fd = self._open_parent(relative.parts[:-1], create=True)
            temporary = f".{relative.name}.{uuid.uuid4().hex}.partial"
            try:
                temporary_fd = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
                try:
                    view = memoryview(data)
                    while view:
                        written = os.write(temporary_fd, view)
                        if written == 0:
                            raise ObjectVerificationError(
                                "filesystem object write made no progress"
                            )
                        view = view[written:]
                    os.fsync(temporary_fd)
                finally:
                    os.close(temporary_fd)
                try:
                    os.link(
                        temporary,
                        relative.name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    return self._verify_retry(parent_fd, relative.name, stored, data)
                os.fsync(parent_fd)
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                os.close(parent_fd)
            self.verify(stored)
            return stored

    def resolve(
        self,
        location: DeltaLocation,
        checksum: str,
        maximum_size: int,
    ) -> StoredObject:
        with self._lock:
            self._ensure_open()
            validate_checksum(checksum)
            validate_maximum_size(maximum_size)
            safe = self._validated_location(location)
            parent_fd = self._open_parent(safe.parts[:-1], create=False)
            try:
                size = self._regular_file_size(
                    parent_fd,
                    safe.name,
                    maximum_size=maximum_size,
                )
            finally:
                os.close(parent_fd)
            stored = StoredObject(location=location, checksum=checksum, size=size)
            self.fetch(stored)
            return stored

    def fetch(self, stored: StoredObject) -> bytes:
        with self._lock:
            self._ensure_open()
            if (
                not isinstance(stored, StoredObject)
                or not isinstance(stored.size, int)
                or isinstance(stored.size, bool)
                or stored.size < 0
            ):
                raise ObjectVerificationError(
                    "filesystem object metadata has an invalid size"
                )
            safe = self._validated_location(stored.location)
            parent_fd = self._open_parent(safe.parts[:-1], create=False)
            try:
                data = self._read_regular_file(
                    parent_fd, safe.name, maximum_size=stored.size
                )
            finally:
                os.close(parent_fd)
            verify_payload(
                data,
                stored.checksum,
                stored.size,
                context="filesystem object",
            )
            return data

    def verify(self, stored: StoredObject) -> None:
        self.fetch(stored)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            os.close(self._root_fd)
            self._root_fd = -1

    def _ensure_open(self) -> None:
        if self._closed:
            raise TransportClosedError("filesystem canonical transport is closed")

    def _validated_location(self, location: DeltaLocation) -> PurePosixPath:
        if not isinstance(location, DeltaLocation) or location.filesystem is None:
            raise ObjectVerificationError(
                "filesystem transport received a non-filesystem location"
            )
        try:
            path = Path(location.filesystem.path)
            relative = path.relative_to(self._root)
            return validate_relative_key(relative.as_posix())
        except (TypeError, ValueError) as exc:
            raise ObjectVerificationError(
                "filesystem object is outside the configured root"
            ) from exc

    def _verify_retry(
        self,
        parent_fd: int,
        name: str,
        requested: StoredObject,
        requested_data: bytes,
    ) -> StoredObject:
        existing = self._read_regular_file(parent_fd, name, maximum_size=requested.size)
        if existing != requested_data:
            raise ImmutableObjectConflict(
                f"immutable object conflict for filesystem key {name}"
            )
        verify_payload(
            existing,
            requested.checksum,
            requested.size,
            context="filesystem retry object",
        )
        return requested

    def _open_parent(self, parts: tuple[str, ...], *, create: bool) -> int:
        descriptor = os.dup(self._root_fd)
        try:
            for part in parts:
                if create:
                    try:
                        os.mkdir(part, mode=0o755, dir_fd=descriptor)
                        os.fsync(descriptor)
                    except FileExistsError:
                        pass
                try:
                    child = os.open(
                        part,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    raise ObjectVerificationError(
                        "filesystem object is outside the configured root: "
                        "a path component is missing, not a directory, or a symlink"
                    ) from exc
                os.close(descriptor)
                descriptor = child
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    @staticmethod
    def _regular_file_size(
        parent_fd: int,
        name: str,
        *,
        maximum_size: int,
    ) -> int:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
        except OSError as exc:
            detail = (
                "symlink"
                if getattr(exc, "errno", None) == errno.ELOOP
                else "unreadable"
            )
            raise ObjectVerificationError(
                f"filesystem object is {detail}: {exc}"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ObjectVerificationError("filesystem object is not a regular file")
            if metadata.st_size > maximum_size:
                raise ObjectVerificationError(
                    "filesystem object size exceeds maximum_size"
                )
            return metadata.st_size
        finally:
            os.close(descriptor)

    @staticmethod
    def _read_regular_file(parent_fd: int, name: str, *, maximum_size: int) -> bytes:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
        except OSError as exc:
            detail = (
                "symlink"
                if getattr(exc, "errno", None) == errno.ELOOP
                else "unreadable"
            )
            raise ObjectVerificationError(
                f"filesystem object is {detail}: {exc}"
            ) from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ObjectVerificationError("filesystem object is not a regular file")
            chunks = []
            remaining = maximum_size + 1
            while remaining and (
                chunk := os.read(descriptor, min(1024 * 1024, remaining))
            ):
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
