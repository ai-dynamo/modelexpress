# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-base canonical framing, attestations, and bounded local snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

import torch

from ..codec import (
    CodecError,
    compress_payload,
    crc32c_hex,
    decode_delta,
    decompress_payload,
    encode_delta,
)
from ..manifest import DeltaLocation, FilesystemLocation, S3Location
from .base import (
    CanonicalBucket,
    CanonicalBucketConsumer,
    CanonicalTensorSpec,
    canonical_tensor_name,
    tensor_nbytes,
)

_SNAPSHOT_SCHEMA = "mx.canonical.snapshot.v2"
_ROOT_SCHEMA = "mx.canonical.root.v1"
_BUCKET_SCHEMA = "mx.canonical.bucket.v1"
_BUCKET_MAGIC = b"MXCDLT01"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_CRC32C = re.compile(r"[0-9a-f]{8}")
_DEFAULT_MAXIMUM_ROOT_BYTES = 64 * 1024 * 1024
_DEFAULT_MAXIMUM_BUCKET_BYTES = 512 * 1024 * 1024
_MAXIMUM_HEADER_BYTES = 8 * 1024 * 1024
_MAXIMUM_RECORDS = 1_000_000


class CanonicalDeltaError(RuntimeError):
    """Canonical bytes or exact-base attestations failed validation."""


def _read_regular_file_bounded(path: Path, maximum_size: int) -> bytes:
    """Read at most one byte beyond a trusted bound without following symlinks."""
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise CanonicalDeltaError(f"canonical base file is unreadable: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CanonicalDeltaError("canonical base file is not a regular file")
        chunks = []
        remaining = maximum_size + 1
        while remaining and (chunk := os.read(descriptor, min(1024 * 1024, remaining))):
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise CanonicalDeltaError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise CanonicalDeltaError(f"non-canonical JSON constant {value!r}")


def _load_json(data: bytes, context: str) -> object:
    try:
        return json.loads(
            data,
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except CanonicalDeltaError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalDeltaError(f"invalid {context}: {exc}") from exc


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _feed_field(digest: Any, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _dtype_table() -> dict[str, torch.dtype]:
    names = (
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    return {
        name: dtype
        for name in names
        if isinstance((dtype := getattr(torch, name, None)), torch.dtype)
    }


_DTYPES = _dtype_table()


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if _DTYPES.get(name) is not dtype:
        raise CanonicalDeltaError(f"unsupported canonical dtype {dtype}")
    return name


def _shape(value: Iterable[object]) -> tuple[int, ...]:
    shape = tuple(value)
    if any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
        for dimension in shape
    ):
        raise CanonicalDeltaError(f"invalid canonical tensor shape {shape!r}")
    return shape


def _expected_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    if dtype not in _DTYPES:
        raise CanonicalDeltaError(f"unsupported canonical dtype {dtype!r}")
    return math.prod(shape) * torch.empty((), dtype=_DTYPES[dtype]).element_size()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().to(device="cpu").contiguous()
    return cpu.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_from_bytes(dtype: str, shape: tuple[int, ...], data: bytes) -> torch.Tensor:
    if len(data) != _expected_bytes(dtype, shape):
        raise CanonicalDeltaError(
            f"tensor byte size {len(data)} does not match dtype={dtype} shape={shape}"
        )
    if not data:
        return torch.empty(shape, dtype=_DTYPES[dtype])
    return (
        torch.frombuffer(bytearray(data), dtype=_DTYPES[dtype]).reshape(shape).clone()
    )


@dataclass(frozen=True)
class CanonicalTensorMetadata:
    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_size: int
    content_digest: str
    blob: str


@dataclass(frozen=True)
class CanonicalFormatIdentity:
    """Semantic identity of one normalized HF state-dict representation."""

    logical_format: str = "hf.state_dict"
    normalization_profile: str = "hf-save-pretrained-v1"
    byte_order: str = "little"
    quantization_profile: str = "none"
    atomic_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        for field, value in (
            ("logical_format", self.logical_format),
            ("normalization_profile", self.normalization_profile),
            ("quantization_profile", self.quantization_profile),
        ):
            if not isinstance(value, str) or not value or len(value) > 1024:
                raise ValueError(f"{field} must be a bounded non-empty string")
        if self.byte_order != "little":
            raise ValueError("canonical byte_order must be 'little'")
        if sys.byteorder != self.byte_order:
            raise ValueError(
                "host byte order cannot produce the canonical little-endian profile"
            )
        if not isinstance(self.atomic_groups, tuple):
            raise ValueError("atomic_groups must be a tuple")
        normalized_groups = []
        seen: set[str] = set()
        for group in self.atomic_groups:
            if not isinstance(group, tuple) or not group:
                raise ValueError("atomic groups must be non-empty tuples")
            normalized = tuple(sorted(canonical_tensor_name(name) for name in group))
            if len(normalized) != len(set(normalized)):
                raise ValueError("an atomic group contains duplicate tensor names")
            overlap = seen.intersection(normalized)
            if overlap:
                raise ValueError(
                    f"tensor {min(overlap)!r} belongs to multiple atomic groups"
                )
            seen.update(normalized)
            normalized_groups.append(normalized)
        canonical_groups = tuple(sorted(normalized_groups))
        if len(canonical_groups) > _MAXIMUM_RECORDS:
            raise ValueError("canonical format has too many atomic groups")
        object.__setattr__(self, "atomic_groups", canonical_groups)


DEFAULT_CANONICAL_FORMAT_IDENTITY = CanonicalFormatIdentity()


@dataclass(frozen=True)
class CanonicalCapture:
    """A capture callback inseparably bound to its canonical representation."""

    callback: Callable[[str, CanonicalBucketConsumer], None]
    format_identity: CanonicalFormatIdentity
    canonical_schema: tuple[CanonicalTensorSpec, ...]
    format_digest: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if not callable(self.callback):
            raise TypeError("canonical capture callback must be callable")
        if not isinstance(self.format_identity, CanonicalFormatIdentity):
            raise TypeError("canonical capture format identity is invalid")
        if not isinstance(self.canonical_schema, tuple) or not self.canonical_schema:
            raise TypeError("canonical capture requires a non-empty tuple HF schema")
        object.__setattr__(
            self,
            "format_digest",
            canonical_format_digest(self.format_identity, self.canonical_schema),
        )

    def __call__(self, version: str, consume: CanonicalBucketConsumer) -> None:
        self.callback(version, consume)


def validate_canonical_format_identity(identity: CanonicalFormatIdentity) -> None:
    """Fail closed for representation profiles this source cannot produce."""
    if not isinstance(identity, CanonicalFormatIdentity):
        raise TypeError("format_identity must be CanonicalFormatIdentity")
    if identity.logical_format != "hf.state_dict":
        raise CanonicalDeltaError(
            f"unsupported canonical logical format {identity.logical_format!r}"
        )
    if identity.normalization_profile != "hf-save-pretrained-v1":
        raise CanonicalDeltaError(
            "unsupported canonical normalization profile "
            f"{identity.normalization_profile!r}"
        )
    if identity.quantization_profile != "none":
        raise CanonicalDeltaError(
            "unsupported canonical quantization profile "
            f"{identity.quantization_profile!r}"
        )


def canonical_capture_units(
    named_sizes: Iterable[tuple[str, int]],
    identity: CanonicalFormatIdentity,
    bucket_bytes: int,
) -> tuple[tuple[tuple[str, ...], ...], tuple[int, ...]]:
    """Validate one supported HF profile and plan indivisible bounded units."""
    validate_canonical_format_identity(identity)
    if not isinstance(bucket_bytes, int) or isinstance(bucket_bytes, bool):
        raise ValueError("bucket_bytes must be a positive integer")
    if bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be a positive integer")

    ordered = tuple(named_sizes)
    names = tuple(name for name, _size in ordered)
    if len(names) != len(set(names)):
        raise CanonicalDeltaError("canonical capture plan contains duplicate names")
    sizes = {}
    for name, size in ordered:
        canonical_name = canonical_tensor_name(name)
        if canonical_name != name:
            raise CanonicalDeltaError(f"capture plan name {name!r} is not canonical")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CanonicalDeltaError(
                f"capture plan tensor {name!r} has an invalid byte size"
            )
        sizes[name] = size

    grouped: dict[str, tuple[str, ...]] = {}
    name_positions = {name: index for index, name in enumerate(names)}
    for group in identity.atomic_groups:
        missing = tuple(name for name in group if name not in sizes)
        if missing:
            raise CanonicalDeltaError(
                f"atomic group references missing canonical tensors {missing!r}"
            )
        positions = sorted(name_positions[name] for name in group)
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise CanonicalDeltaError(
                f"atomic group {group!r} is not contiguous in canonical order"
            )
        ordered_group = tuple(names[index] for index in positions)
        unit_size = sum(sizes[name] for name in ordered_group)
        if unit_size > bucket_bytes:
            raise CanonicalDeltaError(
                f"atomic group {ordered_group!r} size {unit_size} exceeds "
                f"bucket_bytes={bucket_bytes}"
            )
        for name in ordered_group:
            grouped[name] = ordered_group

    units = []
    unit_sizes = []
    visited: set[str] = set()
    for name in names:
        if name in visited:
            continue
        unit = grouped.get(name, (name,))
        unit_size = sum(sizes[item] for item in unit)
        if unit_size > bucket_bytes:
            raise CanonicalDeltaError(
                f"canonical tensor {name!r} size {unit_size} exceeds "
                f"bucket_bytes={bucket_bytes}"
            )
        units.append(unit)
        unit_sizes.append(unit_size)
        visited.update(unit)
    return tuple(units), tuple(unit_sizes)


@dataclass(frozen=True)
class CanonicalSnapshot:
    version: str
    format_digest: str
    target_digest: str
    tensors: tuple[CanonicalTensorMetadata, ...]
    format_identity: CanonicalFormatIdentity = DEFAULT_CANONICAL_FORMAT_IDENTITY


def _identity_dict(identity: CanonicalFormatIdentity) -> dict[str, object]:
    return {
        "atomic_groups": [list(group) for group in identity.atomic_groups],
        "byte_order": identity.byte_order,
        "logical_format": identity.logical_format,
        "normalization_profile": identity.normalization_profile,
        "quantization_profile": identity.quantization_profile,
    }


def _format_digest(
    identity: CanonicalFormatIdentity,
    tensors: Iterable[CanonicalTensorMetadata],
) -> str:
    digest = hashlib.sha256()
    _feed_field(digest, b"mx.canonical.format.v2")
    _feed_field(digest, _json_bytes(_identity_dict(identity)))
    for tensor in tensors:
        _feed_field(digest, tensor.name.encode("utf-8"))
        _feed_field(digest, tensor.dtype.encode("ascii"))
        _feed_field(digest, _json_bytes(tensor.shape))
        _feed_field(digest, str(tensor.byte_size).encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


def canonical_format_digest(
    identity: CanonicalFormatIdentity,
    schema: Sequence[CanonicalTensorSpec],
) -> str:
    """Hash an authoritative HF schema using the snapshot format contract."""
    validate_canonical_format_identity(identity)
    if not schema:
        raise CanonicalDeltaError("canonical HF schema must be non-empty")
    digest = hashlib.sha256()
    _feed_field(digest, b"mx.canonical.format.v2")
    _feed_field(digest, _json_bytes(_identity_dict(identity)))
    names: set[str] = set()
    for spec in schema:
        if not isinstance(spec, CanonicalTensorSpec):
            raise TypeError("canonical HF schema entries must be CanonicalTensorSpec")
        if spec.name in names:
            raise CanonicalDeltaError(
                f"canonical HF schema duplicates tensor {spec.name!r}"
            )
        names.add(spec.name)
        _feed_field(digest, spec.name.encode("utf-8"))
        _feed_field(digest, _dtype_name(spec.dtype).encode("ascii"))
        _feed_field(digest, _json_bytes(spec.shape))
        _feed_field(digest, str(spec.nbytes).encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


def _target_digest(tensors: Iterable[CanonicalTensorMetadata]) -> str:
    digest = hashlib.sha256()
    _feed_field(digest, b"mx.canonical.content.v1")
    for tensor in tensors:
        _feed_field(digest, tensor.name.encode("utf-8"))
        _feed_field(digest, tensor.dtype.encode("ascii"))
        _feed_field(digest, _json_bytes(tensor.shape))
        _feed_field(digest, tensor.content_digest.encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


def _metadata_dict(metadata: CanonicalTensorMetadata) -> dict[str, object]:
    return {
        "blob": metadata.blob,
        "byte_size": metadata.byte_size,
        "content_digest": metadata.content_digest,
        "dtype": metadata.dtype,
        "name": metadata.name,
        "shape": list(metadata.shape),
    }


def _snapshot_bytes(snapshot: CanonicalSnapshot) -> bytes:
    return _json_bytes(
        {
            "format_identity": _identity_dict(snapshot.format_identity),
            "format_digest": snapshot.format_digest,
            "schema": _SNAPSHOT_SCHEMA,
            "target_digest": snapshot.target_digest,
            "tensors": [_metadata_dict(item) for item in snapshot.tensors],
            "version": snapshot.version,
        }
    )


def _require_object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CanonicalDeltaError(f"{context} must be a JSON object")
    return value


def _require_keys(
    value: Mapping[str, object],
    required: set[str],
    context: str,
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    missing = required - value.keys()
    extra = value.keys() - allowed
    if missing or extra:
        raise CanonicalDeltaError(
            f"{context} fields are invalid; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise CanonicalDeltaError(f"{context} must be a non-empty string")
    return value


def _require_int(value: object, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CanonicalDeltaError(f"{context} must be an integer >= {minimum}")
    return value


def _require_digest(value: object, context: str) -> str:
    digest = _require_string(value, context)
    if not _SHA256.fullmatch(digest):
        raise CanonicalDeltaError(f"{context} is not a canonical SHA-256 digest")
    return digest


def _decode_format_identity(value: object) -> CanonicalFormatIdentity:
    item = _require_object(value, "snapshot format_identity")
    _require_keys(
        item,
        {
            "atomic_groups",
            "byte_order",
            "logical_format",
            "normalization_profile",
            "quantization_profile",
        },
        "snapshot format_identity",
    )
    groups_value = item["atomic_groups"]
    if not isinstance(groups_value, list):
        raise CanonicalDeltaError("snapshot atomic_groups must be an array")
    groups = []
    for index, group_value in enumerate(groups_value):
        if not isinstance(group_value, list):
            raise CanonicalDeltaError(f"snapshot atomic group {index} must be an array")
        groups.append(
            tuple(
                _require_string(name, f"snapshot atomic group {index} name")
                for name in group_value
            )
        )
    try:
        identity = CanonicalFormatIdentity(
            logical_format=_require_string(
                item["logical_format"], "snapshot logical_format"
            ),
            normalization_profile=_require_string(
                item["normalization_profile"], "snapshot normalization_profile"
            ),
            byte_order=_require_string(item["byte_order"], "snapshot byte_order"),
            quantization_profile=_require_string(
                item["quantization_profile"], "snapshot quantization_profile"
            ),
            atomic_groups=tuple(groups),
        )
        if identity.atomic_groups != tuple(groups):
            raise ValueError("snapshot atomic_groups are not in canonical order")
        return identity
    except ValueError as exc:
        raise CanonicalDeltaError(f"invalid snapshot format_identity: {exc}") from exc


def _decode_metadata(value: object, context: str) -> CanonicalTensorMetadata:
    item = _require_object(value, context)
    _require_keys(
        item,
        {"blob", "byte_size", "content_digest", "dtype", "name", "shape"},
        context,
    )
    name = canonical_tensor_name(_require_string(item["name"], f"{context}.name"))
    dtype = _require_string(item["dtype"], f"{context}.dtype")
    shape_value = item["shape"]
    if not isinstance(shape_value, list):
        raise CanonicalDeltaError(f"{context}.shape must be an array")
    shape = _shape(shape_value)
    byte_size = _require_int(item["byte_size"], f"{context}.byte_size")
    if byte_size != _expected_bytes(dtype, shape):
        raise CanonicalDeltaError(f"{context}.byte_size does not match dtype and shape")
    content_digest = _require_digest(
        item["content_digest"], f"{context}.content_digest"
    )
    blob = _require_string(item["blob"], f"{context}.blob")
    if blob != content_digest.removeprefix("sha256:"):
        raise CanonicalDeltaError(f"{context}.blob does not match content_digest")
    return CanonicalTensorMetadata(name, dtype, shape, byte_size, content_digest, blob)


class _SnapshotWriter:
    def __init__(
        self,
        store: FilesystemCanonicalBaseStore,
        version: str,
        format_identity: CanonicalFormatIdentity,
    ) -> None:
        self._store = store
        self._version = _require_string(version, "snapshot version")
        if not isinstance(format_identity, CanonicalFormatIdentity):
            raise TypeError("format_identity must be CanonicalFormatIdentity")
        self._format_identity = format_identity
        self._tensors: list[CanonicalTensorMetadata] = []
        self._seen: set[str] = set()
        self._finished = False

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        canonical_name = canonical_tensor_name(name)
        dtype = _dtype_name(tensor.dtype)
        shape = tuple(tensor.shape)
        data = _tensor_bytes(tensor)
        self.add_tensor_bytes(canonical_name, dtype, shape, data)

    def add_tensor_bytes(
        self, name: str, dtype: str, shape: tuple[int, ...], data: bytes
    ) -> None:
        if self._finished:
            raise CanonicalDeltaError("snapshot writer is already finished")
        if len(self._tensors) >= _MAXIMUM_RECORDS:
            raise CanonicalDeltaError("canonical snapshot has too many tensors")
        canonical_name = canonical_tensor_name(name)
        if canonical_name in self._seen:
            raise CanonicalDeltaError("snapshot tensor names must be unique")
        self._seen.add(canonical_name)
        normalized_shape = _shape(shape)
        expected_size = _expected_bytes(dtype, normalized_shape)
        if len(data) != expected_size:
            raise CanonicalDeltaError(
                f"canonical tensor {canonical_name!r} has {len(data)} bytes; "
                f"expected {expected_size}"
            )
        content_digest = _sha256(data)
        blob = content_digest.removeprefix("sha256:")
        self._store._ensure_blob(blob, data)
        self._tensors.append(
            CanonicalTensorMetadata(
                canonical_name,
                dtype,
                normalized_shape,
                len(data),
                content_digest,
                blob,
            )
        )

    def preview(self) -> CanonicalSnapshot:
        if self._finished:
            raise CanonicalDeltaError("snapshot writer is already finished")
        if not self._tensors:
            raise CanonicalDeltaError("canonical snapshot cannot be empty")
        tensors = tuple(self._tensors)
        return CanonicalSnapshot(
            version=self._version,
            format_digest=_format_digest(self._format_identity, tensors),
            target_digest=_target_digest(tensors),
            tensors=tensors,
            format_identity=self._format_identity,
        )

    def finalize(
        self,
        *,
        expected_format_digest: str | None = None,
        expected_target_digest: str | None = None,
    ) -> CanonicalSnapshot:
        snapshot = self.preview()
        if (
            expected_format_digest is not None
            and snapshot.format_digest != expected_format_digest
        ):
            raise CanonicalDeltaError("reconstructed target format digest mismatch")
        if (
            expected_target_digest is not None
            and snapshot.target_digest != expected_target_digest
        ):
            raise CanonicalDeltaError("reconstructed target digest mismatch")
        encoded = _snapshot_bytes(snapshot)
        if len(encoded) > _DEFAULT_MAXIMUM_ROOT_BYTES:
            raise CanonicalDeltaError("canonical snapshot index exceeds maximum size")
        self._finished = True
        self._store._write_snapshot(snapshot)
        return snapshot

    def abort(self) -> None:
        self._finished = True
        self._tensors.clear()


class FilesystemCanonicalBaseStore:
    """Content-addressed exact bases; metadata is immutable per version."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root).resolve()
        self._blobs = self._root / "blobs"
        self._snapshots = self._root / "snapshots"
        self._blobs.mkdir(parents=True, exist_ok=True)
        self._snapshots.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._metadata: dict[tuple[str, str], dict[str, CanonicalTensorMetadata]] = {}

    def begin_snapshot(
        self,
        version: str,
        *,
        format_identity: CanonicalFormatIdentity = DEFAULT_CANONICAL_FORMAT_IDENTITY,
    ) -> _SnapshotWriter:
        self._ensure_open()
        return _SnapshotWriter(self, version, format_identity)

    def create_snapshot(
        self,
        version: str,
        buckets: Iterable[CanonicalBucket],
        *,
        format_identity: CanonicalFormatIdentity = DEFAULT_CANONICAL_FORMAT_IDENTITY,
    ) -> CanonicalSnapshot:
        writer = self.begin_snapshot(version, format_identity=format_identity)
        try:
            for bucket in buckets:
                for name, tensor in bucket:
                    writer.add_tensor(name, tensor)
            return writer.finalize()
        except Exception:
            writer.abort()
            raise

    def open_snapshot(self, version: str) -> CanonicalSnapshot:
        self._ensure_open()
        version = _require_string(version, "snapshot version")
        path = self._snapshot_path(version)
        try:
            encoded = _read_regular_file_bounded(path, _DEFAULT_MAXIMUM_ROOT_BYTES)
        except FileNotFoundError as exc:
            raise CanonicalDeltaError(
                f"canonical base version {version!r} is unavailable"
            ) from exc
        if len(encoded) > _DEFAULT_MAXIMUM_ROOT_BYTES:
            raise CanonicalDeltaError("canonical snapshot index exceeds maximum size")
        document = _load_json(encoded, "canonical snapshot index")
        root = _require_object(document, "snapshot index")
        _require_keys(
            root,
            {
                "format_digest",
                "format_identity",
                "schema",
                "target_digest",
                "tensors",
                "version",
            },
            "snapshot index",
        )
        if root["schema"] != _SNAPSHOT_SCHEMA:
            raise CanonicalDeltaError(
                f"unsupported canonical snapshot schema {root['schema']!r}"
            )
        decoded_version = _require_string(root["version"], "snapshot version")
        if decoded_version != version:
            raise CanonicalDeltaError(
                "canonical snapshot version does not match its lookup key"
            )
        tensor_values = root["tensors"]
        if not isinstance(tensor_values, list) or not tensor_values:
            raise CanonicalDeltaError(
                "canonical snapshot tensors must be a non-empty array"
            )
        if len(tensor_values) > _MAXIMUM_RECORDS:
            raise CanonicalDeltaError("canonical snapshot has too many tensors")
        tensors = tuple(
            _decode_metadata(value, f"snapshot tensor {index}")
            for index, value in enumerate(tensor_values)
        )
        names = [item.name for item in tensors]
        if len(names) != len(set(names)):
            raise CanonicalDeltaError("canonical snapshot tensor names are not unique")
        format_identity = _decode_format_identity(root["format_identity"])
        snapshot = CanonicalSnapshot(
            version=decoded_version,
            format_digest=_require_digest(
                root["format_digest"], "snapshot format_digest"
            ),
            target_digest=_require_digest(
                root["target_digest"], "snapshot target_digest"
            ),
            tensors=tensors,
            format_identity=format_identity,
        )
        if snapshot.format_digest != _format_digest(format_identity, tensors):
            raise CanonicalDeltaError("canonical snapshot format digest mismatch")
        if snapshot.target_digest != _target_digest(tensors):
            raise CanonicalDeltaError("canonical snapshot target digest mismatch")
        self._remember_snapshot(snapshot)
        return snapshot

    def read_tensor_bytes(self, snapshot: CanonicalSnapshot, name: str) -> bytes:
        self._ensure_open()
        metadata = self._metadata_for(snapshot).get(name)
        if metadata is None:
            raise CanonicalDeltaError(
                f"canonical base {snapshot.version!r} has no tensor {name!r}"
            )
        path = self._blobs / metadata.blob
        try:
            data = _read_regular_file_bounded(path, metadata.byte_size)
        except FileNotFoundError as exc:
            raise CanonicalDeltaError(
                f"canonical base blob {metadata.blob!r} is missing"
            ) from exc
        if len(data) != metadata.byte_size or _sha256(data) != metadata.content_digest:
            raise CanonicalDeltaError(
                f"canonical base tensor {name!r} failed content verification"
            )
        return data

    def read_tensor(self, snapshot: CanonicalSnapshot, name: str) -> torch.Tensor:
        metadata = self._metadata_for(snapshot).get(name)
        if metadata is None:
            raise CanonicalDeltaError(
                f"canonical base {snapshot.version!r} has no tensor {name!r}"
            )
        return _tensor_from_bytes(
            metadata.dtype,
            metadata.shape,
            self.read_tensor_bytes(snapshot, name),
        )

    def close(self) -> None:
        self._closed = True
        self._metadata.clear()

    def _ensure_open(self) -> None:
        if self._closed:
            raise CanonicalDeltaError("canonical base store is closed")

    def _snapshot_path(self, version: str) -> Path:
        key = hashlib.sha256(
            b"mx.snapshot.version\0" + version.encode("utf-8")
        ).hexdigest()
        return self._snapshots / f"{key}.json"

    def _remember_snapshot(self, snapshot: CanonicalSnapshot) -> None:
        self._metadata[(snapshot.version, snapshot.target_digest)] = {
            item.name: item for item in snapshot.tensors
        }

    def _metadata_for(
        self, snapshot: CanonicalSnapshot
    ) -> dict[str, CanonicalTensorMetadata]:
        key = (snapshot.version, snapshot.target_digest)
        metadata = self._metadata.get(key)
        if metadata is None:
            metadata = {item.name: item for item in snapshot.tensors}
            self._metadata[key] = metadata
        return metadata

    def attest_snapshot(self, snapshot: CanonicalSnapshot) -> CanonicalSnapshot:
        """Bind a supplied snapshot object to its immutable stored attestation."""
        verified = self.open_snapshot(snapshot.version)
        if verified != snapshot:
            raise CanonicalDeltaError(
                "canonical snapshot attestation does not match immutable storage"
            )
        return verified

    def _ensure_blob(self, blob: str, data: bytes) -> None:
        self._ensure_open()
        destination = self._blobs / blob
        if destination.exists():
            if _read_regular_file_bounded(destination, len(data)) != data:
                raise CanonicalDeltaError(
                    f"immutable canonical blob conflict for {blob}"
                )
            return
        temporary = self._blobs / f".{blob}.{uuid.uuid4().hex}.partial"
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if _read_regular_file_bounded(destination, len(data)) != data:
                    raise CanonicalDeltaError(
                        f"immutable canonical blob conflict for {blob}"
                    )
            self._fsync_directory(self._blobs)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_snapshot(self, snapshot: CanonicalSnapshot) -> None:
        self._ensure_open()
        destination = self._snapshot_path(snapshot.version)
        encoded = _snapshot_bytes(snapshot)
        if destination.exists():
            if _read_regular_file_bounded(destination, len(encoded)) != encoded:
                raise CanonicalDeltaError(
                    f"immutable canonical snapshot conflict for version {snapshot.version!r}"
                )
            return
        temporary = self._snapshots / f".{destination.name}.{uuid.uuid4().hex}.partial"
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if _read_regular_file_bounded(destination, len(encoded)) != encoded:
                    raise CanonicalDeltaError(
                        f"immutable canonical snapshot conflict for version {snapshot.version!r}"
                    )
            self._fsync_directory(self._snapshots)
        finally:
            temporary.unlink(missing_ok=True)
        self._remember_snapshot(snapshot)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class EncodedCanonicalBucket:
    ordinal: int
    checksum: str
    tensor_names: tuple[str, ...]
    decoded_size: int
    compressed_size: int
    data: bytes


@dataclass(frozen=True)
class CanonicalBucketReference:
    ordinal: int
    checksum: str
    size: int
    decoded_size: int
    tensor_names: tuple[str, ...]
    location: DeltaLocation


@dataclass(frozen=True)
class CanonicalTensorCoverage:
    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_size: int
    target_digest: str
    change_state: str
    bucket_ordinal: int | None = None


@dataclass(frozen=True)
class CanonicalRootIndex:
    model_id: str
    base_version: str
    target_version: str
    delta_method: str
    compression_algorithm: str
    format_digest: str
    base_digest: str
    target_digest: str
    buckets: tuple[CanonicalBucketReference, ...]
    tensors: tuple[CanonicalTensorCoverage, ...]


@dataclass(frozen=True)
class CanonicalPublication:
    root_index: CanonicalRootIndex
    root_bytes: bytes | None
    root_checksum: str | None
    target_snapshot: CanonicalSnapshot
    changed: bool


def _location_dict(location: DeltaLocation) -> dict[str, object]:
    if location.filesystem is not None:
        return {"kind": "filesystem", "path": location.filesystem.path}
    if location.s3 is not None:
        value: dict[str, object] = {
            "bucket": location.s3.bucket,
            "key": location.s3.key,
            "kind": "s3",
        }
        if location.s3.object_version is not None:
            value["object_version"] = location.s3.object_version
        return value
    raise CanonicalDeltaError("CANONICAL root supports only filesystem or S3 locations")


def _decode_location(value: object, context: str) -> DeltaLocation:
    location = _require_object(value, context)
    kind = location.get("kind")
    if kind == "filesystem":
        _require_keys(location, {"kind", "path"}, context)
        return DeltaLocation(
            filesystem=FilesystemLocation(
                path=_require_string(location["path"], f"{context}.path")
            )
        )
    if kind == "s3":
        _require_keys(
            location,
            {"bucket", "key", "kind"},
            context,
            optional={"object_version"},
        )
        object_version = location.get("object_version")
        if object_version is not None:
            object_version = _require_string(
                object_version, f"{context}.object_version"
            )
        return DeltaLocation(
            s3=S3Location(
                bucket=_require_string(location["bucket"], f"{context}.bucket"),
                key=_require_string(location["key"], f"{context}.key"),
                object_version=object_version,
            )
        )
    raise CanonicalDeltaError(f"{context} has unsupported transport {kind!r}")


def _coverage_dict(item: CanonicalTensorCoverage) -> dict[str, object]:
    value: dict[str, object] = {
        "byte_size": item.byte_size,
        "change_state": item.change_state,
        "dtype": item.dtype,
        "name": item.name,
        "shape": list(item.shape),
        "target_digest": item.target_digest,
    }
    if item.bucket_ordinal is not None:
        value["bucket_ordinal"] = item.bucket_ordinal
    return value


def _reference_dict(item: CanonicalBucketReference) -> dict[str, object]:
    return {
        "checksum": item.checksum,
        "decoded_size": item.decoded_size,
        "location": _location_dict(item.location),
        "ordinal": item.ordinal,
        "size": item.size,
        "tensor_names": list(item.tensor_names),
    }


def encode_root_index(
    root: CanonicalRootIndex,
    *,
    maximum_bytes: int = _DEFAULT_MAXIMUM_ROOT_BYTES,
) -> bytes:
    encoded = _json_bytes(
        {
            "base_digest": root.base_digest,
            "base_version": root.base_version,
            "buckets": [_reference_dict(item) for item in root.buckets],
            "compression_algorithm": root.compression_algorithm,
            "delta_method": root.delta_method,
            "format_digest": root.format_digest,
            "model_id": root.model_id,
            "schema": _ROOT_SCHEMA,
            "target_digest": root.target_digest,
            "target_version": root.target_version,
            "tensors": [_coverage_dict(item) for item in root.tensors],
        }
    )
    if len(encoded) > maximum_bytes:
        raise CanonicalDeltaError("root index exceeds maximum root index size")
    return encoded


def _encode_bucket(
    *,
    ordinal: int,
    model_id: str,
    base: CanonicalSnapshot,
    target_version: str,
    delta_method: str,
    compression_algorithm: str,
    entries: list[dict[str, object]],
    decoded: bytes,
) -> EncodedCanonicalBucket:
    compressed = compress_payload(compression_algorithm, decoded)
    header = _json_bytes(
        {
            "base_digest": base.target_digest,
            "base_version": base.version,
            "compression_algorithm": compression_algorithm,
            "decoded_size": len(decoded),
            "delta_method": delta_method,
            "entries": entries,
            "format_digest": base.format_digest,
            "model_id": model_id,
            "ordinal": ordinal,
            "schema": _BUCKET_SCHEMA,
            "target_version": target_version,
        }
    )
    if len(header) > _MAXIMUM_HEADER_BYTES:
        raise CanonicalDeltaError("canonical bucket header exceeds maximum size")
    data = _BUCKET_MAGIC + struct.pack(">I", len(header)) + header + compressed
    return EncodedCanonicalBucket(
        ordinal=ordinal,
        checksum=crc32c_hex(data),
        tensor_names=tuple(str(entry["name"]) for entry in entries),
        decoded_size=len(decoded),
        compressed_size=len(compressed),
        data=data,
    )


class CanonicalDeltaEncoder:
    """Stream a complete target against one exact immutable base."""

    def __init__(
        self,
        *,
        model_id: str,
        target_version: str,
        base_store: FilesystemCanonicalBaseStore,
        base: CanonicalSnapshot,
        delta_method: str,
        compression_algorithm: str,
        publish_bucket: Callable[[EncodedCanonicalBucket], DeltaLocation],
        maximum_encoded_ratio: float = 1.0,
        maximum_bucket_bytes: int = _DEFAULT_MAXIMUM_BUCKET_BYTES,
    ) -> None:
        self._model_id = _require_string(model_id, "model_id")
        self._target_version = _require_string(target_version, "target_version")
        if target_version == base.version:
            raise CanonicalDeltaError("target_version must differ from base_version")
        if not math.isfinite(maximum_encoded_ratio) or maximum_encoded_ratio <= 0:
            raise ValueError("maximum_encoded_ratio must be finite and positive")
        if (
            not isinstance(maximum_bucket_bytes, int)
            or isinstance(maximum_bucket_bytes, bool)
            or maximum_bucket_bytes <= 0
        ):
            raise ValueError("maximum_bucket_bytes must be positive")
        try:
            encode_delta(delta_method, b"", b"")
            compress_payload(compression_algorithm, b"")
        except CodecError as exc:
            raise CanonicalDeltaError(str(exc)) from exc
        self._base_store = base_store
        self._base = base_store.attest_snapshot(base)
        self._base_by_name = {item.name: item for item in self._base.tensors}
        if len(self._base_by_name) != len(self._base.tensors) or not self._base.tensors:
            raise CanonicalDeltaError("base snapshot coverage is invalid")
        self._delta_method = delta_method
        self._compression_algorithm = compression_algorithm
        self._publish_bucket = publish_bucket
        self._maximum_encoded_ratio = maximum_encoded_ratio
        self._maximum_bucket_bytes = maximum_bucket_bytes
        self._writer = base_store.begin_snapshot(
            target_version,
            format_identity=self._base.format_identity,
        )
        self._coverage: list[CanonicalTensorCoverage] = []
        self._references: list[CanonicalBucketReference] = []
        self._next_base_index = 0
        self._target_size = 0
        self._finished = False

    def consume_bucket(self, bucket: CanonicalBucket) -> None:
        if self._finished:
            raise CanonicalDeltaError("canonical delta encoder is already finished")
        if not bucket:
            raise CanonicalDeltaError("canonical source emitted an empty bucket")
        if len(bucket) > _MAXIMUM_RECORDS:
            raise CanonicalDeltaError("canonical source bucket has too many tensors")
        bucket_size = sum(
            tensor_nbytes(tensor)
            for _, tensor in bucket
            if isinstance(tensor, torch.Tensor)
        )
        if bucket_size > self._maximum_bucket_bytes:
            raise CanonicalDeltaError(
                "canonical source bucket exceeds maximum decoded size"
            )
        entries: list[dict[str, object]] = []
        deltas: list[bytes] = []
        decoded_offset = 0
        bucket_ordinal = len(self._references)
        for name, tensor in bucket:
            canonical_name = canonical_tensor_name(name)
            if self._next_base_index >= len(self._base.tensors):
                raise CanonicalDeltaError(
                    f"target contains tensor {canonical_name!r} outside the exact base"
                )
            base_metadata = self._base.tensors[self._next_base_index]
            if canonical_name != base_metadata.name:
                raise CanonicalDeltaError(
                    "target does not follow complete exact-base coverage: "
                    f"expected {base_metadata.name!r}, got {canonical_name!r}"
                )
            if not isinstance(tensor, torch.Tensor):
                raise CanonicalDeltaError(
                    f"target tensor {canonical_name!r} is not a tensor"
                )
            dtype = _dtype_name(tensor.dtype)
            shape = tuple(tensor.shape)
            if dtype != base_metadata.dtype or shape != base_metadata.shape:
                raise CanonicalDeltaError(
                    f"target tensor {canonical_name!r} changed canonical dtype or shape"
                )
            if tensor_nbytes(tensor) != base_metadata.byte_size:
                raise CanonicalDeltaError(
                    f"target tensor {canonical_name!r} changed canonical byte size"
                )
            target = _tensor_bytes(tensor)
            target_digest = _sha256(target)
            base_bytes = self._base_store.read_tensor_bytes(self._base, canonical_name)
            if _sha256(base_bytes) != base_metadata.content_digest:
                raise CanonicalDeltaError(
                    f"canonical base tensor {canonical_name!r} failed exact-base verification"
                )
            self._writer.add_tensor_bytes(canonical_name, dtype, shape, target)
            self._target_size += len(target)
            if target_digest == base_metadata.content_digest:
                self._coverage.append(
                    CanonicalTensorCoverage(
                        canonical_name,
                        dtype,
                        shape,
                        len(target),
                        target_digest,
                        "CLEAN",
                    )
                )
            else:
                try:
                    delta = encode_delta(self._delta_method, base_bytes, target)
                except CodecError as exc:
                    raise CanonicalDeltaError(str(exc)) from exc
                entries.append(
                    {
                        "byte_size": len(delta),
                        "dtype": dtype,
                        "name": canonical_name,
                        "offset": decoded_offset,
                        "shape": list(shape),
                        "target_digest": target_digest,
                    }
                )
                deltas.append(delta)
                decoded_offset += len(delta)
                self._coverage.append(
                    CanonicalTensorCoverage(
                        canonical_name,
                        dtype,
                        shape,
                        len(target),
                        target_digest,
                        "DIRTY",
                        bucket_ordinal,
                    )
                )
            self._next_base_index += 1

        if not entries:
            return
        encoded = _encode_bucket(
            ordinal=bucket_ordinal,
            model_id=self._model_id,
            base=self._base,
            target_version=self._target_version,
            delta_method=self._delta_method,
            compression_algorithm=self._compression_algorithm,
            entries=entries,
            decoded=b"".join(deltas),
        )
        if (
            encoded.decoded_size > self._maximum_bucket_bytes
            or len(encoded.data) > self._maximum_bucket_bytes
        ):
            raise CanonicalDeltaError(
                "canonical bucket exceeds maximum encoded or decoded size"
            )
        location = self._publish_bucket(encoded)
        if not isinstance(location, DeltaLocation):
            raise CanonicalDeltaError("publish_bucket must return a DeltaLocation")
        self._references.append(
            CanonicalBucketReference(
                ordinal=encoded.ordinal,
                checksum=encoded.checksum,
                size=len(encoded.data),
                decoded_size=encoded.decoded_size,
                tensor_names=encoded.tensor_names,
                location=location,
            )
        )

    def finish(self) -> CanonicalPublication:
        if self._finished:
            raise CanonicalDeltaError("canonical delta encoder is already finished")
        self._finished = True
        if self._next_base_index != len(self._base.tensors):
            self._writer.abort()
            missing = len(self._base.tensors) - self._next_base_index
            raise CanonicalDeltaError(
                f"target did not provide complete coverage; {missing} base tensors are missing"
            )
        target_snapshot = self._writer.preview()
        if target_snapshot.format_digest != self._base.format_digest:
            self._writer.abort()
            raise CanonicalDeltaError("target format digest does not match exact base")
        root = CanonicalRootIndex(
            model_id=self._model_id,
            base_version=self._base.version,
            target_version=self._target_version,
            delta_method=self._delta_method,
            compression_algorithm=self._compression_algorithm,
            format_digest=self._base.format_digest,
            base_digest=self._base.target_digest,
            target_digest=target_snapshot.target_digest,
            buckets=tuple(self._references),
            tensors=tuple(self._coverage),
        )
        if not self._references:
            if target_snapshot.target_digest != self._base.target_digest:
                self._writer.abort()
                raise CanonicalDeltaError(
                    "clean target digest does not match exact base"
                )
            target_snapshot = self._writer.finalize(
                expected_format_digest=self._base.format_digest,
                expected_target_digest=self._base.target_digest,
            )
            return CanonicalPublication(root, None, None, target_snapshot, False)
        root_bytes = encode_root_index(root)
        physical_size = sum(reference.size for reference in self._references) + len(
            root_bytes
        )
        encoded_ratio = physical_size / self._target_size
        if encoded_ratio > self._maximum_encoded_ratio:
            self._writer.abort()
            raise CanonicalDeltaError(
                "canonical delta is uneconomic for normal delivery: "
                f"encoded ratio {encoded_ratio:.6f} exceeds "
                f"{self._maximum_encoded_ratio:.6f}"
            )
        target_snapshot = self._writer.finalize(
            expected_format_digest=root.format_digest,
            expected_target_digest=root.target_digest,
        )
        return CanonicalPublication(
            root,
            root_bytes,
            crc32c_hex(root_bytes),
            target_snapshot,
            True,
        )

    def abort(self) -> None:
        if not self._finished:
            self._finished = True
            self._writer.abort()


def _decode_reference(value: object, index: int) -> CanonicalBucketReference:
    context = f"root bucket {index}"
    item = _require_object(value, context)
    _require_keys(
        item,
        {"checksum", "decoded_size", "location", "ordinal", "size", "tensor_names"},
        context,
    )
    ordinal = _require_int(item["ordinal"], f"{context}.ordinal")
    if ordinal != index:
        raise CanonicalDeltaError("root bucket ordinals must be contiguous and ordered")
    checksum = _require_string(item["checksum"], f"{context}.checksum")
    if not _CRC32C.fullmatch(checksum):
        raise CanonicalDeltaError(f"{context}.checksum is not bare CRC32C")
    tensor_values = item["tensor_names"]
    if not isinstance(tensor_values, list) or not tensor_values:
        raise CanonicalDeltaError(f"{context}.tensor_names must be a non-empty array")
    tensor_names = tuple(
        canonical_tensor_name(_require_string(name, f"{context}.tensor_names"))
        for name in tensor_values
    )
    if len(tensor_names) != len(set(tensor_names)):
        raise CanonicalDeltaError(f"{context}.tensor_names are not unique")
    return CanonicalBucketReference(
        ordinal=ordinal,
        checksum=checksum,
        size=_require_int(item["size"], f"{context}.size", minimum=1),
        decoded_size=_require_int(item["decoded_size"], f"{context}.decoded_size"),
        tensor_names=tensor_names,
        location=_decode_location(item["location"], f"{context}.location"),
    )


def _decode_coverage(value: object, index: int) -> CanonicalTensorCoverage:
    context = f"root tensor {index}"
    item = _require_object(value, context)
    _require_keys(
        item,
        {"byte_size", "change_state", "dtype", "name", "shape", "target_digest"},
        context,
        optional={"bucket_ordinal"},
    )
    name = canonical_tensor_name(_require_string(item["name"], f"{context}.name"))
    dtype = _require_string(item["dtype"], f"{context}.dtype")
    shape_value = item["shape"]
    if not isinstance(shape_value, list):
        raise CanonicalDeltaError(f"{context}.shape must be an array")
    shape = _shape(shape_value)
    byte_size = _require_int(item["byte_size"], f"{context}.byte_size")
    if byte_size != _expected_bytes(dtype, shape):
        raise CanonicalDeltaError(f"{context}.byte_size does not match dtype and shape")
    change_state = _require_string(item["change_state"], f"{context}.change_state")
    if change_state not in {"CLEAN", "DIRTY"}:
        raise CanonicalDeltaError(f"{context}.change_state is invalid")
    bucket_ordinal = item.get("bucket_ordinal")
    if change_state == "DIRTY":
        bucket_ordinal = _require_int(bucket_ordinal, f"{context}.bucket_ordinal")
    elif bucket_ordinal is not None:
        raise CanonicalDeltaError("clean root coverage cannot carry a bucket ordinal")
    return CanonicalTensorCoverage(
        name=name,
        dtype=dtype,
        shape=shape,
        byte_size=byte_size,
        target_digest=_require_digest(
            item["target_digest"], f"{context}.target_digest"
        ),
        change_state=change_state,
        bucket_ordinal=bucket_ordinal,
    )


def decode_root_index(
    data: bytes,
    expected_checksum: str,
    *,
    maximum_bytes: int = _DEFAULT_MAXIMUM_ROOT_BYTES,
) -> CanonicalRootIndex:
    """Verify physical root bytes before parsing any untrusted metadata."""
    if len(data) > maximum_bytes:
        raise CanonicalDeltaError("root index exceeds maximum root index size")
    if not _CRC32C.fullmatch(expected_checksum):
        raise CanonicalDeltaError("expected root checksum is not bare CRC32C")
    if crc32c_hex(data) != expected_checksum:
        raise CanonicalDeltaError("root checksum mismatch")
    document = _load_json(data, "root index JSON")
    root = _require_object(document, "root index")
    _require_keys(
        root,
        {
            "base_digest",
            "base_version",
            "buckets",
            "compression_algorithm",
            "delta_method",
            "format_digest",
            "model_id",
            "schema",
            "target_digest",
            "target_version",
            "tensors",
        },
        "root index",
    )
    if root["schema"] != _ROOT_SCHEMA:
        raise CanonicalDeltaError(
            f"unsupported canonical root schema {root['schema']!r}"
        )
    bucket_values = root["buckets"]
    tensor_values = root["tensors"]
    if not isinstance(bucket_values, list) or not isinstance(tensor_values, list):
        raise CanonicalDeltaError("root buckets and tensors must be arrays")
    if not tensor_values:
        raise CanonicalDeltaError("root tensor coverage cannot be empty")
    if len(bucket_values) > _MAXIMUM_RECORDS or len(tensor_values) > _MAXIMUM_RECORDS:
        raise CanonicalDeltaError("root index has too many records")
    buckets = tuple(
        _decode_reference(value, index) for index, value in enumerate(bucket_values)
    )
    tensors = tuple(
        _decode_coverage(value, index) for index, value in enumerate(tensor_values)
    )
    names = [item.name for item in tensors]
    if len(names) != len(set(names)):
        raise CanonicalDeltaError("root tensor coverage contains duplicate names")
    dirty_by_bucket: dict[int, list[str]] = {item.ordinal: [] for item in buckets}
    dirty_ordinals: list[int] = []
    for tensor in tensors:
        if tensor.change_state == "DIRTY":
            if tensor.bucket_ordinal not in dirty_by_bucket:
                raise CanonicalDeltaError(
                    "dirty tensor references a missing root bucket"
                )
            dirty_by_bucket[tensor.bucket_ordinal].append(tensor.name)
            dirty_ordinals.append(tensor.bucket_ordinal)
    if dirty_ordinals != sorted(dirty_ordinals):
        raise CanonicalDeltaError("root bucket coverage order is not monotonic")
    for bucket in buckets:
        if tuple(dirty_by_bucket[bucket.ordinal]) != bucket.tensor_names:
            raise CanonicalDeltaError("root bucket tensor coverage is inconsistent")
    return CanonicalRootIndex(
        model_id=_require_string(root["model_id"], "root model_id"),
        base_version=_require_string(root["base_version"], "root base_version"),
        target_version=_require_string(root["target_version"], "root target_version"),
        delta_method=_require_string(root["delta_method"], "root delta_method"),
        compression_algorithm=_require_string(
            root["compression_algorithm"], "root compression_algorithm"
        ),
        format_digest=_require_digest(root["format_digest"], "root format_digest"),
        base_digest=_require_digest(root["base_digest"], "root base_digest"),
        target_digest=_require_digest(root["target_digest"], "root target_digest"),
        buckets=buckets,
        tensors=tensors,
    )


def _decode_bucket_bytes(
    data: bytes,
    reference: CanonicalBucketReference,
    root: CanonicalRootIndex,
    base_store: FilesystemCanonicalBaseStore,
    base: CanonicalSnapshot,
    maximum_bucket_bytes: int,
) -> dict[str, bytes]:
    if len(data) != reference.size:
        raise CanonicalDeltaError(
            f"bucket {reference.ordinal} size does not match root index"
        )
    if crc32c_hex(data) != reference.checksum:
        raise CanonicalDeltaError(
            f"bucket checksum mismatch for ordinal {reference.ordinal}"
        )
    if len(data) < len(_BUCKET_MAGIC) + 4 or not data.startswith(_BUCKET_MAGIC):
        raise CanonicalDeltaError("canonical bucket magic is invalid")
    header_size = struct.unpack(
        ">I", data[len(_BUCKET_MAGIC) : len(_BUCKET_MAGIC) + 4]
    )[0]
    if header_size > _MAXIMUM_HEADER_BYTES:
        raise CanonicalDeltaError("canonical bucket header exceeds maximum size")
    header_start = len(_BUCKET_MAGIC) + 4
    header_end = header_start + header_size
    if header_end > len(data):
        raise CanonicalDeltaError("canonical bucket header is truncated")
    document = _load_json(data[header_start:header_end], "canonical bucket header")
    header = _require_object(document, "bucket header")
    _require_keys(
        header,
        {
            "base_digest",
            "base_version",
            "compression_algorithm",
            "decoded_size",
            "delta_method",
            "entries",
            "format_digest",
            "model_id",
            "ordinal",
            "schema",
            "target_version",
        },
        "bucket header",
    )
    expected_identity = {
        "base_digest": root.base_digest,
        "base_version": root.base_version,
        "compression_algorithm": root.compression_algorithm,
        "delta_method": root.delta_method,
        "format_digest": root.format_digest,
        "model_id": root.model_id,
        "ordinal": reference.ordinal,
        "schema": _BUCKET_SCHEMA,
        "target_version": root.target_version,
    }
    for field, expected in expected_identity.items():
        if header[field] != expected:
            raise CanonicalDeltaError(
                f"canonical bucket {field} does not match root index"
            )
    decoded_size = _require_int(header["decoded_size"], "bucket decoded_size")
    if decoded_size != reference.decoded_size:
        raise CanonicalDeltaError(
            "canonical bucket decoded size does not match root index"
        )
    if decoded_size > maximum_bucket_bytes:
        raise CanonicalDeltaError("canonical bucket decoded size exceeds maximum")
    entry_values = header["entries"]
    if not isinstance(entry_values, list) or not entry_values:
        raise CanonicalDeltaError("canonical bucket entries must be a non-empty array")
    if len(entry_values) > _MAXIMUM_RECORDS:
        raise CanonicalDeltaError("canonical bucket has too many entries")
    entries: list[tuple[str, str, tuple[int, ...], int, int, str]] = []
    next_offset = 0
    for index, value in enumerate(entry_values):
        context = f"bucket entry {index}"
        entry = _require_object(value, context)
        _require_keys(
            entry,
            {"byte_size", "dtype", "name", "offset", "shape", "target_digest"},
            context,
        )
        name = canonical_tensor_name(_require_string(entry["name"], f"{context}.name"))
        dtype = _require_string(entry["dtype"], f"{context}.dtype")
        shape_value = entry["shape"]
        if not isinstance(shape_value, list):
            raise CanonicalDeltaError(f"{context}.shape must be an array")
        shape = _shape(shape_value)
        size = _require_int(entry["byte_size"], f"{context}.byte_size")
        if size != _expected_bytes(dtype, shape):
            raise CanonicalDeltaError(
                f"{context}.byte_size does not match dtype and shape"
            )
        offset = _require_int(entry["offset"], f"{context}.offset")
        if offset != next_offset:
            raise CanonicalDeltaError(
                "canonical bucket entry offsets must be contiguous"
            )
        next_offset += size
        entries.append(
            (
                name,
                dtype,
                shape,
                offset,
                size,
                _require_digest(entry["target_digest"], f"{context}.target_digest"),
            )
        )
    if next_offset != decoded_size:
        raise CanonicalDeltaError(
            "canonical bucket entries do not cover decoded payload"
        )
    if tuple(entry[0] for entry in entries) != reference.tensor_names:
        raise CanonicalDeltaError(
            "canonical bucket entries do not match root tensor names"
        )
    try:
        decoded = decompress_payload(
            root.compression_algorithm,
            data[header_end:],
            expected_size=decoded_size,
        )
    except CodecError as exc:
        raise CanonicalDeltaError(str(exc)) from exc
    base_by_name = {item.name: item for item in base.tensors}
    targets: dict[str, bytes] = {}
    for name, dtype, shape, offset, size, target_digest in entries:
        base_metadata = base_by_name.get(name)
        if (
            base_metadata is None
            or base_metadata.dtype != dtype
            or base_metadata.shape != shape
            or base_metadata.byte_size != size
        ):
            raise CanonicalDeltaError(
                f"canonical bucket tensor {name!r} does not match base format"
            )
        base_bytes = base_store.read_tensor_bytes(base, name)
        try:
            target = decode_delta(
                root.delta_method,
                base_bytes,
                decoded[offset : offset + size],
            )
        except CodecError as exc:
            raise CanonicalDeltaError(str(exc)) from exc
        if _sha256(target) != target_digest:
            raise CanonicalDeltaError(
                f"canonical bucket tensor {name!r} target digest mismatch"
            )
        targets[name] = target
    return targets


def reconstruct_canonical_delta(
    *,
    root_bytes: bytes,
    expected_root_checksum: str,
    base_store: FilesystemCanonicalBaseStore,
    base: CanonicalSnapshot,
    target_store: FilesystemCanonicalBaseStore,
    fetch_bucket: Callable[[CanonicalBucketReference], bytes],
    maximum_bucket_bytes: int = _DEFAULT_MAXIMUM_BUCKET_BYTES,
) -> CanonicalSnapshot:
    """Verify root, exact base, every bucket, and the complete target digest."""
    root = decode_root_index(root_bytes, expected_root_checksum)
    if (
        not isinstance(maximum_bucket_bytes, int)
        or isinstance(maximum_bucket_bytes, bool)
        or maximum_bucket_bytes <= 0
    ):
        raise ValueError("maximum_bucket_bytes must be positive")
    base = base_store.attest_snapshot(base)
    if root.base_version != base.version:
        raise CanonicalDeltaError("root base version does not match exact base")
    if root.format_digest != base.format_digest:
        raise CanonicalDeltaError("root base format digest does not match exact base")
    if root.base_digest != base.target_digest:
        raise CanonicalDeltaError("root base digest does not match exact base")
    base_by_name = {item.name: item for item in base.tensors}
    if tuple(base_by_name) != tuple(item.name for item in root.tensors):
        raise CanonicalDeltaError(
            "root does not provide complete exact-base tensor coverage"
        )
    references = {item.ordinal: item for item in root.buckets}
    fetched: set[int] = set()
    current_ordinal: int | None = None
    current_targets: dict[str, bytes] = {}
    writer = target_store.begin_snapshot(
        root.target_version,
        format_identity=base.format_identity,
    )
    try:
        for coverage in root.tensors:
            base_metadata = base_by_name[coverage.name]
            if (
                coverage.dtype != base_metadata.dtype
                or coverage.shape != base_metadata.shape
                or coverage.byte_size != base_metadata.byte_size
            ):
                raise CanonicalDeltaError(
                    f"root tensor {coverage.name!r} does not match exact-base format"
                )
            if coverage.change_state == "CLEAN":
                if coverage.target_digest != base_metadata.content_digest:
                    raise CanonicalDeltaError(
                        f"clean root tensor {coverage.name!r} target digest differs from base"
                    )
                target = base_store.read_tensor_bytes(base, coverage.name)
            else:
                ordinal = coverage.bucket_ordinal
                if ordinal != current_ordinal:
                    reference = references[ordinal]
                    if current_targets:
                        raise CanonicalDeltaError(
                            "canonical root contains interleaved bucket coverage"
                        )
                    if ordinal in fetched:
                        raise CanonicalDeltaError(
                            "canonical root reuses a completed bucket"
                        )
                    if (
                        reference.size > maximum_bucket_bytes
                        or reference.decoded_size > maximum_bucket_bytes
                    ):
                        raise CanonicalDeltaError(
                            "canonical bucket exceeds maximum encoded or decoded size"
                        )
                    current_targets = _decode_bucket_bytes(
                        fetch_bucket(reference),
                        reference,
                        root,
                        base_store,
                        base,
                        maximum_bucket_bytes,
                    )
                    current_ordinal = ordinal
                    fetched.add(ordinal)
                try:
                    target = current_targets.pop(coverage.name)
                except KeyError as exc:
                    raise CanonicalDeltaError(
                        f"dirty root tensor {coverage.name!r} is missing from its bucket"
                    ) from exc
                if _sha256(target) != coverage.target_digest:
                    raise CanonicalDeltaError(
                        f"dirty root tensor {coverage.name!r} coverage digest mismatch"
                    )
            writer.add_tensor_bytes(
                coverage.name,
                coverage.dtype,
                coverage.shape,
                target,
            )
        if current_targets or fetched != set(references):
            raise CanonicalDeltaError("canonical root contains unused bucket data")
        target_snapshot = writer.finalize(
            expected_format_digest=root.format_digest,
            expected_target_digest=root.target_digest,
        )
    except Exception:
        writer.abort()
        raise
    return target_snapshot
