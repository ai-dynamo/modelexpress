# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical HF identities, one retained exact base, and the V0 S3 payload."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import torch
import zstandard

from ..s3 import S3Uploader, UploadedS3Object

CanonicalBucket = Sequence[tuple[str, torch.Tensor]]

_BUCKET_MAGIC = b"MXCDV0\0"
_SCHEMA = "mx.canonical.delta.v0"
_DELTA = "xor"
_COMPRESSION = "zstd"


class CanonicalError(ValueError):
    """Canonical source or exact-base invariants were violated."""


def canonical_tensor_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise CanonicalError("canonical tensor name must be non-empty")
    while name.startswith("module."):
        name = name.removeprefix("module.")
    if not name or "\x00" in name:
        raise CanonicalError("canonical tensor name is invalid")
    return name


def dtype_name(dtype: torch.dtype) -> str:
    if not isinstance(dtype, torch.dtype):
        raise CanonicalError("canonical tensor dtype must be torch.dtype")
    return str(dtype).removeprefix("torch.")


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    if not isinstance(tensor, torch.Tensor):
        raise CanonicalError("canonical value is not a tensor")
    cpu = tensor.detach().to(device="cpu").contiguous()
    if cpu.numel() == 0:
        return b""
    return cpu.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_from_bytes(
    data: bytes, dtype: torch.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    expected = _tensor_size(dtype, shape)
    if len(data) != expected:
        raise CanonicalError(
            f"tensor bytes have size {len(data)} but expected {expected}"
        )
    if not data:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)


def _tensor_size(dtype: torch.dtype, shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 0
        ):
            raise CanonicalError("canonical tensor shape is invalid")
        count *= dimension
    return count * torch.empty((), dtype=dtype).element_size()


@dataclass(frozen=True)
class CanonicalTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        canonical = canonical_tensor_name(self.name)
        if canonical != self.name:
            raise CanonicalError(f"tensor name {self.name!r} is not canonical")
        normalized_shape = tuple(self.shape)
        _tensor_size(self.dtype, normalized_shape)
        object.__setattr__(self, "shape", normalized_shape)

    @property
    def nbytes(self) -> int:
        return _tensor_size(self.dtype, self.shape)


@dataclass(frozen=True)
class CanonicalTensorMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    byte_size: int
    content_digest: str

    @property
    def spec(self) -> CanonicalTensorSpec:
        return CanonicalTensorSpec(self.name, self.shape, self.dtype)


@dataclass(frozen=True)
class CanonicalSnapshot:
    version: str
    format_digest: str
    target_digest: str
    tensors: tuple[CanonicalTensorMetadata, ...]

    @property
    def schema(self) -> tuple[CanonicalTensorSpec, ...]:
        return tuple(tensor.spec for tensor in self.tensors)


@dataclass(frozen=True)
class _TensorSource:
    path: Path
    source_name: str | None = None


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _feed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def format_digest(schema: Sequence[CanonicalTensorSpec]) -> str:
    if not schema:
        raise CanonicalError("canonical HF schema cannot be empty")
    digest = hashlib.sha256()
    _feed(digest, b"mx.canonical.format.v0")
    seen: set[str] = set()
    for spec in schema:
        if not isinstance(spec, CanonicalTensorSpec):
            raise CanonicalError("canonical HF schema entry is invalid")
        if spec.name in seen:
            raise CanonicalError(f"canonical HF schema duplicates {spec.name!r}")
        seen.add(spec.name)
        _feed(digest, spec.name.encode())
        _feed(digest, dtype_name(spec.dtype).encode())
        _feed(digest, canonical_json(spec.shape))
        _feed(digest, str(spec.nbytes).encode())
    return f"sha256:{digest.hexdigest()}"


def target_digest(tensors: Sequence[CanonicalTensorMetadata]) -> str:
    digest = hashlib.sha256()
    _feed(digest, b"mx.canonical.content.v0")
    for tensor in tensors:
        _feed(digest, tensor.name.encode())
        _feed(digest, dtype_name(tensor.dtype).encode())
        _feed(digest, canonical_json(tensor.shape))
        _feed(digest, tensor.content_digest.encode())
    return f"sha256:{digest.hexdigest()}"


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _checkpoint_owners(root: Path) -> dict[str, Path]:
    from safetensors import safe_open

    if root.is_file():
        files = [root]
    else:
        index = root / "model.safetensors.index.json"
        if index.is_file():
            document = json.loads(index.read_text(encoding="utf-8"))
            weight_map = document.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise CanonicalError("HF safetensors index has no weight_map")
            owners = {}
            for name, relative in weight_map.items():
                if not isinstance(name, str) or not isinstance(relative, str):
                    raise CanonicalError("HF safetensors weight_map is invalid")
                path = (root / relative).resolve()
                try:
                    path.relative_to(root.resolve())
                except ValueError as exc:
                    raise CanonicalError(
                        "HF safetensors index escapes its checkpoint root"
                    ) from exc
                if name in owners:
                    raise CanonicalError(f"duplicate HF checkpoint key {name!r}")
                owners[name] = path
            return owners
        files = sorted(root.glob("*.safetensors"))
    if not files:
        raise CanonicalError("launch checkpoint has no safetensors files")
    owners: dict[str, Path] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in owners:
                    raise CanonicalError(f"duplicate HF checkpoint key {name!r}")
                owners[name] = path.resolve()
    return owners


def attest_hf_checkpoint(
    checkpoint: str | os.PathLike[str],
    version: str,
    *,
    maximum_tensor_bytes: int,
) -> tuple[CanonicalSnapshot, dict[str, _TensorSource]]:
    """Attest an existing HF checkpoint tensor-by-tensor without copying it."""
    from safetensors import safe_open

    root = Path(checkpoint).resolve()
    owners = _checkpoint_owners(root)
    canonical_owners = {}
    for source_name, path in owners.items():
        name = canonical_tensor_name(source_name)
        if name in canonical_owners:
            raise CanonicalError(f"canonical HF checkpoint duplicates {name!r}")
        canonical_owners[name] = (source_name, path)
    metadata = []
    sources = {}
    for name in sorted(canonical_owners):
        source_name, path = canonical_owners[name]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(source_name)
        data = tensor_bytes(tensor)
        if len(data) > maximum_tensor_bytes:
            raise CanonicalError(
                f"launch tensor {name!r} exceeds bucket_bytes={maximum_tensor_bytes}"
            )
        metadata.append(
            CanonicalTensorMetadata(
                name,
                tuple(tensor.shape),
                tensor.dtype,
                len(data),
                _sha256(data),
            )
        )
        sources[name] = _TensorSource(path, source_name)
        del tensor, data
    tensors = tuple(metadata)
    snapshot = CanonicalSnapshot(
        version=version,
        format_digest=format_digest(tuple(item.spec for item in tensors)),
        target_digest=target_digest(tensors),
        tensors=tensors,
    )
    return snapshot, sources


class _Candidate:
    def __init__(self, store: RetainedBaseStore, version: str) -> None:
        self._store = store
        self._version = version
        self._directory = store._root / f"candidate-{uuid.uuid4().hex}"
        self._directory.mkdir(parents=True)
        self._metadata = []
        self._sources = {}
        self._next = 0
        self._snapshot: CanonicalSnapshot | None = None
        self._promoted = False

    def add(self, name: str, tensor: torch.Tensor) -> bytes:
        if self._snapshot is not None:
            raise CanonicalError("candidate is already finalized")
        base = self._store.current
        if self._next >= len(base.tensors):
            raise CanonicalError(f"target tensor {name!r} is outside the exact base")
        expected = base.tensors[self._next]
        canonical = canonical_tensor_name(name)
        if (
            canonical != expected.name
            or tuple(tensor.shape) != expected.shape
            or tensor.dtype != expected.dtype
        ):
            raise CanonicalError(
                f"target tensor {canonical!r} does not match exact-base schema entry "
                f"{expected.name!r}"
            )
        data = tensor_bytes(tensor)
        if len(data) != expected.byte_size:
            raise CanonicalError(f"target tensor {canonical!r} changed byte size")
        digest = _sha256(data)
        path = self._directory / f"{self._next:08d}.tensor"
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self._metadata.append(
            CanonicalTensorMetadata(
                canonical,
                tuple(tensor.shape),
                tensor.dtype,
                len(data),
                digest,
            )
        )
        self._sources[canonical] = _TensorSource(path)
        self._next += 1
        return data

    def finalize(self) -> CanonicalSnapshot:
        if self._snapshot is not None:
            return self._snapshot
        base = self._store.current
        if self._next != len(base.tensors):
            raise CanonicalError(
                "target did not provide complete canonical HF coverage"
            )
        tensors = tuple(self._metadata)
        snapshot = CanonicalSnapshot(
            self._version,
            format_digest(tuple(item.spec for item in tensors)),
            target_digest(tensors),
            tensors,
        )
        if snapshot.format_digest != base.format_digest:
            raise CanonicalError("target format differs from the exact retained base")
        self._snapshot = snapshot
        return snapshot

    def abort(self) -> None:
        if not self._promoted:
            shutil.rmtree(self._directory, ignore_errors=True)


class RetainedBaseStore:
    """Keep exactly one verified base; launch bytes remain in the HF checkpoint."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._snapshot: CanonicalSnapshot | None = None
        self._sources: dict[str, _TensorSource] = {}
        self._owned_directory: Path | None = None

    @property
    def current(self) -> CanonicalSnapshot:
        if self._snapshot is None:
            raise CanonicalError("exact retained base is not initialized")
        return self._snapshot

    def seed_launch(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        maximum_tensor_bytes: int,
    ) -> CanonicalSnapshot:
        snapshot, sources = attest_hf_checkpoint(
            checkpoint,
            "0",
            maximum_tensor_bytes=maximum_tensor_bytes,
        )
        self._snapshot = snapshot
        self._sources = sources
        self._owned_directory = None
        return snapshot

    def read(self, name: str) -> bytes:
        from safetensors import safe_open

        snapshot = self.current
        metadata = next((item for item in snapshot.tensors if item.name == name), None)
        source = self._sources.get(name)
        if metadata is None or source is None:
            raise CanonicalError(f"exact retained base has no tensor {name!r}")
        if source.source_name is None:
            with source.path.open("rb") as handle:
                data = handle.read(metadata.byte_size + 1)
        else:
            with safe_open(str(source.path), framework="pt", device="cpu") as handle:
                data = tensor_bytes(handle.get_tensor(source.source_name))
        if len(data) != metadata.byte_size or _sha256(data) != metadata.content_digest:
            raise CanonicalError(
                f"exact retained base tensor {name!r} failed attestation"
            )
        return data

    def begin_candidate(self, version: str) -> _Candidate:
        if (
            not isinstance(version, str)
            or not version
            or version == self.current.version
        ):
            raise CanonicalError("target version must differ from exact retained base")
        return _Candidate(self, version)

    def promote(self, candidate: _Candidate) -> CanonicalSnapshot:
        snapshot = candidate.finalize()
        previous = self._owned_directory
        self._snapshot = snapshot
        self._sources = dict(candidate._sources)
        self._owned_directory = candidate._directory
        candidate._promoted = True
        if previous is not None and previous != self._owned_directory:
            shutil.rmtree(previous, ignore_errors=True)
        return snapshot


@dataclass(frozen=True)
class CanonicalPublication:
    payload: UploadedS3Object
    snapshot: CanonicalSnapshot
    candidate: _Candidate
    root_bytes: bytes


def _xor(base: bytes, target: bytes) -> bytes:
    if len(base) != len(target):
        raise CanonicalError("xor delta inputs differ in length")
    return np.bitwise_xor(
        np.frombuffer(base, dtype=np.uint8),
        np.frombuffer(target, dtype=np.uint8),
    ).tobytes()


def _s3_dict(stored: UploadedS3Object) -> dict[str, object]:
    result: dict[str, object] = {
        "bucket": stored.object.bucket,
        "checksum": stored.object.checksum,
        "key": stored.object.key,
        "size": stored.size,
    }
    if stored.object.object_version is not None:
        result["object_version"] = stored.object.object_version
    return result


def _revision_key(model_id: str, target_version: str, filename: str) -> str:
    return (
        f"models/{quote(model_id, safe='')}/revisions/"
        f"{quote(target_version, safe='')}/canonical/{filename}"
    )


class CanonicalDeltaEncoder:
    """Stream canonical target buckets against the one exact retained base."""

    def __init__(
        self,
        *,
        model_id: str,
        target_version: str,
        base_store: RetainedBaseStore,
        uploader: S3Uploader,
        bucket_bytes: int,
    ) -> None:
        self._model_id = model_id
        self._target_version = target_version
        self._base_store = base_store
        self._base = base_store.current
        self._uploader = uploader
        self._bucket_bytes = bucket_bytes
        self._candidate = base_store.begin_candidate(target_version)
        self._coverage = []
        self._buckets = []
        self._next = 0
        self._finished = False

    def consume_bucket(self, bucket: CanonicalBucket) -> None:
        if self._finished:
            raise CanonicalError("canonical encoder is already finished")
        if not bucket:
            raise CanonicalError("canonical source emitted an empty bucket")
        total = sum(tensor.nbytes for _name, tensor in bucket)
        if total > self._bucket_bytes:
            raise CanonicalError("canonical source exceeded its bucket memory bound")
        entries = []
        decoded = bytearray()
        bucket_ordinal = len(self._buckets)
        for name, tensor in bucket:
            if self._next >= len(self._base.tensors):
                raise CanonicalError("target exceeds exact-base coverage")
            base_metadata = self._base.tensors[self._next]
            target = self._candidate.add(name, tensor)
            base = self._base_store.read(base_metadata.name)
            digest = _sha256(target)
            coverage: dict[str, object] = {
                "byte_size": len(target),
                "dtype": dtype_name(tensor.dtype),
                "name": base_metadata.name,
                "shape": list(tensor.shape),
                "target_digest": digest,
            }
            if digest == base_metadata.content_digest:
                coverage["state"] = "clean"
            else:
                delta = _xor(base, target)
                offset = len(decoded)
                decoded.extend(delta)
                entries.append(
                    {
                        "byte_size": len(delta),
                        "dtype": dtype_name(tensor.dtype),
                        "name": base_metadata.name,
                        "offset": offset,
                        "shape": list(tensor.shape),
                        "target_digest": digest,
                    }
                )
                coverage["state"] = "dirty"
                coverage["bucket_ordinal"] = bucket_ordinal
            self._coverage.append(coverage)
            self._next += 1
        if not entries:
            return
        compressed = zstandard.ZstdCompressor(level=3).compress(bytes(decoded))
        header = canonical_json(
            {
                "base_digest": self._base.target_digest,
                "base_version": self._base.version,
                "compression": _COMPRESSION,
                "decoded_size": len(decoded),
                "delta": _DELTA,
                "entries": entries,
                "format_digest": self._base.format_digest,
                "model_id": self._model_id,
                "ordinal": bucket_ordinal,
                "schema": f"{_SCHEMA}.bucket",
                "target_version": self._target_version,
            }
        )
        encoded = _BUCKET_MAGIC + struct.pack(">I", len(header)) + header + compressed
        stored = self._uploader.put(
            _revision_key(
                self._model_id,
                self._target_version,
                f"bucket-{bucket_ordinal:08d}.mxcd",
            ),
            encoded,
        )
        self._buckets.append(
            {
                "decoded_size": len(decoded),
                "object": _s3_dict(stored),
                "ordinal": bucket_ordinal,
                "tensors": [entry["name"] for entry in entries],
            }
        )

    def finish(self) -> CanonicalPublication:
        if self._finished:
            raise CanonicalError("canonical encoder is already finished")
        self._finished = True
        if self._next != len(self._base.tensors):
            self._candidate.abort()
            raise CanonicalError(
                "target did not provide complete canonical HF coverage"
            )
        try:
            snapshot = self._candidate.finalize()
            root = {
                "base_digest": self._base.target_digest,
                "base_version": self._base.version,
                "buckets": self._buckets,
                "encoding": {"compression": _COMPRESSION, "delta": _DELTA},
                "format_digest": snapshot.format_digest,
                "model_id": self._model_id,
                "schema": _SCHEMA,
                "target_digest": snapshot.target_digest,
                "target_version": self._target_version,
                "tensors": self._coverage,
            }
            root_bytes = canonical_json(root)
            payload = self._uploader.put(
                _revision_key(self._model_id, self._target_version, "root.json"),
                root_bytes,
            )
            return CanonicalPublication(
                payload=payload,
                snapshot=snapshot,
                candidate=self._candidate,
                root_bytes=root_bytes,
            )
        except Exception:
            self._candidate.abort()
            raise

    def abort(self) -> None:
        if not self._finished:
            self._finished = True
        self._candidate.abort()
