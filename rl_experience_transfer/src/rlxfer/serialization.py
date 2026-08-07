# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safe JSON metadata and separately transferable tensor buffers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from enum import Enum
from types import ModuleType
from typing import Any, Protocol, cast
from uuid import UUID

import numpy as np

from . import model
from .errors import IntegrityError, SerializationError
from .model import ExperienceBatch

_FORMAT = "rlxfer-json"
_SERIALIZER_VERSION = 1
_PathPart = str | int
_CATALOG_FIELDS = (
    "name",
    "path",
    "kind",
    "dtype",
    "shape",
    "stride",
    "layout",
    "original_device",
    "wire_device",
    "nbytes",
    "sha256",
)


@dataclass(frozen=True, slots=True)
class SerializationLimits:
    """Resource limits applied to metadata and logical tensor payloads.

    Limits are checked while serializing and before deserialization restores any
    tensor buffer. ``max_depth`` and ``max_items`` apply to the encoded JSON tree.
    Tensor byte limits count logical contiguous values, not backing-storage spans.
    """

    max_metadata_bytes: int = 16 * 1024 * 1024
    max_depth: int = 64
    max_items: int = 100_000
    max_tensor_count: int = 4_096
    max_tensor_bytes: int = 2 * 1024 * 1024 * 1024
    max_total_tensor_bytes: int = 8 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        for setting in fields(self):
            value = getattr(self, setting.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{setting.name} must be a positive integer")


_DEFAULT_LIMITS = SerializationLimits()


def _torch() -> ModuleType | None:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        return None


def _is_torch_tensor(value: object) -> bool:
    torch = _torch()
    return torch is not None and isinstance(value, torch.Tensor)


def _synchronize(torch: Any, tensor: Any) -> None:
    backend = getattr(torch, tensor.device.type, None)
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize):
        synchronize(tensor.device)


def _tensor_bytes(value: object) -> bytes:
    """Return logical tensor values in contiguous row-major order."""

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise SerializationError("NumPy object arrays are not safe to serialize")
        return value.tobytes(order="C")
    if _is_torch_tensor(value):
        torch = cast(Any, _torch())
        tensor = cast(Any, value).detach()
        if tensor.layout != torch.strided:
            raise SerializationError(f"unsupported PyTorch layout: {tensor.layout}")
        if tensor.device.type != "cpu":
            _synchronize(torch, tensor)
            tensor = tensor.to("cpu")
        tensor = tensor.contiguous()
        return cast(bytes, tensor.view(torch.uint8).numpy().tobytes())
    raise SerializationError(f"unsupported tensor value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class BufferSegment:
    """One tensor buffer plus the complete transport-neutral tensor catalog entry.

    ``owner`` keeps a staged or direct tensor alive until transfer completion. A
    transport that needs bytes (notably the filesystem transport) calls
    :meth:`materialize`. ``stride`` is a normalized C-contiguous stride measured
    in elements for both NumPy and PyTorch tensors.
    """

    name: str
    path: tuple[_PathPart, ...]
    kind: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    layout: str
    original_device: str
    wire_device: str
    nbytes: int
    sha256: str | None = None
    data: bytes | memoryview | None = field(default=None, repr=False, compare=False)
    owner: object | None = field(default=None, repr=False, compare=False)

    def materialize(self) -> bytes:
        """Return transfer bytes, staging a retained tensor owner when necessary."""

        if self.data is not None:
            return bytes(self.data)
        if self.owner is not None:
            return _tensor_bytes(self.owner)
        raise IntegrityError(f"buffer {self.name!r} has neither data nor an owner")

    def catalog_entry(self) -> dict[str, object]:
        """Return the JSON-safe transport metadata for this buffer."""

        result = {name: getattr(self, name) for name in _CATALOG_FIELDS}
        result.update({name: list(getattr(self, name)) for name in ("path", "shape", "stride")})
        return result

    @classmethod
    def from_catalog_entry(
        cls,
        value: Mapping[str, Any],
        *,
        data: bytes | memoryview | None = None,
        owner: object | None = None,
    ) -> BufferSegment:
        """Rebuild a segment from validated JSON-safe transport metadata."""

        return cls(
            name=str(value["name"]),
            path=tuple(
                item
                for item in value["path"]
                if isinstance(item, (str, int)) and not isinstance(item, bool)
            ),
            kind=str(value["kind"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),
            stride=tuple(int(item) for item in value["stride"]),
            layout=str(value["layout"]),
            original_device=str(value["original_device"]),
            wire_device=str(value["wire_device"]),
            nbytes=int(value["nbytes"]),
            sha256=str(value["sha256"]) if value["sha256"] is not None else None,
            data=data,
            owner=owner,
        )


@dataclass(frozen=True, slots=True)
class SerializedExperience:
    """Deterministic UTF-8 JSON metadata and its external tensor buffers."""

    metadata: bytes = field(repr=False)
    buffers: tuple[BufferSegment, ...] = ()

    @property
    def nbytes(self) -> int:
        """Return the total encoded size without materializing direct buffers."""

        return len(self.metadata) + sum(buffer.nbytes for buffer in self.buffers)


def validate_transfer_limits(
    *,
    metadata_bytes: int,
    tensor_sizes: Iterable[int],
    limits: SerializationLimits | None = None,
) -> None:
    """Reject a transfer catalog that could exceed configured resource bounds."""

    active = limits or _DEFAULT_LIMITS
    if (
        isinstance(metadata_bytes, bool)
        or not isinstance(metadata_bytes, int)
        or metadata_bytes < 0
    ):
        raise SerializationError("metadata size must be a non-negative integer")
    if metadata_bytes > active.max_metadata_bytes:
        raise SerializationError(
            f"metadata size {metadata_bytes} exceeds byte limit {active.max_metadata_bytes}"
        )
    total = 0
    for count, size in enumerate(tensor_sizes, start=1):
        if count > active.max_tensor_count:
            raise SerializationError(f"tensor count exceeds limit {active.max_tensor_count}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SerializationError("tensor size must be a non-negative integer")
        if size > active.max_tensor_bytes:
            raise SerializationError(
                f"tensor size {size} exceeds per-tensor byte limit {active.max_tensor_bytes}"
            )
        total += size
        if total > active.max_total_tensor_bytes:
            raise SerializationError(
                f"total tensor size {total} exceeds byte limit {active.max_total_tensor_bytes}"
            )


class BufferManager(Protocol):
    """Stages tensor-like values and reconstructs their logical values."""

    def stage(
        self,
        value: object,
        *,
        name: str,
        path: tuple[_PathPart, ...],
        checksum: bool,
    ) -> BufferSegment: ...

    def restore(self, segment: BufferSegment, *, preserve_device: bool) -> object: ...


class DefaultBufferManager:
    """Default NumPy/PyTorch buffer manager with optional CPU or pinned staging."""

    def __init__(self, *, cpu_staging: bool = True, pinned_memory: bool = False) -> None:
        self.cpu_staging = cpu_staging
        self.pinned_memory = pinned_memory

    def stage(
        self,
        value: object,
        *,
        name: str,
        path: tuple[_PathPart, ...],
        checksum: bool,
    ) -> BufferSegment:
        if isinstance(value, np.ndarray):
            return self._stage_numpy(value, name=name, path=path, checksum=checksum)
        if _is_torch_tensor(value):
            return self._stage_torch(value, name=name, path=path, checksum=checksum)
        raise SerializationError(f"unsupported tensor value: {type(value).__name__}")

    def restore(self, segment: BufferSegment, *, preserve_device: bool) -> object:
        raw = segment.materialize()
        if segment.kind == "numpy":
            dtype = _numpy_dtype(segment.dtype)
            return np.frombuffer(raw, dtype=dtype).copy().reshape(segment.shape)
        if segment.kind == "torch":
            torch = _torch()
            if torch is None:
                raise SerializationError("PyTorch is required to deserialize a PyTorch tensor")
            dtype = _torch_dtype(torch, segment.dtype)
            byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            tensor = byte_tensor.view(dtype).reshape(segment.shape).clone()
            if preserve_device and segment.original_device != "cpu":
                try:
                    tensor = tensor.to(segment.original_device)
                except (RuntimeError, ValueError) as error:
                    raise SerializationError(
                        f"cannot restore tensor {segment.name!r} on {segment.original_device!r}"
                    ) from error
            return tensor
        raise SerializationError(f"unsupported tensor kind: {segment.kind!r}")

    def _stage_numpy(
        self,
        value: np.ndarray[Any, Any],
        *,
        name: str,
        path: tuple[_PathPart, ...],
        checksum: bool,
    ) -> BufferSegment:
        if value.dtype.hasobject:
            raise SerializationError("NumPy object arrays are not safe to serialize")
        owner = np.ascontiguousarray(value)
        raw = memoryview(cast(Any, owner)).cast("B")
        return BufferSegment(
            name=name,
            path=path,
            kind="numpy",
            dtype=value.dtype.str,
            shape=tuple(value.shape),
            stride=_contiguous_stride(tuple(value.shape)),
            layout="strided",
            original_device="cpu",
            wire_device="cpu",
            nbytes=len(raw),
            sha256=_digest(raw) if checksum else None,
            data=raw,
            owner=owner,
        )

    def _stage_torch(
        self,
        value: object,
        *,
        name: str,
        path: tuple[_PathPart, ...],
        checksum: bool,
    ) -> BufferSegment:
        torch = cast(Any, _torch())
        tensor = cast(Any, value).detach()
        if tensor.layout != torch.strided:
            raise SerializationError(f"unsupported PyTorch layout: {tensor.layout}")
        original_device = str(tensor.device)
        staged = tensor.contiguous().view(-1).view(tensor.shape)
        should_stage = self.cpu_staging or tensor.device.type == "cpu"
        if should_stage and tensor.device.type != "cpu":
            _synchronize(torch, tensor)
            staged = staged.to("cpu")
        if should_stage and self.pinned_memory and not staged.is_pinned():
            staged = staged.pin_memory()
        data: bytes | memoryview | None = None
        if should_stage:
            data = memoryview(staged.view(torch.uint8).numpy()).cast("B")
        raw = bytes(data) if data is not None else (_tensor_bytes(staged) if checksum else None)
        return BufferSegment(
            name=name,
            path=path,
            kind="torch",
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=tuple(tensor.shape),
            stride=tuple(staged.stride()),
            layout=str(tensor.layout).removeprefix("torch."),
            original_device=original_device,
            wire_device="cpu" if should_stage else original_device,
            nbytes=tensor.numel() * tensor.element_size(),
            sha256=_digest(raw) if raw is not None and checksum else None,
            data=data,
            owner=staged,
        )


class ExperienceSerializer(Protocol):
    """Transport-independent experience serializer contract."""

    def serialize(self, experience: ExperienceBatch) -> SerializedExperience: ...

    def deserialize(self, serialized: SerializedExperience) -> ExperienceBatch: ...


@dataclass(slots=True)
class _EncodeState:
    buffers: list[BufferSegment] = field(default_factory=list)
    catalogs: list[dict[str, object]] = field(default_factory=list)
    tensor_bytes: int = 0


class JsonExperienceSerializer:
    """Versioned safe serializer using JSON metadata and raw tensor buffers."""

    def __init__(
        self,
        *,
        inline_threshold: int = 0,
        checksum: bool = True,
        cpu_staging: bool = True,
        pinned_memory: bool = False,
        preserve_device: bool = False,
        buffer_manager: BufferManager | None = None,
        allowed_types: Iterable[type[object]] = (),
        limits: SerializationLimits | None = None,
    ) -> None:
        if inline_threshold < 0:
            raise ValueError("inline_threshold must be non-negative")
        self.inline_threshold = inline_threshold
        self.checksum = checksum
        self.preserve_device = preserve_device
        self.buffer_manager = buffer_manager or DefaultBufferManager(
            cpu_staging=cpu_staging,
            pinned_memory=pinned_memory,
        )
        self._types = _safe_types(allowed_types)
        self.limits = limits or _DEFAULT_LIMITS

    def serialize(self, experience: ExperienceBatch) -> SerializedExperience:
        """Validate and encode an experience without pickle or executable metadata."""

        if not isinstance(experience, ExperienceBatch):
            raise SerializationError("serialize expects an ExperienceBatch")
        _check_source_limits(experience, self.limits)
        state = _EncodeState()
        try:
            experience.validate()
            root = self._encode(experience, (), state)
        except RecursionError as error:
            raise SerializationError("metadata nesting exceeds Python recursion safety") from error
        document = {
            "format": _FORMAT,
            "serializer_version": _SERIALIZER_VERSION,
            "schema_version": experience.metadata.schema_version,
            "root": root,
            "tensors": state.catalogs,
        }
        _check_json_limits(document, self.limits)
        metadata = _dump_document(document, self.limits)
        return SerializedExperience(metadata=metadata, buffers=tuple(state.buffers))

    def deserialize(self, serialized: SerializedExperience) -> ExperienceBatch:
        """Validate the complete catalog and all buffers before object construction."""

        document = _load_document(serialized.metadata, self.limits)
        catalog = _validate_catalog(
            document.get("tensors"),
            serialized.buffers,
            self.limits,
        )
        values: dict[str, object] = {}
        for name, segment in catalog.items():
            values[name] = self.buffer_manager.restore(
                segment,
                preserve_device=self.preserve_device,
            )
        result = self._decode(document["root"], values)
        if not isinstance(result, ExperienceBatch):
            raise SerializationError("metadata root is not an ExperienceBatch")
        if result.metadata.schema_version != document["schema_version"]:
            raise IntegrityError("root schema version disagrees with serializer envelope")
        result.validate()
        return result

    def validate_metadata(self, metadata: bytes) -> tuple[BufferSegment, ...]:
        """Validate an envelope and catalog before allocating external buffers."""

        return validate_metadata(metadata, limits=self.limits)

    def _encode(
        self,
        value: object,
        path: tuple[_PathPart, ...],
        state: _EncodeState,
    ) -> object:
        if isinstance(value, np.ndarray) or _is_torch_tensor(value):
            name = f"tensor-{len(state.catalogs):06d}"
            segment = self.buffer_manager.stage(
                value,
                name=name,
                path=path,
                checksum=self.checksum,
            )
            _check_encoded_segment(segment, state, self.limits)
            entry = segment.catalog_entry()
            if segment.nbytes <= self.inline_threshold:
                if _base64_size(segment.nbytes) > self.limits.max_metadata_bytes:
                    raise SerializationError(
                        f"inline tensor {name!r} cannot fit within metadata byte limit "
                        f"{self.limits.max_metadata_bytes}"
                    )
                entry["inline"] = base64.b64encode(segment.materialize()).decode("ascii")
            else:
                entry["external"] = True
                state.buffers.append(segment)
            state.catalogs.append(entry)
            state.tensor_bytes += segment.nbytes
            return {"$type": "tensor", "name": name}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise SerializationError("non-finite floats are not valid metadata")
            return value
        if isinstance(value, bytes):
            return {"$type": "bytes", "data": base64.b64encode(value).decode("ascii")}
        if isinstance(value, UUID):
            return {"$type": "uuid", "value": str(value)}
        if isinstance(value, datetime):
            return {"$type": "datetime", "value": value.isoformat()}
        if isinstance(value, Enum):
            key = _type_key(type(value))
            if key not in self._types:
                raise SerializationError(f"enum type is not allowed: {key}")
            return {
                "$type": "enum",
                "class": key,
                "value": self._encode(value.value, path, state),
            }
        if is_dataclass(value) and not isinstance(value, type):
            key = _type_key(type(value))
            if key not in self._types:
                raise SerializationError(f"dataclass type is not allowed: {key}")
            encoded_fields = {
                field.name: self._encode(
                    _normalized_field(value, field.name),
                    (*path, field.name),
                    state,
                )
                for field in fields(value)
            }
            return {
                "$type": "dataclass",
                "class": key,
                "fields": encoded_fields,
            }
        if isinstance(value, Mapping):
            encoded = [
                [
                    self._encode(key, (*path, "<key>"), state),
                    self._encode(item, (*path, str(key)), state),
                ]
                for key, item in value.items()
            ]
            encoded.sort(key=lambda pair: _canonical(pair[0]))
            return {"$type": "mapping", "items": encoded}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            kind = "tuple" if isinstance(value, tuple) else "list"
            return {
                "$type": kind,
                "items": [
                    self._encode(item, (*path, index), state) for index, item in enumerate(value)
                ],
            }
        raise SerializationError(f"unsupported metadata value at {path!r}: {type(value).__name__}")

    def _decode(self, value: object, tensors: Mapping[str, object]) -> object:
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if not isinstance(value, dict) or not isinstance(value.get("$type"), str):
            raise SerializationError("malformed tagged metadata value")
        tag = value["$type"]
        if tag == "tensor":
            name = _required_str(value, "name")
            try:
                return tensors[name]
            except KeyError as error:
                raise IntegrityError(f"tensor {name!r} is absent from the catalog") from error
        if tag == "bytes":
            return _decode_base64(_required_str(value, "data"), "bytes metadata")
        if tag == "uuid":
            try:
                return UUID(_required_str(value, "value"))
            except ValueError as error:
                raise SerializationError("invalid UUID metadata") from error
        if tag == "datetime":
            try:
                return datetime.fromisoformat(_required_str(value, "value"))
            except ValueError as error:
                raise SerializationError("invalid datetime metadata") from error
        if tag == "enum":
            cls = self._allowed_class(value)
            if not issubclass(cls, Enum):
                raise SerializationError("encoded enum class is not an Enum")
            return cls(self._decode(value.get("value"), tensors))
        if tag == "dataclass":
            cls = self._allowed_class(value)
            encoded_fields = value.get("fields")
            if not isinstance(encoded_fields, dict):
                raise SerializationError("dataclass fields must be an object")
            expected = {field.name for field in fields(cast(Any, cls))}
            if set(encoded_fields) != expected:
                raise SerializationError(
                    f"fields for {_type_key(cls)} differ: expected {sorted(expected)!r}"
                )
            decoded_fields = {
                name: self._decode(item, tensors) for name, item in encoded_fields.items()
            }
            instance = cls(**decoded_fields)
            if isinstance(instance, model.TensorPayload):
                instance.stride = _element_stride(instance.data)
                instance.layout = _tensor_layout(instance.data)
                instance.device = _tensor_device(instance.data)
                instance.nbytes = _logical_tensor_nbytes(instance.data)
            return instance
        if tag in {"mapping", "list", "tuple"}:
            items = value.get("items")
            if not isinstance(items, list):
                raise SerializationError(f"{tag} items must be a list")
            if tag == "mapping":
                result: dict[object, object] = {}
                for pair in items:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise SerializationError("mapping entries must be key/value pairs")
                    key = self._decode(pair[0], tensors)
                    try:
                        if key in result:
                            raise SerializationError("decoded mapping keys must be unique")
                        result[key] = self._decode(pair[1], tensors)
                    except TypeError as error:
                        raise SerializationError("decoded mapping key is unhashable") from error
                return result
            sequence = [self._decode(item, tensors) for item in items]
            return tuple(sequence) if tag == "tuple" else sequence
        raise SerializationError(f"unknown metadata tag: {tag!r}")

    def _allowed_class(self, value: Mapping[str, object]) -> type[object]:
        key = _required_str(value, "class")
        try:
            return self._types[key]
        except KeyError as error:
            raise SerializationError(f"metadata class is not allowed: {key}") from error


class AuthenticatedExperienceSerializer:
    """HMAC-SHA256 authentication wrapper for the safe JSON serializer."""

    def __init__(
        self,
        keys: Mapping[str, bytes],
        *,
        signing_key_id: str,
        serializer: JsonExperienceSerializer | None = None,
    ) -> None:
        self.serializer = serializer or JsonExperienceSerializer()
        self.limits = self.serializer.limits
        self._keys = dict(keys)
        if signing_key_id not in self._keys:
            raise ValueError("signing_key_id must identify a configured key")
        for key_id, key in self._keys.items():
            if not _valid_key_id(key_id):
                raise ValueError(f"invalid authentication key ID: {key_id!r}")
            if not isinstance(key, bytes) or len(key) < 32:
                raise ValueError("authentication keys must contain at least 32 bytes")
        self.signing_key_id = signing_key_id

    def serialize(self, experience: ExperienceBatch) -> SerializedExperience:
        """Serialize and authenticate the complete metadata and tensor payload."""

        serialized = self.serializer.serialize(experience)
        document = _load_document(serialized.metadata, self.limits)
        document["authentication"] = {
            "algorithm": "hmac-sha256",
            "key_id": self.signing_key_id,
            "signature": _signature(
                self._keys[self.signing_key_id],
                self.signing_key_id,
                serialized.metadata,
                serialized.buffers,
            ),
        }
        return replace(serialized, metadata=_dump_document(document, self.limits))

    def deserialize(self, serialized: SerializedExperience) -> ExperienceBatch:
        """Authenticate all bytes before schema reconstruction."""

        document = _load_document(
            serialized.metadata,
            self.limits,
            allow_authentication=True,
        )
        authentication = cast(dict[str, object], document.pop("authentication", None))
        if not authentication:
            raise IntegrityError("authenticated serializer requires a signature")
        key_id = cast(str, authentication["key_id"])
        try:
            key = self._keys[key_id]
        except KeyError as error:
            raise IntegrityError(f"unknown authentication key ID: {key_id!r}") from error
        metadata = _dump_document(document, self.limits)
        expected = _signature(key, key_id, metadata, serialized.buffers)
        if not hmac.compare_digest(expected, cast(str, authentication["signature"])):
            raise IntegrityError("experience authentication failed")
        return self.serializer.deserialize(replace(serialized, metadata=metadata))

    def validate_metadata(self, metadata: bytes) -> tuple[BufferSegment, ...]:
        """Validate a signed envelope and its catalog before buffer allocation."""

        return validate_metadata(metadata, limits=self.limits)


def _check_source_limits(root: object, limits: SerializationLimits) -> None:
    """Bound recursive input before schema validation or tensor staging."""

    item_count = 0
    tensor_count = 0
    tensor_bytes = 0
    active: set[int] = set()
    stack: list[tuple[object, int, bool]] = [(root, 0, False)]
    while stack:
        value, depth, leaving = stack.pop()
        identity = id(value)
        if leaving:
            active.remove(identity)
            continue
        if depth > limits.max_depth:
            raise SerializationError(f"metadata nesting depth exceeds limit {limits.max_depth}")
        item_count += 1
        if item_count > limits.max_items:
            raise SerializationError(f"metadata item count exceeds limit {limits.max_items}")
        if isinstance(value, np.ndarray) or _is_torch_tensor(value):
            nbytes = _logical_tensor_nbytes(value)
            tensor_count += 1
            tensor_bytes += nbytes
            if tensor_count > limits.max_tensor_count:
                raise SerializationError(f"tensor count exceeds limit {limits.max_tensor_count}")
            if nbytes > limits.max_tensor_bytes:
                raise SerializationError(
                    f"tensor size {nbytes} exceeds per-tensor byte limit {limits.max_tensor_bytes}"
                )
            if tensor_bytes > limits.max_total_tensor_bytes:
                raise SerializationError(
                    f"total tensor size {tensor_bytes} exceeds byte limit "
                    f"{limits.max_total_tensor_bytes}"
                )
            continue
        if isinstance(value, bytes):
            if _base64_size(len(value)) > limits.max_metadata_bytes:
                raise SerializationError(
                    f"encoded bytes metadata cannot fit within metadata byte limit "
                    f"{limits.max_metadata_bytes}"
                )
            continue
        if isinstance(value, str) and len(value) > limits.max_metadata_bytes:
            raise SerializationError(
                f"string metadata cannot fit within metadata byte limit {limits.max_metadata_bytes}"
            )

        children: list[object] | None = None
        if isinstance(value, Enum):
            children = [value.value]
        elif is_dataclass(value) and not isinstance(value, type):
            children = [getattr(value, item.name) for item in fields(value)]
        elif isinstance(value, Mapping):
            if len(value) > limits.max_items:
                raise SerializationError(f"metadata item count exceeds limit {limits.max_items}")
            children = []
            for key, item in value.items():
                children.extend((key, item))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) > limits.max_items:
                raise SerializationError(f"metadata item count exceeds limit {limits.max_items}")
            children = list(value)
        if children is None:
            continue
        if identity in active:
            raise SerializationError("cyclic metadata is not supported")
        active.add(identity)
        stack.append((value, depth, True))
        stack.extend((child, depth + 1, False) for child in children)


def _check_json_limits(root: object, limits: SerializationLimits) -> None:
    item_count = 0
    stack: list[tuple[object, int]] = [(root, 0)]
    while stack:
        value, depth = stack.pop()
        if depth > limits.max_depth:
            raise SerializationError(f"metadata nesting depth exceeds limit {limits.max_depth}")
        item_count += 1
        if isinstance(value, dict):
            item_count += len(value)
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
        if item_count > limits.max_items:
            raise SerializationError(f"metadata item count exceeds limit {limits.max_items}")


def _check_metadata_size(metadata: bytes, limits: SerializationLimits) -> None:
    if len(metadata) > limits.max_metadata_bytes:
        raise SerializationError(
            f"metadata size {len(metadata)} exceeds byte limit {limits.max_metadata_bytes}"
        )


def _check_encoded_segment(
    segment: BufferSegment,
    state: _EncodeState,
    limits: SerializationLimits,
) -> None:
    if (
        isinstance(segment.nbytes, bool)
        or not isinstance(segment.nbytes, int)
        or segment.nbytes < 0
    ):
        raise SerializationError(f"tensor {segment.name!r} has invalid nbytes")
    if len(state.catalogs) >= limits.max_tensor_count:
        raise SerializationError(f"tensor count exceeds limit {limits.max_tensor_count}")
    if segment.nbytes > limits.max_tensor_bytes:
        raise SerializationError(
            f"tensor {segment.name!r} size {segment.nbytes} exceeds per-tensor byte limit "
            f"{limits.max_tensor_bytes}"
        )
    total = state.tensor_bytes + segment.nbytes
    if total > limits.max_total_tensor_bytes:
        raise SerializationError(
            f"total tensor size {total} exceeds byte limit {limits.max_total_tensor_bytes}"
        )


def _logical_tensor_nbytes(value: object) -> int:
    if isinstance(value, np.ndarray):
        return int(value.size) * int(value.dtype.itemsize)
    if _is_torch_tensor(value):
        tensor = cast(Any, value)
        return int(tensor.numel()) * int(tensor.element_size())
    raise SerializationError(f"unsupported tensor value: {type(value).__name__}")


def _normalized_field(value: object, name: str) -> object:
    field_value = getattr(value, name)
    if not isinstance(value, model.TensorPayload):
        return field_value
    if name == "stride":
        return _contiguous_stride(value.shape)
    if name == "layout":
        return "strided"
    return field_value


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = [0] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = running
        running *= max(shape[index], 1)
    return tuple(stride)


def _element_stride(value: object) -> tuple[int, ...] | None:
    if isinstance(value, np.ndarray):
        itemsize = value.dtype.itemsize
        if itemsize <= 0 or any(size % itemsize for size in value.strides):
            raise SerializationError("NumPy byte strides are not aligned to the element size")
        return tuple(size // itemsize for size in value.strides)
    if _is_torch_tensor(value):
        return tuple(int(size) for size in cast(Any, value).stride())
    return None


def _tensor_layout(value: object) -> str:
    if isinstance(value, np.ndarray):
        return "strided"
    if _is_torch_tensor(value):
        return str(cast(Any, value).layout).removeprefix("torch.")
    return "strided"


def _tensor_device(value: object) -> str:
    return str(getattr(value, "device", "cpu"))


def _base64_size(nbytes: int) -> int:
    return 4 * ((nbytes + 2) // 3)


def _safe_types(extra: Iterable[type[object]]) -> dict[str, type[object]]:
    classes: list[type[object]] = []
    for value in vars(model).values():
        if isinstance(value, type) and (is_dataclass(value) or issubclass(value, Enum)):
            classes.append(value)
    classes.extend(extra)
    result: dict[str, type[object]] = {}
    for cls in classes:
        if not (is_dataclass(cls) or issubclass(cls, Enum)):
            raise ValueError(f"allowed type must be a dataclass or Enum: {cls!r}")
        key = _type_key(cls)
        if key in result and result[key] is not cls:
            raise ValueError(f"duplicate allowed type key: {key}")
        result[key] = cls
    return result


def _load_document(
    metadata: bytes,
    limits: SerializationLimits = _DEFAULT_LIMITS,
    *,
    allow_authentication: bool = False,
) -> dict[str, object]:
    if not isinstance(metadata, bytes):
        raise SerializationError("serializer metadata must be bytes")
    _check_metadata_size(metadata, limits)
    try:
        value = json.loads(metadata.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise SerializationError(f"invalid UTF-8 JSON metadata: {error}") from error
    _check_json_limits(value, limits)
    if not isinstance(value, dict):
        raise SerializationError("serializer metadata must be a JSON object")
    if value.get("format") != _FORMAT:
        raise SerializationError(f"unsupported serializer format: {value.get('format')!r}")
    if value.get("serializer_version") != _SERIALIZER_VERSION:
        raise SerializationError(
            f"unsupported serializer version: {value.get('serializer_version')!r}"
        )
    if value.get("schema_version") != model.SCHEMA_VERSION:
        raise SerializationError(f"unsupported schema version: {value.get('schema_version')!r}")
    expected = {"format", "serializer_version", "schema_version", "root", "tensors"}
    authentication = value.get("authentication")
    if allow_authentication and authentication is not None:
        _validate_authentication(authentication)
        expected.add("authentication")
    if set(value) != expected:
        raise SerializationError("serializer envelope has unknown or missing fields")
    if "root" not in value:
        raise SerializationError("serializer metadata is missing root")
    return cast(dict[str, object], value)


def validate_metadata(
    metadata: bytes,
    *,
    limits: SerializationLimits | None = None,
) -> tuple[BufferSegment, ...]:
    """Validate metadata and return catalog-only descriptors for transfer planning."""

    active_limits = limits or _DEFAULT_LIMITS
    document = _load_document(metadata, active_limits, allow_authentication=True)
    return tuple(
        _validate_catalog(
            document.get("tensors"), (), active_limits, require_buffers=False
        ).values()
    )


def _check_catalog_limits(value: object, limits: SerializationLimits) -> None:
    if not isinstance(value, list):
        raise SerializationError("tensor catalog must be a list")
    if len(value) > limits.max_tensor_count:
        raise SerializationError(f"tensor count exceeds limit {limits.max_tensor_count}")
    sizes: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise SerializationError("tensor catalog entries must be objects")
        nbytes = item.get("nbytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise SerializationError(f"tensor catalog entry {index} has invalid nbytes")
        sizes.append(nbytes)
    validate_transfer_limits(metadata_bytes=0, tensor_sizes=sizes, limits=limits)


def _validate_catalog(
    value: object,
    buffers: tuple[BufferSegment, ...],
    limits: SerializationLimits,
    *,
    require_buffers: bool = True,
) -> dict[str, BufferSegment]:
    _check_catalog_limits(value, limits)
    items = cast(list[object], value)
    supplied = {segment.name: segment for segment in buffers}
    if len(supplied) != len(buffers):
        raise IntegrityError("external buffer names must be unique")
    result: dict[str, BufferSegment] = {}
    external_names: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise SerializationError("tensor catalog entries must be objects")
        segment = _segment_from_catalog(
            cast(dict[str, object], item), supplied, require_buffer=require_buffers
        )
        if segment.name in result:
            raise IntegrityError(f"duplicate tensor catalog name: {segment.name!r}")
        if require_buffers or segment.data is not None:
            raw = segment.materialize()
            if len(raw) != segment.nbytes:
                raise IntegrityError(
                    f"buffer {segment.name!r} size mismatch: expected "
                    f"{segment.nbytes}, got {len(raw)}"
                )
            if segment.sha256 is not None and _digest(raw) != segment.sha256:
                raise IntegrityError(f"buffer {segment.name!r} checksum mismatch")
            segment = replace(segment, data=raw, owner=None)
        result[segment.name] = segment
        if item.get("external") is True:
            external_names.add(segment.name)
    if require_buffers and external_names != set(supplied):
        raise IntegrityError(
            f"external buffer catalog mismatch: expected {sorted(external_names)!r}, "
            f"got {sorted(supplied)!r}"
        )
    return result


def _segment_from_catalog(
    entry: dict[str, object],
    supplied: Mapping[str, BufferSegment],
    *,
    require_buffer: bool = True,
) -> BufferSegment:
    name = _required_str(entry, "name")
    kind = _required_str(entry, "kind")
    dtype = _required_str(entry, "dtype")
    shape = _integer_tuple(entry.get("shape"), "shape", non_negative=True)
    stride = _integer_tuple(entry.get("stride"), "stride", non_negative=False)
    if len(shape) != len(stride):
        raise SerializationError(f"tensor {name!r} shape and stride rank differ")
    expected_stride = _contiguous_stride(shape)
    if stride != expected_stride:
        raise IntegrityError(
            f"tensor {name!r} catalog stride mismatch: expected normalized "
            f"element stride {expected_stride}, got {stride}"
        )
    layout = _required_str(entry, "layout")
    if layout != "strided":
        raise SerializationError(f"tensor {name!r} has unsupported layout {layout!r}")
    original_device = _required_str(entry, "original_device")
    wire_device = _required_str(entry, "wire_device")
    path = _path(entry.get("path"))
    nbytes = entry.get("nbytes")
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
        raise SerializationError(f"tensor {name!r} has invalid nbytes")
    expected_size = _expected_nbytes(kind, dtype, shape)
    if expected_size != nbytes:
        raise IntegrityError(
            f"tensor {name!r} catalog size mismatch: shape/dtype require "
            f"{expected_size}, got {nbytes}"
        )
    checksum = entry.get("sha256")
    if checksum is not None and (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise SerializationError(f"tensor {name!r} has invalid sha256")
    external = entry.get("external") is True
    inline = entry.get("inline")
    if external == (inline is not None):
        raise SerializationError(f"tensor {name!r} must be exactly one of inline or external")
    if inline is not None:
        if not isinstance(inline, str):
            raise SerializationError(f"tensor {name!r} inline payload must be base64 text")
        expected_inline_size = _base64_size(nbytes)
        if len(inline) != expected_inline_size:
            raise IntegrityError(
                f"tensor {name!r} inline size mismatch: expected "
                f"{expected_inline_size} base64 characters, got {len(inline)}"
            )
    expected_keys = {*_CATALOG_FIELDS, "external" if external else "inline"}
    if set(entry) != expected_keys:
        raise SerializationError(f"tensor {name!r} has unknown or missing catalog fields")
    catalog = BufferSegment(
        name=name,
        path=path,
        kind=kind,
        dtype=dtype,
        shape=shape,
        stride=stride,
        layout=layout,
        original_device=original_device,
        wire_device=wire_device,
        nbytes=nbytes,
        sha256=checksum,
        data=_decode_base64(inline, f"tensor {name!r}") if inline is not None else None,
    )
    if not external:
        return catalog
    if not require_buffer:
        return catalog
    try:
        provided = supplied[name]
    except KeyError as error:
        raise IntegrityError(f"external buffer {name!r} is missing") from error
    if provided.catalog_entry() != catalog.catalog_entry():
        raise IntegrityError(f"external buffer {name!r} metadata disagrees with catalog")
    return provided


def _expected_nbytes(kind: str, dtype: str, shape: tuple[int, ...]) -> int:
    count = math.prod(shape)
    if kind == "numpy":
        itemsize = _numpy_dtype(dtype).itemsize
    elif kind == "torch":
        torch = _torch()
        if torch is None:
            raise SerializationError("PyTorch is required to validate a PyTorch tensor")
        itemsize = torch.empty((), dtype=_torch_dtype(torch, dtype)).element_size()
    else:
        raise SerializationError(f"unsupported tensor kind: {kind!r}")
    return count * itemsize


def _numpy_dtype(value: str) -> np.dtype[Any]:
    try:
        dtype = np.dtype(value)
    except TypeError as error:
        raise SerializationError(f"invalid NumPy dtype: {value!r}") from error
    if dtype.hasobject:
        raise SerializationError("NumPy object dtypes are not safe to deserialize")
    return dtype


def _torch_dtype(torch: Any, value: str) -> Any:
    allowed = {
        name: getattr(torch, name)
        for name in (
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
        )
        if hasattr(torch, name)
    }
    try:
        return allowed[value.removeprefix("torch.")]
    except KeyError as error:
        raise SerializationError(f"unsupported PyTorch dtype: {value!r}") from error


def _integer_tuple(value: object, field: str, *, non_negative: bool) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or (non_negative and item < 0)
        for item in value
    ):
        raise SerializationError(f"tensor catalog {field} must be a list of integers")
    return tuple(value)


def _path(value: object) -> tuple[_PathPart, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (str, int)) for item in value
    ):
        raise SerializationError("tensor catalog path must contain strings and integers")
    return tuple(value)


def _required_str(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise SerializationError(f"metadata {key!r} must be a non-empty string")
    return result


def _decode_base64(value: str, context: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise SerializationError(f"invalid base64 in {context}") from error


def _digest(value: bytes | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _signature(
    key: bytes,
    key_id: str,
    metadata: bytes,
    buffers: tuple[BufferSegment, ...],
) -> str:
    signature = hmac.new(key, digestmod=hashlib.sha256)
    protected = _canonical({"algorithm": "hmac-sha256", "key_id": key_id}).encode()
    for part in (protected, metadata):
        signature.update(len(part).to_bytes(8, "big"))
        signature.update(part)
    for buffer in buffers:
        part = buffer.materialize()
        signature.update(len(part).to_bytes(8, "big"))
        signature.update(part)
    return signature.hexdigest()


def _valid_key_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 128
        and value.isascii()
        and all(character.isalnum() or character in "._-" for character in value)
    )


def _validate_authentication(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"algorithm", "key_id", "signature"}:
        raise SerializationError("authentication metadata is malformed")
    if value.get("algorithm") != "hmac-sha256" or not _valid_key_id(value.get("key_id")):
        raise SerializationError("authentication algorithm or key ID is invalid")
    signature = value.get("signature")
    if (
        not isinstance(signature, str)
        or len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        raise SerializationError("authentication signature is invalid")


def _dump_document(value: object, limits: SerializationLimits) -> bytes:
    _check_json_limits(value, limits)
    try:
        metadata = _canonical(value).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise SerializationError(f"metadata encoding failed: {error}") from error
    _check_metadata_size(metadata, limits)
    return metadata


def _type_key(value: type[object]) -> str:
    return f"{value.__module__}:{value.__qualname__}"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SerializationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result
