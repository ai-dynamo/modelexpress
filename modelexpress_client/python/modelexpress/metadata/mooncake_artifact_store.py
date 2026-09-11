# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side Mooncake cache for file-backed framework artifacts.

This module deliberately does not talk to the ModelExpress server. Mooncake is
used as an external artifact transport keyed by the same compatibility identity
used by the other artifact transport; backend selection policy is owned by the
lifecycle coordinator.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import struct
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from uuid import uuid4

import torch

from .. import envs, p2p_pb2
from ..mooncake_env import mx_mc_env_override
from .artifact_manifest import (
    DEFAULT_ARTIFACT_TRANSFER_CHUNK_SIZE,
    _crc32c_hex,
    artifact_manifest_id,
)
from .source_id import compute_mx_source_id

if TYPE_CHECKING:
    from .artifact_transfer import ArtifactBundle, ArtifactTransfer

logger = logging.getLogger("modelexpress.metadata.mooncake_artifact_store")

_store_lock = threading.RLock()
_shared_store: _MooncakeNativeStore | None = None
_shared_store_config: _MooncakeNativeConfig | None = None
_store_atexit_registered = False


class MooncakeArtifactCacheMiss(LookupError):
    """Raised when a deterministic artifact cache key is not present."""


class MooncakeArtifactCacheStale(MooncakeArtifactCacheMiss):
    """Raised when a Mooncake artifact exists but cannot be trusted."""

    def __init__(self, message: str, *, cache_key: str, fingerprint: str) -> None:
        super().__init__(message)
        self.cache_key = cache_key
        self.fingerprint = fingerprint


class MooncakeArtifactCacheUnavailable(RuntimeError):
    """Raised when Mooncake native store is not installed or configured."""


_ARTIFACT_ENVELOPE_HEADER = struct.Struct("!8sHQ")
_ARTIFACT_ENVELOPE_MAGIC = b"MXMCART1"
# This is part of the MXMCART1 wire layout, not a deployment tuning knob.
# Changing it requires a new protocol magic so mixed-version replicas cannot
# interpret the same Mooncake object with different payload boundaries.
_ARTIFACT_ENVELOPE_BYTES = 4 * 1024 * 1024
_REPAIR_LOCK_HEADER = struct.Struct("!8sI")
_REPAIR_LOCK_MAGIC = b"MXMCRPR1"
_REPAIR_LOCK_BYTES = 4096
_REPAIR_LOCK_STALE_SECONDS = 10 * 60
_RECEIVE_SLICE_BYTES = DEFAULT_ARTIFACT_TRANSFER_CHUNK_SIZE


@dataclass(frozen=True)
class _MooncakeArtifactEnvelope:
    artifact_id: str
    manifest_bytes: bytes


@dataclass(frozen=True)
class _FetchedMooncakeArtifact:
    buffers: tuple[torch.Tensor, ...]
    manifest: p2p_pb2.ArtifactManifest
    artifact_id: str
    header: p2p_pb2.GetArtifactManifestHeaderResponse


@dataclass(frozen=True)
class _RepairLock:
    token: str
    pid: int
    hostname: str
    created_at: float


def install_from_mooncake(
    transfer: ArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
    *,
    node_rank: int,
    accelerator: str,
) -> p2p_pb2.GetArtifactManifestHeaderResponse:
    """Fetch one artifact from Mooncake and stage it for normal installation."""
    start = time.perf_counter()
    cache_key = compute_artifact_cache_key(
        transfer,
        identity,
        node_rank=node_rank,
        accelerator=accelerator,
    )
    logger.debug(
        "[Mooncake] artifact query start: name=%s key=%s node_rank=%s "
        "accelerator=%s source_identity=%s",
        transfer.name,
        cache_key,
        node_rank,
        accelerator,
        _identity_debug(identity),
    )
    with _store_session() as store:
        fetched = _fetch_artifact_object(store, cache_key)

    target_header = _header_with_transfer_target_paths(transfer, fetched.header)
    _prepare_target_files(target_header.files)
    try:
        transferred_size = _write_payload_files(fetched, target_header)
    except Exception:
        _cleanup_target_files(target_header.files)
        raise

    elapsed = time.perf_counter() - start
    logger.info(
        "[TIMING] Mooncake artifact fetch complete: name=%s artifact_id=%s "
        "key=%s files=%d chunks=%d size=%.2f MiB elapsed=%.3fs "
        "throughput=%.2f Gbps",
        transfer.name,
        fetched.artifact_id,
        cache_key,
        len(fetched.header.files),
        len(fetched.manifest.chunks),
        transferred_size / (1024 * 1024),
        elapsed,
        _gbps(transferred_size, elapsed),
    )
    return target_header


def publish_to_mooncake(
    transfer: ArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
    bundle: ArtifactBundle,
    *,
    node_rank: int,
    accelerator: str,
    repair_from_fingerprint: str | None = None,
) -> str:
    """Publish one prepared artifact as one atomic Mooncake object."""
    if artifact_manifest_id(bundle.manifest) != bundle.artifact_id:
        raise ValueError("artifact bundle id does not match its manifest")

    start = time.perf_counter()
    cache_key = compute_artifact_cache_key(
        transfer,
        identity,
        node_rank=node_rank,
        accelerator=accelerator,
    )
    logger.info(
        "[Mooncake] artifact publish start: name=%s key=%s node_rank=%s "
        "accelerator=%s artifact_id=%s files=%d chunks=%d total_size=%d "
        "source_identity=%s",
        transfer.name,
        cache_key,
        node_rank,
        accelerator,
        bundle.artifact_id,
        len(bundle.manifest.files),
        len(bundle.manifest.chunks),
        sum(file.size for file in bundle.manifest.files),
        _identity_debug(identity),
    )
    buffers = _bundle_buffers(bundle)
    payload_size = sum(_tensor_size(buffer) for buffer in buffers[1:])
    with _store_session() as store:
        if repair_from_fingerprint is None:
            _put_artifact_object(store, cache_key, buffers, transfer.name)
        else:
            _repair_artifact_object(
                store,
                cache_key,
                buffers,
                transfer_name=transfer.name,
                stale_fingerprint=repair_from_fingerprint,
            )

    elapsed = time.perf_counter() - start
    logger.info(
        "[TIMING] Mooncake artifact publish operation complete: "
        "name=%s artifact_id=%s "
        "key=%s files=%d chunks=%d size=%.2f MiB elapsed=%.3fs "
        "throughput=%.2f Gbps repair=%s",
        transfer.name,
        bundle.artifact_id,
        cache_key,
        len(bundle.manifest.files),
        len(bundle.manifest.chunks),
        payload_size / (1024 * 1024),
        elapsed,
        _gbps(payload_size, elapsed),
        repair_from_fingerprint is not None,
    )
    return cache_key


def compute_artifact_cache_key(
    transfer: ArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
    *,
    node_rank: int,
    accelerator: str,
) -> str:
    """Generate the deterministic Mooncake key for one artifact.

    The digest reuses the canonical compatibility identity shared with the
    P2P/server path, plus the artifact type, artifact name, node rank, and
    accelerator backend. The returned key directly addresses the complete
    Mooncake object; no manifest or chunk suffix is appended.
    """
    namespace = envs.MX_ARTIFACT_MOONCAKE_NAMESPACE or "modelexpress/artifacts"
    digest = sha256()
    digest.update(compute_mx_source_id(identity).encode("ascii"))
    digest.update(str(transfer.mx_source_type).encode())
    digest.update(transfer.name.encode())
    digest.update(str(node_rank).encode())
    digest.update((accelerator or "").encode())
    try:
        framework = p2p_pb2.BackendFramework.Name(identity.backend_framework)
    except ValueError:
        framework = "BACKEND_FRAMEWORK_UNKNOWN"
    framework = framework.removeprefix("BACKEND_FRAMEWORK_").lower()

    model_name = identity.model_name.rstrip("/")
    model_slug = model_name.rsplit("/", 1)[-1] if model_name else "unknown"
    model_slug = "".join(
        char.lower() if char.isalnum() else "-" for char in model_slug
    ).strip("-")[:96] or "unknown"

    return (
        f"{namespace.rstrip('/')}/{framework}/{model_slug}/"
        f"{transfer.name}/{digest.hexdigest()}"
    )


def _identity_debug(identity: p2p_pb2.SourceIdentity) -> str:
    """Keep protobuf identity logs single-line and readable."""
    return " ".join(str(identity).split()) or "<empty>"


def _repair_key(cache_key: str) -> str:
    return f"{cache_key}/repair"


@dataclass(frozen=True)
class _MooncakeNativeConfig:
    local_hostname: str
    metadata_server: str
    master_server: str
    protocol: str
    device_name: str
    global_segment_size: int
    local_buffer_size: int


class _MooncakeNativeStore:
    """Registered-buffer wrapper around MooncakeDistributedStore."""

    def __init__(self, config: _MooncakeNativeConfig) -> None:
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except Exception as exc:
            raise MooncakeArtifactCacheUnavailable(
                "Mooncake Python package is not installed or mooncake.store "
                f"could not be imported: {exc}"
            ) from exc

        self._store = MooncakeDistributedStore()
        ret = self._store.setup(
            config.local_hostname,
            config.metadata_server,
            config.global_segment_size,
            config.local_buffer_size,
            config.protocol,
            config.device_name,
            config.master_server,
        )
        if ret != 0:
            raise MooncakeArtifactCacheUnavailable(
                "Mooncake Python package is installed, but connecting to the "
                "Mooncake store failed during setup: "
                f"ret={ret}, local_hostname={config.local_hostname}, "
                f"metadata_server={config.metadata_server}, "
                f"master_server={config.master_server}, "
                f"protocol={config.protocol}, device_name={config.device_name}"
            )
        self._artifact_replicate_config = ReplicateConfig()
        self._artifact_replicate_config.with_soft_pin = True
        self._repair_replicate_config = ReplicateConfig()
        self._repair_replicate_config.with_soft_pin = False

    def get_size(self, key: str) -> int | None:
        get_size = getattr(self._store, "get_size", None)
        if not callable(get_size):
            raise MooncakeArtifactCacheUnavailable(
                "Mooncake store does not expose get_size"
            )
        result = int(get_size(key))
        logger.debug(
            "[Mooncake] get_size completed: key=%s result=%s",
            key,
            _result_debug(result),
        )
        if result == -704:
            return None
        if result < 0:
            raise MooncakeArtifactCacheUnavailable(
                f"Mooncake get_size failed: key={key!r} "
                f"result={_result_debug(result)}"
            )
        return result

    def get_buffers(
        self,
        key: str,
        sizes: list[int],
    ) -> tuple[torch.Tensor, ...] | None:
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError(f"invalid Mooncake get buffer sizes for {key}: {sizes}")
        buffers = tuple(
            torch.empty(size, dtype=torch.uint8, device="cpu") for size in sizes
        )
        pointers, buffer_sizes = _buffer_descriptors(buffers, key=key)
        with self._registered_buffers(pointers, buffer_sizes, key=key):
            result = self._store.batch_get_into_multi_buffers(
                [key],
                [pointers],
                [buffer_sizes],
            )
        if not result:
            raise MooncakeArtifactCacheUnavailable(
                f"Mooncake get returned no result for key {key!r}"
            )
        result_code = int(result[0])
        expected_size = sum(buffer_sizes)
        logger.debug(
            "[Mooncake] get completed: key=%s buffers=%d expected_bytes=%d "
            "result=%s",
            key,
            len(buffers),
            expected_size,
            _result_debug(result_code),
        )
        if result_code == -704:
            return None
        if result_code < 0:
            raise MooncakeArtifactCacheUnavailable(
                "Mooncake get failed: "
                f"key={key!r} expected_bytes={expected_size} "
                f"result={_result_debug(result_code)}"
            )
        if result_code != expected_size:
            raise MooncakeArtifactCacheUnavailable(
                "Mooncake get returned an unexpected object size: "
                f"key={key!r} expected={expected_size} actual={result_code}"
            )
        return buffers

    def put_buffers(
        self,
        key: str,
        buffers: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        soft_pin: bool,
    ) -> int:
        buffers = tuple(buffers)
        pointers, buffer_sizes = _buffer_descriptors(buffers, key=key)
        config = (
            self._artifact_replicate_config
            if soft_pin
            else self._repair_replicate_config
        )
        with self._registered_buffers(pointers, buffer_sizes, key=key):
            result = self._store.batch_put_from_multi_buffers(
                [key],
                [pointers],
                [buffer_sizes],
                config,
            )
        result_code = int(result[0]) if result else -1
        logger.debug(
            "[Mooncake] put completed: key=%s buffers=%d bytes=%d result=%s",
            key,
            len(buffers),
            sum(buffer_sizes),
            _result_debug(result_code),
        )
        return result_code

    def get_bytes(self, key: str, expected_size: int | None = None) -> bytes | None:
        size = expected_size if expected_size is not None else self.get_size(key)
        if size is None:
            return None
        if size <= 0:
            raise ValueError(f"invalid Mooncake get size for {key}: {size}")
        buffers = self.get_buffers(key, [size])
        if buffers is None:
            return None
        return _tensor_to_bytes(buffers[0])

    def put_bytes(self, key: str, data: bytes, *, soft_pin: bool = True) -> int:
        tensor = _bytes_to_tensor(data)
        return self.put_buffers(key, (tensor,), soft_pin=soft_pin)

    def remove(self, key: str) -> int:
        remove = getattr(self._store, "remove", None)
        if not callable(remove):
            raise MooncakeArtifactCacheUnavailable(
                "Mooncake store does not expose remove"
            )
        result = int(remove(key))
        logger.debug(
            "[Mooncake] remove completed: key=%s result=%s",
            key,
            _result_debug(result),
        )
        return result

    def _register_buffer(self, buffer_ptr: int, buffer_size: int, *, key: str) -> None:
        register = getattr(self._store, "register_buffer", None)
        if not callable(register):
            raise MooncakeArtifactCacheUnavailable(
                "Mooncake store does not expose register_buffer; "
                "multi-buffer artifact transfers require registered buffers"
            )
        try:
            result = register(buffer_ptr, buffer_size)
        except Exception as exc:
            raise MooncakeArtifactCacheUnavailable(
                f"Mooncake register_buffer failed for key {key!r}: {exc}"
            ) from exc
        if result not in (None, 0):
            raise MooncakeArtifactCacheUnavailable(
                f"Mooncake register_buffer returned {result} for key {key!r}"
            )

    def _unregister_buffer(self, buffer_ptr: int, *, key: str) -> None:
        unregister = getattr(self._store, "unregister_buffer", None)
        if not callable(unregister):
            logger.warning(
                "Mooncake store does not expose unregister_buffer for key %r",
                key,
            )
            return
        try:
            result = unregister(buffer_ptr)
            if result not in (None, 0):
                logger.warning(
                    "Mooncake unregister_buffer returned %s for key %r",
                    result,
                    key,
                )
        except Exception:
            logger.warning(
                "Mooncake unregister_buffer failed for key %r",
                key,
                exc_info=True,
            )

    @contextmanager
    def _registered_buffers(
        self,
        pointers: list[int],
        sizes: list[int],
        *,
        key: str,
    ):
        registered: list[int] = []
        try:
            for pointer, size in zip(pointers, sizes, strict=True):
                self._register_buffer(pointer, size, key=key)
                registered.append(pointer)
            yield
        finally:
            for pointer in reversed(registered):
                self._unregister_buffer(pointer, key=key)

    def close(self) -> None:
        self._store.close()


def _new_store() -> _MooncakeNativeStore:
    """Return the process-local Mooncake store, creating it lazily."""
    global _shared_store, _shared_store_config, _store_atexit_registered
    config = _mooncake_native_config()
    with _store_lock:
        if _shared_store is None:
            _shared_store = _MooncakeNativeStore(config)
            _shared_store_config = config
            logger.info(
                "[MX][Mooncake] native store initialized: config_path=%s "
                "local_hostname=%s metadata_server=%s master_server=%s "
                "protocol=%s device_name=%s global_segment_size=%d "
                "local_buffer_size=%d",
                _mooncake_config_path() or "<env>",
                config.local_hostname,
                config.metadata_server,
                config.master_server,
                config.protocol,
                config.device_name,
                config.global_segment_size,
                config.local_buffer_size,
            )
            if not _store_atexit_registered:
                atexit.register(_close_shared_store)
                _store_atexit_registered = True
        elif _shared_store_config != config:
            raise MooncakeArtifactCacheUnavailable(
                "Mooncake native config changed while the process-local store "
                "is active; use one configuration per process"
            )
        else:
            logger.debug("[Mooncake] reusing process-local native store")
        return _shared_store


@contextmanager
def _store_session():
    """Serialize one complete artifact operation on the shared native store.

    The ModelExpress Mooncake cluster is configured under ``MX_MC_*`` while the
    native Mooncake libraries read ``MC_*``. Keep the ``MX_MC_*`` variables
    promoted for the duration of the operation so both the config snapshot and
    the native libraries observe the ModelExpress cluster, then restore the
    previous ``MC_*`` values.
    """
    with _store_lock:
        with mx_mc_env_override():
            yield _new_store()


def _close_shared_store() -> None:
    global _shared_store, _shared_store_config
    with _store_lock:
        store = _shared_store
        _shared_store = None
        _shared_store_config = None
        if store is not None:
            _close_store(store)


def _mooncake_native_config() -> _MooncakeNativeConfig:
    config_path = _mooncake_config_path()
    raw: dict[str, object] = {}
    if config_path:
        with open(config_path, encoding="utf-8") as file:
            raw = json.load(file)

    # Called under mx_mc_env_override(), so the native MC_* names already
    # carry the ModelExpress (MX_MC_*) values when set, and keep the
    # caller-supplied MC_* values otherwise. Reading only the native names
    # here means new Mooncake settings are picked up without per-variable
    # plumbing.
    config = _MooncakeNativeConfig(
        local_hostname=(
            _str_config(raw, "local_hostname", os.getenv("MC_LOCAL_HOSTNAME", ""))
            or _default_local_hostname()
        ),
        metadata_server=_required_config(
            raw,
            "metadata_server",
            os.getenv("MC_METADATA_ADDR", ""),
        ),
        master_server=_required_config(
            raw,
            "master_server_address",
            os.getenv("MC_MASTER_SERVER", ""),
            aliases=("master_server",),
        ),
        protocol=_str_config(
            raw,
            "protocol",
            os.getenv("MC_PROTOCOL", "rdma"),
        ),
        device_name=_str_config(
            raw,
            "device_name",
            os.getenv("MC_DEVICE_NAME", ""),
            aliases=("rdma_devices",),
        ),
        # Artifact clients must use a separately managed Mooncake store
        # segment.  Do not mount a segment in the vLLM/Modelexpress process;
        # otherwise its exit would release the only in-memory artifact copy.
        global_segment_size=0,
        local_buffer_size=_int_config(
            raw,
            "local_buffer_size",
            int(envs.MX_ARTIFACT_MOONCAKE_POOL_BYTES),
        ),
    )
    return config


def _mooncake_config_path() -> str:
    return (
        os.getenv("MX_MOONCAKE_CONFIG_PATH")
        or os.getenv("MOONCAKE_CONFIG_PATH", "")
    ).strip()


def _required_config(
    raw: dict[str, object],
    key: str,
    fallback: str,
    *,
    aliases: tuple[str, ...] = (),
) -> str:
    value = _str_config(raw, key, fallback, aliases=aliases)
    if not value:
        names = ", ".join((key, *aliases))
        raise MooncakeArtifactCacheUnavailable(
            f"Mooncake native config missing required field: {names}"
        )
    return value


def _str_config(
    raw: dict[str, object],
    key: str,
    fallback: str,
    *,
    aliases: tuple[str, ...] = (),
) -> str:
    for name in (key, *aliases):
        value = raw.get(name)
        if value is not None:
            return str(value).strip()
    return str(fallback or "").strip()


def _int_config(
    raw: dict[str, object],
    key: str,
    fallback: int,
) -> int:
    value = raw.get(key)
    if value is None:
        return int(fallback)
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    multipliers = {
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
    }
    for suffix, multiplier in sorted(
        multipliers.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * multiplier)
    return int(text)


def _default_local_hostname() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return socket.gethostname()


def _encode_artifact_envelope(*, artifact_id: str, manifest_bytes: bytes) -> bytes:
    artifact_bytes = artifact_id.encode("ascii")
    header = _ARTIFACT_ENVELOPE_HEADER.pack(
        _ARTIFACT_ENVELOPE_MAGIC,
        len(artifact_bytes),
        len(manifest_bytes),
    )
    payload = header + artifact_bytes + manifest_bytes
    limit = _ARTIFACT_ENVELOPE_BYTES
    if len(payload) > limit:
        raise ValueError(
            "Mooncake artifact envelope is too large: "
            f"{len(payload)} bytes > protocol limit {limit}"
        )
    return payload + bytes(limit - len(payload))


def _decode_artifact_envelope(data: bytes) -> _MooncakeArtifactEnvelope:
    header_size = _ARTIFACT_ENVELOPE_HEADER.size
    if len(data) < header_size:
        raise RuntimeError("Mooncake artifact envelope is truncated")
    magic, artifact_len, manifest_len = _ARTIFACT_ENVELOPE_HEADER.unpack_from(data)
    if magic != _ARTIFACT_ENVELOPE_MAGIC:
        raise RuntimeError("Mooncake artifact envelope magic mismatch")
    artifact_end = header_size + artifact_len
    manifest_end = artifact_end + manifest_len
    if manifest_end > len(data):
        raise RuntimeError("Mooncake artifact envelope length exceeds fetched bytes")
    try:
        artifact_id = data[header_size:artifact_end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "Mooncake artifact envelope contains a non-ASCII id"
        ) from exc
    if not artifact_id or not manifest_len:
        raise RuntimeError("Mooncake artifact envelope contains an empty field")
    return _MooncakeArtifactEnvelope(
        artifact_id=artifact_id,
        manifest_bytes=data[artifact_end:manifest_end],
    )


def _bundle_buffers(bundle: ArtifactBundle) -> tuple[torch.Tensor, ...]:
    envelope = _encode_artifact_envelope(
        artifact_id=bundle.artifact_id,
        manifest_bytes=bundle.manifest.SerializeToString(),
    )
    buffers = [_bytes_to_tensor(envelope)]
    files = list(bundle.manifest.files)
    for chunk in bundle.manifest.chunks:
        path = Path(files[chunk.file_index].path).resolve(strict=True)
        data = _read_file_range(path, chunk.file_offset, chunk.length)
        checksum = _crc32c_hex(data)
        if checksum != chunk.checksum:
            raise RuntimeError(
                "artifact chunk checksum changed while publishing: "
                f"path={path} chunk={chunk.chunk_index} "
                f"expected={chunk.checksum} actual={checksum}"
            )
        buffers.append(_bytes_to_tensor(data))
    return tuple(buffers)


def _fetch_artifact_object(
    store: _MooncakeNativeStore,
    cache_key: str,
) -> _FetchedMooncakeArtifact:
    object_size = store.get_size(cache_key)
    if object_size is None:
        raise MooncakeArtifactCacheMiss(f"Mooncake artifact miss: {cache_key}")
    if object_size <= 0:
        _raise_stale_artifact(
            cache_key,
            fingerprint=sha256(b"").hexdigest(),
            reason=f"invalid object size: {object_size}",
        )

    buffers = store.get_buffers(cache_key, _receive_buffer_sizes(object_size))
    if buffers is None:
        # The object may be evicted between get_size and the actual transfer.
        raise MooncakeArtifactCacheMiss(f"Mooncake artifact miss: {cache_key}")
    fingerprint = _buffers_fingerprint(buffers)
    envelope_size = _ARTIFACT_ENVELOPE_BYTES
    try:
        if object_size < envelope_size:
            raise RuntimeError(
                f"artifact object is shorter than its envelope: "
                f"{object_size} < {envelope_size}"
            )
        envelope = _decode_artifact_envelope(_tensor_to_bytes(buffers[0]))
        manifest = p2p_pb2.ArtifactManifest()
        manifest.ParseFromString(envelope.manifest_bytes)
        artifact_id = artifact_manifest_id(manifest)
        if envelope.artifact_id != artifact_id:
            raise RuntimeError(
                "artifact_id mismatch: "
                f"envelope={envelope.artifact_id} computed={artifact_id}"
            )
        header = _header_from_manifest(manifest, artifact_id=artifact_id)
        _validate_manifest_header(
            header,
            list(manifest.chunks),
            expected_artifact_id=artifact_id,
        )
        expected_size = envelope_size + sum(file.size for file in manifest.files)
        if object_size != expected_size:
            raise RuntimeError(
                "artifact object size mismatch: "
                f"expected={expected_size} actual={object_size}"
            )
        _validate_payload_checksums(buffers[1:], manifest)
    except Exception as exc:
        _raise_stale_artifact(
            cache_key,
            fingerprint=fingerprint,
            reason=str(exc),
        )

    return _FetchedMooncakeArtifact(
        buffers=buffers,
        manifest=manifest,
        artifact_id=artifact_id,
        header=header,
    )


def _receive_buffer_sizes(object_size: int) -> list[int]:
    envelope_size = _ARTIFACT_ENVELOPE_BYTES
    first_size = min(object_size, envelope_size)
    sizes = [first_size]
    remaining = object_size - first_size
    while remaining:
        size = min(remaining, _RECEIVE_SLICE_BYTES)
        sizes.append(size)
        remaining -= size
    return sizes


class _PayloadReader:
    def __init__(self, buffers: tuple[torch.Tensor, ...]) -> None:
        self._buffers = tuple(
            memoryview(buffer.numpy()).cast("B") for buffer in buffers
        )
        self._buffer_index = 0
        self._offset = 0

    def read(self, length: int) -> bytes:
        if length <= 0:
            raise ValueError(f"invalid Mooncake artifact chunk length: {length}")
        remaining = length
        parts: list[bytes] = []
        while remaining:
            if self._buffer_index >= len(self._buffers):
                raise RuntimeError(
                    f"Mooncake artifact payload is truncated by {remaining} bytes"
                )
            current = self._buffers[self._buffer_index]
            available = len(current) - self._offset
            take = min(remaining, available)
            if take:
                parts.append(bytes(current[self._offset : self._offset + take]))
                self._offset += take
                remaining -= take
            if self._offset == len(current):
                self._buffer_index += 1
                self._offset = 0
        return b"".join(parts)


def _validate_payload_checksums(
    buffers: tuple[torch.Tensor, ...],
    manifest: p2p_pb2.ArtifactManifest,
) -> None:
    reader = _PayloadReader(buffers)
    for chunk in manifest.chunks:
        data = reader.read(chunk.length)
        checksum = _crc32c_hex(data)
        if checksum != chunk.checksum:
            raise RuntimeError(
                "artifact chunk checksum mismatch: "
                f"chunk={chunk.chunk_index} expected={chunk.checksum} "
                f"actual={checksum}"
            )


def _write_payload_files(
    fetched: _FetchedMooncakeArtifact,
    target_header: p2p_pb2.GetArtifactManifestHeaderResponse,
) -> int:
    reader = _PayloadReader(fetched.buffers[1:])
    transferred_size = 0
    for chunk in fetched.manifest.chunks:
        data = reader.read(chunk.length)
        _write_file_range(
            Path(target_header.files[chunk.file_index].path),
            chunk.file_offset,
            data,
        )
        transferred_size += len(data)
    return transferred_size


def _put_artifact_object(
    store: _MooncakeNativeStore,
    cache_key: str,
    buffers: tuple[torch.Tensor, ...],
    transfer_name: str,
) -> None:
    rc = store.put_buffers(cache_key, buffers, soft_pin=True)
    if rc != 0:
        raise MooncakeArtifactCacheUnavailable(
            "Mooncake put artifact failed: "
            f"name={transfer_name} key={cache_key!r} "
            f"bytes={sum(_tensor_size(buffer) for buffer in buffers)} "
            f"result={_result_debug(rc)}"
        )


def _repair_artifact_object(
    store: _MooncakeNativeStore,
    cache_key: str,
    buffers: tuple[torch.Tensor, ...],
    *,
    transfer_name: str,
    stale_fingerprint: str,
) -> None:
    repair_key = _repair_key(cache_key)
    token = uuid4().hex
    lock_data = _encode_repair_lock(token)
    if not _acquire_repair_lock(
        store,
        repair_key,
        lock_data,
        token=token,
        transfer_name=transfer_name,
        cache_key=cache_key,
    ):
        logger.info(
            "[Mooncake] artifact repair delegated because another publisher "
            "owns the repair key: name=%s key=%s repair_key=%s",
            transfer_name,
            cache_key,
            repair_key,
        )
        return

    try:
        try:
            current = _fetch_artifact_object(store, cache_key)
        except MooncakeArtifactCacheStale as exc:
            if exc.fingerprint != stale_fingerprint:
                logger.warning(
                    "[Mooncake] artifact repair skipped because the stored "
                    "object changed after the miss: name=%s key=%s "
                    "expected_fingerprint=%s actual_fingerprint=%s",
                    transfer_name,
                    cache_key,
                    stale_fingerprint,
                    exc.fingerprint,
                )
                return
            _remove_object_with_retry(store, cache_key)
        except MooncakeArtifactCacheMiss:
            current = None
        else:
            logger.info(
                "[Mooncake] artifact repair skipped because another publisher "
                "already installed a valid object: name=%s key=%s artifact_id=%s",
                transfer_name,
                cache_key,
                current.artifact_id,
            )
            return
        _put_artifact_object(store, cache_key, buffers, transfer_name)
    finally:
        _release_repair_lock(store, repair_key, token)


def _encode_repair_lock(token: str) -> bytes:
    payload = json.dumps(
        {
            "token": token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": time.time(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    header = _REPAIR_LOCK_HEADER.pack(_REPAIR_LOCK_MAGIC, len(payload))
    if len(header) + len(payload) > _REPAIR_LOCK_BYTES:
        raise ValueError("Mooncake repair marker is too large")
    return header + payload + bytes(_REPAIR_LOCK_BYTES - len(header) - len(payload))


def _decode_repair_lock(data: bytes) -> _RepairLock | None:
    try:
        if len(data) != _REPAIR_LOCK_BYTES:
            return None
        magic, payload_length = _REPAIR_LOCK_HEADER.unpack_from(data)
        if magic != _REPAIR_LOCK_MAGIC:
            return None
        start = _REPAIR_LOCK_HEADER.size
        end = start + payload_length
        if end > len(data):
            return None
        payload = json.loads(data[start:end].decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        token = payload.get("token")
        pid = payload.get("pid")
        hostname = payload.get("hostname")
        created_at = payload.get("created_at")
        if not isinstance(token, str) or not token:
            return None
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return None
        if not isinstance(hostname, str) or not hostname:
            return None
        if (
            not isinstance(created_at, (int, float))
            or isinstance(created_at, bool)
            or not isfinite(float(created_at))
            or float(created_at) <= 0
        ):
            return None
        return _RepairLock(
            token=token,
            pid=pid,
            hostname=hostname,
            created_at=float(created_at),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, struct.error, TypeError):
        return None


def _acquire_repair_lock(
    store: _MooncakeNativeStore,
    repair_key: str,
    lock_data: bytes,
    *,
    token: str,
    transfer_name: str,
    cache_key: str,
) -> bool:
    """Acquire a repair marker, reclaiming one older than ten minutes.

    Mooncake objects do not expose compare-and-delete. The repeated read and
    token checks below make stale takeover best-effort rather than a strict
    distributed lease. A repair that legitimately runs for more than ten
    minutes can be superseded; every repair still revalidates the corrupt
    artifact fingerprint before replacing it.
    """

    _put_repair_lock(store, repair_key, lock_data, transfer_name=transfer_name)
    observed_data = store.get_bytes(repair_key)
    observed_lock = (
        _decode_repair_lock(observed_data) if observed_data is not None else None
    )
    if observed_lock is not None and observed_lock.token == token:
        return True
    if observed_data is None:
        logger.warning(
            "[Mooncake] repair marker disappeared during acquisition; "
            "leaving repair to another publisher: name=%s key=%s repair_key=%s",
            transfer_name,
            cache_key,
            repair_key,
        )
        return False
    if observed_lock is None:
        logger.warning(
            "[Mooncake] repair marker is invalid and will not be reclaimed "
            "automatically: name=%s key=%s repair_key=%s",
            transfer_name,
            cache_key,
            repair_key,
        )
        return False

    age = time.time() - observed_lock.created_at
    if age < _REPAIR_LOCK_STALE_SECONDS:
        return False

    confirmed_data = store.get_bytes(repair_key)
    if confirmed_data != observed_data:
        logger.info(
            "[Mooncake] stale repair marker changed before takeover; "
            "leaving it untouched: name=%s key=%s repair_key=%s",
            transfer_name,
            cache_key,
            repair_key,
        )
        return False

    logger.warning(
        "[Mooncake] reclaiming stale artifact repair marker: name=%s key=%s "
        "repair_key=%s owner_host=%s owner_pid=%s age=%.1fs timeout=%ss",
        transfer_name,
        cache_key,
        repair_key,
        observed_lock.hostname,
        observed_lock.pid,
        age,
        _REPAIR_LOCK_STALE_SECONDS,
    )
    _remove_object_with_retry(store, repair_key)
    _put_repair_lock(store, repair_key, lock_data, transfer_name=transfer_name)
    acquired_data = store.get_bytes(repair_key)
    acquired_lock = (
        _decode_repair_lock(acquired_data) if acquired_data is not None else None
    )
    return acquired_lock is not None and acquired_lock.token == token


def _put_repair_lock(
    store: _MooncakeNativeStore,
    repair_key: str,
    lock_data: bytes,
    *,
    transfer_name: str,
) -> None:
    # Keep this tiny coordination object protected from normal pressure
    # eviction for at least the store's soft-pin window.  Explicit removal is
    # still allowed, and stale-owner recovery below takes over after ten
    # minutes, so soft pinning does not turn the marker into a permanent lock.
    rc = store.put_bytes(repair_key, lock_data, soft_pin=True)
    # OBJECT_ALREADY_EXISTS (-705) means the repair key already exists.
    # Acquisition is decided by reading the marker and comparing its token
    # in _acquire_repair_lock().
    if rc not in (0, -705):
        raise MooncakeArtifactCacheUnavailable(
            "Mooncake put repair marker failed: "
            f"name={transfer_name} key={repair_key!r} "
            f"result={_result_debug(rc)}"
        )


def _release_repair_lock(
    store: _MooncakeNativeStore,
    repair_key: str,
    token: str,
) -> None:
    try:
        current = store.get_bytes(repair_key)
        if current is None:
            return
        current_lock = _decode_repair_lock(current)
        if current_lock is None or current_lock.token != token:
            logger.warning(
                "[Mooncake] repair key ownership changed before release; "
                "leaving it untouched: key=%s",
                repair_key,
            )
            return
        _remove_object_with_retry(store, repair_key)
    except Exception:
        logger.warning(
            "[Mooncake] failed to release repair key; manual cleanup may be "
            "required: key=%s",
            repair_key,
            exc_info=True,
        )


def _raise_stale_artifact(
    cache_key: str,
    *,
    fingerprint: str,
    reason: str,
) -> NoReturn:
    """Report a corrupt object without deleting potentially newer data."""
    raise MooncakeArtifactCacheStale(
        f"Mooncake artifact is stale: key={cache_key} reason={reason}",
        cache_key=cache_key,
        fingerprint=fingerprint,
    )


def _remove_object_with_retry(store: _MooncakeNativeStore, key: str) -> None:
    retries = max(0, int(envs.MX_ARTIFACT_MOONCAKE_DELETE_RETRIES))
    delay = max(0.0, float(envs.MX_ARTIFACT_MOONCAKE_DELETE_RETRY_DELAY_SECS))
    # OBJECT_NOT_FOUND is -704; OBJECT_HAS_LEASE is -706.
    for attempt in range(retries + 1):
        ret = store.remove(key)
        if ret in (0, -704):
            return
        if ret != -706 or attempt >= retries:
            raise RuntimeError(
                f"Mooncake remove returned {ret} for key {key!r}"
            )
        time.sleep(delay)


def _close_store(store) -> None:
    try:
        store.close()
    except Exception:
        logger.debug("Failed to close Mooncake artifact store", exc_info=True)


def _result_debug(result: int | None) -> str:
    names = {
        -704: "OBJECT_NOT_FOUND",
        -705: "OBJECT_ALREADY_EXISTS",
        -706: "OBJECT_HAS_LEASE",
    }
    if result is None:
        return "EMPTY_RESULT"
    return f"{names.get(result, 'OK' if result >= 0 else 'ERROR')}({result})"


def _buffer_descriptors(
    buffers: tuple[torch.Tensor, ...],
    *,
    key: str,
) -> tuple[list[int], list[int]]:
    if not buffers:
        raise ValueError(f"Mooncake object must have at least one buffer: {key}")
    pointers: list[int] = []
    sizes: list[int] = []
    for buffer in buffers:
        if (
            buffer.dtype != torch.uint8
            or buffer.device.type != "cpu"
            or not buffer.is_contiguous()
        ):
            raise ValueError(
                "Mooncake artifact buffers must be contiguous CPU uint8 tensors: "
                f"key={key!r} dtype={buffer.dtype} device={buffer.device}"
            )
        size = _tensor_size(buffer)
        if size <= 0:
            raise ValueError(f"Mooncake object contains an empty buffer: {key}")
        pointers.append(buffer.data_ptr())
        sizes.append(size)
    return pointers, sizes


def _tensor_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _buffers_fingerprint(buffers: tuple[torch.Tensor, ...]) -> str:
    digest = sha256()
    for buffer in buffers:
        digest.update(memoryview(buffer.numpy()).cast("B"))
    return digest.hexdigest()


def _bytes_to_tensor(data: bytes) -> torch.Tensor:
    if not data:
        raise ValueError("Mooncake remote tensor API does not support empty tensors")
    return torch.frombuffer(bytearray(data), dtype=torch.uint8)


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.uint8:
        raise RuntimeError(f"Mooncake artifact tensor dtype mismatch: {tensor.dtype}")
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _read_file_range(path: Path, offset: int, length: int) -> bytes:
    if length <= 0:
        raise ValueError(f"invalid Mooncake artifact chunk length: {length}")
    with path.open("rb") as file:
        file.seek(offset)
        data = file.read(length)
    if len(data) != length:
        raise OSError(
            f"short read for Mooncake artifact chunk: {path} "
            f"offset={offset} length={length} got={len(data)}"
        )
    return data


def _write_file_range(path: Path, offset: int, data: bytes) -> None:
    with path.open("r+b") as file:
        file.seek(offset)
        file.write(data)


def _header_from_manifest(
    manifest: p2p_pb2.ArtifactManifest,
    *,
    artifact_id: str,
) -> p2p_pb2.GetArtifactManifestHeaderResponse:
    return p2p_pb2.GetArtifactManifestHeaderResponse(
        artifact_id=artifact_id,
        manifest_version=manifest.manifest_version,
        mx_source_type=manifest.mx_source_type,
        total_size=sum(file.size for file in manifest.files),
        file_count=len(manifest.files),
        chunk_count=len(manifest.chunks),
        chunk_size=manifest.chunk_size,
        files=manifest.files,
    )


def _header_with_transfer_target_paths(
    transfer: ArtifactTransfer,
    header: p2p_pb2.GetArtifactManifestHeaderResponse,
) -> p2p_pb2.GetArtifactManifestHeaderResponse:
    target_paths = transfer.target_file_paths()
    from .artifact_transfer import _header_with_target_file_paths

    return _header_with_target_file_paths(header, target_paths)


def _prepare_target_files(files) -> None:
    from .artifact_transfer import _prepare_target_files

    _prepare_target_files(files)


def _cleanup_target_files(files) -> None:
    from .artifact_transfer import _cleanup_target_files

    _cleanup_target_files(files)


def _validate_manifest_header(
    header: p2p_pb2.GetArtifactManifestHeaderResponse,
    chunks: list[p2p_pb2.ArtifactManifestChunk],
    *,
    expected_artifact_id: str,
) -> None:
    from .artifact_transfer import _validate_fetched_artifact_manifest

    _validate_fetched_artifact_manifest(header, chunks, expected_artifact_id)


def _gbps(size_bytes: int, elapsed_secs: float) -> float:
    if elapsed_secs <= 0:
        return 0.0
    return size_bytes * 8 / elapsed_secs / 1_000_000_000
