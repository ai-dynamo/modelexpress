# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side Mooncake cache for file-backed framework artifacts.

This module deliberately does not talk to the ModelExpress server. Mooncake is
used as a best-effort external cache keyed by the same compatibility identity
used by the existing artifact path; P2P metadata remains the fallback source of
truth for live workers.
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
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import torch

from .. import envs, p2p_pb2
from ..mooncake_env import mx_mc_env_override
from .artifact_manifest import _crc32c_hex, artifact_manifest_id

if TYPE_CHECKING:
    from .artifact_transfer import ArtifactBundle, P2PArtifactTransfer

logger = logging.getLogger("modelexpress.metadata.mooncake_artifact_cache")

_store_lock = threading.RLock()
_shared_store: _MooncakeNativeStore | None = None
_shared_store_config: _MooncakeNativeConfig | None = None
_store_atexit_registered = False


class MooncakeArtifactCacheMiss(LookupError):
    """Raised when a deterministic artifact cache key is not present."""


class MooncakeArtifactCacheUnavailable(RuntimeError):
    """Raised when Mooncake native store is not installed or configured."""


_MANIFEST_FRAME = struct.Struct("!Q")
_MANIFEST_ENVELOPE_HEADER = struct.Struct("!8sHHQ")
_MANIFEST_ENVELOPE_MAGIC = b"MXMCGEN1"


@dataclass(frozen=True)
class _MooncakeManifestEnvelope:
    generation_id: str
    artifact_id: str
    manifest_bytes: bytes


def install_from_mooncake(
    transfer: P2PArtifactTransfer,
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
    logger.info(
        "[Mooncake] artifact query start: name=%s key=%s node_rank=%s "
        "accelerator=%s source_identity=%s",
        transfer.name,
        cache_key,
        node_rank,
        accelerator,
        _identity_debug(identity),
    )
    with _store_session() as store:
        manifest_query_start = time.perf_counter()
        manifest_bytes = store.get_bytes(_manifest_key(cache_key))
        logger.info(
            "[Mooncake] manifest query result: name=%s key=%s result=%s "
            "bytes=%s elapsed=%.3fs",
            transfer.name,
            _manifest_key(cache_key),
            "HIT" if manifest_bytes is not None else "MISS_OR_EVICTED",
            len(manifest_bytes) if manifest_bytes is not None else 0,
            time.perf_counter() - manifest_query_start,
        )
        if manifest_bytes is None:
            logger.warning(
                "[Mooncake] artifact manifest unavailable; it may have expired "
                "or been evicted: name=%s key=%s source_identity=%s",
                transfer.name,
                cache_key,
                _identity_debug(identity),
            )
            raise MooncakeArtifactCacheMiss(f"Mooncake artifact miss: {cache_key}")
        envelope = _decode_manifest_envelope(manifest_bytes)
        manifest = p2p_pb2.ArtifactManifest()
        manifest.ParseFromString(envelope.manifest_bytes)
        artifact_id = artifact_manifest_id(manifest)
        if envelope.artifact_id and envelope.artifact_id != artifact_id:
            logger.warning(
                "[Mooncake] manifest artifact_id mismatch: name=%s key=%s "
                "envelope_artifact_id=%s computed_artifact_id=%s; treating as stale",
                transfer.name,
                cache_key,
                envelope.artifact_id,
                artifact_id,
            )
            raise MooncakeArtifactCacheMiss(
                f"Mooncake manifest artifact id mismatch for {cache_key}"
            )
        # Old manifests predate generations and stored chunks under artifact_id.
        generation_id = envelope.generation_id or artifact_id
        logger.info(
            "[Mooncake] manifest decoded: name=%s key=%s artifact_id=%s "
            "generation_id=%s files=%d chunks=%d total_size=%d",
            transfer.name,
            cache_key,
            artifact_id,
            generation_id,
            len(manifest.files),
            len(manifest.chunks),
            sum(file.size for file in manifest.files),
        )
        header = _header_from_manifest(manifest, artifact_id=artifact_id)
        _validate_manifest_header(
            header,
            manifest.chunks,
            expected_artifact_id=artifact_id,
        )

        target_header = _header_with_transfer_target_paths(transfer, header)
        _prepare_target_files(target_header.files)
        try:
            transferred_size = 0
            for chunk in manifest.chunks:
                chunk_start = time.perf_counter()
                chunk_key = _chunk_key(
                    cache_key, generation_id, chunk.chunk_index
                )
                data = store.get_bytes(
                    chunk_key,
                    expected_size=chunk.length,
                )
                if data is None:
                    logger.warning(
                        "[Mooncake] artifact chunk unavailable; it may have "
                        "expired or been evicted: name=%s key=%s chunk=%d "
                        "generation_id=%s expected_bytes=%d",
                        transfer.name,
                        chunk_key,
                        chunk.chunk_index,
                        generation_id,
                        chunk.length,
                    )
                    raise MooncakeArtifactCacheMiss(
                        "Mooncake artifact chunk miss: "
                        f"{cache_key}/{artifact_id}/chunk/{chunk.chunk_index}"
                    )
                if len(data) != chunk.length:
                    raise RuntimeError(
                        "Mooncake artifact chunk size mismatch: "
                        f"{len(data)} != {chunk.length}"
                    )
                checksum = _crc32c_hex(data)
                if checksum != chunk.checksum:
                    logger.warning(
                        "[Mooncake] artifact chunk checksum mismatch: name=%s "
                        "key=%s chunk=%d expected=%s actual=%s; treating as stale",
                        transfer.name,
                        chunk_key,
                        chunk.chunk_index,
                        chunk.checksum,
                        checksum,
                    )
                    raise RuntimeError(
                        "Mooncake artifact chunk checksum mismatch: "
                        f"expected {chunk.checksum}, got {checksum}"
                    )
                _write_file_range(
                    Path(target_header.files[chunk.file_index].path),
                    chunk.file_offset,
                    data,
                )
                transferred_size += len(data)
                chunk_elapsed = time.perf_counter() - chunk_start
                logger.debug(
                    "[TIMING] Mooncake artifact chunk fetch: name=%s chunk=%d "
                    "bytes=%d elapsed=%.3fs throughput=%.2f Gbps",
                    transfer.name,
                    chunk.chunk_index,
                    len(data),
                    chunk_elapsed,
                    _gbps(len(data), chunk_elapsed),
                )
        except Exception:
            _cleanup_target_files(target_header.files)
            raise

        elapsed = time.perf_counter() - start
        logger.info(
            "[TIMING] Mooncake artifact fetch complete: name=%s artifact_id=%s "
            "generation_id=%s "
            "key=%s files=%d chunks=%d size=%.2f MiB elapsed=%.3fs throughput=%.2f Gbps",
            transfer.name,
            artifact_id,
            generation_id,
            cache_key,
            len(header.files),
            len(manifest.chunks),
            transferred_size / (1024 * 1024),
            elapsed,
            _gbps(transferred_size, elapsed),
        )
        return target_header


def publish_to_mooncake(
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
    bundle: ArtifactBundle,
    *,
    node_rank: int,
    accelerator: str,
) -> str:
    """Publish one prepared artifact bundle to Mooncake.

    Chunks are written under a fresh generation before the fixed manifest is
    switched. The manifest key acts as the commit marker. Old generations are
    intentionally retained so targets that already read the previous manifest
    can finish without being affected by this publication.
    """
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
    with _store_session() as store:
        generation_id = uuid4().hex
        total_size = 0
        files = list(bundle.manifest.files)
        for chunk in bundle.manifest.chunks:
            chunk_start = time.perf_counter()
            chunk_key = _chunk_key(cache_key, generation_id, chunk.chunk_index)
            path = Path(files[chunk.file_index].path).resolve(strict=True)
            data = _read_file_range(path, chunk.file_offset, chunk.length)
            if len(data) != chunk.length:
                raise RuntimeError(
                    f"artifact chunk size changed while publishing: {path}"
                )
            checksum = _crc32c_hex(data)
            if checksum != chunk.checksum:
                raise RuntimeError(
                    f"artifact chunk checksum changed while publishing: {path}"
                )
            rc = store.put_bytes(
                chunk_key,
                data,
            )
            if rc != 0:
                raise MooncakeArtifactCacheUnavailable(
                    "Mooncake put chunk failed: "
                    f"name={transfer.name} key={chunk_key!r} "
                    f"generation_id={generation_id} chunk={chunk.chunk_index} "
                    f"bytes={len(data)} result={_result_debug(rc)}"
                )
            total_size += len(data)
            chunk_elapsed = time.perf_counter() - chunk_start
            logger.debug(
                "[TIMING] Mooncake artifact chunk publish: name=%s chunk=%d "
                "bytes=%d key=%s elapsed=%.3fs throughput=%.2f Gbps",
                transfer.name,
                chunk.chunk_index,
                len(data),
                chunk_key,
                chunk_elapsed,
                _gbps(len(data), chunk_elapsed),
            )

        # Remove the old commit marker. Mooncake keys are immutable, so the
        # marker must be deleted before the new manifest can be installed.
        remove_start = time.perf_counter()
        _remove_manifest_with_retry(store, _manifest_key(cache_key))
        logger.debug(
            "[TIMING] Mooncake old manifest removed: name=%s key=%s elapsed=%.3fs",
            transfer.name,
            _manifest_key(cache_key),
            time.perf_counter() - remove_start,
        )
        manifest_bytes = _encode_manifest_envelope(
            generation_id=generation_id,
            artifact_id=bundle.artifact_id,
            manifest_bytes=bundle.manifest.SerializeToString(),
        )
        rc = store.put_bytes(
            _manifest_key(cache_key),
            manifest_bytes,
        )
        if rc != 0:
            raise MooncakeArtifactCacheUnavailable(
                "Mooncake put manifest failed: "
                f"name={transfer.name} key={_manifest_key(cache_key)!r} "
                f"generation_id={generation_id} bytes={len(manifest_bytes)} "
                f"result={_result_debug(rc)}"
            )
        logger.info(
            "[Mooncake] manifest committed: name=%s key=%s artifact_id=%s "
            "generation_id=%s bytes=%d",
            transfer.name,
            _manifest_key(cache_key),
            bundle.artifact_id,
            generation_id,
            len(manifest_bytes),
        )

        elapsed = time.perf_counter() - start
        logger.info(
            "[TIMING] Mooncake artifact publish complete: name=%s artifact_id=%s "
            "generation_id=%s "
            "key=%s files=%d chunks=%d size=%.2f MiB elapsed=%.3fs throughput=%.2f Gbps",
            transfer.name,
            bundle.artifact_id,
            generation_id,
            cache_key,
            len(bundle.manifest.files),
            len(bundle.manifest.chunks),
            total_size / (1024 * 1024),
            elapsed,
            _gbps(total_size, elapsed),
        )
        return cache_key


def compute_artifact_cache_key(
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
    *,
    node_rank: int,
    accelerator: str,
) -> str:
    """Generate the deterministic Mooncake key for one artifact.

    The digest is derived from the serialized source identity, artifact type,
    artifact name, node rank, and accelerator backend. The readable key prefix
    identifies the framework, model, and cache type; callers append
    ``/manifest`` or a generation and chunk suffix when addressing individual
    Mooncake objects.
    """
    from .artifact_lifecycle import _identity_bytes

    namespace = envs.MX_ARTIFACT_MOONCAKE_NAMESPACE or "modelexpress/artifacts"
    digest = sha256()
    digest.update(_identity_bytes(identity))
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


def _manifest_key(cache_key: str) -> str:
    return f"{cache_key}/manifest"


def _chunk_key(cache_key: str, generation_id: str, chunk_index: int) -> str:
    return f"{cache_key}/{generation_id}/chunk/{chunk_index}"


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
    """Small bytes-oriented wrapper around mooncake.store.MooncakeDistributedStore."""

    def __init__(self, config: _MooncakeNativeConfig) -> None:
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except Exception as exc:
            raise MooncakeArtifactCacheUnavailable(
                "mooncake.store is not importable"
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
                "MooncakeDistributedStore.setup failed: "
                f"ret={ret}, local_hostname={config.local_hostname}, "
                f"metadata_server={config.metadata_server}, "
                f"master_server={config.master_server}, "
                f"protocol={config.protocol}, device_name={config.device_name}"
            )
        self._replicate_config = ReplicateConfig()
        self._replicate_config.with_soft_pin = bool(
            envs.MX_ARTIFACT_MOONCAKE_ENABLE_SOFT_PIN
        )

    def get_bytes(self, key: str, expected_size: int | None = None) -> bytes | None:
        size = expected_size or int(envs.MX_ARTIFACT_MOONCAKE_MANIFEST_BYTES)
        if size <= 0:
            raise ValueError(f"invalid Mooncake get size for {key}: {size}")
        tensor = torch.empty(size, dtype=torch.uint8, device="cpu")
        buffer_ptr = tensor.data_ptr()
        buffer_size = tensor.numel() * tensor.element_size()
        self._register_buffer(buffer_ptr, buffer_size, key=key)
        try:
            result = self._store.batch_get_into_multi_buffers(
                [key],
                [[buffer_ptr]],
                [[buffer_size]],
            )
        finally:
            self._unregister_buffer(buffer_ptr, key=key)
        logger.debug(
            "[Mooncake] get completed: key=%s expected_bytes=%d result=%s",
            key,
            buffer_size,
            _result_debug(result[0] if result else None),
        )
        if not result:
            raise MooncakeArtifactCacheUnavailable(
                f"Mooncake get returned no result for key {key!r}"
            )
        result_code = int(result[0])
        # Mooncake uses OBJECT_NOT_FOUND for an absent or evicted key.  Do not
        # turn transport failures into cache misses: callers use that
        # distinction to decide whether to report a cache miss or a broken data
        # plane before falling back to P2P.
        if result_code == -704:
            return None
        if result_code < 0:
            logger.warning(
                "[Mooncake] get failed: key=%s expected_bytes=%d result=%s",
                key,
                buffer_size,
                _result_debug(result_code),
            )
            raise MooncakeArtifactCacheUnavailable(
                "Mooncake get failed: "
                f"key={key!r} expected_bytes={buffer_size} "
                f"result={_result_debug(result_code)}"
            )
        return _tensor_to_bytes(tensor)

    def put_bytes(self, key: str, data: bytes) -> int:
        tensor = _bytes_to_tensor(data)
        buffer_ptr = tensor.data_ptr()
        buffer_size = tensor.numel() * tensor.element_size()
        self._register_buffer(buffer_ptr, buffer_size, key=key)
        try:
            result = self._store.batch_put_from_multi_buffers(
                [key],
                [[buffer_ptr]],
                [[buffer_size]],
                self._replicate_config,
            )
        finally:
            self._unregister_buffer(buffer_ptr, key=key)
        logger.debug(
            "[Mooncake] put completed: key=%s bytes=%d result=%s",
            key,
            buffer_size,
            _result_debug(result[0] if result else None),
        )
        if not result:
            return -1
        return int(result[0])

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
                "[Mooncake] native store initialized: protocol=%s device=%s "
                "global_segment_bytes=%d local_buffer_bytes=%d",
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
    config_path = (
        os.getenv("MX_MOONCAKE_CONFIG_PATH")
        or os.getenv("MOONCAKE_CONFIG_PATH", "")
    ).strip()
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
    logger.info(
        "[MX][Mooncake] native config: config_path=%s local_hostname=%s "
        "metadata_server=%s master_server=%s protocol=%s device_name=%s "
        "global_segment_size=%d local_buffer_size=%d",
        config_path or "<env>",
        config.local_hostname,
        config.metadata_server,
        config.master_server,
        config.protocol,
        config.device_name,
        config.global_segment_size,
        config.local_buffer_size,
    )
    return config


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


def _encode_manifest_frame(data: bytes) -> bytes:
    limit = int(envs.MX_ARTIFACT_MOONCAKE_MANIFEST_BYTES)
    frame_size = _MANIFEST_FRAME.size + len(data)
    if frame_size > limit:
        raise ValueError(
            "Mooncake artifact manifest is too large: "
            f"{frame_size} bytes > MX_ARTIFACT_MOONCAKE_MANIFEST_BYTES={limit}"
        )
    return _MANIFEST_FRAME.pack(len(data)) + data + bytes(limit - frame_size)


def _encode_manifest_envelope(
    *, generation_id: str, artifact_id: str, manifest_bytes: bytes
) -> bytes:
    generation_bytes = generation_id.encode("ascii")
    artifact_bytes = artifact_id.encode("ascii")
    header = _MANIFEST_ENVELOPE_HEADER.pack(
        _MANIFEST_ENVELOPE_MAGIC,
        len(generation_bytes),
        len(artifact_bytes),
        len(manifest_bytes),
    )
    payload = header + generation_bytes + artifact_bytes + manifest_bytes
    limit = int(envs.MX_ARTIFACT_MOONCAKE_MANIFEST_BYTES)
    if len(payload) > limit:
        raise ValueError(
            "Mooncake artifact manifest envelope is too large: "
            f"{len(payload)} bytes > MX_ARTIFACT_MOONCAKE_MANIFEST_BYTES={limit}"
        )
    return payload + bytes(limit - len(payload))


def _decode_manifest_envelope(data: bytes) -> _MooncakeManifestEnvelope:
    # Backward compatibility with manifests written before generations.
    if not data.startswith(_MANIFEST_ENVELOPE_MAGIC):
        manifest_bytes = _decode_manifest_frame(data)
        manifest = p2p_pb2.ArtifactManifest()
        manifest.ParseFromString(manifest_bytes)
        artifact_id = artifact_manifest_id(manifest)
        return _MooncakeManifestEnvelope(
            generation_id=artifact_id,
            artifact_id=artifact_id,
            manifest_bytes=manifest_bytes,
        )

    header_size = _MANIFEST_ENVELOPE_HEADER.size
    if len(data) < header_size:
        raise RuntimeError("Mooncake manifest envelope is truncated")
    magic, generation_len, artifact_len, manifest_len = (
        _MANIFEST_ENVELOPE_HEADER.unpack_from(data)
    )
    if magic != _MANIFEST_ENVELOPE_MAGIC:
        raise RuntimeError("Mooncake manifest envelope magic mismatch")
    start = header_size
    generation_end = start + generation_len
    artifact_end = generation_end + artifact_len
    manifest_end = artifact_end + manifest_len
    if manifest_end > len(data):
        raise RuntimeError("Mooncake manifest envelope length exceeds fetched bytes")
    try:
        generation_id = data[start:generation_end].decode("ascii")
        artifact_id = data[generation_end:artifact_end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Mooncake manifest envelope contains non-ASCII ids") from exc
    if not generation_id or not artifact_id:
        raise RuntimeError("Mooncake manifest envelope contains an empty id")
    return _MooncakeManifestEnvelope(
        generation_id=generation_id,
        artifact_id=artifact_id,
        manifest_bytes=data[artifact_end:manifest_end],
    )


def _decode_manifest_frame(data: bytes) -> bytes:
    if len(data) < _MANIFEST_FRAME.size:
        raise RuntimeError("Mooncake artifact manifest frame is truncated")
    (length,) = _MANIFEST_FRAME.unpack_from(data)
    end = _MANIFEST_FRAME.size + length
    if end > len(data):
        raise RuntimeError(
            "Mooncake artifact manifest frame length exceeds fetched bytes: "
            f"{end} > {len(data)}"
        )
    return data[_MANIFEST_FRAME.size : end]


def _remove_manifest_with_retry(store: _MooncakeNativeStore, key: str) -> None:
    retries = max(0, int(envs.MX_ARTIFACT_MOONCAKE_DELETE_RETRIES))
    delay = max(0.0, float(envs.MX_ARTIFACT_MOONCAKE_DELETE_RETRY_DELAY_SECS))
    # OBJECT_NOT_FOUND is -704; OBJECT_HAS_LEASE is -706.
    for attempt in range(retries + 1):
        ret = store.remove(key)
        if ret in (0, -704):
            return
        if ret != -706 or attempt >= retries:
            raise RuntimeError(
                f"Mooncake remove manifest returned {ret} for key {key!r}"
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
        -706: "OBJECT_HAS_LEASE",
    }
    if result is None:
        return "EMPTY_RESULT"
    return f"{names.get(result, 'OK' if result >= 0 else 'ERROR')}({result})"


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
    transfer: P2PArtifactTransfer,
    header: p2p_pb2.GetArtifactManifestHeaderResponse,
) -> p2p_pb2.GetArtifactManifestHeaderResponse:
    target_paths = getattr(transfer, "_target_tar_paths")()
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
