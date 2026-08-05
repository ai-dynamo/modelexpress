# SPDX-License-Identifier: Apache-2.0

"""Canonical V0 receiver preparation and host-local materialization helpers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import struct
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from modelexpress.refit.api import PreparedUpdate
from modelexpress.refit.manifest import (
    ChangeState,
    DeltaTransferMethod,
    RevisionLifecycleState,
)
from modelexpress.refit.source.canonical import (
    CanonicalDeltaError,
    FilesystemCanonicalBaseStore,
    reconstruct_canonical_delta,
)


@dataclass(frozen=True)
class PreparedPayload:
    """Host-private materialization paired with one receiver's immutable identity."""

    identity: Any
    model_path: Path
    canonical_store: Any | None = None
    canonical_snapshot: Any | None = None
    noop: bool = False
    materialized_sha256: str | None = None
    materialized_size: int | None = None
    materialized_device: int | None = None
    materialized_inode: int | None = None
    materialized_mtime_ns: int | None = None
    materialized_ctime_ns: int | None = None
    preparation_lock_path: Path | None = None


_SAFETENSORS_DTYPES = {
    "bool": "BOOL",
    "uint8": "U8",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "float16": "F16",
    "bfloat16": "BF16",
    "float32": "F32",
    "float64": "F64",
    "complex64": "C64",
    "complex128": "C128",
    "float8_e4m3fn": "F8_E4M3",
    "float8_e5m2": "F8_E5M2",
}


def modelexpress_model_cache_root(cache_root: str | Path, model_id: str) -> Path:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("ModelExpress model_id must be nonempty")
    namespace = quote(model_id, safe="")
    if namespace in {".", ".."}:
        namespace = "".join(f"%{byte:02X}" for byte in model_id.encode("utf-8"))
    return Path(cache_root) / namespace


@contextmanager
def exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _materialized_file_attestation(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("host-local materialization must be a regular file")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise ValueError("host-local materialization changed during attestation")
        return {
            "hf_ctime_ns": after.st_ctime_ns,
            "hf_device": after.st_dev,
            "hf_inode": after.st_ino,
            "hf_mtime_ns": after.st_mtime_ns,
            "hf_sha256": digest.hexdigest(),
            "hf_size": total,
        }
    finally:
        os.close(descriptor)


def materialized_file_identity(path: Path) -> dict[str, int]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode):
            raise ValueError("host-local materialization must be a regular file")
        return {
            "hf_ctime_ns": current.st_ctime_ns,
            "hf_device": current.st_dev,
            "hf_inode": current.st_ino,
            "hf_mtime_ns": current.st_mtime_ns,
            "hf_size": current.st_size,
        }
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def seed_base_from_safetensors(
    store: FilesystemCanonicalBaseStore,
    version: str,
    files: list[str | Path],
):
    """Import an HF safetensors checkpoint into the exact-base store tensor by tensor."""

    from safetensors import safe_open

    if not files:
        raise ValueError("V0 initial base requires safetensors files")
    writer = store.begin_snapshot(version)
    try:
        with ExitStack() as stack:
            handles = [
                stack.enter_context(
                    safe_open(str(Path(path)), framework="pt", device="cpu")
                )
                for path in sorted(map(Path, files))
            ]
            owners: dict[str, Any] = {}
            for handle in handles:
                for name in handle.keys():
                    if name in owners:
                        raise ValueError(f"duplicate safetensors key {name!r}")
                    owners[name] = handle
            for name in sorted(owners):
                writer.add_tensor(name, owners[name].get_tensor(name))
        return writer.finalize()
    except BaseException:
        writer.abort()
        raise


def materialize_snapshot_to_safetensors(
    store: Any, snapshot: Any, target_dir: str | Path
) -> Path:
    """Write a canonical snapshot as one bounded, loader-compatible safetensors file."""

    destination = Path(target_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tensors = sorted(snapshot.tensors, key=lambda item: item.name)
    header: dict[str, Any] = {}
    offset = 0
    for tensor in tensors:
        try:
            dtype = _SAFETENSORS_DTYPES[tensor.dtype]
        except KeyError as exc:
            raise ValueError(
                f"V0 cannot materialize safetensors dtype {tensor.dtype!r}"
            ) from exc
        end = offset + tensor.byte_size
        header[tensor.name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    encoded = json.dumps(
        header, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    target = destination / "model.safetensors"
    temporary = destination / f".{target.name}.{uuid.uuid4().hex}.partial"
    try:
        with temporary.open("xb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            for tensor in tensors:
                handle.write(store.read_tensor_bytes(snapshot, tensor.name))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o444)
        _fsync_directory(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_modelexpress_s3_transport(server_args: Any, *, client: Any = None):
    """Build the only Phase 2 production transport: immutable S3 objects."""

    from modelexpress.refit.transport.s3 import S3CanonicalTransport

    bucket = server_args.modelexpress_delta_s3_bucket
    if not isinstance(bucket, str) or not bucket.strip():
        raise ValueError("ModelExpress Phase 2 requires an S3 bucket")
    if client is None:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=server_args.modelexpress_delta_s3_endpoint,
            config=Config(
                connect_timeout=5.0,
                read_timeout=5.0,
                retries={"mode": "standard", "total_max_attempts": 3},
                tcp_keepalive=True,
            ),
        )
    return S3CanonicalTransport(
        bucket=bucket.strip(),
        prefix=server_args.modelexpress_delta_s3_prefix,
        client=client,
    )


class CanonicalV0Preparer:
    """Prepare a CANONICAL target fetched from object storage.

    ``base_store`` and prepared targets are host-local, model-ID-namespaced
    reconstruction state shared by co-located ranks. Production payload transport
    is the configured S3 transport.
    """

    def __init__(
        self,
        *,
        model_id: str,
        receiver_incarnation: str,
        model_generation: Callable[[], int],
        base_store: FilesystemCanonicalBaseStore,
        base_snapshot: Callable[[], Any],
        target_root: str | Path,
        catalog: Any,
        transport: Any,
    ) -> None:
        self._model_id = model_id
        self._receiver_incarnation = receiver_incarnation
        self._model_generation = model_generation
        self._base_store = base_store
        self._base_snapshot = base_snapshot
        self._target_root = Path(target_root)
        self._catalog = catalog
        self._transport = transport

    def prepare(self, version: str) -> PreparedPayload:
        record = self._catalog.get_revision(self._model_id, version)
        if record.state not in {
            RevisionLifecycleState.READY,
            RevisionLifecycleState.COMMITTED,
        }:
            raise ValueError("target revision is not ready")
        manifest = record.manifest
        base = self._base_store.attest_snapshot(self._base_snapshot())
        delta = self._validate_manifest(manifest, version, base)
        lock_path = self._target_root.parent / "prepare.lock"
        with exclusive_file_lock(lock_path):
            base = self._base_store.attest_snapshot(self._base_snapshot())
            delta = self._validate_manifest(manifest, version, base)
            return self._prepare_locked(manifest, delta, base)

    def _validate_manifest(self, manifest: Any, version: str, base: Any) -> Any:
        if manifest.model_id != self._model_id or manifest.version != version:
            raise ValueError("catalog returned the wrong target revision")
        if manifest.transfer_method is not DeltaTransferMethod.CANONICAL:
            raise ValueError("V0 supports CANONICAL revisions only")
        if manifest.base_version != base.version:
            raise ValueError("target revision does not use the exact installed base")
        if manifest.base_digest != base.target_digest:
            raise ValueError(
                "target revision base digest does not match installed base"
            )
        if manifest.format_digest != base.format_digest:
            raise ValueError("target revision format does not match installed base")
        if len(manifest.ranks) != 1 or manifest.ranks[0].trainer_rank != 0:
            raise ValueError("CANONICAL V0 requires exactly one trainer-rank-0 entry")
        delta = manifest.ranks[0].delta
        if delta is None:
            raise ValueError("CANONICAL V0 requires one rank delta")
        if delta.change_state is ChangeState.CLEAN:
            if delta.location is not None or delta.checksum is not None:
                raise ValueError("clean CANONICAL revision cannot reference an object")
            if manifest.target_digest != base.target_digest:
                raise ValueError(
                    "clean CANONICAL target digest must equal its exact base"
                )
        elif (
            delta.change_state is not ChangeState.DIRTY
            or delta.location is None
            or delta.checksum is None
        ):
            raise ValueError(
                "dirty CANONICAL revision requires one root-index reference"
            )
        return delta

    def _prepare_locked(self, manifest: Any, delta: Any, base: Any) -> PreparedPayload:
        version_key = manifest.version.encode("utf-8").hex()
        version_root = self._target_root / version_key
        hf_root = version_root / "hf"
        marker_path = version_root / "prepared.json"
        noop = delta.change_state is ChangeState.CLEAN
        materialized_attestation = None
        expected_marker = {
            "base_digest": base.target_digest,
            "base_version": base.version,
            "format_digest": manifest.format_digest,
            "model_id": self._model_id,
            "noop": noop,
            "target_digest": manifest.target_digest,
            "target_version": manifest.version,
        }
        marker = self._read_marker(marker_path)
        if marker is not None and any(
            marker.get(key) != value for key, value in expected_marker.items()
        ):
            raise ValueError(
                "host-local prepared marker does not match target revision"
            )

        target = self._open_cached_target(manifest.version)
        if target is not None:
            self._validate_target(target, manifest)
            if noop:
                if marker is None:
                    _write_json_atomic(marker_path, expected_marker)
            else:
                checkpoint = hf_root / "model.safetensors"
                try:
                    actual_attestation = _materialized_file_attestation(checkpoint)
                except (OSError, ValueError):
                    actual_attestation = None
                expected_attestation = (
                    None
                    if marker is None
                    else {
                        "hf_ctime_ns": marker.get("hf_ctime_ns"),
                        "hf_device": marker.get("hf_device"),
                        "hf_inode": marker.get("hf_inode"),
                        "hf_mtime_ns": marker.get("hf_mtime_ns"),
                        "hf_sha256": marker.get("hf_sha256"),
                        "hf_size": marker.get("hf_size"),
                    }
                )
                if actual_attestation != expected_attestation:
                    materialize_snapshot_to_safetensors(
                        self._base_store, target, hf_root
                    )
                    actual_attestation = _materialized_file_attestation(checkpoint)
                    _write_json_atomic(
                        marker_path, {**expected_marker, **actual_attestation}
                    )
                materialized_attestation = actual_attestation
            return self._payload(
                base,
                target,
                hf_root,
                noop=noop,
                materialized_attestation=materialized_attestation,
            )

        if noop:
            target = self._copy_clean_target(base, manifest.version)
        else:
            root_object = self._transport.resolve(
                delta.location, delta.checksum, 64 * 1024 * 1024
            )
            root_bytes = self._transport.fetch(root_object)

            def fetch_bucket(reference):
                stored = self._transport.resolve(
                    reference.location, reference.checksum, reference.size
                )
                return self._transport.fetch(stored)

            target = reconstruct_canonical_delta(
                root_bytes=root_bytes,
                expected_root_checksum=delta.checksum,
                base_store=self._base_store,
                base=base,
                target_store=self._base_store,
                fetch_bucket=fetch_bucket,
            )
            self._validate_target(target, manifest)
            materialize_snapshot_to_safetensors(self._base_store, target, hf_root)

        if noop:
            marker_document = expected_marker
        else:
            materialized_attestation = _materialized_file_attestation(
                hf_root / "model.safetensors"
            )
            marker_document = {
                **expected_marker,
                **materialized_attestation,
            }
        _write_json_atomic(marker_path, marker_document)
        return self._payload(
            base,
            target,
            hf_root,
            noop=noop,
            materialized_attestation=materialized_attestation,
        )

    def _open_cached_target(self, version: str) -> Any | None:
        try:
            return self._base_store.open_snapshot(version)
        except CanonicalDeltaError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return None
            raise

    @staticmethod
    def _read_marker(path: Path) -> dict[str, Any] | None:
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except FileNotFoundError:
            return None
        if not isinstance(value, dict):
            raise ValueError("host-local prepared marker must be an object")
        return value

    def _copy_clean_target(self, base: Any, version: str) -> Any:
        writer = self._base_store.begin_snapshot(
            version, format_identity=base.format_identity
        )
        try:
            for metadata in base.tensors:
                writer.add_tensor(
                    metadata.name,
                    self._base_store.read_tensor(base, metadata.name),
                )
            return writer.finalize(
                expected_format_digest=base.format_digest,
                expected_target_digest=base.target_digest,
            )
        except BaseException:
            writer.abort()
            raise

    @staticmethod
    def _validate_target(target: Any, manifest: Any) -> None:
        if (
            target.target_digest != manifest.target_digest
            or target.format_digest != manifest.format_digest
        ):
            raise ValueError("reconstructed target does not match revision identity")

    def _payload(
        self,
        base: Any,
        target: Any,
        hf_root: Path,
        *,
        noop: bool,
        materialized_attestation: dict[str, Any] | None,
    ) -> PreparedPayload:
        identity = PreparedUpdate(
            model_id=self._model_id,
            base_version=base.version,
            base_digest=base.target_digest,
            target_version=target.version,
            target_digest=target.target_digest,
            format_digest=target.format_digest,
            receiver_incarnation=self._receiver_incarnation,
            model_generation=self._model_generation(),
        )
        return PreparedPayload(
            identity=identity,
            model_path=hf_root,
            canonical_store=self._base_store,
            canonical_snapshot=target,
            noop=noop,
            materialized_sha256=(
                None
                if materialized_attestation is None
                else materialized_attestation["hf_sha256"]
            ),
            materialized_size=(
                None
                if materialized_attestation is None
                else materialized_attestation["hf_size"]
            ),
            materialized_device=(
                None
                if materialized_attestation is None
                else materialized_attestation["hf_device"]
            ),
            materialized_inode=(
                None
                if materialized_attestation is None
                else materialized_attestation["hf_inode"]
            ),
            materialized_mtime_ns=(
                None
                if materialized_attestation is None
                else materialized_attestation["hf_mtime_ns"]
            ),
            materialized_ctime_ns=(
                None
                if materialized_attestation is None
                else materialized_attestation["hf_ctime_ns"]
            ),
            preparation_lock_path=self._target_root.parent / "prepare.lock",
        )

    def commit(self, payload: PreparedPayload) -> None:
        if payload.canonical_store is None or payload.canonical_snapshot is None:
            raise ValueError("prepared payload has no canonical target")
        if payload.canonical_store is not self._base_store:
            raise ValueError("prepared target does not belong to the host-local store")
        snapshot = payload.canonical_snapshot
        self._base_snapshot = lambda: snapshot
