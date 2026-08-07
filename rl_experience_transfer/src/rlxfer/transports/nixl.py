# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional NIXL data plane with a filesystem control plane."""

from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Any

from rlxfer.errors import (
    DeliveryError,
    MissingDependencyError,
    TransferTimeoutError,
    TransportError,
)
from rlxfer.observability import Metrics, NullMetrics
from rlxfer.serialization import (
    BufferSegment,
    SerializationLimits,
    SerializedExperience,
    validate_metadata,
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
from rlxfer.transports.filesystem import FileSystemTransport


@dataclass(slots=True)
class _Source:
    owners: tuple[Any, ...]
    registrations: tuple[Any, ...]
    receipt: DeliveryReceipt


class NixlTransport:
    """Pull-based NIXL transport retaining producer buffers until settlement.

    The producer owns and keeps every registered source buffer immutable until the
    delivery is acknowledged or rejected. The consumer owns destination buffers;
    their NIXL registration ends after transfer completion, while their storage
    remains alive through ``BufferSegment.owner``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        agent_name: str | None = None,
        backend: str = "UCX",
        max_queue: int = 128,
        transfer_timeout: float = 30.0,
        poll_interval: float = 0.0005,
        target_device: str | None = None,
        allow_cpu_staging: bool = True,
        metrics: Metrics | None = None,
        limits: SerializationLimits | None = None,
    ) -> None:
        if transfer_timeout <= 0 or poll_interval <= 0:
            raise ValueError("transfer_timeout and poll_interval must be positive")
        try:
            self._nixl = import_module("nixl")
            self._torch = import_module("torch")
        except ImportError as error:
            raise MissingDependencyError(
                "NIXL transport requires the optional 'nixl' and 'torch' packages"
            ) from error
        self._backend = backend.upper()
        config = self._nixl.nixl_agent_config(backends=[self._backend])
        self._agent = self._nixl.nixl_agent(agent_name or f"rlxfer-{uuid.uuid4().hex}", config)
        if self._backend not in self._agent.backends:
            raise TransportError(f"NIXL backend {self._backend!r} is unavailable")
        self._limits = limits or SerializationLimits()
        control_limits = replace(
            self._limits,
            max_metadata_bytes=self._limits.max_metadata_bytes * 2 + 1024 * 1024,
        )
        self._control = FileSystemTransport(
            Path(path) / "control",
            max_queue=max_queue,
            poll_interval=0.01,
            limits=control_limits,
        )
        self._transfer_timeout = transfer_timeout
        self._poll_interval = poll_interval
        self._target_device = target_device
        self._allow_cpu_staging = allow_cpu_staging
        self._metrics = metrics or NullMetrics()
        self._sources: dict[str, _Source] = {}
        self._receipts: dict[str, DeliveryReceipt] = {}
        self._lock = RLock()
        self._closing = False
        self._closed = False

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> NixlTransport:
        return cls(**dict(options))

    @property
    def capabilities(self) -> TransportCapabilities:
        with self._lock:
            memory_types = set(self._agent.get_backend_mem_types(self._backend))
        accelerators = frozenset({"cuda"} if "VRAM_SEG" in memory_types else ())
        return TransportCapabilities(
            name=f"nixl:{self._backend.lower()}",
            zero_copy=False,
            cpu_buffers="DRAM_SEG" in memory_types,
            accelerator_buffers=accelerators,
            remote=True,
            scatter_gather=True,
            asynchronous=True,
            acknowledgements=True,
            persistence=False,
            max_transfer_size=(
                self._limits.max_metadata_bytes + self._limits.max_total_tensor_bytes
            ),
            requires_registration=True,
            delivery_guarantee="at-least-once while the producer remains alive",
        )

    def _as_source_tensor(self, segment: BufferSegment) -> Any:
        owner = segment.owner
        if isinstance(owner, self._torch.Tensor):
            tensor = owner.detach()
        elif owner is not None:
            try:
                tensor = self._torch.as_tensor(owner)
            except (TypeError, ValueError, RuntimeError):
                tensor = self._torch.frombuffer(
                    bytearray(segment.materialize()), dtype=self._torch.uint8
                )
        else:
            tensor = self._torch.frombuffer(
                bytearray(segment.materialize()), dtype=self._torch.uint8
            )
        tensor = tensor.contiguous()
        if tensor.numel() * tensor.element_size() != segment.nbytes:
            tensor = self._torch.frombuffer(
                bytearray(segment.materialize()), dtype=self._torch.uint8
            )
        if tensor.device.type not in {"cpu", "cuda"}:
            if not self._allow_cpu_staging:
                raise TransportError(f"NIXL backend cannot register device {tensor.device.type!r}")
            tensor = tensor.cpu().contiguous()
        return tensor

    @staticmethod
    def _memory_type(tensor: Any) -> str:
        return "VRAM" if tensor.device.type == "cuda" else "DRAM"

    @staticmethod
    def _descriptor(tensor: Any) -> tuple[int, int, int]:
        device_id = int(tensor.get_device())
        return (
            int(tensor.data_ptr()),
            int(tensor.numel() * tensor.element_size()),
            max(0, device_id),
        )

    def _register(self, tensors: tuple[Any, ...]) -> tuple[Any, ...]:
        groups: dict[tuple[str, str], list[Any]] = {}
        for tensor in tensors:
            groups.setdefault((self._memory_type(tensor), str(tensor.device)), []).append(tensor)
        registrations: list[Any] = []
        with self._lock:
            try:
                for group in groups.values():
                    registrations.append(
                        self._agent.register_memory(group, backends=[self._backend])
                    )
            except BaseException:
                self._deregister(tuple(registrations))
                raise
        return tuple(registrations)

    def _deregister(self, registrations: tuple[Any, ...]) -> None:
        with self._lock:
            for registration in reversed(registrations):
                try:
                    self._agent.deregister_memory(registration, backends=[self._backend])
                except Exception:
                    self._metrics.increment("cleanup_failures", attributes={"transport": "nixl"})

    def _sync(self, tensors: tuple[Any, ...]) -> None:
        devices = {str(tensor.device) for tensor in tensors if tensor.device.type == "cuda"}
        for device in devices:
            self._torch.cuda.synchronize(device)

    def publish(
        self,
        payload: SerializedExperience,
        *,
        experience_id: str,
        idempotency_key: str,
        timeout: float | None = None,
        max_retries: int = 3,
    ) -> DeliveryReceipt:
        with self._lock:
            return self._publish_locked(
                payload,
                experience_id=experience_id,
                idempotency_key=idempotency_key,
                timeout=timeout,
                max_retries=max_retries,
            )

    def _publish_locked(
        self,
        payload: SerializedExperience,
        *,
        experience_id: str,
        idempotency_key: str,
        timeout: float | None,
        max_retries: int,
    ) -> DeliveryReceipt:
        if self._closed or self._closing:
            raise TransportError("NIXL transport is closed")
        validate_transfer_limits(
            metadata_bytes=len(payload.metadata),
            tensor_sizes=(segment.nbytes for segment in payload.buffers),
            limits=self._limits,
        )
        expected = tuple(
            segment
            for segment in validate_metadata(payload.metadata, limits=self._limits)
            if segment.data is None
        )
        if tuple((item.name, item.nbytes) for item in expected) != tuple(
            (item.name, item.nbytes) for item in payload.buffers
        ):
            raise DeliveryError("NIXL payload buffers disagree with serializer metadata")
        self._reap()
        staging_started = time.perf_counter()
        tensors = tuple(self._as_source_tensor(segment) for segment in payload.buffers)
        self._sync(tensors)
        self._metrics.observe("staging_copy_latency_seconds", time.perf_counter() - staging_started)
        registration_started = time.perf_counter()
        registrations = self._register(tensors)
        self._metrics.observe(
            "buffer_registration_latency_seconds",
            time.perf_counter() - registration_started,
        )
        catalogs: list[dict[str, Any]] = []
        for segment, tensor in zip(payload.buffers, tensors, strict=True):
            address, nbytes, device_id = self._descriptor(tensor)
            catalog = segment.catalog_entry()
            catalog.update(
                wire_device=str(tensor.device),
                address=address,
                region_size=nbytes,
                device_id=device_id,
                memory_type=self._memory_type(tensor),
            )
            catalogs.append(catalog)
        control = {
            "version": 1,
            "backend": self._backend,
            "agent_metadata": base64.b64encode(self._agent.get_agent_metadata()).decode("ascii"),
            "metadata": base64.b64encode(payload.metadata).decode("ascii"),
            "buffers": catalogs,
        }
        try:
            receipt = self._control.publish(
                SerializedExperience(
                    metadata=json.dumps(control, separators=(",", ":"), sort_keys=True).encode(),
                    buffers=(),
                ),
                experience_id=experience_id,
                idempotency_key=idempotency_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        except BaseException:
            self._deregister(registrations)
            raise
        self._receipts.setdefault(receipt.receipt_id, receipt)
        if (
            receipt.wait(0.0).state is not ReceiptState.EXPIRED
            or receipt.receipt_id in self._sources
        ):
            self._deregister(registrations)
        else:
            self._sources[receipt.receipt_id] = _Source(tensors, registrations, receipt)
        self._metrics.increment("produced_batches")
        self._metrics.increment("bytes_transferred", payload.nbytes)
        return DeliveryReceipt(
            receipt_id=receipt.receipt_id,
            experience_id=receipt.experience_id,
            idempotency_key=receipt.idempotency_key,
            accepted_at=receipt.accepted_at,
            _wait=lambda wait_timeout: self._wait_source(receipt.receipt_id, wait_timeout),
        )

    def _wait_source(self, record_id: str, timeout: float | None) -> ReceiptResult:
        with self._lock:
            try:
                receipt = self._receipts[record_id]
            except KeyError as error:
                raise DeliveryError("unknown NIXL source receipt") from error
        result = receipt.wait(timeout)
        if result.state.terminal:
            self._release_source(record_id)
        return result

    def _release_source(self, record_id: str) -> None:
        with self._lock:
            source = self._sources.pop(record_id, None)
            if source is not None:
                self._deregister(source.registrations)

    def _reap(self) -> None:
        with self._lock:
            for record_id, source in tuple(self._sources.items()):
                result = source.receipt.wait(0.0)
                if result.state is not ReceiptState.EXPIRED:
                    self._release_source(record_id)

    def _target(self, item: Mapping[str, Any]) -> Any:
        source_device = str(item["wire_device"])
        requested = self._target_device or source_device
        if requested.startswith("cuda") and not self._torch.cuda.is_available():
            if not self._allow_cpu_staging:
                raise TransportError("CUDA destination requested but CUDA is unavailable")
            requested = "cpu"
        return self._torch.empty(int(item["nbytes"]), dtype=self._torch.uint8, device=requested)

    def _decode_control(self, payload: SerializedExperience) -> dict[str, Any]:
        try:
            value = json.loads(payload.metadata)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
            raise DeliveryError("invalid NIXL control metadata") from error
        if not isinstance(value, dict) or value.get("version") != 1:
            raise DeliveryError("unsupported NIXL control metadata version")
        if value.get("backend") != self._backend or not isinstance(value.get("buffers"), list):
            raise DeliveryError("incompatible NIXL backend or buffer catalog")
        return value

    def _validate_control_catalog(
        self, metadata: bytes, items: Sequence[Mapping[str, Any]]
    ) -> None:
        expected = tuple(
            segment
            for segment in validate_metadata(metadata, limits=self._limits)
            if segment.data is None
        )
        if len(expected) != len(items):
            raise DeliveryError("NIXL and serializer buffer catalogs differ in length")
        for segment, item in zip(expected, items, strict=True):
            if any(item.get(key) != value for key, value in segment.catalog_entry().items()):
                raise DeliveryError(
                    f"NIXL catalog for buffer {segment.name!r} disagrees with metadata"
                )
            if (
                item.get("region_size") != segment.nbytes
                or isinstance(item.get("address"), bool)
                or not isinstance(item.get("address"), int)
                or int(item["address"]) <= 0
                or isinstance(item.get("device_id"), bool)
                or not isinstance(item.get("device_id"), int)
                or int(item["device_id"]) < 0
                or item.get("memory_type") not in {"DRAM", "VRAM"}
            ):
                raise DeliveryError(f"NIXL descriptor for buffer {segment.name!r} is invalid")

    def _transfer_group(
        self,
        remote_name: str,
        items: list[Mapping[str, Any]],
        targets: list[Any],
        deadline: float,
    ) -> None:
        source_type = str(items[0]["memory_type"])
        remote = self._agent.get_xfer_descs(
            [
                (
                    int(item["address"]),
                    int(item["region_size"]),
                    int(item["device_id"]),
                )
                for item in items
            ],
            source_type,
        )
        local = self._agent.get_xfer_descs(targets)
        handle = self._agent.initialize_xfer(
            "READ", local, remote, remote_name, backends=[self._backend]
        )
        try:
            transfer_started = time.perf_counter()
            state = self._agent.transfer(handle)
            while state == "PROC":
                if time.monotonic() >= deadline:
                    raise TransferTimeoutError("NIXL transfer timed out")
                time.sleep(self._poll_interval)
                state = self._agent.check_xfer_state(handle)
            if state != "DONE":
                raise TransportError(f"NIXL transfer failed with state {state!r}")
            self._metrics.observe(
                "transfer_latency_seconds", time.perf_counter() - transfer_started
            )
        finally:
            with suppress(Exception):
                self._agent.release_xfer_handle(handle)

    def receive(self, timeout: float | None = None) -> TransportDelivery | None:
        with self._lock:
            return self._receive_locked(timeout)

    def _receive_locked(self, timeout: float | None) -> TransportDelivery | None:
        if self._closed or self._closing:
            raise TransportError("NIXL transport is closed")
        control_delivery = self._control.receive(timeout)
        if control_delivery is None:
            return None
        registrations: tuple[Any, ...] = ()
        remote_name: str | None = None
        try:
            control = self._decode_control(control_delivery.payload)
            items = [item for item in control["buffers"] if isinstance(item, dict)]
            if len(items) != len(control["buffers"]):
                raise DeliveryError("invalid NIXL buffer catalog entry")
            metadata = base64.b64decode(control["metadata"], validate=True)
            self._validate_control_catalog(metadata, items)
            targets = tuple(self._target(item) for item in items)
            registrations = self._register(targets)
            if items:
                remote_name = self._agent.add_remote_agent(
                    base64.b64decode(control["agent_metadata"], validate=True)
                )
                grouped: dict[
                    tuple[str, int, str, str],
                    tuple[list[Mapping[str, Any]], list[Any]],
                ] = {}
                for item, target in zip(items, targets, strict=True):
                    key = (
                        str(item["memory_type"]),
                        int(item["device_id"]),
                        self._memory_type(target),
                        str(target.device),
                    )
                    selected_items, selected_targets = grouped.setdefault(key, ([], []))
                    selected_items.append(item)
                    selected_targets.append(target)
                deadline = time.monotonic() + self._transfer_timeout
                for selected_items, selected_targets in grouped.values():
                    self._transfer_group(
                        remote_name,
                        selected_items,
                        selected_targets,
                        deadline,
                    )
                self._sync(targets)
            buffers = tuple(
                BufferSegment.from_catalog_entry(item, owner=target)
                for item, target in zip(items, targets, strict=True)
            )
            self._metrics.increment("consumed_batches")
            return TransportDelivery(
                token=control_delivery.token,
                experience_id=control_delivery.experience_id,
                idempotency_key=control_delivery.idempotency_key,
                payload=SerializedExperience(
                    metadata=metadata,
                    buffers=buffers,
                ),
                attempt=control_delivery.attempt,
                published_at=control_delivery.published_at,
                max_retries=control_delivery.max_retries,
            )
        except BaseException as error:
            self._metrics.increment("transfer_failures", attributes={"transport": "nixl"})
            with suppress(Exception):
                self._control.nack(
                    control_delivery.token, f"NIXL receive failed: {error}", retry=True
                )
            raise
        finally:
            if remote_name is not None:
                with suppress(Exception):
                    self._agent.remove_remote_agent(remote_name)
            self._deregister(registrations)

    def ack(self, token: str) -> None:
        with self._lock:
            self._control.ack(token)
            self._reap()

    def nack(self, token: str, reason: str, *, retry: bool = True) -> None:
        with self._lock:
            self._control.nack(token, reason, retry=retry)
            self._reap()

    def reject(self, token: str, reason: str) -> None:
        with self._lock:
            self._control.reject(token, reason)
            self._reap()

    def cancel(self, receipt_id: str, reason: str = "cancelled") -> None:
        """Cancel a pending pull before releasing its registered source buffers."""

        with self._lock:
            self._control.cancel(receipt_id, reason)
            self._release_source(receipt_id)

    def health(self) -> HealthStatus:
        with self._lock:
            self._reap()
            control = self._control.health()
            return HealthStatus(
                not self._closed and not self._closing and control.healthy,
                "closed" if self._closed or self._closing else control.detail,
                control.queue_depth,
            )

    def close(self, timeout: float | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closing = True
        deadline = time.monotonic() + max(0.0, timeout or 0.0)
        while time.monotonic() < deadline:
            self._reap()
            with self._lock:
                if not self._sources:
                    break
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        with self._lock:
            source_ids = tuple(self._sources)
        for record_id in source_ids:
            try:
                self.cancel(record_id, "producer shutdown")
            except DeliveryError as error:
                with self._lock:
                    self._closing = False
                raise TransferTimeoutError(
                    "cannot close NIXL transport while a delivery is inflight"
                ) from error
        with self._lock:
            self._control.close()
            self._closed = True
            self._closing = False
