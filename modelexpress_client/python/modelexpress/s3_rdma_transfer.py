# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL OBJ transfer manager for S3-over-RDMA reads into device memory.

Drives NIXL's OBJ plugin with the generic accelerated engine
(``accelerated=true``, no ``type``), which speaks the cuObject
``x-amz-rdma-token`` protocol: each READ descriptor becomes one signed,
body-less ranged GET and the object server RDMA-writes the bytes straight
into the registered destination. There is no HTTP fallback on this path.

The manager owns a private, short-lived NIXL agent (loopback transfers, no
listen thread) and expects a single calling thread: NIXL's default sync mode
for such agents is NONE, so all registrations happen before any transfer is
posted.
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urlparse

import torch

from ._nixl import load_nixl_api
from .accelerators import AcceleratorBackend

logger = logging.getLogger("modelexpress.s3_rdma_transfer")

NIXL_AVAILABLE = False
NixlAgent = None
NixlAgentConfig = None
_nixl_api = load_nixl_api()
if _nixl_api is not None:
    NixlAgent = _nixl_api.nixl_agent
    NixlAgentConfig = _nixl_api.nixl_agent_config
    NIXL_AVAILABLE = True

OBJ_BACKEND = "OBJ"
OBJ_MEM_TYPE = "OBJ"
HOST_MEM_TYPE = "DRAM"
# Reads and slots have no 4 GiB cap. cuObject mints an RDMA token for at most
# 4 GiB - 64 KiB, but NIXL splits a larger read into ranged GETs of that size.
# Registering a slot of 4 GiB or more needs the cuObject
# from CUDA 13.4 or newer; older releases fail the registration, and the
# strategy falls back to the next loader.
_CUOBJ_CLIENT_LIB = "libcuobjclient.so.1"

# Agents and buffers whose transfers could not be drained. Keeping them
# referenced prevents RDMA writes from landing in memory the allocator has
# already handed to someone else.
_QUARANTINE: list[tuple[Any, list[torch.Tensor]]] = []


class S3RdmaTransferError(RuntimeError):
    """A NIXL OBJ transfer failed or could not be posted."""


class S3RdmaBackendUnavailable(S3RdmaTransferError):
    """The OBJ plugin, its accelerated engine, or the RDMA fabric is not usable."""


def _cuobj_client_loadable() -> bool:
    """Check that the cuObject client library resolves through the dynamic linker."""
    try:
        ctypes.CDLL(_CUOBJ_CLIENT_LIB)
        return True
    except OSError:
        return False


def is_s3_rdma_available() -> bool:
    """Return whether NIXL and the cuObject client library are present.

    This is a cheap, side-effect-free probe. Whether the accelerated engine is
    compiled in and the RDMA fabric is reachable is only known once the
    backend is created (see :meth:`S3RdmaTransferManager.initialize`).
    """
    if not NIXL_AVAILABLE:
        return False
    if not _cuobj_client_loadable():
        logger.debug("NIXL OBJ not available: %s not loadable", _CUOBJ_CLIENT_LIB)
        return False
    return True


@dataclass(frozen=True)
class S3RdmaEndpoint:
    """S3 endpoint handed to the NIXL OBJ backend.

    Credentials are deliberately absent: NIXL resolves them through the AWS
    SDK's default provider chain, the same sources boto3 reads.
    """

    bucket: str
    endpoint_url: str
    region: str
    ca_bundle: str | None = None

    @property
    def scheme(self) -> str:
        return urlparse(self.endpoint_url).scheme or "http"

    def backend_params(self, num_threads: int) -> dict[str, str]:
        """Build OBJ backend init parameters for the generic accelerated engine."""
        params = {
            "accelerated": "true",
            "bucket": self.bucket,
            "endpoint_override": self.endpoint_url,
            "scheme": self.scheme,
            "region": self.region,
            "use_virtual_addressing": "false",
            "num_threads": str(num_threads),
        }
        if self.ca_bundle:
            params["ca_bundle"] = self.ca_bundle
        return params


class S3RdmaTransferManager:
    """Post ranged S3-over-RDMA reads from objects into registered device slots.

    Each posted transfer carries exactly one descriptor into its own slot
    registration, so a failure never leaves sibling descriptors running
    inside the same request.
    Slots are contiguous tensors on this manager's device, addressed by
    registration order, and a slot takes no new read while one is in flight.
    """

    def __init__(
        self,
        agent_name: str,
        endpoint: S3RdmaEndpoint,
        *,
        device_id: int,
        num_threads: int,
        accelerator_backend: AcceleratorBackend,
    ):
        self._agent_name = agent_name
        self._endpoint = endpoint
        self._device_id = device_id
        self._num_threads = num_threads
        self._accelerator_backend = accelerator_backend
        self._mem_type = accelerator_backend.nixl_mem_type
        self._slot_mem_type = self._mem_type
        self._slot_dev_id = device_id
        self._agent: Any = None
        self._obj_reg: Any = None
        self._slot_reg: Any = None
        self._slots: list[torch.Tensor] = []
        self._inflight: dict[Any, int] = {}

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def inflight(self) -> int:
        return len(self._inflight)

    def initialize(self) -> None:
        """Create the agent and the accelerated OBJ backend; fail fast if RDMA is unusable."""
        if self._agent is not None:
            return
        if not NIXL_AVAILABLE:
            raise S3RdmaBackendUnavailable("NIXL is not installed")

        self._accelerator_backend.set_device(self._device_id)
        agent = NixlAgent(self._agent_name, NixlAgentConfig(backends=[]))
        if OBJ_BACKEND not in agent.get_plugin_list():
            raise S3RdmaBackendUnavailable("NIXL OBJ plugin is not available")
        try:
            agent.create_backend(OBJ_BACKEND, self._endpoint.backend_params(self._num_threads))
        except Exception as exc:
            raise S3RdmaBackendUnavailable(
                f"NIXL OBJ accelerated backend could not be created: {exc}"
            ) from exc
        mem_types = agent.get_backend_mem_types(OBJ_BACKEND)
        if f"{self._mem_type}_SEG" not in mem_types:
            raise S3RdmaBackendUnavailable(
                f"NIXL OBJ backend does not support {self._mem_type} "
                f"(advertised: {mem_types}); the RDMA path is not ready"
            )
        self._agent = agent
        logger.info(
            "NIXL OBJ agent '%s' ready on device %d (endpoint=%s, bucket=%s, threads=%d)",
            self._agent_name,
            self._device_id,
            self._endpoint.endpoint_url,
            self._endpoint.bucket,
            self._num_threads,
        )

    def register_objects(self, objects: Sequence[tuple[str, int]]) -> list[int]:
        """Register (key, size) objects; returns the device id assigned to each.

        The OBJ backend maps device ids to object keys, so ids must be unique.
        Registration makes no network call.
        """
        self._require_agent()
        if self._obj_reg is not None:
            raise S3RdmaTransferError("objects are already registered")
        if not objects:
            raise S3RdmaTransferError("no objects to register")
        regions = [(0, size, dev_id, key) for dev_id, (key, size) in enumerate(objects)]
        self._obj_reg = self._agent.register_memory(regions, OBJ_MEM_TYPE, backends=[OBJ_BACKEND])
        return list(range(len(objects)))

    def register_slots(self, slots: Sequence[torch.Tensor], *, host: bool = False) -> None:
        """Register each staging slot as its own region, in device or host memory."""
        self._require_agent()
        if self._slot_reg is not None:
            raise S3RdmaTransferError("slots are already registered")
        if not slots:
            raise S3RdmaTransferError("no staging slots to register")
        if host:
            device, mem_type, dev_id = torch.device("cpu"), HOST_MEM_TYPE, 0
        else:
            device = self._accelerator_backend.torch_device(self._device_id)
            mem_type, dev_id = self._mem_type, self._device_id
        regions = []
        for idx, slot in enumerate(slots):
            nbytes = slot.numel() * slot.element_size()
            if slot.device != device:
                raise S3RdmaTransferError(f"staging slot {idx} is on {slot.device}, expected {device}")
            if not slot.is_contiguous():
                raise S3RdmaTransferError(f"staging slot {idx} is not contiguous")
            regions.append((slot.data_ptr(), nbytes, dev_id, ""))
        try:
            self._slot_reg = self._agent.register_memory(regions, mem_type, backends=[OBJ_BACKEND])
        except Exception as exc:
            raise S3RdmaTransferError(f"could not register {mem_type} staging slots: {exc}") from exc
        self._slot_mem_type, self._slot_dev_id = mem_type, dev_id
        self._slots = list(slots)

    def post(self, slot: int, obj_dev_id: int, offset: int, length: int) -> Any:
        """Post one ranged read of ``length`` bytes at ``offset`` into ``slot``."""
        self._require_agent()
        if not 0 <= slot < len(self._slots):
            raise S3RdmaTransferError(f"slot {slot} is not registered")
        if slot in self._inflight.values():
            raise S3RdmaTransferError(f"slot {slot} already has a read in flight")
        slot_tensor = self._slots[slot]
        capacity = slot_tensor.numel() * slot_tensor.element_size()
        if length <= 0 or length > capacity:
            raise S3RdmaTransferError(f"invalid read length {length} for a {capacity}-byte slot")
        local = self._agent.get_xfer_descs(
            [(slot_tensor.data_ptr(), length, self._slot_dev_id)], self._slot_mem_type
        )
        remote = self._agent.get_xfer_descs([(offset, length, obj_dev_id)], OBJ_MEM_TYPE)
        try:
            handle = self._agent.initialize_xfer(
                "READ", local, remote, self._agent.name, backends=[OBJ_BACKEND]
            )
        except Exception as exc:
            raise S3RdmaTransferError(f"could not create OBJ read request: {exc}") from exc
        try:
            state = self._agent.transfer(handle)
        except Exception as exc:
            self.release(handle)
            raise S3RdmaTransferError(f"OBJ read could not be posted: {exc}") from exc
        if state == "ERR":
            self.release(handle)
            raise S3RdmaTransferError("OBJ read could not be posted")
        self._inflight[handle] = slot
        return handle

    def poll(self, handle: Any) -> bool:
        """Return True when the read is complete; raise S3RdmaTransferError on failure."""
        try:
            state = self._agent.check_xfer_state(handle)
        except Exception as exc:
            self._inflight.pop(handle, None)
            raise S3RdmaTransferError(f"OBJ read failed: {exc}") from exc
        if state == "DONE":
            self._inflight.pop(handle, None)
            return True
        if state == "ERR":
            self._inflight.pop(handle, None)
            raise S3RdmaTransferError("OBJ read failed")
        return False

    def release(self, handle: Any) -> None:
        """Release a finished request handle, logging rather than raising."""
        self._inflight.pop(handle, None)
        try:
            self._agent.release_xfer_handle(handle)
        except Exception as exc:
            logger.warning("Failed to release NIXL OBJ request handle: %s", exc)

    def drain(self, timeout_s: float) -> bool:
        """Wait for every in-flight read to finish (success or failure).

        Returns False if some reads are still running at the deadline.
        """
        deadline = time.monotonic() + timeout_s
        while self._inflight:
            for handle in list(self._inflight):
                try:
                    done = self.poll(handle)
                except S3RdmaTransferError:
                    done = True
                if done:
                    self.release(handle)
            if not self._inflight:
                break
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.0001)
        return True

    def shutdown(self, *, drain_timeout_s: float = 30.0) -> None:
        """Drain, deregister, and drop the agent.

        If reads cannot be drained, the agent and slots are quarantined for the
        life of the process instead of being freed.
        """
        if self._agent is None:
            return
        if self._inflight and not self.drain(drain_timeout_s):
            logger.error(
                "NIXL OBJ agent '%s' has %d reads that did not finish; quarantining "
                "the agent and %d staging slots instead of freeing them",
                self._agent_name,
                len(self._inflight),
                len(self._slots),
            )
            _QUARANTINE.append((self._agent, list(self._slots)))
            self._reset()
            return
        for reg in (self._slot_reg, self._obj_reg):
            if reg is None:
                continue
            try:
                self._agent.deregister_memory(reg, backends=[OBJ_BACKEND])
            except Exception as exc:
                logger.warning("Failed to deregister NIXL OBJ memory: %s", exc)
        self._reset()
        logger.info("NIXL OBJ agent '%s' shut down", self._agent_name)

    def _reset(self) -> None:
        self._agent = None
        self._obj_reg = None
        self._slot_reg = None
        self._slots = []
        self._inflight = {}

    def _require_agent(self) -> None:
        if self._agent is None:
            raise S3RdmaTransferError("NIXL OBJ agent not initialized")
