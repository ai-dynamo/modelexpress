# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inference-side weight receiver for RL refit via ModelExpress.

Wraps NixlTransferManager + MxClient to discover updated weights
published by the training side, pull them via RDMA, and yield
``(name, tensor)`` pairs compatible with vLLM's ``model.load_weights()``.

Typical usage in a vLLM worker extension::

    receiver = MxRefitReceiver("inference-0", device_id=0, mx_server_url="mx-server:8001")
    receiver.initialize(model_tensors=dict(model.named_parameters()))

    source = receiver.poll_for_source(model_name="Qwen/Qwen2.5-1.5B")
    if source is not None:
        for name, tensor in receiver.receive_weights(source):
            ...  # load into model
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from .client import MxClient
from .nixl_transfer import NixlTransferManager, is_nixl_available
from .types import TensorDescriptor
from . import p2p_pb2

logger = logging.getLogger("modelexpress.refit_receiver")


# Maps the dtype string the publisher writes into TensorDescriptor.dtype to a
# torch.dtype. Module-scope so all receiver paths share one definition (and so
# we don't rebuild it on every receive_weights_scratch call).
_DTYPE_MAP: dict[str, torch.dtype] = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class SourceRef:
    """Lightweight handle to a discovered weight source on the MX Server."""
    mx_source_id: str
    worker_id: str
    model_name: str
    worker_rank: int
    training_step: int


class MxRefitReceiver:
    """Receives updated weights from a training process via ModelExpress RDMA.

    One instance per GPU rank on the inference side. Discovers training
    sources via the MX Server, pulls weight tensors over NIXL RDMA,
    and yields them for ``model.load_weights()``.

    Args:
        agent_name: Unique NIXL agent name (e.g. ``"inference-rank-0"``).
        device_id: CUDA device index for this inference rank.
        mx_server_url: gRPC address of the ModelExpress server.
        listen_port: Optional NIXL listen port for P2P metadata exchange.
    """

    def __init__(
        self,
        agent_name: str,
        device_id: int,
        mx_server_url: str = "localhost:8001",
        listen_port: int | None = None,
    ):
        self._agent_name = agent_name
        self._device_id = device_id
        self._mx_server_url = mx_server_url
        self._listen_port = listen_port

        self._nixl: NixlTransferManager | None = None
        self._client: MxClient | None = None
        self._initialized = False
        self._current_step = -1
        # Stable per-instance worker_id consumed by publish flows that
        # require a non-empty worker_id (e.g.
        # MxV2RefitReceiver.publish_self_as_source).  Assigned eagerly
        # so the attribute exists even before initialize() runs;
        # initialize() refreshes it after the NIXL agent boots.
        self._worker_id = f"{self._agent_name}-{uuid.uuid4().hex[:12]}"

    @property
    def current_step(self) -> int:
        """The most recently received training step."""
        return self._current_step

    def initialize(self, model_tensors: dict[str, torch.Tensor] | None = None) -> None:
        """Initialize NIXL agent, MX client, and optionally register receive buffers.

        Args:
            model_tensors: If provided, registers these tensors with NIXL as
                receive buffers. For tensor-name-matched transfers, the source's
                tensors are written directly into these buffers. If *None*,
                the caller must register tensors separately.
        """
        if not is_nixl_available():
            raise RuntimeError(
                "NIXL is not available. Install nixl or build from source."
            )

        self._nixl = NixlTransferManager(
            agent_name=self._agent_name,
            device_id=self._device_id,
            listen_port=self._listen_port,
        )
        self._nixl.initialize()

        if model_tensors is not None:
            self._nixl.register_tensors(model_tensors)
            logger.info(
                f"Registered {len(model_tensors)} receive buffers with NIXL"
            )

        self._client = MxClient(server_url=self._mx_server_url)
        self._initialized = True
        logger.info(
            f"MxRefitReceiver initialized: agent={self._agent_name}, "
            f"device={self._device_id}"
        )

    def poll_for_source(
        self,
        model_name: str,
        min_step: int | None = None,
        status_filter: int = p2p_pb2.SOURCE_STATUS_READY,
        timeout_seconds: float = 0,
    ) -> SourceRef | None:
        """Check the MX Server for a training source with updated weights.

        Args:
            model_name: Model name to filter on (must match publisher's identity).
            min_step: If set, only return sources with ``training_step >= min_step``.
                Defaults to ``current_step + 1`` to only find newer versions.
            timeout_seconds: If > 0, poll repeatedly until a source is found
                or timeout is reached. If 0, check once and return immediately.

        Returns:
            A :class:`SourceRef` if a matching source was found, else *None*.

        Note:
            ``training_step`` is published in ``SourceIdentity.extra_parameters``
            but ``ListSourcesResponse.instances`` only carries
            ``SourceInstanceRef`` (no ``extra_parameters``). To honor the
            ``min_step`` contract, this method does a per-candidate
            ``get_metadata`` lookup so it can read ``training_step`` from the
            publisher's full ``SourceIdentity``. A future server-side fix
            (adding ``training_step`` to ``SourceInstanceRef``) will let us
            drop the extra round-trip.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before poll_for_source()")

        if min_step is None:
            min_step = self._current_step + 1

        deadline = time.perf_counter() + timeout_seconds

        while True:
            try:
                response = self._client.list_sources(
                    status_filter=status_filter,
                )
            except Exception as e:  # noqa: BLE001 — log + retry on transient gRPC error
                logger.warning(f"list_sources failed: {e}")
                if time.perf_counter() >= deadline:
                    return None
                time.sleep(0.5)
                continue

            for instance in response.instances:
                if instance.model_name != model_name:
                    continue

                # Resolve training_step from the publisher's SourceIdentity so
                # min_step can be enforced. Skip candidates whose metadata is
                # unreachable or whose step is below the threshold.
                step = self._resolve_training_step(instance)
                if step is None or step < min_step:
                    continue

                return SourceRef(
                    mx_source_id=instance.mx_source_id,
                    worker_id=instance.worker_id,
                    model_name=instance.model_name,
                    worker_rank=instance.worker_rank,
                    training_step=step,
                )

            if time.perf_counter() >= deadline:
                return None
            time.sleep(0.5)

    def _resolve_training_step(self, instance: Any) -> int | None:
        """Fetch the publisher's ``training_step`` from MX Server metadata.

        ``SourceInstanceRef`` (returned by ``list_sources``) doesn't expose
        ``extra_parameters``, so we do a follow-up ``get_metadata`` to read
        ``training_step`` from ``SourceIdentity.extra_parameters``. Returns
        ``None`` if the metadata isn't available or the step can't be
        parsed — caller should treat this as "skip candidate".
        """
        try:
            meta = self._client.get_metadata(instance.mx_source_id, instance.worker_id)
        except Exception as e:  # noqa: BLE001 — gRPC failures are per-candidate, not fatal
            logger.debug(f"get_metadata failed for {instance.worker_id}: {e}")
            return None
        if not getattr(meta, "found", False):
            return None
        identity = getattr(meta, "identity", None)
        if identity is None:
            return None
        extra = getattr(identity, "extra_parameters", None) or {}
        raw = extra.get("training_step") if hasattr(extra, "get") else None
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.debug(f"training_step={raw!r} not parseable as int; skipping")
            return None

    def receive_weights(
        self,
        source: SourceRef,
        timeout_seconds: float = 300.0,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Receive weights from a discovered source via NIXL RDMA.

        Fetches the source's NIXL metadata and tensor descriptors from the
        MX Server, establishes an RDMA connection, and transfers weight
        tensors into locally registered buffers.

        Args:
            source: A :class:`SourceRef` obtained from :meth:`poll_for_source`.
            timeout_seconds: Maximum time to wait for the RDMA transfer.

        Yields:
            ``(name, tensor)`` pairs suitable for ``model.load_weights()``.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before receive_weights()")

        meta_resp = self._client.get_metadata(
            mx_source_id=source.mx_source_id,
            worker_id=source.worker_id,
        )
        if not meta_resp.found:
            raise RuntimeError(
                f"Source {source.mx_source_id}/{source.worker_id} not found on MX Server"
            )

        worker = meta_resp.worker
        # Filter out V2 sidecar TensorDescriptors (name="__mx_v2_meta__",
        # addr=0, size=0). The V2 publisher uses them to smuggle metadata
        # past the MX server's field-dropping; they aren't real RDMA
        # targets. Leaving them in the source_tensors list propagates a
        # (0,0,0) descriptor into prep_xfer_dlist which UCX rejects.
        source_tensors = [
            TensorDescriptor(
                name=t.name,
                addr=t.addr,
                size=t.size,
                device_id=t.device_id,
                dtype=t.dtype,
            )
            for t in worker.tensors
            if not t.name.startswith("__mx_") and t.size > 0
        ]

        transferred, skipped, elapsed = self._nixl.receive_from_source(
            source_metadata=worker.nixl_metadata,
            source_tensors=source_tensors,
            timeout_seconds=timeout_seconds,
        )

        logger.info(
            f"RDMA transfer complete: {transferred} bytes, "
            f"{len(source_tensors)} tensors, {elapsed:.2f}s "
            f"(step={source.training_step})"
        )

        self._current_step = source.training_step

        for td in source_tensors:
            if td.name in self._nixl._tensors:
                yield td.name, self._nixl._tensors[td.name]

    def receive_weights_scratch(
        self,
        source: SourceRef,
        timeout_seconds: float = 300.0,
        tensor_shapes: dict[str, tuple[int, ...]] | None = None,
        include_names: set[str] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Receive weights into scratch GPU buffers via NIXL RDMA.

        Unlike :meth:`receive_weights` which requires pre-registered model
        buffers with matching tensor names, this method allocates temporary
        GPU tensors that match the source's layout, transfers via RDMA, and
        yields the results. The caller feeds these through
        ``model.load_weights()`` which handles name mapping and tensor fusion.

        This is the correct approach when the source (trainer) publishes
        HuggingFace-format weights but the target (vLLM) uses fused internal
        parameter names.

        Args:
            source: A :class:`SourceRef` obtained from :meth:`poll_for_source`.
            timeout_seconds: Maximum time to wait for the RDMA transfer.

        Yields:
            ``(name, tensor)`` pairs in HF checkpoint format.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before receive_weights_scratch()")

        meta_resp = self._client.get_metadata(
            mx_source_id=source.mx_source_id,
            worker_id=source.worker_id,
        )
        if not meta_resp.found:
            raise RuntimeError(
                f"Source {source.mx_source_id}/{source.worker_id} not found on MX Server"
            )

        worker = meta_resp.worker
        # Filter out V2 sidecar TensorDescriptors (name="__mx_v2_meta__",
        # addr=0, size=0). The V2 publisher uses them to smuggle metadata
        # past the MX server's field-dropping; they aren't real RDMA
        # targets. Leaving them in the source_tensors list propagates a
        # (0,0,0) descriptor into prep_xfer_dlist which UCX rejects.
        source_tensors = [
            TensorDescriptor(
                name=t.name,
                addr=t.addr,
                size=t.size,
                device_id=t.device_id,
                dtype=t.dtype,
            )
            for t in worker.tensors
            if not t.name.startswith("__mx_") and t.size > 0
            # Wire-prune: when the caller knows which tensors it wants (EP expert
            # filter, non-expert dedup across EP sources), pull ONLY those over
            # RDMA instead of the source's full published set.
            and (include_names is None or t.name in include_names)
        ]

        scratch_tensors: dict[str, torch.Tensor] = {}
        scratch_shapes: dict[str, tuple[int, ...]] = {}
        for td in source_tensors:
            dt = _DTYPE_MAP.get(td.dtype, torch.bfloat16)
            elem_size = torch.tensor([], dtype=dt).element_size()
            numel = td.size // elem_size
            scratch_tensors[td.name] = torch.empty(
                numel, dtype=dt, device=f"cuda:{self._device_id}"
            )
            scratch_shapes[td.name] = (numel,)

        logger.info(
            f"Allocated {len(scratch_tensors)} scratch buffers "
            f"({sum(t.numel() * t.element_size() for t in scratch_tensors.values()) / 1e9:.2f} GB)"
        )

        self._nixl.register_tensors(scratch_tensors)

        transferred, skipped, elapsed = self._nixl.receive_from_source(
            source_metadata=worker.nixl_metadata,
            source_tensors=source_tensors,
            timeout_seconds=timeout_seconds,
        )

        bandwidth_gbps = (transferred * 8) / (elapsed * 1e9) if elapsed > 0 else 0.0
        logger.info(
            f"RDMA transfer complete: {transferred / 1e9:.2f} GB, "
            f"{len(source_tensors)} tensors, {elapsed:.2f}s, "
            f"{bandwidth_gbps:.1f} Gbps (step={source.training_step})"
        )

        self._current_step = source.training_step

        for name, tensor in scratch_tensors.items():
            if tensor_shapes and name in tensor_shapes:
                tensor = tensor.view(tensor_shapes[name])
            yield name, tensor

    def pull_to(
        self,
        source: SourceRef,
        requests: list[tuple[str, tuple[int, int] | None, torch.Tensor]],
        timeout_seconds: float = 300.0,
    ) -> tuple[int, int, float]:
        """Pull specific sub-slices of source tensors into specific dest views.

        v1 sliced-pull primitive — the bandwidth-optimal mixed-TP data plane.
        Each receiver pulls only the bytes it actually needs from each
        source rank, instead of pulling the full source manifest and
        slicing on the host (the v0 fallback in
        :meth:`receive_weights_scratch`).

        Args:
            source: ``SourceRef`` from :meth:`discover_v2_sources` / poll.
            requests: per-slice pull spec, each a tuple of:

                * ``name``: source tensor name (must match an entry in the
                  source's published manifest).
                * ``source_subslice``: ``(lo_elements, hi_elements)`` along
                  the source tensor's flat byte order, OR ``None`` to pull
                  the full source tensor. The element unit lets callers
                  express "rows [lo, hi)" naturally for axis-0 sharded
                  tensors; the helper converts to bytes using the source's
                  declared dtype. To pull at byte granularity, pass
                  ``(byte_lo, byte_hi)`` and the dest_view's element size
                  must agree.
                * ``dest_view``: local destination, **must be contiguous**.
                  Its byte size must equal the slice's byte size (after
                  conversion). For axis-0 narrows of a pre-allocated dest
                  buffer (e.g. ``buffers[name].narrow(0, lo, hi - lo)``)
                  the view is contiguous and direct RDMA works. For
                  axis-1 narrows the view is non-contiguous; the caller
                  must fall back to ``receive_weights_scratch`` + host
                  copy for those.
            timeout_seconds: as for :meth:`receive_weights`.

        Returns:
            ``(total_bytes_transferred, num_slices, elapsed_seconds)``.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before pull_to()")

        meta_resp = self._client.get_metadata(
            mx_source_id=source.mx_source_id,
            worker_id=source.worker_id,
        )
        if not meta_resp.found:
            raise RuntimeError(
                f"Source {source.mx_source_id}/{source.worker_id} not found on MX Server"
            )
        worker = meta_resp.worker
        # Filter v2 sidecar descriptors.
        source_tensors = [
            TensorDescriptor(
                name=t.name, addr=t.addr, size=t.size,
                device_id=t.device_id, dtype=t.dtype,
            )
            for t in worker.tensors
            if not t.name.startswith("__mx_") and t.size > 0
        ]
        by_name = {t.name: t for t in source_tensors}

        # Build SlicedTransferRequest list. source_subslice is in elements
        # along the FLAT byte order — we convert to bytes using the source
        # tensor's dtype element size.
        from .nixl_transfer import SlicedTransferRequest

        slice_requests: list[SlicedTransferRequest] = []
        for name, subslice, dest_view in requests:
            src = by_name.get(name)
            if src is None:
                raise RuntimeError(f"pull_to: tensor {name!r} not in source manifest")
            elem_size = torch.tensor([], dtype=_DTYPE_MAP.get(src.dtype, torch.bfloat16)).element_size()
            if subslice is None:
                source_offset_bytes = 0
                slice_bytes = src.size
            else:
                lo, hi = subslice
                source_offset_bytes = int(lo) * elem_size
                slice_bytes = int(hi - lo) * elem_size
            slice_requests.append(SlicedTransferRequest(
                name=name,
                source_offset_bytes=source_offset_bytes,
                slice_bytes=slice_bytes,
                dest_view=dest_view,
            ))

        transferred, num_slices, elapsed = self._nixl.receive_sliced_from_source(
            source_metadata=worker.nixl_metadata,
            source_tensors=source_tensors,
            slice_requests=slice_requests,
            timeout_seconds=timeout_seconds,
        )
        self._current_step = source.training_step
        logger.info(
            f"pull_to: {num_slices} slices, {transferred / 1e9:.2f} GB, "
            f"{elapsed:.2f}s (step={source.training_step})"
        )
        return transferred, num_slices, elapsed

    def receive_weights_from_metadata(
        self,
        nixl_metadata: bytes,
        source_tensors: list[TensorDescriptor],
        training_step: int,
        timeout_seconds: float = 300.0,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Receive weights when metadata is already available (bypasses MX Server query).

        Useful when the orchestrator passes metadata directly instead of
        having the worker poll the MX Server.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() first")

        transferred, skipped, elapsed = self._nixl.receive_from_source(
            source_metadata=nixl_metadata,
            source_tensors=source_tensors,
            timeout_seconds=timeout_seconds,
        )

        logger.info(
            f"RDMA transfer (direct metadata): {transferred} bytes, "
            f"{len(source_tensors)} tensors, {elapsed:.2f}s"
        )

        self._current_step = training_step

        for td in source_tensors:
            if td.name in self._nixl._tensors:
                yield td.name, self._nixl._tensors[td.name]

    def prefetch_source(self, mx_source_id: str, worker_id: str) -> str:
        """Resolve + cache the NIXL remote-agent handle for a published source.

        Calls ``MxClient.get_metadata`` once to obtain the source's NIXL
        metadata blob, then loads it into the local NIXL agent via
        ``add_remote_agent``. Returns the resulting remote-agent-name —
        the caller passes this string to :meth:`receive_segment` so per-
        segment RDMA reads avoid the gRPC round-trip + metadata load.

        Caches results in ``self._remote_agents`` keyed by ``(source_id,
        worker_id)`` so repeated calls for the same source are O(1).

        This is the metadata-plane half of the rank-to-rank contract; the
        data-plane half is :meth:`receive_segment`.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before prefetch_source()")
        if not hasattr(self, "_remote_agents"):
            self._remote_agents: dict[tuple[str, str], str] = {}

        key = (mx_source_id, worker_id)
        if key in self._remote_agents:
            return self._remote_agents[key]

        meta_resp = self._client.get_metadata(
            mx_source_id=mx_source_id, worker_id=worker_id
        )
        if not meta_resp.found:
            raise RuntimeError(
                f"prefetch_source: source {mx_source_id}/{worker_id} not on MX server"
            )
        remote_agent_name = self._nixl._agent.add_remote_agent(
            meta_resp.worker.nixl_metadata
        )
        self._remote_agents[key] = remote_agent_name
        logger.info(
            "prefetch_source: cached agent=%s for source_id=%s worker_id=%s "
            "(metadata=%d bytes)",
            remote_agent_name,
            mx_source_id,
            worker_id,
            len(meta_resp.worker.nixl_metadata),
        )
        return remote_agent_name

    def receive_segment(
        self,
        *,
        remote_agent_name: str,
        source_addr: int,
        byte_count: int,
        target_addr: int,
        source_device_id: int = 0,
        timeout_seconds: float = 60.0,
    ) -> float:
        """One-shot rank-to-rank RDMA READ — the Gen 3 data-plane primitive.

        Issues a single NIXL READ for ``byte_count`` bytes from
        ``source_addr`` (on the remote GPU identified by ``remote_agent_name``
        + ``source_device_id``) into ``target_addr`` on the local GPU.

        Unlike :meth:`receive_weights` this is name-free and tensor-free —
        the caller pre-computes the absolute byte addresses on both sides
        and the segment count is exactly 1. This is what
        :class:`VerlMxRolloutLoader` (and the PrimeRL ``mx_v2`` worker)
        iterates over once per :class:`SegmentPlan` after the planner has
        intersected source ownerships with receiver requests.

        Args:
            remote_agent_name: Cached agent string from
                :meth:`prefetch_source`.
            source_addr: Absolute GPU address on the source side, in bytes.
                For a sharded source this is typically
                ``ownership.nixl_addr + source_range[0] * row_stride``.
            byte_count: Bytes to read. Must match a contiguous range on
                both sides (the receiver does no scatter).
            target_addr: Absolute GPU address on the local side, in bytes.
                Typically ``request.target_addr + request.target_offset +
                target_range[0] * row_stride``.
            source_device_id: CUDA device index on the source side.
                Defaults to 0 (matches how MxTrainingPublisher writes its
                shard descriptors today).
            timeout_seconds: Max time to wait for the transfer.

        Returns:
            Elapsed seconds for the transfer.

        Raises:
            TimeoutError: if NIXL doesn't complete within ``timeout_seconds``.
            RuntimeError: if NIXL reports an error state.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before receive_segment()")
        if self._nixl is None:
            raise RuntimeError("NIXL not initialized")

        torch.cuda.set_device(self._device_id)
        start = time.perf_counter()

        # Build the 1-entry descriptor list for both sides.
        remote_desc = [(source_addr, byte_count, source_device_id)]
        local_desc = [(target_addr, byte_count, self._device_id)]

        src_prepped = self._nixl._agent.prep_xfer_dlist(
            agent_name=remote_agent_name,
            xfer_list=remote_desc,
            mem_type="cuda",
            backends=["UCX"],
        )
        dst_prepped = self._nixl._agent.prep_xfer_dlist(
            agent_name="",
            xfer_list=local_desc,
            mem_type="cuda",
            backends=["UCX"],
        )

        handle = self._nixl._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=dst_prepped,
            local_indices=[0],
            remote_xfer_side=src_prepped,
            remote_indices=[0],
            backends=["UCX"],
        )
        self._nixl._agent.transfer(handle)

        wait_start = time.perf_counter()
        while True:
            if time.perf_counter() - wait_start >= timeout_seconds:
                self._nixl._agent.release_xfer_handle(handle)
                raise TimeoutError(
                    f"receive_segment: transfer of {byte_count} bytes timed out"
                )
            status = self._nixl._agent.check_xfer_state(handle)
            if status in ("DONE", "SUCCESS"):
                self._nixl._agent.release_xfer_handle(handle)
                break
            if status in ("ERR", "ERROR", "FAIL"):
                self._nixl._agent.release_xfer_handle(handle)
                raise RuntimeError(f"receive_segment: transfer failed status={status}")
            time.sleep(0.0005)

        # RDMA writes bypass CUDA streams — sync so subsequent kernels see them.
        torch.cuda.synchronize(self._device_id)
        return time.perf_counter() - start

    def shutdown(self) -> None:
        """Release NIXL agent and close gRPC channel."""
        if self._nixl is not None:
            self._nixl.shutdown()
            self._nixl = None
        if self._client is not None:
            self._client.close()
            self._client = None
        self._initialized = False
        logger.info(f"MxRefitReceiver shut down: {self._agent_name}")
