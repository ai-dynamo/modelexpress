# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Engine-agnostic receiver for the no-gather slice-resharding weight refit.

``ReshardReceiver`` owns everything an inference engine needs to pull a resharded
weight update over NIXL that is NOT engine-specific: build this rank's NIXL agent,
discover + P2P-handshake the trainer shards, capture the model's own load
geometry, build the pull plan, allocate + register the receive/staging buffers,
and per refit RDMA the needed slices in + cast dtype-mismatched sources.

The two engine-specific steps are abstract hooks a subclass implements:

  * :meth:`_capture` - run the engine's ``load_weights`` with zero-storage
    placeholders (on a meta twin for a quantized model) to record where each
    source lands, and report the load-time param layout to size the buffers.
  * :meth:`_install` - install the RDMA'd receive buffers into the live params
    (a plain copy for bf16, or re-quantize via the engine's post-load path).

So an sglang / trtllm receiver only implements those two hooks; discover, plan,
transport, buffers and the router dtype-cast are shared here.
"""

from __future__ import annotations

import logging

import torch

from modelexpress.client import MxClient
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.refit.reshard.cuda_pool import classic_cuda_alloc
from modelexpress.refit.reshard.rendezvous import gather_sources
from modelexpress.refit.reshard.transfer_plan import execute_transfer, plan_transfer
from modelexpress.refit.reshard.transport import (
    NixlReshardTransport,
    ReadDescriptor,
)
from modelexpress.refit.reshard.types import CaptureResult, UnsupportedReshard

logger = logging.getLogger("modelexpress.refit.reshard.receiver")


def _replay_ops(tensor: torch.Tensor, op_chain: tuple) -> torch.Tensor:
    """Replay a captured loader view chain on a staged full-source tensor."""
    value = tensor
    for op_name, args, frozen_kwargs in op_chain:
        kwargs = dict(frozen_kwargs)
        if op_name == "__getitem__":
            value = value.__getitem__(*args)
        else:
            value = getattr(value, op_name)(*args, **kwargs)
    return value


class ReshardReceiver:
    """Pull-mode slice-resharding weight receiver (engine-agnostic).

    Lifecycle: construct once (builds the NIXL agent + metadata client), then
    call :meth:`update_weights` per weight update. The first call lazily discovers the
    trainer shards, captures geometry, and builds the plan + buffers (cached);
    every refit re-reads the same trainer buffer addresses (now holding the
    step's refreshed weights).
    """

    def __init__(
        self,
        *,
        model_name: str,
        mx_server: str,
        agent_name: str,
        local_rank: int,
        global_rank: int,
        num_trainer_sources: int,
        device: "torch.device",
        listen_port: int,
        timeout: float = 1200.0,
    ) -> None:
        """Build this rank's NIXL agent + metadata client.

        Args:
            model_name: the served model name (the shared ``[model] name`` both
                trainer and inference inherit) - the rendezvous identity key.
            mx_server: ``host:port`` of the modelexpress metadata server.
            agent_name: this rank's NIXL agent name.
            local_rank: device index (the NIXL device id).
            global_rank: rendezvous rank (``rank_offset + local_rank``).
            num_trainer_sources: number of trainer ranks publishing shards (all
                must be discovered before planning, since a slice can fan in
                across ranks).
            device: the torch device receive buffers are allocated on.
            listen_port: NIXL listen port for this rank's agent. The receiver
                needs a listen thread (MX's P2P metadata exchange is
                bidirectional); the caller owns port assignment so it can avoid
                colliding with a colocated trainer publisher (which listens on
                ``MX_METADATA_PORT + device_id``).
            timeout: rendezvous / per-pull timeout seconds.
        """
        self._device = device
        self._model_name = model_name
        self._num_trainer_sources = num_trainer_sources
        self._timeout = timeout
        self._global_rank = global_rank

        # TODO(transport-agnostic): the receiver is engine-agnostic but still
        # transport-bound to NIXL (this manager, NixlReshardTransport, and the
        # fetch_remote_and_wait P2P handshake in _prepare). Abstract these behind
        # a transport interface so non-NIXL backends can plug in.
        self._manager = NixlTransferManager(
            agent_name=agent_name, device_id=local_rank, listen_port=listen_port
        )
        self._manager.initialize()
        self._mx_client = MxClient(server_url=mx_server)

        self._plan = None  # built lazily on the first refit
        self._transport: NixlReshardTransport | None = None
        self._recv_buffers: dict[
            str, torch.Tensor
        ] = {}  # param_name -> receive buffer at load-time layout
        self._staging: dict[
            str, torch.Tensor
        ] = {}  # dtype-convert param -> bf16 staging (RDMA target)
        self._staging_ptr: dict[str, int] = {}
        self._full_staging: dict[str, torch.Tensor] = {}
        self._full_staging_ptr: dict[str, int] = {}
        self._param_ptr: dict[
            str, int
        ] = {}  # segment param_name -> receive-buffer data_ptr

        logger.info(
            "[reshard] receiver init: agent=%s global_rank=%d trainer_sources=%d",
            agent_name,
            global_rank,
            num_trainer_sources,
        )

    # ------------------------------------------------------------- engine hooks
    def _capture(self, manifest: list) -> "tuple[CaptureResult, dict]":
        """Record where each published source lands in the engine's load-time
        param layout, without moving data.

        Returns ``(capture, param_layout)`` where ``param_layout`` is
        ``{param_name: (shape, dtype)}`` at the LOAD-TIME layout (bf16, pre-quant)
        - used to size the receive buffers. For a quantized model this is captured
        on a fresh meta twin (the live params are post-quantization); for a bf16
        model it may be the live model directly."""
        raise NotImplementedError

    def _install(self, recv_buffers: dict) -> None:
        """Install the RDMA'd receive buffers into the live params.

        For a bf16 model this is effectively making the buffers the live params;
        for a quantized model it re-runs the engine's post-load processing
        (quantize + derive) with the buffers as the load-time params. Must be
        CUDA-graph-safe (write into the graph-bound storage)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ prepare
    def _prepare(self, timeout: float) -> None:
        """One-time: discover trainer shards, connect their agents, capture load
        geometry, build the pull plan, and allocate + register buffers."""
        logger.info(
            "[reshard] _prepare: discovering %d trainer source(s) (timeout=%.0fs)",
            self._num_trainer_sources,
            timeout,
        )
        sources, session_to_agent, session_to_device, agent_endpoints = gather_sources(
            self._mx_client,
            expected_trainers=self._num_trainer_sources,
            model_name=self._model_name,
            role="inference",
            rank=self._global_rank,
            timeout=timeout,
        )
        logger.info(
            "[reshard] _prepare: discovered %d source(s), %d agent(s); P2P-fetching remote metadata",
            len(sources),
            len(agent_endpoints),
        )
        # P2P memory handshake (mirrors MX's vLLM RDMA path): fetch each trainer's
        # NIXL metadata (incl. its memory registrations) via its listen thread, so
        # prep_xfer_dlist can resolve the remote addresses. The central
        # add_remote_agent(blob) path does NOT convey the registrations.
        for agent_name, endpoint in agent_endpoints.items():
            host, port_str = endpoint.rsplit(":", 1)
            self._manager.fetch_remote_and_wait(
                agent_name, host, int(port_str), timeout_seconds=timeout
            )

        manifest = [
            (name, src.dtype, tuple(src.global_shape)) for name, src in sources.items()
        ]
        logger.info(
            "[reshard] _prepare: capturing geometry over %d manifest entries",
            len(manifest),
        )
        capture, param_layout = self._capture(manifest)
        logger.info(
            "[reshard] _prepare: captured %d copies, %d unsupported",
            len(capture.copies),
            len(capture.unsupported),
        )

        # The plan encodes THIS discovery's topology: each trainer's registered
        # buffer addresses, per-source shard boundaries, and fan-in across ranks.
        # It is built once and reused every step (see the guard in
        # update_weights), which assumes the trainer set + their shard layout +
        # their buffer addresses are stable for the run.
        plan = plan_transfer(capture, sources)
        if plan.fallback:
            # Fallback params are dropped from the RDMA plan and never pulled or
            # installed, so they would silently keep their initial (base-model)
            # weights for the entire run. Until the full-pull/loader path exists
            # (TODO), fail loudly rather than serve stale weights.
            raise UnsupportedReshard(
                f"[reshard] {len(plan.fallback)} source(s) need the unimplemented "
                f"full-pull path (unsupported reshard ops); refusing to serve stale "
                f"weights. Params: {plan.fallback[:10]}"
            )
        self._transport = NixlReshardTransport(
            self._manager, session_to_agent, session_to_device, timeout_seconds=timeout
        )
        self._plan = plan

        # dtype-mismatched sources (e.g. a bf16-served router for an fp32 dest):
        # one persistent bf16 STAGING buffer per convert param, registered as an
        # RDMA target (classic cudaMalloc so the HCA can RDMA into it); each refit
        # we RDMA into staging then cast staging -> the (load-time) receive buffer.
        self._staging = {}
        self._staging_ptr = {}
        if plan.converts:
            with classic_cuda_alloc():
                self._staging = {
                    c.param_name: torch.empty(
                        c.dest_shape, dtype=c.src_dtype, device=self._device
                    )
                    for c in plan.converts
                }
            self._manager.register_tensors(
                {f"__stage__{n}": t for n, t in self._staging.items()}
            )
            self._staging_ptr = {n: t.data_ptr() for n, t in self._staging.items()}

        # Descriptor-heavy strided copies pull each complete source into one
        # persistent contiguous staging tensor, then replay captured loader views
        # locally. Each source shard contributes one bounded descriptor.
        self._full_staging = {}
        self._full_staging_ptr = {}
        if plan.full_pulls:
            with classic_cuda_alloc():
                self._full_staging = {
                    full_pull.src_name: torch.empty(
                        full_pull.global_shape,
                        dtype=full_pull.dtype,
                        device=self._device,
                    )
                    for full_pull in plan.full_pulls
                }
            self._manager.register_tensors(
                {
                    f"__full__{name}": tensor
                    for name, tensor in self._full_staging.items()
                }
            )
            self._full_staging_ptr = {
                name: tensor.data_ptr() for name, tensor in self._full_staging.items()
            }

        # Receive buffers: one per captured param at its CAPTURED (load-time)
        # shape/dtype, classic cudaMalloc, registered once. The live params are
        # NOT RDMA targets; _install() writes the buffers into the live params.
        # Segment params (captured == served) are the RDMA targets - register them
        # + point _param_ptr at them. Convert params (router) are captured fp32 ->
        # their bf16 staging is the RDMA target and the refit casts into the buffer.
        all_params = sorted({c.param_name for c in capture.copies})
        seg_params = {seg.param_name for seg in plan.segments}
        self._recv_buffers = {}
        with classic_cuda_alloc():
            for name in all_params:
                shape, dtype = param_layout[name]
                self._recv_buffers[name] = torch.empty(
                    tuple(shape), dtype=dtype, device=self._device
                )
        self._param_ptr = {}
        if seg_params:
            self._manager.register_tensors(
                {f"__recv__{n}": self._recv_buffers[n] for n in seg_params}
            )
            for name in seg_params:
                self._param_ptr[name] = self._recv_buffers[name].data_ptr()

        logger.info(
            "[reshard] prepared: %d descriptor(s), %d full-pull source(s), "
            "%d convert(s), %.1f MB/pull, %d descriptor(s) saved, "
            "%.1f MB extra wire, %d unbounded source(s), %d fallback",
            plan.descriptor_count(),
            len(plan.full_pulls),
            len(plan.converts),
            plan.bytes_planned() / 1e6,
            plan.descriptor_savings(),
            plan.extra_wire_bytes() / 1e6,
            len(plan.unbounded_sources),
            len(plan.fallback),
        )

    # ----------------------------------------------------------- update_weights
    @torch.no_grad()
    def update_weights(self, step: int, *, timeout: float | None = None) -> dict:
        """RDMA-pull the needed slices into the receive buffers, cast the
        dtype-mismatched ones, then install into the live params."""
        timeout = timeout if timeout is not None else self._timeout
        # TODO(re-plan on topology change): the plan is built once and cached, so
        # a mid-run change in the trainer set - a trainer restart (new buffer
        # addresses), a reshard (new shard boundaries / fan-in), or scaling the
        # trainer count - is NOT picked up; every step re-reads the first
        # discovery's addresses. Adapt the plan when topology changes (e.g.
        # re-discover + rebuild if a version/epoch token or address set differs).
        if self._plan is None:
            self._prepare(timeout)
        assert self._plan is not None and self._transport is not None

        # RDMA the sliced bf16 into the receive buffers (segments) and per-param
        # staging (dtype-convert / router). No live param is written by RDMA.
        stats = execute_transfer(
            self._plan,
            resolve_param_ptr=lambda name: self._param_ptr[name],
            transport=self._transport,
        )
        if self._plan.full_pulls:
            full_descriptors = [
                ReadDescriptor(
                    session=segment.session,
                    src_addr=segment.src_addr,
                    dst_addr=(
                        self._full_staging_ptr[full_pull.src_name] + segment.dst_byte
                    ),
                    nbytes=segment.nbytes,
                )
                for full_pull in self._plan.full_pulls
                for segment in full_pull.segments
            ]
            self._transport.read(full_descriptors)
            for full_pull in self._plan.full_pulls:
                full_tensor = self._full_staging[full_pull.src_name]
                for copy in full_pull.copies:
                    source_view = _replay_ops(full_tensor, copy.op_chain)
                    receive_buffer = self._recv_buffers[copy.param_name]
                    destination = receive_buffer.as_strided(
                        copy.dest_shape,
                        copy.dest_stride,
                        receive_buffer.storage_offset() + copy.dest_offset,
                    )
                    destination.copy_(source_view)
            stats["segments"] += len(full_descriptors)
            stats["bytes"] += sum(descriptor.nbytes for descriptor in full_descriptors)
        if self._plan.converts:
            conv_descs = [
                ReadDescriptor(
                    session=seg.session,
                    src_addr=seg.src_addr,
                    dst_addr=self._staging_ptr[c.param_name] + seg.dst_byte,
                    nbytes=seg.nbytes,
                )
                for c in self._plan.converts
                for seg in c.segments
            ]
            self._transport.read(conv_descs)
            # Cast the served bf16 staging into the (fp32) receive buffer - a torch
            # op, so the RDMA never crosses dtypes. _install writes the buffer.
            for c in self._plan.converts:
                self._recv_buffers[c.param_name].copy_(self._staging[c.param_name])
            stats["segments"] += len(conv_descs)
            stats["bytes"] += sum(descriptor.nbytes for descriptor in conv_descs)

        self._install(self._recv_buffers)
        torch.cuda.synchronize(self._device)

        metrics = {
            "step": step,
            "bytes_received": stats["bytes"],
            "segments": stats["segments"],
            "converts": len(self._plan.converts),
            "full_pull_sources": len(self._plan.full_pulls),
            "exact_descriptors": self._plan.exact_descriptor_count,
            "descriptor_savings": self._plan.descriptor_savings(),
            "extra_wire_bytes": self._plan.extra_wire_bytes(),
            "unbounded_sources": len(self._plan.unbounded_sources),
            "fallback": len(stats["fallback"]),
        }
        logger.info(
            "[reshard] refit step=%d bytes=%.1fMB descriptors=%d "
            "(saved=%d, extra_wire=%.1fMB) full_pulls=%d converts=%d "
            "unbounded=%d fallback=%d",
            step,
            stats["bytes"] / 1e6,
            stats["segments"],
            self._plan.descriptor_savings(),
            self._plan.extra_wire_bytes() / 1e6,
            len(self._plan.full_pulls),
            len(self._plan.converts),
            len(self._plan.unbounded_sources),
            len(stats["fallback"]),
        )
        return metrics
