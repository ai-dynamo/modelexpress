# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RDMA P2P loading strategy: receive weights from an existing source via NIXL."""

from __future__ import annotations

import logging
import time

from .. import envs
from ..adapter import EngineAdapter, StrategyFailed
from .base import (
    LoadContext,
    LoadStrategy,
    SourceTransferError,
    _as_load_result,
    register_tensors,
)
from .context import LoadResult
from ..metadata.payload import (
    accelerators_compatible,
    worker_tensor_count,
    worker_tensor_descriptors,
)
from ..nixl_transfer import is_nixl_available
from ..source_selection import (
    configured_policy_label,
    get_configured_selector,
)
from ..metrics import metrics as selection_metrics
from ..transfer_safety import check_transfer_allowed
from ..types import TensorDescriptor
from .. import p2p_pb2

logger = logging.getLogger("modelexpress.strategy_rdma")

MAX_SOURCE_RETRIES = 3


class RdmaStrategy(LoadStrategy):
    """Load weights via RDMA P2P transfer from an existing source.

    Overrides load() entirely since RDMA has a fundamentally different flow:
    prepare target storage -> RDMA receive -> register + publish.
    """

    name = "rdma"
    requires = (EngineAdapter.discover_tensors,)

    def rollback(self, ctx: LoadContext) -> None:
        """Clean up NIXL state from a failed RDMA target attempt."""
        if ctx.nixl_manager is not None:
            ctx.nixl_manager.shutdown()
        ctx.tensors = {}
        ctx.nixl_manager = None

    def is_available(self, ctx: LoadContext) -> bool:
        if not ctx.p2p_enabled:
            return False
        if not super().is_available(ctx):
            return False
        if not is_nixl_available():
            return False

        if not ctx.accelerator_backend.supports_rdma_p2p():
            logger.info(
                f"[Worker {ctx.global_rank}] Backend "
                f"{ctx.accelerator_backend.name} does not support RDMA P2P, skipping"
            )
            return False

        # Decentralized backends (k8s-service) serve their own
        # metadata; skip the central-server precondition for them.
        # Strict `is True` check so MagicMock's auto-attribute doesn't
        # masquerade as the flag in tests.
        server_addr = envs.MODEL_EXPRESS_URL or envs.MX_SERVER_ADDRESS
        requires_p2p = getattr(ctx.mx_client, "REQUIRES_P2P_METADATA", False) is True
        if not server_addr and not requires_p2p:
            logger.info(f"[Worker {ctx.global_rank}] No MX server configured, skipping RDMA")
            return False

        allowed, reason = check_transfer_allowed(ctx.model_config)
        if not allowed:
            logger.info(
                f"[Worker {ctx.global_rank}] RDMA transfer disabled: {reason}"
            )
            return False

        return True

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        """Load from a READY source or raise StrategyFailed for fallback.

        Source discovery and metadata misses do not mutate the target model.
        A failed transfer retries the next candidate. If it may have mutated
        the model, the adapter first replaces the model with a fresh instance.
        """
        result = _as_load_result(result)
        candidates = self._find_source_instances(ctx)
        if not candidates:
            logger.info(f"[Worker {ctx.global_rank}] No RDMA source available, skipping")
            raise StrategyFailed("No RDMA source available", mutated=False)

        attempts = candidates[:MAX_SOURCE_RETRIES]
        policy = configured_policy_label()
        needs_outer_reinit = False
        for attempt_index, instance in enumerate(attempts):
            mx_source_id = instance.mx_source_id
            worker_id = instance.worker_id
            logger.info(
                f"[Worker {ctx.global_rank}] Source attempt: "
                f"source_attempt_index={attempt_index} "
                f"source_worker_id={worker_id} mx_source_id={mx_source_id}"
            )

            try:
                source_worker = self._fetch_worker_metadata(
                    ctx, mx_source_id, worker_id,
                )
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Failed to fetch metadata for worker {worker_id}: {e}. "
                    f"Trying next candidate."
                )
                selection_metrics.record_metadata_failure(policy)
                selection_metrics.record_attempt(policy, "metadata_miss")
                continue

            if source_worker is None:
                selection_metrics.record_attempt(policy, "metadata_miss")
                continue

            if not self._accelerator_compatible(ctx, source_worker, worker_id):
                continue

            logger.info(
                f"[Worker {ctx.global_rank}] Trying source worker {worker_id} "
                f"({worker_tensor_count(source_worker)} tensors)"
            )

            selection_metrics.record_selection(policy, worker_id)
            transfer_start = time.perf_counter()
            try:
                out = self._load_as_target(
                    result, ctx, source_worker, mx_source_id, worker_id,
                )
            except StrategyFailed as e:
                has_next_candidate = attempt_index + 1 < len(attempts)
                selection_metrics.observe_transfer_seconds(
                    policy,
                    "retry" if has_next_candidate else "fallback",
                    time.perf_counter() - transfer_start,
                )
                selection_metrics.record_attempt(
                    policy,
                    "transfer_retry" if has_next_candidate else "transfer_fallback",
                )
                if not has_next_candidate:
                    if needs_outer_reinit and not e.mutated:
                        raise StrategyFailed(str(e), mutated=True) from e
                    raise

                logger.warning(
                    f"[Worker {ctx.global_rank}] RDMA source worker {worker_id} "
                    f"failed: {e}. Trying next candidate."
                )
                try:
                    self.rollback(ctx)
                except Exception as cleanup_error:
                    raise StrategyFailed(
                        f"Failed to clean up target after source worker "
                        f"{worker_id} failed: {cleanup_error}",
                        mutated=True,
                    ) from cleanup_error
                if e.mutated:
                    try:
                        result = ctx.adapter.reinit_for_retry(result)
                    except Exception as reinit_error:
                        raise StrategyFailed(
                            f"Failed to reinitialize target after source worker "
                            f"{worker_id} failed: {reinit_error}",
                            mutated=True,
                        ) from reinit_error
                    needs_outer_reinit = True
                continue
            except BaseException:
                selection_metrics.observe_transfer_seconds(
                    policy, "fallback", time.perf_counter() - transfer_start
                )
                selection_metrics.record_attempt(policy, "transfer_fallback")
                raise
            selection_metrics.observe_transfer_seconds(
                policy, "success", time.perf_counter() - transfer_start
            )
            selection_metrics.record_attempt(policy, "success")
            return out

        tried = min(len(candidates), MAX_SOURCE_RETRIES)
        logger.warning(
            f"[Worker {ctx.global_rank}] Tried {tried} of {len(candidates)} source workers "
            f"(max retries={MAX_SOURCE_RETRIES}), falling through"
        )
        # An internal reinit returns a new result, but the outer strategy chain
        # still owns the original result that the adapter cleared.
        raise StrategyFailed(
            "No RDMA source succeeded", mutated=needs_outer_reinit,
        )

    def _find_source_instances(
        self, ctx: LoadContext,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        """Return READY source instances ranked by the configured selector.

        Filters listed instances to the target's worker_rank, then delegates
        ordering to the policy named by MX_P2P_SOURCE_SELECTOR (default
        ``random``). The retry slice (MAX_SOURCE_RETRIES) is applied by the
        caller in load(), so the selector controls ordering only.
        """
        try:
            list_resp = ctx.mx_client.list_sources(
                identity=ctx.identity,
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
            if not list_resp.instances:
                logger.debug(f"[Worker {ctx.global_rank}] No ready source instances found")
                return []

            rank_matched = [
                inst for inst in list_resp.instances
                if inst.worker_rank == ctx.worker_rank
            ]

            # Drop accelerator-incompatible sources before ordering and the
            # retry-cap slice, so the selector only ranks eligible peers and a
            # compatible source can never be pushed past MAX_SOURCE_RETRIES by
            # incompatible ones. The post-GetMetadata check in load() stays as
            # defense-in-depth (empty refs, stale records, metadata drift).
            target_accelerator = ctx.accelerator_backend.name
            candidates = [
                inst for inst in rank_matched
                if accelerators_compatible(
                    target_accelerator,
                    inst.accelerator,
                    mx_source_type=ctx.identity.mx_source_type,
                    quantization=ctx.identity.quantization,
                    dtype=ctx.identity.dtype,
                )
            ]

            selector = get_configured_selector()
            select_start = time.perf_counter()
            ordered = selector.order(candidates, ctx)
            select_seconds = time.perf_counter() - select_start

            selection_metrics.observe_candidates(
                selector.name, "listed", len(list_resp.instances)
            )
            selection_metrics.observe_candidates(
                selector.name, "rank_matched", len(rank_matched)
            )
            selection_metrics.observe_candidates(
                selector.name, "accelerator_matched", len(candidates)
            )
            selection_metrics.observe_selection_seconds(selector.name, select_seconds)

            # Surface the source-published load the client saw per candidate, so a
            # dashboard can compare how load_aware steers vs random/rendezvous.
            for inst in ordered:
                selection_metrics.set_source_load(
                    inst.worker_id, getattr(inst, "source_load", 0.0)
                )

            logger.info(
                f"[Worker {ctx.global_rank}] Source selection: "
                f"source_selector={selector.name} "
                f"source_candidates_total={len(list_resp.instances)} "
                f"source_candidates_rank_matched={len(rank_matched)} "
                f"source_candidates_accelerator_matched={len(candidates)}"
            )
            if ordered:
                logger.debug(
                    f"[Worker {ctx.global_rank}] Ranked source workers: "
                    f"{[inst.worker_id for inst in ordered]}"
                )
            return ordered

        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Error listing sources, falling through: {e}"
            )
            return []

    def _accelerator_compatible(
        self,
        ctx: LoadContext,
        source_worker: p2p_pb2.WorkerMetadata,
        worker_id: str,
    ) -> bool:
        """Return whether source and target accelerator metadata are compatible.

        Defense-in-depth re-check on the authoritative ``WorkerMetadata`` after
        GetMetadata: the pre-slice filter in ``_find_source_instances`` uses the
        lightweight ``SourceInstanceRef.accelerator``, which may be empty on old
        servers or drift between list and fetch.
        """
        target_accelerator = ctx.accelerator_backend.name
        source_accelerator = source_worker.accelerator
        if accelerators_compatible(
            target_accelerator,
            source_accelerator,
            mx_source_type=ctx.identity.mx_source_type,
            quantization=ctx.identity.quantization,
            dtype=ctx.identity.dtype,
            defer_unknown=False,
        ):
            return True

        logger.info(
            f"[Worker {ctx.global_rank}] Skipping source worker {worker_id}: "
            f"accelerator mismatch source={source_accelerator!r}, "
            f"target={target_accelerator!r}"
        )
        return False

    def _fetch_worker_metadata(
        self,
        ctx: LoadContext,
        mx_source_id: str,
        worker_id: str,
    ) -> p2p_pb2.WorkerMetadata | None:
        """Fetch tensor metadata for one worker."""
        fetch_start = time.perf_counter()
        metadata_resp = ctx.mx_client.get_metadata(
            mx_source_id=mx_source_id,
            worker_id=worker_id,
        )
        if not metadata_resp.found:
            logger.debug(
                f"[Worker {ctx.global_rank}] Metadata not found for worker {worker_id}, skipping"
            )
            return None
        worker = metadata_resp.worker
        has_tensor_descriptors = bool(worker_tensor_descriptors(worker))
        if not has_tensor_descriptors and not worker.worker_grpc_endpoint:
            logger.debug(
                f"[Worker {ctx.global_rank}] Worker {worker_id} has no tensors "
                f"and no P2P endpoint, skipping"
            )
            return None
        # A worker ID requires endpoint validation. Without one, fetch only
        # when the metadata response did not already include the manifest.
        needs_manifest_prefetch = worker_id != "" or not has_tensor_descriptors
        if worker.worker_grpc_endpoint and needs_manifest_prefetch:
            self._prefetch_tensor_manifest(ctx, worker, mx_source_id, worker_id)
        fetch_time = time.perf_counter() - fetch_start
        mode = "P2P (lightweight)" if worker.worker_grpc_endpoint else "centralized"
        tensor_count = worker_tensor_count(worker)
        logger.info(
            f"[Worker {ctx.global_rank}] [TIMING] GetMetadata ({mode}): "
            f"{fetch_time:.3f}s, {tensor_count} tensors"
        )
        return worker

    def _prefetch_tensor_manifest(
        self,
        ctx: LoadContext,
        source_worker: p2p_pb2.WorkerMetadata,
        mx_source_id: str,
        worker_id: str,
    ) -> None:
        """Fetch once before target preparation and retain it for the transfer."""
        from ..metadata.worker_server import fetch_tensor_manifest

        manifest_start = time.perf_counter()
        tensor_protos, manifest_bytes = fetch_tensor_manifest(
            endpoint=source_worker.worker_grpc_endpoint,
            mx_source_id=mx_source_id,
            worker_id=worker_id,
        )
        # Store the validated descriptors on the metadata object so the
        # transfer can reuse them without a second manifest RPC.
        source_worker.tensor_source.ClearField("tensors")
        source_worker.tensor_source.tensors.extend(tensor_protos)
        manifest_time = time.perf_counter() - manifest_start
        logger.info(
            f"[Worker {ctx.global_rank}] [TIMING] P2P tensor manifest: "
            f"{manifest_time:.3f}s ({len(tensor_protos)} tensors, "
            f"{manifest_bytes} bytes)"
        )

    def _load_as_target(
        self,
        result: LoadResult,
        ctx: LoadContext,
        source_worker,
        mx_source_id: str,
        source_worker_id: str,
    ) -> LoadResult:
        """Receive fully-processed weights via RDMA from an existing source."""
        try:
            result = ctx.adapter.prepare_rdma_target(result)
            result = ctx.adapter.before_rdma_receive(result)
            self._receive_from_peer(result, ctx, source_worker, mx_source_id)
            return ctx.adapter.after_rdma_receive(result)
        except StrategyFailed:
            raise
        except Exception as e:
            raise StrategyFailed(str(e), mutated=True) from e

    def _receive_from_peer(
        self,
        result: LoadResult,
        ctx: LoadContext,
        source_worker,
        mx_source_id: str,
    ) -> None:
        """Receive fully-processed tensors via RDMA from the detected source."""
        receive_start = time.perf_counter()
        register_tensors(result, ctx)

        is_p2p = bool(source_worker.worker_grpc_endpoint)
        remote_agent_name_override = None

        if is_p2p:
            # _fetch_worker_metadata() prefetched and generation-validated
            # this manifest before _load_as_target() prepared target tensors.
            tensor_protos = worker_tensor_descriptors(source_worker)
            source_tensors = [
                TensorDescriptor(
                    name=t.name, addr=t.addr, size=t.size,
                    device_id=t.device_id, dtype=t.dtype,
                )
                for t in tensor_protos
            ]
            nixl_fetch_start = time.perf_counter()
            ep = source_worker.metadata_endpoint
            host, port_str = ep.rsplit(":", 1)
            ctx.nixl_manager.fetch_remote_and_wait(
                remote_agent_name=source_worker.agent_name,
                ip=host,
                port=int(port_str),
            )
            nixl_fetch_time = time.perf_counter() - nixl_fetch_start
            logger.info(
                f"[Worker {ctx.global_rank}] [TIMING] P2P NIXL metadata fetch: "
                f"{nixl_fetch_time:.3f}s"
            )
            remote_agent_name_override = source_worker.agent_name
        else:
            source_tensors = [
                TensorDescriptor(
                    name=t.name, addr=t.addr, size=t.size,
                    device_id=t.device_id, dtype=t.dtype,
                )
                for t in worker_tensor_descriptors(source_worker)
            ]

        logger.info(
            f"[Worker {ctx.global_rank}] Receiving {len(source_tensors)} tensors from source"
            f"{' (P2P)' if is_p2p else ''}"
        )

        # Cross-family (heterogeneous) transfers must name the exact same tensor
        # set on both sides: a name diff can mean vendor-specific hidden/derived
        # tensors, which would leave part of the target at dummy values while
        # RDMA reports success. Same-family transfers tolerate subset transfers.
        target_accelerator = ctx.accelerator_backend.name
        source_accelerator = source_worker.accelerator
        require_exact_match = bool(
            target_accelerator
            and source_accelerator
            and target_accelerator != source_accelerator
        )

        transfer_start = time.perf_counter()
        try:
            bytes_transferred, tensor_count, _ = ctx.nixl_manager.receive_from_source(
                source_metadata=source_worker.nixl_metadata,
                source_tensors=source_tensors,
                timeout_seconds=300.0,
                remote_agent_name=remote_agent_name_override,
                require_exact_match=require_exact_match,
            )
        except Exception as e:
            raise SourceTransferError(f"RDMA receive failed: {e}") from e
        transfer_time = time.perf_counter() - transfer_start

        bandwidth_gbps = (bytes_transferred * 8) / (transfer_time * 1e9) if transfer_time > 0 else 0
        logger.info(
            f"[Worker {ctx.global_rank}] [TIMING] RDMA transfer complete: "
            f"{tensor_count} tensors, {bytes_transferred / 1e9:.2f} GB, "
            f"{transfer_time:.3f}s, {bandwidth_gbps:.1f} Gbps"
        )

        ctx.accelerator_backend.synchronize()

        total_time = time.perf_counter() - receive_start
        logger.info(f"[Worker {ctx.global_rank}] [TIMING] Total receive time: {total_time:.2f}s")
