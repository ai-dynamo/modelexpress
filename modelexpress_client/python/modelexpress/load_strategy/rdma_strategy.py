# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RDMA P2P loading strategy: receive weights from an existing source via NIXL.

Supports v1 (static-model) sources and v2 (training-step-aware) sources.
When v2 candidates are present, the freshest is picked deterministically
(trainer > inference_replica, then highest ``training_step``, then most
recent ``updated_at``) and the receive path switches to scratch buffers
so HF-named trainer tensors get fed through vLLM's ``load_weights``
HF→fused merger. v1 sources keep the legacy random-shuffle pick and
direct-buffer NIXL receive.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field

import torch

from ..adapter import EngineAdapter, StrategyFailed
from .base import (
    LoadContext,
    LoadStrategy,
    SourceTransferError,
    _as_load_result,
    register_tensors,
)
from .context import LoadResult
from ..nixl_transfer import is_nixl_available
from ..transfer_safety import check_transfer_allowed
from ..types import TensorDescriptor
from .. import p2p_pb2

logger = logging.getLogger("modelexpress.strategy_rdma")

MAX_SOURCE_RETRIES = 3


def _enrich_identity_v2(
    ctx: LoadContext,
    *,
    training_step: int,
    role: str,
) -> None:
    """Stamp v2 markers onto ``ctx.identity.extra_parameters``.

    Additive: v1 receivers still see the source via the existing fields
    (model_name, tp/pp/ep) and ignore ``extra_parameters``. v2 receivers
    parse the markers via :func:`nemo_rl_v2.parse_v2_extras` and use the
    ``training_step`` to pick the freshest source.

    Keeps the format byte-identical to what
    :meth:`MxV2RefitReceiver.publish_self_as_source` writes, so a v2
    receiver can't tell a loader-published replica from a refit-published
    replica.
    """
    extras = ctx.identity.extra_parameters
    extras["mx_v2"] = "1"
    extras["role"] = role
    extras["training_step"] = str(int(training_step))
    extras["worker_rank"] = str(ctx.worker_rank)
    extras["training_framework"] = "nemo_rl"
    ctx.loaded_version = int(training_step)


@dataclass
class _RdmaCandidate:
    """A discovered source with cached metadata and v2 extras (if any).

    The metadata fetch is done once during discovery so we can both inspect
    the v2 extras for sorting/filtering AND pass the same blob into the
    receive path without re-fetching.
    """
    instance: object  # p2p_pb2.SourceInstanceRef
    worker_meta: object  # p2p_pb2.WorkerMetadata
    v2_extras: dict[str, str] = field(default_factory=dict)

    @property
    def mx_source_id(self) -> str:
        return self.instance.mx_source_id

    @property
    def worker_id(self) -> str:
        return self.instance.worker_id

    @property
    def is_v2(self) -> bool:
        return self.v2_extras.get("mx_v2") == "1"

    @property
    def training_step(self) -> int:
        try:
            return int(self.v2_extras.get("training_step", "0"))
        except (TypeError, ValueError):
            return 0

    @property
    def role(self) -> str:
        return self.v2_extras.get("role", "")


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
            try:
                ctx.nixl_manager.shutdown()
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Failed to shut down NIXL manager: {e}"
                )
        ctx.tensors = {}
        ctx.nixl_manager = None

    def is_available(self, ctx: LoadContext) -> bool:
        if not super().is_available(ctx):
            return False
        if not is_nixl_available():
            return False

        # Decentralized backends (k8s-service) serve their own
        # metadata; skip the central-server precondition for them.
        # Strict `is True` check so MagicMock's auto-attribute doesn't
        # masquerade as the flag in tests.
        server_addr = os.environ.get("MODEL_EXPRESS_URL") or os.environ.get("MX_SERVER_ADDRESS")
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

        Source discovery and metadata misses do not mutate the target model and
        therefore raise clean StrategyFailed errors. Once _load_as_target()
        prepares target storage, failures are treated as mutated because the
        engine may have initialized or transformed model tensors, and those
        failures are raised immediately instead of trying another source.
        """
        result = _as_load_result(result)
        candidates = self._find_source_instances(ctx)
        if not candidates:
            logger.info(f"[Worker {ctx.global_rank}] No RDMA source available, skipping")
            raise StrategyFailed("No RDMA source available", mutated=False)

        for cand in candidates[:MAX_SOURCE_RETRIES]:
            logger.info(
                f"[Worker {ctx.global_rank}] Trying source worker {cand.worker_id} "
                f"({len(cand.worker_meta.tensors)} tensors, "
                f"{'v2 ' + cand.role + ' step=' + str(cand.training_step) if cand.is_v2 else 'v1'})"
            )
            # Do not try another source after target preparation starts. The
            # adapter may have initialized or transformed model tensors, and a
            # failed receive may have partially written weights. The chain will
            # re-initialize the model before trying the next loading strategy.
            return self._load_as_target(result, ctx, cand)

        tried = min(len(candidates), MAX_SOURCE_RETRIES)
        logger.warning(
            f"[Worker {ctx.global_rank}] Tried {tried} of {len(candidates)} source workers "
            f"(max retries={MAX_SOURCE_RETRIES}), falling through"
        )
        # Only pre-target metadata/discovery misses reach here. Failures after
        # target preparation are raised from _load_as_target() as mutated=True.
        raise StrategyFailed("No RDMA source succeeded", mutated=False)

    def _find_source_instances(
        self, ctx: LoadContext,
    ) -> list[_RdmaCandidate]:
        """Discover ready RDMA sources, with v2 awareness when present.

        When any candidate carries v2 metadata (``mx_v2="1"``), we filter to
        v2 sources only and sort them deterministically by
        (role_priority, -training_step, -updated_at). Optionally applies a
        MoE expert-coverage filter when ``ctx.needed_experts_per_layer`` is
        set. Otherwise falls back to the v1 random-shuffle behavior.

        Metadata is fetched once per candidate here and cached on the
        returned `_RdmaCandidate` so the receive path doesn't need to
        re-fetch.
        """
        try:
            # We deliberately do NOT pass ctx.identity as a filter — v2
            # publishers (trainers in particular) set parallelism fields to 0
            # because their topology differs from the inference target. The
            # server-side identity match would drop them. Filter client-side
            # by model_name + the parsed v2 extras instead.
            list_resp = ctx.mx_client.list_sources(
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Error listing sources, falling through: {e}"
            )
            return []

        if not list_resp.instances:
            logger.debug(f"[Worker {ctx.global_rank}] No ready source instances found")
            return []

        candidates: list[_RdmaCandidate] = []
        target_model = ctx.identity.model_name
        for inst in list_resp.instances:
            if inst.model_name != target_model:
                continue
            if inst.worker_rank != ctx.worker_rank:
                continue
            metadata_resp = self._fetch_metadata_response(
                ctx, inst.mx_source_id, inst.worker_id,
            )
            if metadata_resp is None:
                continue
            worker_meta = metadata_resp.worker
            v2_extras = self._parse_v2_extras_safe(metadata_resp)
            candidates.append(_RdmaCandidate(
                instance=inst, worker_meta=worker_meta, v2_extras=v2_extras,
            ))

        v2_candidates = [c for c in candidates if c.is_v2]
        if v2_candidates:
            ranked = self._rank_v2_candidates(v2_candidates, ctx)
            logger.info(
                f"[Worker {ctx.global_rank}] Found {len(v2_candidates)} v2 candidate(s); "
                f"picked freshest role={ranked[0].role if ranked else 'n/a'} "
                f"step={ranked[0].training_step if ranked else 'n/a'}"
            )
            return ranked

        # No v2 sources — fall back to existing v1 behavior.
        random.shuffle(candidates)
        logger.info(
            f"[Worker {ctx.global_rank}] Found {len(candidates)} v1 source worker(s)"
        )
        return candidates

    def _parse_v2_extras_safe(self, metadata_resp) -> dict[str, str]:
        """Wrap nemo_rl_v2.parse_v2_extras with the full GetMetadataResponse.

        Must pass the FULL response (with both ``.identity`` and ``.worker``),
        not just the worker. The v2 trainer publish uses the sidecar
        TensorDescriptor in ``meta.worker.tensors``; v2 inference_replica
        publish puts the markers in ``meta.identity.extra_parameters``. If
        ``identity`` is missing the inference_replica goes undetected and the
        loader falsely treats it as a v1 source.
        """
        try:
            from ..nemo_rl_v2 import parse_v2_extras
        except ImportError:
            return {}
        return parse_v2_extras(metadata_resp)

    def _rank_v2_candidates(
        self, candidates: list[_RdmaCandidate], ctx: LoadContext,
    ) -> list[_RdmaCandidate]:
        """Sort v2 candidates by freshness; optionally MoE-filter.

        Sort key: (role_priority, -training_step, -updated_at). Trainer
        beats inference_replica; within a role, highest training_step
        wins; tiebreak on most recent update. Inference replicas that
        don't cover ``ctx.needed_experts_per_layer`` (when set) are
        dropped.
        """
        needed = getattr(ctx, "needed_experts_per_layer", None)

        def _covers_needed(c: _RdmaCandidate) -> bool:
            if not needed or c.role == "trainer":
                # Trainer is always authoritative (publishes full set).
                return True
            owned_blob = c.v2_extras.get("owned_experts_per_layer", "")
            if not owned_blob:
                return False
            owned: dict[int, set[int]] = {}
            for chunk in owned_blob.split("|"):
                if ":" not in chunk:
                    continue
                lid, ids = chunk.split(":", 1)
                try:
                    owned[int(lid.lstrip("L"))] = {
                        int(e) for e in ids.split(",") if e.strip()
                    }
                except ValueError:
                    return False
            return all(
                req.issubset(owned.get(layer, set()))
                for layer, req in needed.items()
            )

        filtered = [c for c in candidates if _covers_needed(c)]

        def _key(c: _RdmaCandidate):
            role_priority = 0 if c.role == "trainer" else 1
            updated_at = getattr(c.instance, "updated_at", 0) or 0
            return (role_priority, -c.training_step, -int(updated_at))

        filtered.sort(key=_key)
        return filtered

    def _fetch_metadata_response(
        self,
        ctx: LoadContext,
        mx_source_id: str,
        worker_id: str,
    ):
        """Fetch the full GetMetadataResponse for one worker.

        Returns the full response (with both ``.identity`` and ``.worker``)
        so callers can parse v2 metadata from ``identity.extra_parameters``,
        not just the worker-side sidecar. Returns None when not found or
        the worker has no tensors and no P2P endpoint.
        """
        fetch_start = time.perf_counter()
        try:
            metadata_resp = ctx.mx_client.get_metadata(
                mx_source_id=mx_source_id,
                worker_id=worker_id,
            )
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] get_metadata failed for {worker_id}: {e}"
            )
            return None
        if not metadata_resp.found:
            logger.debug(
                f"[Worker {ctx.global_rank}] Metadata not found for worker {worker_id}, skipping"
            )
            return None
        worker = metadata_resp.worker
        if not worker.tensors and not worker.worker_grpc_endpoint:
            logger.debug(
                f"[Worker {ctx.global_rank}] Worker {worker_id} has no tensors "
                f"and no P2P endpoint, skipping"
            )
            return None
        fetch_time = time.perf_counter() - fetch_start
        mode = "P2P (lightweight)" if worker.worker_grpc_endpoint else "centralized"
        tensor_count = len(worker.tensors)
        logger.info(
            f"[Worker {ctx.global_rank}] [TIMING] GetMetadata ({mode}): "
            f"{fetch_time:.3f}s, {tensor_count} tensors"
        )
        return metadata_resp

    def _load_as_target(
        self,
        result: LoadResult,
        ctx: LoadContext,
        cand: _RdmaCandidate,
    ) -> LoadResult:
        """Receive fully-processed weights via RDMA from an existing source.

        v1 sources go through the direct-buffer path (register vLLM's
        ``named_parameters`` as NIXL receive buffers and RDMA-pull into
        them). v2 sources go through the scratch-buffer path so HF-named
        trainer tensors are routed through ``model.load_weights`` —
        otherwise the HF→fused merger never runs and ``qkv_proj`` would
        mismatch (see DEBUGGING_POSTMORTEM §3 in the nemo-rl tree).
        """
        try:
            if cand.is_v2:
                result = self._load_as_v2_target(result, ctx, cand)
            else:
                result = ctx.adapter.prepare_rdma_target(result)
                result = ctx.adapter.before_rdma_receive(result)
                self._receive_from_peer(
                    result, ctx, cand.worker_meta, cand.mx_source_id,
                )
                result = ctx.adapter.after_rdma_receive(result)
            # Record the loaded version so the post-load self-publish can
            # tag this worker as an inference_replica at the same step.
            if cand.is_v2:
                ctx.loaded_version = cand.training_step
            return result
        except StrategyFailed:
            raise
        except Exception as e:
            raise StrategyFailed(str(e), mutated=True) from e

    def _load_as_v2_target(
        self,
        result: LoadResult,
        ctx: LoadContext,
        cand: _RdmaCandidate,
    ) -> LoadResult:
        """Pull a v2 trainer/replica into scratch buffers and apply via load_weights.

        Bypasses ``register_tensors`` (no NIXL registration of the model's
        named_parameters here) because the v2 trainer publishes HF state-dict
        names while vLLM exposes fused internals. The scratch path allocates
        temp CUDA buffers sized to the publisher's tensor list, RDMA-pulls
        into them, and yields ``(hf_name, tensor)`` pairs that vLLM's
        ``model.load_weights`` knows how to merge.
        """
        from ..nemo_rl_v2 import MxV2RefitReceiver, V2SourceCandidate
        from ..refit_receiver import SourceRef
        from ..shape_descriptors import decode_registry

        # Build a SourceRef + V2SourceCandidate from the cached metadata so
        # we can call into the v2 receiver's pull path without re-doing
        # discovery.
        ref = SourceRef(
            mx_source_id=cand.mx_source_id,
            worker_id=cand.worker_id,
            model_name=cand.instance.model_name,
            worker_rank=cand.instance.worker_rank,
            training_step=cand.training_step,
        )
        registry_blob = cand.v2_extras.get("shape_registry", "")
        registry = decode_registry(registry_blob) if registry_blob else None
        v2_cand = V2SourceCandidate(
            ref=ref,
            role=cand.role,
            worker_rank=cand.instance.worker_rank,
            registry=registry,
            owned_experts_per_layer={},
            updated_at=int(getattr(cand.instance, "updated_at", 0) or 0),
        )

        agent_name = f"mx-loader-r{ctx.worker_rank}"
        mx_server_url = (
            os.environ.get("MODEL_EXPRESS_URL")
            or os.environ.get("MX_SERVER_ADDRESS")
            or "modelexpress-server:8001"
        )
        receiver = MxV2RefitReceiver(
            agent_name=agent_name,
            device_id=ctx.device_id,
            mx_server_url=mx_server_url,
            worker_rank=ctx.worker_rank,
        )
        receiver.initialize(model_tensors=None)

        tensor_shapes: dict[str, tuple[int, ...]] = {}
        if registry:
            for td in registry.get("tensors", []):
                tensor_shapes[td.name] = tuple(int(s) for s in td.global_shape)

        # Prepare the model storage if not yet done (allocates the param
        # buffers via DummyModelLoader so load_weights has somewhere to
        # write). prepare_rdma_target is idempotent on already-prepared
        # models in practice; vLLM's DummyModelLoader just zeros existing
        # params.
        result = ctx.adapter.prepare_rdma_target(result)

        pull_start = time.perf_counter()

        def _weights_iter():
            return receiver._receiver.receive_weights_scratch(
                v2_cand.ref,
                timeout_seconds=300.0,
                tensor_shapes=tensor_shapes or None,
            )

        # Stream the pulled (hf_name, tensor) pairs into vLLM's load_weights
        # via the existing adapter hook. vLLM applies its HF→fused merger
        # (qkv_proj, gate_up_proj, etc.) inside this call.
        result = ctx.adapter.apply_weight_iter(result, _weights_iter())
        pull_time = time.perf_counter() - pull_start
        logger.info(
            f"[Worker {ctx.global_rank}] [TIMING] v2 scratch receive + load: "
            f"{pull_time:.3f}s"
        )
        torch.cuda.current_stream().synchronize()
        result = ctx.adapter.after_weight_iter_load(result)

        # Register the now-populated model params with NIXL so the chain's
        # post-load publish_metadata exposes this worker as a tree fan-out
        # source. We register vLLM-internal named_parameters (post-merge),
        # not the trainer's HF-named scratch buffers — subsequent v2
        # receivers pulling from us will see vLLM-internal names, which
        # vLLM's load_weights handles as identity (no merge needed).
        register_tensors(result, ctx)

        # Tag this worker as a v2 inference_replica at the training_step
        # we just loaded, so v2-aware receivers see it as a tree-fan-out
        # candidate. v1 receivers ignore extra_parameters and still see
        # the same source as before — backwards compatible.
        _enrich_identity_v2(
            ctx, training_step=cand.training_step, role="inference_replica",
        )
        return result

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
            from ..metadata.worker_server import fetch_tensor_manifest

            manifest_start = time.perf_counter()
            logger.info(
                f"[Worker {ctx.global_rank}] P2P mode: fetching tensor manifest from "
                f"{source_worker.worker_grpc_endpoint}"
            )
            tensor_protos, manifest_bytes = fetch_tensor_manifest(
                endpoint=source_worker.worker_grpc_endpoint,
                mx_source_id=mx_source_id,
            )
            manifest_time = time.perf_counter() - manifest_start
            source_tensors = [
                TensorDescriptor(
                    name=t.name, addr=t.addr, size=t.size,
                    device_id=t.device_id, dtype=t.dtype,
                )
                for t in tensor_protos
            ]
            logger.info(
                f"[Worker {ctx.global_rank}] [TIMING] P2P tensor manifest: "
                f"{manifest_time:.3f}s ({len(source_tensors)} tensors, "
                f"{manifest_bytes} bytes)"
            )

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
                for t in source_worker.tensors
            ]

        logger.info(
            f"[Worker {ctx.global_rank}] Receiving {len(source_tensors)} tensors from source"
            f"{' (P2P)' if is_p2p else ''}"
        )

        transfer_start = time.perf_counter()
        try:
            bytes_transferred, tensor_count, _ = ctx.nixl_manager.receive_from_source(
                source_metadata=source_worker.nixl_metadata,
                source_tensors=source_tensors,
                timeout_seconds=300.0,
                remote_agent_name=remote_agent_name_override,
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

        torch.cuda.synchronize()

        total_time = time.perf_counter() - receive_start
        logger.info(f"[Worker {ctx.global_rank}] [TIMING] Total receive time: {total_time:.2f}s")
