#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-2 cluster harness for inference-side EP fan-out.

Demonstrates the substrate-level wire-savings property: a receiver
holding a subset of experts pulls only the byte ranges for its local
experts, never the non-local ones. The harness colocates N trainer
publishers and M rollout receivers as threads on a single GPU pod
(same shape as ``bench_verl_rank_to_rank.py``), with each publisher
owning a contiguous block of MoE expert tensors and each receiver
pulling only its EP-rank-local experts via the new
``SliceRequest.required_experts`` filter that this PR adds.

For each cycle, the harness:

1. The N publishers register their per-rank expert blocks. Trainer EP
   layout: linear partition over ``num_experts``, ``num_publishers``
   ranks.
2. Each receiver builds a ``SliceRequest`` per layer, with
   ``required_experts`` set to the receiver's EP-rank-local set via
   :func:`modelexpress.compute_local_expert_ids`.
3. ``plan_coverage`` intersects ownerships against requests; the
   substrate planner emits ``SegmentPlan``s only for publisher ranks
   whose ``owned_expert_ids`` intersects the receiver's
   ``required_experts``.
4. Each receiver issues one :meth:`MxRefitReceiver.receive_segment`
   per emitted ``SegmentPlan``. Bytes-on-the-wire are summed; per-
   receiver byte distribution is compared against the full-expert
   pull baseline.
5. The harness reports the byte savings ratio and validates the
   §10.4 E-series-EP pass condition: each receiver pulls within
   ±10% of ``local_experts / total_experts`` of the baseline expert
   bytes.

The harness runs two cells back-to-back:

- **Matched EP** (``num_publishers == num_receivers``): trainer and
  inference have the same EP world size. Each receiver should pull
  from exactly one publisher.
- **Mixed EP** (``num_publishers > num_receivers``): trainer EP is
  wider than inference EP. Each receiver pulls from multiple
  publishers because its local-expert set spans multiple trainer
  ranks.

This validates the substrate property under both topology shapes
and confirms the planner handles cross-EP-shape transfers.

What this validates beyond Tier-1:

- ``SliceRequest.required_experts`` actually drives expert filtering
  in ``plan_coverage`` against live ownership rows.
- The ``MxRefitReceiver.receive_segment`` data plane composes with
  per-expert filtering — no wasted pulls for non-local experts.
- The byte-savings property predicted by the substrate matches what
  hits the wire.

What this does NOT validate (deliberate scope cuts, matching the
verl Tier-2 harness):

- Cross-node RDMA — we colocate on one pod (single-pod loopback);
  absolute bandwidth numbers are not representative of a real
  cross-host run. Cross-host is the Tier-3 gap (open).
- Real MoE model state dicts — the harness uses synthetic per-expert
  tensors sized to fit on one GPU.
- Mixed dtype / quantization — bf16 only.

Usage:
    python bench_moe_ep_fanout.py \\
        --mx-server-url=modelexpress-server.kavin.svc.cluster.local:8001 \\
        --num-experts=128 --num-layers=4 --expert-rows=512 --expert-cols=512 \\
        --cycles=5 --cuda-device=0 \\
        --matched-publishers=4 --matched-receivers=4 \\
        --mixed-publishers=4 --mixed-receivers=2 \\
        --output-json=results.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import threading
import time
import uuid

import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench_moe_ep_fanout")

# Import after logging is configured so child loggers inherit.
from modelexpress import (  # noqa: E402
    MxRefitReceiver,
    MxTrainingPublisher,
    SliceOwnership,
    SliceRequest,
    compute_local_expert_ids,
    plan_coverage,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BenchConfig:
    mx_server_url: str = "modelexpress-server:8001"
    model_name: str = "synthetic/moe-ep-fanout-bench"

    # Defaults match Qwen3-30B-A3B shape but scaled down for one GPU.
    num_experts: int = 128
    num_layers: int = 4
    expert_rows: int = 512  # synthetic expert tensor rows (Qwen3-MoE-30B-A3B has ~4096+)
    expert_cols: int = 512  # synthetic expert tensor cols
    dtype: str = "torch.bfloat16"

    matched_publishers: int = 4  # trainer EP world size (matched cell)
    matched_receivers: int = 4   # inference EP world size (matched cell)
    mixed_publishers: int = 4    # trainer EP world size (mixed cell)
    mixed_receivers: int = 2     # inference EP world size (mixed cell)

    cycles: int = 5
    timeout_s: float = 60.0
    cuda_device: int = 0


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _expert_bytes(cfg: BenchConfig) -> int:
    """Bytes per single expert tensor."""
    return cfg.expert_rows * cfg.expert_cols * 2  # bf16 = 2 bytes


def _make_expert_block(cfg: BenchConfig, expert_ids: list[int], device: torch.device) -> torch.Tensor:
    """Build a contiguous block of expert tensors owned by one publisher.

    Shape: (len(expert_ids), expert_rows, expert_cols). Filled with
    a deterministic pattern keyed off the expert id for checksum.
    """
    block = torch.empty(
        (len(expert_ids), cfg.expert_rows, cfg.expert_cols),
        dtype=torch.bfloat16,
        device=device,
    )
    for i, eid in enumerate(expert_ids):
        # Deterministic pattern per expert id — receiver can verify.
        block[i] = float(eid) / 1000.0
    return block


# ---------------------------------------------------------------------------
# Trainer thread
# ---------------------------------------------------------------------------


class TrainerThread(threading.Thread):
    """One publisher rank. Publishes its block of expert tensors per layer."""

    def __init__(
        self,
        rank: int,
        ep_world_size: int,
        cfg: BenchConfig,
        cycle_barrier: threading.Barrier,
        results: dict,
    ):
        super().__init__(daemon=True, name=f"trainer-{rank}")
        self.rank = rank
        self.ep_world_size = ep_world_size
        self.cfg = cfg
        self.barrier = cycle_barrier
        self.results = results
        self.owned_expert_ids = compute_local_expert_ids(
            rank, ep_world_size, cfg.num_experts, placement="linear"
        )

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as e:
            logger.exception(f"trainer-{self.rank} crashed: {e}")
            self.results[f"trainer_{self.rank}_error"] = str(e)

    def _run_inner(self) -> None:
        device = torch.device(f"cuda:{self.cfg.cuda_device}")
        worker_id = f"trainer-{self.rank}-{uuid.uuid4().hex[:6]}"

        # Build one expert block per layer.
        # Each layer's expert block is a separate registered tensor.
        layer_blocks: dict[str, torch.Tensor] = {}
        layer_tensor_names: list[str] = []
        for layer in range(self.cfg.num_layers):
            tensor_name = f"model.layers.{layer}.experts.w13_weight"
            block = _make_expert_block(self.cfg, list(self.owned_expert_ids), device)
            layer_blocks[tensor_name] = block
            layer_tensor_names.append(tensor_name)

        pub = MxTrainingPublisher(
            mx_server_url=self.cfg.mx_server_url,
            worker_id=worker_id,
            framework_name="moe-ep-bench-trainer",
        )
        pub.initialize(model_name=self.cfg.model_name)
        pub.register_tensors(layer_blocks)

        # The substrate's v1 publisher publishes the tensors with name-only
        # metadata. For EP filtering we need to publish per-expert ownership
        # via a separate metadata channel — but for this bench the receiver
        # builds the SliceOwnership entries directly from known config
        # (since this is a controlled harness, not catalog-driven discovery).
        # See bench_verl_rank_to_rank.py for the same simplification.

        # Build the SliceOwnership list this rank advertises (passed to the
        # main thread via results dict for the receivers to consume).
        owner_per_layer: list[SliceOwnership] = []
        for layer_idx, tname in enumerate(layer_tensor_names):
            block = layer_blocks[tname]
            # NIXL address — pull from the underlying torch storage.
            nixl_addr = block.data_ptr()
            byte_size = block.numel() * block.element_size()
            owner_per_layer.append(SliceOwnership(
                model_name=self.cfg.model_name,
                tensor_name=tname,
                global_shape=(self.cfg.num_experts, self.cfg.expert_rows, self.cfg.expert_cols),
                dtype=self.cfg.dtype,
                placement_kind="SHARD",
                shard_axis=0,
                local_shard_range=(min(self.owned_expert_ids), max(self.owned_expert_ids) + 1),
                worker_rank=self.rank,
                nixl_addr=nixl_addr,
                byte_size=byte_size,
                device_id=self.cfg.cuda_device,
                is_expert=True,
                expert_axis=0,
                owned_expert_ids=self.owned_expert_ids,
            ))
        self.results[f"trainer_{self.rank}_ownerships"] = owner_per_layer
        self.results[f"trainer_{self.rank}_publisher"] = pub
        self.results[f"trainer_{self.rank}_blocks"] = layer_blocks

        for cycle in range(self.cfg.cycles):
            # Refresh block contents per cycle (simulates trainer step).
            for layer_idx, tname in enumerate(layer_tensor_names):
                block = layer_blocks[tname]
                for i, eid in enumerate(self.owned_expert_ids):
                    block[i] = float(eid) / 1000.0 + cycle * 0.01

            pub.publish(version=cycle + 1)
            logger.info(
                f"trainer-{self.rank}: cycle {cycle} published "
                f"{len(self.owned_expert_ids)} experts × {self.cfg.num_layers} layers"
            )
            self.barrier.wait()


# ---------------------------------------------------------------------------
# Receiver thread
# ---------------------------------------------------------------------------


class ReceiverThread(threading.Thread):
    """One inference rank. Pulls only its EP-rank-local experts."""

    def __init__(
        self,
        rank: int,
        ep_world_size: int,
        cfg: BenchConfig,
        cycle_barrier: threading.Barrier,
        results: dict,
        shared_ownerships: dict,  # populated by trainers before this starts
        num_publishers: int,
    ):
        super().__init__(daemon=True, name=f"receiver-{rank}")
        self.rank = rank
        self.ep_world_size = ep_world_size
        self.cfg = cfg
        self.barrier = cycle_barrier
        self.results = results
        self.shared = shared_ownerships
        self.num_publishers = num_publishers
        self.required_experts = compute_local_expert_ids(
            rank, ep_world_size, cfg.num_experts, placement="linear"
        )

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as e:
            logger.exception(f"receiver-{self.rank} crashed: {e}")
            self.results[f"receiver_{self.rank}_error"] = str(e)

    def _run_inner(self) -> None:
        device = torch.device(f"cuda:{self.cfg.cuda_device}")
        worker_id = f"receiver-{self.rank}-{uuid.uuid4().hex[:6]}"

        # Allocate target buffers — one per layer, sized for local experts only.
        target_blocks: dict[str, torch.Tensor] = {}
        for layer in range(self.cfg.num_layers):
            tname = f"model.layers.{layer}.experts.w13_weight"
            target_blocks[tname] = torch.zeros(
                (len(self.required_experts), self.cfg.expert_rows, self.cfg.expert_cols),
                dtype=torch.bfloat16,
                device=device,
            )

        recv = MxRefitReceiver(
            mx_server_url=self.cfg.mx_server_url,
            worker_id=worker_id,
            framework_name="moe-ep-bench-receiver",
        )
        recv.initialize(model_tensors=target_blocks)

        for cycle in range(self.cfg.cycles):
            self.barrier.wait()  # wait for all publishers to publish

            # Build SliceRequest list for this receiver: one per layer.
            # We need the request range to cover the expert sub-range we want.
            # For linear placement (contiguous required experts), this is
            # one contiguous range per layer.
            requests: list[SliceRequest] = []
            lo, hi = min(self.required_experts), max(self.required_experts) + 1
            req_required = frozenset(self.required_experts)
            for layer_idx, tname in enumerate(target_blocks.keys()):
                target_block = target_blocks[tname]
                requests.append(SliceRequest(
                    tensor_name=tname,
                    global_range=(lo, hi),
                    shard_axis=0,
                    dtype=self.cfg.dtype,
                    receiver_rank=self.rank,
                    target_addr=target_block.data_ptr(),
                    required_experts=req_required,
                ))

            # Collect ownerships from all publishers.
            all_ownerships: list[SliceOwnership] = []
            for pub_rank in range(self.num_publishers):
                key = f"trainer_{pub_rank}_ownerships"
                if key in self.shared:
                    all_ownerships.extend(self.shared[key])

            t0 = time.monotonic()
            plan = plan_coverage(all_ownerships, requests)
            plan_time = time.monotonic() - t0

            # Validate the plan is complete + matches expectations.
            assert plan.complete, f"receiver-{self.rank}: plan incomplete: {plan.missing}"
            source_ranks_used = sorted({seg.source.worker_rank for seg in plan.segments})

            # Execute the plan (issue receive_segment per SegmentPlan).
            # Note: this bench uses receive_segment as the data plane primitive
            # added in PR #349. The plan's expert filtering is what we're
            # validating — fewer SegmentPlans = fewer bytes pulled.
            t0 = time.monotonic()
            bytes_pulled = 0
            for seg in plan.segments:
                source = seg.source
                # Resolve remote agent for this source rank.
                source_worker_id = self.shared.get(
                    f"trainer_{source.worker_rank}_worker_id"
                )
                if source_worker_id is None:
                    # Fall back to looking it up via the publisher object.
                    pub_obj = self.shared.get(f"trainer_{source.worker_rank}_publisher")
                    if pub_obj is not None:
                        source_worker_id = pub_obj._worker_id  # noqa: SLF001
                if source_worker_id is None:
                    raise RuntimeError(
                        f"receiver-{self.rank}: could not resolve worker_id for "
                        f"source rank {source.worker_rank}"
                    )
                # Prefetch the remote agent handle (cached after first call).
                remote_agent = recv.prefetch_source(
                    mx_source_id=pub_obj._source_id if pub_obj else "",  # noqa: SLF001
                    worker_id=source_worker_id,
                )
                # Pull the segment.
                source_addr = source.nixl_addr + seg.source_range[0] * (
                    source.byte_size // (source.local_shard_range[1] - source.local_shard_range[0])
                )
                target_addr_full = seg.request.target_addr + seg.target_range[0] * (
                    self.cfg.expert_rows * self.cfg.expert_cols * 2
                )
                recv.receive_segment(
                    remote_agent_name=remote_agent,
                    source_addr=source_addr,
                    byte_count=seg.byte_count,
                    target_addr=target_addr_full,
                    source_device_id=source.device_id,
                    timeout_seconds=self.cfg.timeout_s,
                )
                bytes_pulled += seg.byte_count
            xfer_time = time.monotonic() - t0

            # Compute baseline: full-expert pull would be num_experts × layers × expert_bytes.
            baseline_bytes = (
                self.cfg.num_experts * self.cfg.num_layers * _expert_bytes(self.cfg)
            )
            local_expert_bytes = (
                len(self.required_experts) * self.cfg.num_layers * _expert_bytes(self.cfg)
            )
            savings_ratio = baseline_bytes / max(bytes_pulled, 1)
            # Expected: bytes_pulled ≈ local_expert_bytes (within rounding).
            expected_ratio = (
                len(self.required_experts) / self.cfg.num_experts
            )
            actual_ratio = bytes_pulled / baseline_bytes

            cycle_result = {
                "cycle": cycle,
                "receiver_rank": self.rank,
                "ep_world_size": self.ep_world_size,
                "required_experts": list(self.required_experts),
                "segment_count": len(plan.segments),
                "source_ranks_used": source_ranks_used,
                "bytes_pulled": bytes_pulled,
                "baseline_bytes": baseline_bytes,
                "local_expert_bytes": local_expert_bytes,
                "savings_vs_full_expert": savings_ratio,
                "actual_ratio_of_baseline": actual_ratio,
                "expected_ratio_of_baseline": expected_ratio,
                "plan_time_seconds": plan_time,
                "xfer_time_seconds": xfer_time,
                "gbps": (bytes_pulled * 8) / max(xfer_time, 1e-9) / 1e9,
            }
            self.results.setdefault(f"receiver_{self.rank}_cycles", []).append(cycle_result)
            logger.info(
                f"receiver-{self.rank} cycle={cycle} "
                f"sources={source_ranks_used} bytes={bytes_pulled} "
                f"savings={savings_ratio:.2f}x "
                f"(expected ~{1.0 / expected_ratio:.2f}x), "
                f"xfer={xfer_time:.3f}s ({cycle_result['gbps']:.2f} Gbps)"
            )


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


def _run_cell(
    cell_name: str,
    cfg: BenchConfig,
    num_publishers: int,
    num_receivers: int,
) -> dict:
    logger.info(
        f"=== {cell_name}: trainer EP={num_publishers}, inference EP={num_receivers}, "
        f"experts={cfg.num_experts}, layers={cfg.num_layers}, "
        f"expert_shape=({cfg.expert_rows},{cfg.expert_cols}) bf16 ==="
    )

    results: dict = {"cell": cell_name}
    # +1 barrier wait per cycle: publishers wait once after publish, receivers
    # wait once before pull. So barrier party count = pubs + recvs.
    barrier = threading.Barrier(num_publishers + num_receivers)

    trainers = [
        TrainerThread(r, num_publishers, cfg, barrier, results)
        for r in range(num_publishers)
    ]
    for t in trainers:
        t.start()

    # Give trainers a moment to populate ownerships dict.
    time.sleep(0.5)

    receivers = [
        ReceiverThread(r, num_receivers, cfg, barrier, results, results, num_publishers)
        for r in range(num_receivers)
    ]
    for r in receivers:
        r.start()

    for t in trainers + receivers:
        t.join(timeout=cfg.timeout_s * cfg.cycles + 60)
        if t.is_alive():
            logger.error(f"{t.name} did not exit in time")

    # Aggregate.
    per_cycle: list[dict] = []
    for cycle in range(cfg.cycles):
        cycle_summary = {"cycle": cycle, "receivers": []}
        for r in range(num_receivers):
            cycles = results.get(f"receiver_{r}_cycles", [])
            if cycle < len(cycles):
                cycle_summary["receivers"].append(cycles[cycle])
        per_cycle.append(cycle_summary)

    # Pass-condition check.
    pass_conditions: list[dict] = []
    for r in range(num_receivers):
        cycles = results.get(f"receiver_{r}_cycles", [])
        for c in cycles:
            actual = c["actual_ratio_of_baseline"]
            expected = c["expected_ratio_of_baseline"]
            tol = 0.10  # ±10% per §10.4 E-series-EP
            within_tol = abs(actual - expected) <= tol
            pass_conditions.append({
                "receiver_rank": r,
                "cycle": c["cycle"],
                "actual_ratio": actual,
                "expected_ratio": expected,
                "abs_diff": abs(actual - expected),
                "tolerance": tol,
                "passes": within_tol,
            })

    all_pass = all(pc["passes"] for pc in pass_conditions)
    return {
        "cell": cell_name,
        "config": {
            "num_publishers": num_publishers,
            "num_receivers": num_receivers,
            "num_experts": cfg.num_experts,
            "num_layers": cfg.num_layers,
            "expert_bytes_per_tensor": _expert_bytes(cfg),
        },
        "per_cycle": per_cycle,
        "pass_conditions": pass_conditions,
        "all_pass": all_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mx-server-url", default="modelexpress-server:8001")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--expert-rows", type=int, default=512)
    parser.add_argument("--expert-cols", type=int, default=512)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--matched-publishers", type=int, default=4)
    parser.add_argument("--matched-receivers", type=int, default=4)
    parser.add_argument("--mixed-publishers", type=int, default=4)
    parser.add_argument("--mixed-receivers", type=int, default=2)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    cfg = BenchConfig(
        mx_server_url=args.mx_server_url,
        num_experts=args.num_experts,
        num_layers=args.num_layers,
        expert_rows=args.expert_rows,
        expert_cols=args.expert_cols,
        cycles=args.cycles,
        cuda_device=args.cuda_device,
        matched_publishers=args.matched_publishers,
        matched_receivers=args.matched_receivers,
        mixed_publishers=args.mixed_publishers,
        mixed_receivers=args.mixed_receivers,
    )

    matched = _run_cell(
        "Matched EP", cfg,
        num_publishers=cfg.matched_publishers,
        num_receivers=cfg.matched_receivers,
    )
    mixed = _run_cell(
        "Mixed EP (trainer wider)", cfg,
        num_publishers=cfg.mixed_publishers,
        num_receivers=cfg.mixed_receivers,
    )

    summary = {
        "config": dataclasses.asdict(cfg),
        "matched_ep": matched,
        "mixed_ep": mixed,
        "all_pass": matched["all_pass"] and mixed["all_pass"],
    }

    logger.info("=== SUMMARY ===")
    logger.info(f"Matched EP: all_pass={matched['all_pass']}")
    logger.info(f"Mixed EP:   all_pass={mixed['all_pass']}")
    logger.info(f"Overall:    all_pass={summary['all_pass']}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results written to {args.output_json}")

    if not summary["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
