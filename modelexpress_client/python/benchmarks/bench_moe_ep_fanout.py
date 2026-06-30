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

1. The N publishers register one expert-block per "layer" (where one
   layer is one expert tensor of shape (num_experts, rows, cols)
   shard-published as a block along the expert axis). Trainer EP
   layout: linear partition over ``num_experts``.
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
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
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

# Import modelexpress after logging is configured so child loggers inherit.
from modelexpress import (  # noqa: E402
    MxRefitReceiver,
    MxTrainingPublisher,
    SliceOwnership,
    SliceRequest,
    compute_local_expert_ids,
    plan_coverage,
    summarize_plan,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BenchConfig:
    mx_server_url: str = "modelexpress-server:8001"
    model_name: str = "synthetic/moe-ep-fanout-bench"

    # Defaults match Qwen3-30B-A3B shape but scaled down for one GPU.
    # 128 experts × num_layers × expert_rows × expert_cols × 2 bytes (bf16).
    # At 128 × 4 × 512 × 512 × 2 = 256 MiB total expert bytes; per-rank ownership
    # is num_local_experts × num_layers × 512KB.
    num_experts: int = 128
    num_layers: int = 4
    expert_rows: int = 512
    expert_cols: int = 512
    dtype: str = "torch.bfloat16"

    matched_publishers: int = 4
    matched_receivers: int = 4
    mixed_publishers: int = 4
    mixed_receivers: int = 2

    cycles: int = 5
    timeout_s: float = 60.0
    cuda_device: int = 0


# ---------------------------------------------------------------------------
# Trainer thread
# ---------------------------------------------------------------------------


class TrainerThread(threading.Thread):
    """One publisher rank. Holds a block of expert tensors per layer."""

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
        self.cycle_barrier = cycle_barrier
        self.results = results

        self.owned_expert_ids = compute_local_expert_ids(
            rank, ep_world_size, cfg.num_experts, placement="linear"
        )
        self.expert_lo = self.owned_expert_ids[0]
        self.expert_hi = self.owned_expert_ids[-1] + 1

        self.publisher: MxTrainingPublisher | None = None
        self.mx_source_id: str | None = None
        self.layer_blocks: dict[str, torch.Tensor] = {}
        self.ready_event = threading.Event()

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as e:
            logger.exception("trainer-%d crashed: %s", self.rank, e)
            self.results[f"trainer_{self.rank}_error"] = str(e)
            try:
                self.cycle_barrier.abort()
            except Exception:
                pass

    def _run_inner(self) -> None:
        torch.cuda.set_device(self.cfg.cuda_device)
        logger.info(
            "trainer-%d (ep_ws=%d): owns experts [%d, %d) (%d experts) × %d layers",
            self.rank, self.ep_world_size, self.expert_lo, self.expert_hi,
            len(self.owned_expert_ids), self.cfg.num_layers,
        )

        dtype = torch.bfloat16
        # Allocate per-layer expert blocks. Each block: (num_local_experts, rows, cols).
        # Filled with a deterministic per-(layer, expert) pattern so receivers can
        # checksum-verify they got the right bytes.
        named_tensors: dict[str, torch.Tensor] = {}
        for layer in range(self.cfg.num_layers):
            tname = f"layer_{layer}_experts"
            block = torch.empty(
                (len(self.owned_expert_ids), self.cfg.expert_rows, self.cfg.expert_cols),
                dtype=dtype,
                device=f"cuda:{self.cfg.cuda_device}",
            )
            for i, eid in enumerate(self.owned_expert_ids):
                block[i].fill_(float(layer * 1000 + eid + 1))
            self.layer_blocks[tname] = block
            named_tensors[tname] = block

        # Stand up publisher.
        agent_name = f"moe-ep-bench-trainer-{self.rank}-{uuid.uuid4().hex[:6]}"
        self.publisher = MxTrainingPublisher(
            agent_name=agent_name,
            device_id=self.cfg.cuda_device,
            mx_server_url=self.cfg.mx_server_url,
        )
        self.publisher.initialize(
            model_name=self.cfg.model_name,
            expert_parallel_size=self.ep_world_size,
            training_framework="moe-ep-bench",
        )

        for cycle in range(self.cfg.cycles):
            # Refresh contents per cycle (simulates trainer step).
            for layer in range(self.cfg.num_layers):
                tname = f"layer_{layer}_experts"
                for i, eid in enumerate(self.owned_expert_ids):
                    self.layer_blocks[tname][i].fill_(
                        float(layer * 1000 + eid + 1 + cycle * 0.1)
                    )

            t0 = time.perf_counter()
            self.mx_source_id = self.publisher.publish_weights(
                named_tensors=named_tensors,
                step=cycle,
                worker_rank=self.rank,
            )
            self.publisher.mark_ready(worker_rank=self.rank)
            publish_dt = time.perf_counter() - t0

            total_bytes = sum(t.numel() * t.element_size() for t in named_tensors.values())
            logger.info(
                "trainer-%d cycle=%d source_id=%s publish=%.3fs (bytes=%d)",
                self.rank, cycle, self.mx_source_id, publish_dt, total_bytes,
            )

            self.ready_event.set()
            self.cycle_barrier.wait()
            self.ready_event.clear()


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
        trainers: list[TrainerThread],
        cycle_barrier: threading.Barrier,
        results: dict,
    ):
        super().__init__(daemon=True, name=f"receiver-{rank}")
        self.rank = rank
        self.ep_world_size = ep_world_size
        self.cfg = cfg
        self.trainers = trainers
        self.cycle_barrier = cycle_barrier
        self.results = results

        self.required_experts = compute_local_expert_ids(
            rank, ep_world_size, cfg.num_experts, placement="linear"
        )
        self.expert_lo = self.required_experts[0]
        self.expert_hi = self.required_experts[-1] + 1

        self.receiver: MxRefitReceiver | None = None
        self.target_blocks: dict[str, torch.Tensor] = {}

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as e:
            logger.exception("receiver-%d crashed: %s", self.rank, e)
            self.results[f"receiver_{self.rank}_error"] = str(e)
            try:
                self.cycle_barrier.abort()
            except Exception:
                pass

    def _run_inner(self) -> None:
        torch.cuda.set_device(self.cfg.cuda_device)
        logger.info(
            "receiver-%d (ep_ws=%d): wants experts [%d, %d) (%d experts) × %d layers",
            self.rank, self.ep_world_size, self.expert_lo, self.expert_hi,
            len(self.required_experts), self.cfg.num_layers,
        )

        dtype = torch.bfloat16
        # Allocate target buffers sized for local experts only.
        for layer in range(self.cfg.num_layers):
            tname = f"layer_{layer}_experts"
            self.target_blocks[tname] = torch.zeros(
                (len(self.required_experts), self.cfg.expert_rows, self.cfg.expert_cols),
                dtype=dtype,
                device=f"cuda:{self.cfg.cuda_device}",
            )

        agent_name = f"moe-ep-bench-receiver-{self.rank}-{uuid.uuid4().hex[:6]}"
        self.receiver = MxRefitReceiver(
            agent_name=agent_name,
            device_id=self.cfg.cuda_device,
            mx_server_url=self.cfg.mx_server_url,
        )
        self.receiver.initialize(model_tensors=self.target_blocks)

        # The byte stride per "row along expert axis" — i.e. one expert's worth.
        per_expert_bytes = self.cfg.expert_rows * self.cfg.expert_cols * 2  # bf16

        per_cycle_metrics: list[dict] = []
        for cycle in range(self.cfg.cycles):
            for t in self.trainers:
                t.ready_event.wait(timeout=30.0)

            # Build SliceOwnership entries from the published trainer state.
            # One ownership per (trainer, layer) — the publisher rank owns a
            # contiguous block of experts on axis 0 of each layer's tensor.
            ownerships: list[SliceOwnership] = []
            for t in self.trainers:
                for layer in range(self.cfg.num_layers):
                    tname = f"layer_{layer}_experts"
                    block = t.layer_blocks[tname]
                    ownerships.append(SliceOwnership(
                        model_name=self.cfg.model_name,
                        tensor_name=tname,
                        global_shape=(self.cfg.num_experts, self.cfg.expert_rows, self.cfg.expert_cols),
                        dtype=self.cfg.dtype,
                        placement_kind="SHARD",
                        shard_axis=0,
                        local_shard_range=(t.expert_lo, t.expert_hi),
                        worker_rank=t.rank,
                        nixl_addr=int(block.data_ptr()),
                        byte_size=block.numel() * block.element_size(),
                        device_id=self.cfg.cuda_device,
                        is_expert=True,
                        expert_axis=0,
                        owned_expert_ids=t.owned_expert_ids,
                    ))

            # Build one SliceRequest per layer. The receiver wants
            # experts in [expert_lo, expert_hi) — contiguous range, since
            # we use linear placement.
            req_required = frozenset(self.required_experts)
            requests: list[SliceRequest] = []
            for layer in range(self.cfg.num_layers):
                tname = f"layer_{layer}_experts"
                target_block = self.target_blocks[tname]
                requests.append(SliceRequest(
                    tensor_name=tname,
                    global_range=(self.expert_lo, self.expert_hi),
                    shard_axis=0,
                    dtype=self.cfg.dtype,
                    receiver_rank=self.rank,
                    target_addr=int(target_block.data_ptr()),
                    target_offset=0,
                    required_experts=req_required,
                ))

            t0 = time.perf_counter()
            plan = plan_coverage(sources=ownerships, requests=requests)
            plan.raise_if_incomplete()
            plan_dt = time.perf_counter() - t0
            summary = summarize_plan(plan)

            # Resolve per-source NIXL agent handles (cached after first cycle).
            t0 = time.perf_counter()
            agents: dict[int, str] = {}
            for t in self.trainers:
                if t.rank in agents:
                    continue
                agents[t.rank] = self.receiver.prefetch_source(
                    mx_source_id=t.mx_source_id,
                    worker_id=t.publisher._worker_id,  # noqa: SLF001
                )
            prefetch_dt = time.perf_counter() - t0

            # Drive each SegmentPlan through receive_segment.
            t0 = time.perf_counter()
            total_bytes = 0
            for seg in plan.segments:
                src_addr = (
                    seg.source.nixl_addr
                    + seg.source_range[0] * per_expert_bytes
                )
                tgt_addr = (
                    seg.request.target_addr
                    + seg.request.target_offset
                    + seg.target_range[0] * per_expert_bytes
                )
                self.receiver.receive_segment(
                    remote_agent_name=agents[seg.source.worker_rank],
                    source_addr=src_addr,
                    byte_count=seg.byte_count,
                    target_addr=tgt_addr,
                    source_device_id=seg.source.device_id,
                    timeout_seconds=self.cfg.timeout_s,
                )
                total_bytes += seg.byte_count
            xfer_dt = time.perf_counter() - t0

            # Checksum verify: every expert i in our buffer at layer L
            # should be filled with float(L * 1000 + (lo + i) + 1 + cycle * 0.1).
            verified, errors = self._verify_checksum(cycle)

            # Baseline: full-expert pull would be num_experts × layers × per_expert_bytes.
            baseline_bytes = self.cfg.num_experts * self.cfg.num_layers * per_expert_bytes
            actual_ratio = total_bytes / baseline_bytes
            expected_ratio = len(self.required_experts) / self.cfg.num_experts
            within_tolerance = abs(actual_ratio - expected_ratio) <= 0.10  # ±10%
            savings_vs_baseline = baseline_bytes / max(total_bytes, 1)
            expected_savings = 1.0 / expected_ratio

            gbps = (total_bytes * 8) / (xfer_dt * 1e9) if xfer_dt > 0 else float("inf")
            metric = {
                "cycle": cycle,
                "receiver_rank": self.rank,
                "ep_world_size": self.ep_world_size,
                "required_experts_count": len(self.required_experts),
                "total_experts": self.cfg.num_experts,
                "segment_count": summary["segment_count"],
                "source_ranks_used": summary["source_ranks_used"],
                "bytes_transferred": total_bytes,
                "baseline_bytes": baseline_bytes,
                "actual_ratio_of_baseline": actual_ratio,
                "expected_ratio_of_baseline": expected_ratio,
                "savings_vs_baseline": savings_vs_baseline,
                "expected_savings": expected_savings,
                "within_tolerance": within_tolerance,
                "verified": verified,
                "checksum_errors": errors,
                "plan_seconds": plan_dt,
                "prefetch_seconds": prefetch_dt,
                "xfer_seconds": xfer_dt,
                "gbps": gbps,
            }
            per_cycle_metrics.append(metric)
            logger.info(
                "receiver-%d cycle=%d sources=%s bytes=%d savings=%.2fx (expected ~%.2fx) "
                "tol_pass=%s verified=%s xfer=%.3fs (%.2f Gbps)",
                self.rank, cycle, summary["source_ranks_used"], total_bytes,
                savings_vs_baseline, expected_savings, within_tolerance,
                verified, xfer_dt, gbps,
            )
            self.cycle_barrier.wait()

        self.results[f"receiver_{self.rank}_metrics"] = per_cycle_metrics

    def _verify_checksum(self, cycle: int) -> tuple[bool, list[str]]:
        """Confirm each layer's local-expert block is filled with the
        cycle-specific pattern from the matching trainer rank."""
        errors: list[str] = []
        per_expert_bytes = self.cfg.expert_rows * self.cfg.expert_cols * 2  # noqa: F841
        for layer in range(self.cfg.num_layers):
            tname = f"layer_{layer}_experts"
            block = self.target_blocks[tname]
            for i, eid in enumerate(self.required_experts):
                expected = float(layer * 1000 + eid + 1 + cycle * 0.1)
                # Sample first element of the i-th expert.
                actual = float(block[i, 0, 0].item())
                if abs(actual - expected) > 0.5:  # bf16 tolerance
                    errors.append(
                        f"layer={layer} expert_id={eid} expected={expected} actual={actual}"
                    )
                    if len(errors) >= 5:
                        errors.append(f"... (more errors suppressed)")
                        return False, errors
        return len(errors) == 0, errors


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
        "=== %s: trainer EP=%d, inference EP=%d, experts=%d, layers=%d, "
        "expert_shape=(%d,%d) bf16 ===",
        cell_name, num_publishers, num_receivers,
        cfg.num_experts, cfg.num_layers, cfg.expert_rows, cfg.expert_cols,
    )

    results: dict = {"cell": cell_name}
    # Barrier: all publishers + all receivers wait at the cycle boundary.
    # Receivers also wait at the END of each cycle so publishers can mutate
    # the shard for the next cycle. So parties = pubs + recvs.
    barrier = threading.Barrier(num_publishers + num_receivers)

    trainers = [
        TrainerThread(r, num_publishers, cfg, barrier, results)
        for r in range(num_publishers)
    ]
    for t in trainers:
        t.start()

    # Give trainers a moment to populate publisher / mx_source_id / layer_blocks.
    time.sleep(0.5)

    receivers = [
        ReceiverThread(r, num_receivers, cfg, trainers, barrier, results)
        for r in range(num_receivers)
    ]
    for r in receivers:
        r.start()

    for t in trainers + receivers:
        t.join(timeout=cfg.timeout_s * cfg.cycles + 60)
        if t.is_alive():
            logger.error("%s did not exit in time", t.name)

    # Surface crashes.
    crashed: list[str] = []
    for key, val in results.items():
        if key.endswith("_error"):
            crashed.append(f"{key}: {val}")
    if crashed:
        logger.error("%s crashed threads:\n  %s", cell_name, "\n  ".join(crashed))

    # Aggregate.
    per_cycle: list[dict] = []
    pass_conditions: list[dict] = []
    for cycle in range(cfg.cycles):
        cycle_summary = {"cycle": cycle, "receivers": []}
        for r in range(num_receivers):
            metrics = results.get(f"receiver_{r}_metrics", [])
            if cycle < len(metrics):
                m = metrics[cycle]
                cycle_summary["receivers"].append(m)
                pass_conditions.append({
                    "receiver_rank": r,
                    "cycle": cycle,
                    "actual_ratio": m["actual_ratio_of_baseline"],
                    "expected_ratio": m["expected_ratio_of_baseline"],
                    "abs_diff": abs(m["actual_ratio_of_baseline"] - m["expected_ratio_of_baseline"]),
                    "tolerance": 0.10,
                    "passes": m["within_tolerance"] and m["verified"],
                    "verified": m["verified"],
                    "checksum_errors": m["checksum_errors"],
                })
        per_cycle.append(cycle_summary)

    # all_pass requires:
    # (1) every (receiver, cycle) within ±10% AND verified
    # (2) we actually got results for every (receiver, cycle) — no crashes
    expected_data_points = num_receivers * cfg.cycles
    all_pass = (
        len(pass_conditions) == expected_data_points
        and all(pc["passes"] for pc in pass_conditions)
        and not crashed
    )

    return {
        "cell": cell_name,
        "config": {
            "num_publishers": num_publishers,
            "num_receivers": num_receivers,
            "num_experts": cfg.num_experts,
            "num_layers": cfg.num_layers,
            "expert_rows": cfg.expert_rows,
            "expert_cols": cfg.expert_cols,
            "per_expert_bytes": cfg.expert_rows * cfg.expert_cols * 2,
        },
        "per_cycle": per_cycle,
        "pass_conditions": pass_conditions,
        "expected_data_points": expected_data_points,
        "observed_data_points": len(pass_conditions),
        "crashed_threads": crashed,
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
    logger.info(
        "Matched EP: all_pass=%s observed=%d/%d crashed=%d",
        matched["all_pass"], matched["observed_data_points"],
        matched["expected_data_points"], len(matched["crashed_threads"]),
    )
    logger.info(
        "Mixed EP:   all_pass=%s observed=%d/%d crashed=%d",
        mixed["all_pass"], mixed["observed_data_points"],
        mixed["expected_data_points"], len(mixed["crashed_threads"]),
    )
    logger.info("Overall:    all_pass=%s", summary["all_pass"])

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results written to %s", args.output_json)

    if not summary["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
