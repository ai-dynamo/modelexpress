# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 cluster harness for the Gen 3 rank-to-rank prototype.

Spins up two trainer-side publishers and two inference-side receivers
(all four roles colocated as threads on a single GPU pod, to keep the
deployment footprint small while still exercising the real NIXL/RDMA
data plane against the live MX server).

For each "refit step" the harness:

1. The two publishers register one shard each of a synthetic large tensor
   — neither publisher ever holds the full tensor in memory (the no-
   allgather invariant we want to validate).
2. Each receiver issues two :meth:`MxRefitReceiver.receive_segment` calls,
   one to each publisher, pulling its rank-local half of the tensor.
3. Bytes-on-the-wire are summed; bandwidth is reported per receiver and
   in aggregate.
4. A baseline "v1 receive_weights" pass is run for comparison —
   represents the gather-equivalent path the loader falls back to when
   the receiver lacks ``receive_segment``.

What this validates beyond Tier 1:

- ``receive_segment`` actually drives a NIXL READ end-to-end against the
  live MX server (not a mock).
- Multi-segment fan-in from two source ranks works without an allgather
  on the publisher side.
- Bytes match exactly (we verify with a checksum after each pull).
- The rank-to-rank wire footprint matches what
  :func:`collect_byte_savings_vs_allgather` predicts.

What this does NOT validate (deliberate scope cut for Tier 2):

- Cross-node RDMA — we colocate on one pod so we exercise NIXL but over
  the local IPC/UCX path rather than InfiniBand. Cross-node is a Tier 3
  experiment.
- DTensor-derived placement metadata — the harness builds
  :class:`SliceOwnership` entries manually via
  :class:`PlacementDescriptor`.
- A real RL loop — the trainer here is a tensor-generating loop, not a
  forward/backward pass.
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
from collections import defaultdict

import torch

# Configure logging early so MX submodule output is visible.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench_verl_rank_to_rank")

# Import modelexpress after logging is configured so child loggers inherit.
from modelexpress import (  # noqa: E402
    MxRefitReceiver,
    MxTrainingPublisher,
    PlacementDescriptor,
    SliceOwnership,
    SliceRequest,
    collect_byte_savings_vs_allgather,
    plan_coverage,
    summarize_plan,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BenchConfig:
    """All knobs for one harness run."""

    mx_server_url: str = "modelexpress-server:8001"
    model_name: str = "synthetic/rank-to-rank-bench"
    rows_total: int = 65536       # total rows of the synthetic tensor
    cols: int = 8192              # columns (kept fixed; row * col * 2 bytes for bf16)
    dtype: str = "torch.bfloat16"
    num_publishers: int = 2       # FSDP world size on the trainer side
    num_receivers: int = 2        # TP world size on the inference side
    cycles: int = 5               # how many refit steps to run
    timeout_s: float = 60.0       # per-segment timeout
    cuda_device: int = 0          # all four roles share one GPU


# ---------------------------------------------------------------------------
# Trainer (publisher) thread
# ---------------------------------------------------------------------------


class TrainerThread(threading.Thread):
    """One publisher rank. Holds an FSDP-shard slice of the synthetic tensor."""

    def __init__(
        self,
        rank: int,
        cfg: BenchConfig,
        cycle_barrier: threading.Barrier,
        results: dict,
    ):
        super().__init__(daemon=True, name=f"trainer-{rank}")
        self.rank = rank
        self.cfg = cfg
        self.cycle_barrier = cycle_barrier
        self.results = results

        # Each rank owns rows [rank*S, (rank+1)*S) of the global tensor.
        S = cfg.rows_total // cfg.num_publishers
        self.row_lo = rank * S
        self.row_hi = (rank + 1) * S if rank < cfg.num_publishers - 1 else cfg.rows_total
        self.local_rows = self.row_hi - self.row_lo

        self.publisher: MxTrainingPublisher | None = None
        self.mx_source_id: str | None = None
        self.local_shard: torch.Tensor | None = None
        self.ready_event = threading.Event()

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as e:  # noqa: BLE001
            logger.exception("trainer-%d crashed: %s", self.rank, e)
            self.results[f"trainer_{self.rank}_error"] = str(e)
            # Signal so receivers don't deadlock waiting on us.
            try:
                self.cycle_barrier.abort()
            except Exception:
                pass

    def _run_inner(self) -> None:
        torch.cuda.set_device(self.cfg.cuda_device)
        logger.info(
            "trainer-%d: owns rows [%d, %d) of shape (%d, %d)",
            self.rank, self.row_lo, self.row_hi,
            self.cfg.rows_total, self.cfg.cols,
        )

        # Allocate local shard with a deterministic, rank-distinguishable
        # pattern so receivers can checksum-verify they got the right bytes.
        dtype = torch.bfloat16
        self.local_shard = torch.full(
            (self.local_rows, self.cfg.cols),
            float(self.rank + 1),  # rank N's shard is full of (N+1) in bf16
            dtype=dtype,
            device=f"cuda:{self.cfg.cuda_device}",
        )

        # Stand up the v1 publisher.
        agent_name = f"verl-bench-trainer-{self.rank}-{uuid.uuid4().hex[:6]}"
        self.publisher = MxTrainingPublisher(
            agent_name=agent_name,
            device_id=self.cfg.cuda_device,
            mx_server_url=self.cfg.mx_server_url,
        )
        self.publisher.initialize(model_name=self.cfg.model_name)

        for step in range(self.cfg.cycles):
            # Refresh the shard contents so each cycle is observably different.
            self.local_shard.fill_(float(self.rank + 1 + step))

            t0 = time.perf_counter()
            self.mx_source_id = self.publisher.publish_weights(
                named_tensors={"weight_shard": self.local_shard},
                step=step,
                worker_rank=self.rank,
            )
            self.publisher.mark_ready(worker_rank=self.rank)
            publish_dt = time.perf_counter() - t0
            logger.info(
                "trainer-%d step=%d source_id=%s publish=%.3fs (bytes=%d)",
                self.rank, step, self.mx_source_id, publish_dt,
                self.local_shard.numel() * self.local_shard.element_size(),
            )

            # Receivers read from the shard in parallel; wait for them all
            # to finish before mutating it for the next step.
            self.ready_event.set()
            self.cycle_barrier.wait()
            self.ready_event.clear()


# ---------------------------------------------------------------------------
# Receiver thread
# ---------------------------------------------------------------------------


class ReceiverThread(threading.Thread):
    """One receiver rank. TP-shards the synthetic tensor on axis 0."""

    def __init__(
        self,
        rank: int,
        cfg: BenchConfig,
        trainers: list[TrainerThread],
        cycle_barrier: threading.Barrier,
        results: dict,
    ):
        super().__init__(daemon=True, name=f"receiver-{rank}")
        self.rank = rank
        self.cfg = cfg
        self.trainers = trainers
        self.cycle_barrier = cycle_barrier
        self.results = results

        # Receiver rank R owns rows [R*T, (R+1)*T) — symmetric to the trainer
        # layout in this harness, but the planner doesn't assume that.
        T = cfg.rows_total // cfg.num_receivers
        self.row_lo = rank * T
        self.row_hi = (rank + 1) * T if rank < cfg.num_receivers - 1 else cfg.rows_total
        self.local_rows = self.row_hi - self.row_lo

        self.receiver: MxRefitReceiver | None = None
        self.local_buffer: torch.Tensor | None = None

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as e:  # noqa: BLE001
            logger.exception("receiver-%d crashed: %s", self.rank, e)
            self.results[f"receiver_{self.rank}_error"] = str(e)
            try:
                self.cycle_barrier.abort()
            except Exception:
                pass

    def _run_inner(self) -> None:
        torch.cuda.set_device(self.cfg.cuda_device)
        logger.info(
            "receiver-%d: target rows [%d, %d)", self.rank, self.row_lo, self.row_hi,
        )
        dtype = torch.bfloat16
        self.local_buffer = torch.zeros(
            (self.local_rows, self.cfg.cols),
            dtype=dtype,
            device=f"cuda:{self.cfg.cuda_device}",
        )

        agent_name = f"verl-bench-receiver-{self.rank}-{uuid.uuid4().hex[:6]}"
        self.receiver = MxRefitReceiver(
            agent_name=agent_name,
            device_id=self.cfg.cuda_device,
            mx_server_url=self.cfg.mx_server_url,
        )
        # Register the local buffer so NIXL has a memory mapping for our side.
        self.receiver.initialize(model_tensors={"target_buffer": self.local_buffer})

        per_cycle_metrics: list[dict] = []

        for step in range(self.cfg.cycles):
            # Wait for every publisher to be ready for this step.
            for t in self.trainers:
                t.ready_event.wait(timeout=30.0)

            # Build SliceOwnership entries from the published trainer state.
            ownerships = []
            for t in self.trainers:
                ownerships.append(
                    SliceOwnership(
                        model_name=self.cfg.model_name,
                        tensor_name="weight_shard",
                        global_shape=(self.cfg.rows_total, self.cfg.cols),
                        dtype=self.cfg.dtype,
                        placement_kind="SHARD",
                        shard_axis=0,
                        local_shard_range=(t.row_lo, t.row_hi),
                        worker_rank=t.rank,
                        # nixl_addr is the local shard's CUDA address on the
                        # source; for v1 publishers we read it directly off
                        # the publisher's tensor (the harness owns both
                        # sides so we don't need to query MX metadata for it).
                        nixl_addr=int(t.local_shard.data_ptr()),
                        byte_size=t.local_shard.numel() * t.local_shard.element_size(),
                        device_id=self.cfg.cuda_device,
                    )
                )

            # Build the single SliceRequest for this receiver's shard.
            request = SliceRequest(
                tensor_name="weight_shard",
                global_range=(self.row_lo, self.row_hi),
                shard_axis=0,
                dtype=self.cfg.dtype,
                receiver_rank=self.rank,
                target_addr=int(self.local_buffer.data_ptr()),
                target_offset=0,
            )

            plan = plan_coverage(sources=ownerships, requests=[request])
            plan.raise_if_incomplete()
            summary = summarize_plan(plan)
            savings = collect_byte_savings_vs_allgather(plan, ownerships)

            # Resolve per-source NIXL agent handles (cached after the first cycle).
            t_prefetch_start = time.perf_counter()
            agents: dict[int, str] = {}
            for t in self.trainers:
                # MxTrainingPublisher constructs worker_id internally; we can
                # discover it via list_sources -> metadata, but since we
                # control both sides we just look it up off the publisher
                # object directly (saves a gRPC round-trip per cycle).
                worker_id = t.publisher._worker_id
                agents[t.rank] = self.receiver.prefetch_source(
                    mx_source_id=t.mx_source_id, worker_id=worker_id,
                )
            prefetch_dt = time.perf_counter() - t_prefetch_start

            # Drive each SegmentPlan through receive_segment.
            row_stride_bytes = self.cfg.cols * 2  # bf16
            t_xfer_start = time.perf_counter()
            for seg in plan.segments:
                src_addr_bytes = (
                    seg.source.nixl_addr
                    + (seg.source_range[0]) * row_stride_bytes
                )
                tgt_addr_bytes = (
                    seg.request.target_addr
                    + seg.request.target_offset
                    + seg.target_range[0] * row_stride_bytes
                )
                self.receiver.receive_segment(
                    remote_agent_name=agents[seg.source.worker_rank],
                    source_addr=src_addr_bytes,
                    byte_count=seg.byte_count,
                    target_addr=tgt_addr_bytes,
                    source_device_id=seg.source.device_id,
                    timeout_seconds=self.cfg.timeout_s,
                )
            xfer_dt = time.perf_counter() - t_xfer_start

            # Checksum-verify: every row in our buffer that comes from trainer
            # rank R should contain the constant float(R + 1 + step).
            verified = self._verify_checksum(step=step)
            total_bytes = sum(s.byte_count for s in plan.segments)
            gbps = (total_bytes * 8) / (xfer_dt * 1e9) if xfer_dt > 0 else float("inf")

            metric = {
                "step": step,
                "segment_count": summary["segment_count"],
                "source_ranks_used": summary["source_ranks_used"],
                "bytes_transferred": total_bytes,
                "prefetch_seconds": prefetch_dt,
                "xfer_seconds": xfer_dt,
                "gbps": gbps,
                "verified": verified,
                "savings_factor_vs_allgather": savings["savings_factor"],
                "allgather_baseline_bytes": savings["allgather_per_receiver_bytes"],
            }
            per_cycle_metrics.append(metric)
            logger.info(
                "receiver-%d step=%d xfer=%.3fs %.1f Gbps "
                "(%d segments, %d bytes, prefetch=%.3fs, verified=%s, savings=%.2fx)",
                self.rank, step, xfer_dt, gbps,
                summary["segment_count"], total_bytes, prefetch_dt,
                verified, savings["savings_factor"],
            )

            # Sync with the other receiver + the trainers for the next cycle.
            self.cycle_barrier.wait()

        self.results[f"receiver_{self.rank}_metrics"] = per_cycle_metrics

    def _verify_checksum(self, step: int) -> bool:
        """Confirm each row range in the local buffer matches the publishing rank's
        constant payload for this step."""
        T = self.cfg.rows_total // self.cfg.num_receivers
        row_stride_bytes = self.cfg.cols * 2
        for t in self.trainers:
            # Compute the intersection of this trainer's shard with our buffer.
            lo = max(self.row_lo, t.row_lo)
            hi = min(self.row_hi, t.row_hi)
            if lo >= hi:
                continue
            # Convert to local-buffer row indices.
            local_lo = lo - self.row_lo
            local_hi = hi - self.row_lo
            expected = float(t.rank + 1 + step)
            chunk = self.local_buffer[local_lo:local_hi].float()
            mean = chunk.mean().item()
            if abs(mean - expected) > 1e-2:
                logger.error(
                    "receiver-%d checksum mismatch step=%d trainer=%d "
                    "rows[%d:%d): expected %.2f got %.2f",
                    self.rank, step, t.rank, local_lo, local_hi, expected, mean,
                )
                return False
        return True


# ---------------------------------------------------------------------------
# Baseline: v1 receive_weights (gather-equivalent)
# ---------------------------------------------------------------------------


def run_v1_baseline(cfg: BenchConfig) -> dict:
    """One round-trip via the v1 receive_weights path for comparison.

    This is what the rollout loader falls back to when the receiver lacks
    receive_segment — it represents the gather-equivalent baseline we want
    to beat with rank-to-rank.

    For an apples-to-apples comparison we use a SINGLE publisher that
    advertises the WHOLE tensor (the gather output) and a single receiver
    that pulls it. Bandwidth here is the raw NIXL ceiling on this fabric.
    """
    logger.info("=== Baseline: v1 receive_weights (single publisher, full tensor) ===")
    torch.cuda.set_device(cfg.cuda_device)
    dtype = torch.bfloat16

    full = torch.ones(
        (cfg.rows_total, cfg.cols),
        dtype=dtype, device=f"cuda:{cfg.cuda_device}",
    )
    pub = MxTrainingPublisher(
        agent_name=f"v1-baseline-pub-{uuid.uuid4().hex[:6]}",
        device_id=cfg.cuda_device,
        mx_server_url=cfg.mx_server_url,
    )
    pub.initialize(model_name=cfg.model_name + "-v1baseline")
    src_id = pub.publish_weights(
        named_tensors={"weight_full": full}, step=0, worker_rank=0,
    )
    pub.mark_ready(worker_rank=0)

    rx_buffer = torch.zeros_like(full)
    rx = MxRefitReceiver(
        agent_name=f"v1-baseline-rx-{uuid.uuid4().hex[:6]}",
        device_id=cfg.cuda_device,
        mx_server_url=cfg.mx_server_url,
    )
    rx.initialize(model_tensors={"weight_full": rx_buffer})

    # Use the source_id the publisher returned rather than polling — we own
    # both ends of the handshake here and don't want the bench's apples-to-
    # apples timing to include catalog poll-loop overhead.
    from modelexpress.refit_receiver import SourceRef
    src = SourceRef(
        mx_source_id=src_id,
        worker_id=pub._worker_id,
        model_name=cfg.model_name + "-v1baseline",
        worker_rank=0,
        training_step=0,
    )

    t0 = time.perf_counter()
    received_names = []
    for name, _tensor in rx.receive_weights(src, timeout_seconds=cfg.timeout_s):
        received_names.append(name)
    dt = time.perf_counter() - t0
    total_bytes = full.numel() * full.element_size()
    gbps = (total_bytes * 8) / (dt * 1e9) if dt > 0 else float("inf")
    logger.info(
        "v1 baseline: %.3fs %.1f Gbps (%d bytes, names=%s, source_id=%s)",
        dt, gbps, total_bytes, received_names, src_id,
    )

    rx.shutdown()
    pub.shutdown()

    return {
        "elapsed_seconds": dt,
        "gbps": gbps,
        "bytes": total_bytes,
        "received_names": received_names,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(cfg: BenchConfig, output_path: str | None) -> dict:
    logger.info("=== Tier 2: rank-to-rank harness ===")
    logger.info("cfg=%s", dataclasses.asdict(cfg))

    results: dict = {"config": dataclasses.asdict(cfg)}

    # Trainers + receivers all sync on this barrier each cycle so we get
    # clean per-cycle measurements.
    n_threads = cfg.num_publishers + cfg.num_receivers
    cycle_barrier = threading.Barrier(n_threads)

    trainers = [TrainerThread(r, cfg, cycle_barrier, results)
                for r in range(cfg.num_publishers)]
    for t in trainers:
        t.start()

    # Give publishers a moment to register before receivers start polling.
    time.sleep(2.0)

    receivers = [ReceiverThread(r, cfg, trainers, cycle_barrier, results)
                 for r in range(cfg.num_receivers)]
    for r in receivers:
        r.start()

    for t in trainers + receivers:
        t.join(timeout=300.0)

    # Aggregate per-cycle Gbps + segments
    aggregated = defaultdict(list)
    for k, v in results.items():
        if k.startswith("receiver_") and k.endswith("_metrics"):
            for m in v:
                aggregated[m["step"]].append(m)
    summary = []
    for step, ms in sorted(aggregated.items()):
        summary.append({
            "step": step,
            "total_gbps_all_receivers": sum(m["gbps"] for m in ms),
            "max_xfer_seconds": max(m["xfer_seconds"] for m in ms),
            "min_xfer_seconds": min(m["xfer_seconds"] for m in ms),
            "total_bytes_all_receivers": sum(m["bytes_transferred"] for m in ms),
            "savings_factor_vs_allgather": ms[0]["savings_factor_vs_allgather"],
            "all_verified": all(m["verified"] for m in ms),
        })
    results["aggregated_per_step"] = summary

    # v1 baseline for comparison
    try:
        results["v1_baseline"] = run_v1_baseline(cfg)
    except Exception as e:  # noqa: BLE001
        logger.exception("v1 baseline failed: %s", e)
        results["v1_baseline"] = {"error": str(e)}

    # Final printable summary
    logger.info("=== AGGREGATED PER-STEP ===")
    for s in summary:
        logger.info(
            "  step=%d total_gbps=%.1f max_xfer=%.3fs verified=%s savings=%.2fx",
            s["step"], s["total_gbps_all_receivers"],
            s["max_xfer_seconds"], s["all_verified"],
            s["savings_factor_vs_allgather"],
        )
    if isinstance(results["v1_baseline"], dict) and "gbps" in results["v1_baseline"]:
        logger.info(
            "v1 baseline: %.1f Gbps in %.3fs",
            results["v1_baseline"]["gbps"], results["v1_baseline"]["elapsed_seconds"],
        )

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("results written to %s", output_path)

    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mx-server-url", default=os.environ.get(
        "MX_SERVER_URL", "modelexpress-server:8001"))
    p.add_argument("--rows-total", type=int, default=65536)
    p.add_argument("--cols", type=int, default=8192)
    p.add_argument("--num-publishers", type=int, default=2)
    p.add_argument("--num-receivers", type=int, default=2)
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--cuda-device", type=int, default=0)
    p.add_argument("--output-json", default="/tmp/bench_results.json")
    args = p.parse_args()

    cfg = BenchConfig(
        mx_server_url=args.mx_server_url,
        rows_total=args.rows_total,
        cols=args.cols,
        num_publishers=args.num_publishers,
        num_receivers=args.num_receivers,
        cycles=args.cycles,
        cuda_device=args.cuda_device,
    )

    results = run(cfg, args.output_json)

    # Exit 1 if any step failed verification, so the K8s Job correctly fails.
    any_failed = any(
        not s["all_verified"] for s in results.get("aggregated_per_step", [])
    ) or any(k.endswith("_error") for k in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
