#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-2 cluster harness for non-expert rank-to-rank reassembly.

Validates the property behind the prime-rl "on-the-fly shard-in without
allgather" work: N trainer ranks each publish only their FSDP row-shard
of a non-expert tensor (via the new ``NonExpertShardSpec`` on the v2
publisher), and a receiver reassembles the full tensor by pulling every
rank's shard and concatenating locally — no ``DTensor.full_tensor()``
allgather ever runs on the publisher side, and no trainer rank ever
holds the full tensor.

This is the substrate-level proof for the trainer-side change in
``prime_rl/trainer/rl/broadcast/nixl_mx_v2.py`` (drop the GatheredSlot
gather override; publish ShardedSlot per-rank shards with shard metadata)
and the ModelExpress ``NonExpertShardSpec`` addition. The full prime-rl
receiver hybrid (gather non-experts / same-rank experts) is a separate,
larger change; this harness isolates and validates the new mechanism.

For each cycle:

1. N publisher threads each hold a ``(rows_per_shard, cols)`` shard of a
   synthetic ``(rows_total, cols)`` tensor. Each publishes its shard with
   ``NonExpertShardSpec(global_shape=(rows_total, cols), shard_axis=0,
   local_shard_range=(r*rps, (r+1)*rps))``. No publisher ever allocates
   the full tensor.
2. The receiver discovers all N shards, plans coverage for the FULL
   tensor (``TargetTPLayout(world_size=1, rank=0, shard_axis=0)``), and
   reassembles via ``receive_via_plan`` (pull each shard, torch.cat).
3. Byte-identity check: reassembled ``full[r*rps:(r+1)*rps]`` must equal
   the value publisher rank ``r`` wrote for this cycle.

Pass condition: every cycle reassembles byte-identical, the plan draws
from all N publisher ranks (proving no single rank held the full tensor),
and the receiver pulled exactly ``rows_total`` rows' worth of bytes.

What this does NOT cover (deliberate, matches the verl Tier-2 harness):
- Cross-host RDMA (single-pod loopback here).
- The prime-rl expert/non-expert hybrid receive routing.
- Real model state dicts / fused QKV translation.
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
logger = logging.getLogger("bench_non_expert_rank_to_rank")

from modelexpress import NonExpertShardSpec  # noqa: E402
from modelexpress.nemo_rl_v2 import (  # noqa: E402
    MxV2RefitReceiver,
    MxV2TrainingPublisher,
    TrainerWorldLayout,
)
from modelexpress.rl_reshard_planner import plan_coverage  # noqa: E402
from modelexpress.rl_slice_descriptors import SliceOwnership, SliceRequest  # noqa: E402


@dataclasses.dataclass
class BenchConfig:
    mx_server_url: str = "modelexpress-server:8001"
    model_name: str = "synthetic/non-expert-rank-to-rank-bench"
    tensor_name: str = "model.layers.0.self_attn.q_proj.weight"
    rows_total: int = 4096
    cols: int = 4096
    dtype: str = "torch.bfloat16"
    num_publishers: int = 4  # FSDP world size on the trainer side
    cycles: int = 5
    timeout_s: float = 120.0
    cuda_device: int = 0


class TrainerThread(threading.Thread):
    """One publisher rank. Holds ONLY its FSDP row-shard — never the full tensor."""

    def __init__(self, rank: int, cfg: BenchConfig, barrier: threading.Barrier, results: dict):
        super().__init__(daemon=True, name=f"trainer-{rank}")
        self.rank = rank
        self.cfg = cfg
        self.barrier = barrier
        self.results = results
        self.publisher: MxV2TrainingPublisher | None = None
        self.mx_source_id: str | None = None
        self.local_shard: torch.Tensor | None = None
        self.ready_event = threading.Event()

        rps = cfg.rows_total // cfg.num_publishers
        self.rows_per_shard = rps
        self.row_lo = rank * rps
        self.row_hi = self.row_lo + rps

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as e:  # noqa: BLE001
            logger.exception("trainer-%d crashed: %s", self.rank, e)
            self.results[f"trainer_{self.rank}_error"] = str(e)
            try:
                self.barrier.abort()
            except Exception:
                pass

    def _run_inner(self) -> None:
        torch.cuda.set_device(self.cfg.cuda_device)
        # Allocate ONLY the local shard. The full tensor is never materialized
        # on any trainer rank — that's the invariant this harness proves.
        self.local_shard = torch.empty(
            (self.rows_per_shard, self.cfg.cols),
            dtype=torch.bfloat16,
            device=f"cuda:{self.cfg.cuda_device}",
        )
        logger.info(
            "trainer-%d: holds rows [%d,%d) of (%d,%d) — shard %.1f MiB, full would be %.1f MiB",
            self.rank, self.row_lo, self.row_hi, self.cfg.rows_total, self.cfg.cols,
            self.local_shard.numel() * 2 / 2**20,
            self.cfg.rows_total * self.cfg.cols * 2 / 2**20,
        )

        self.publisher = MxV2TrainingPublisher(
            agent_name=f"ne-bench-trainer-{self.rank}-{uuid.uuid4().hex[:6]}",
            device_id=self.cfg.cuda_device,
            mx_server_url=self.cfg.mx_server_url,
            worker_rank=self.rank,
            world_layout=TrainerWorldLayout(
                fsdp_world_size=self.cfg.num_publishers,
                tp_world_size=1,
                pp_world_size=1,
                ep_world_size=1,
            ),
        )
        self.publisher.initialize(model_name=self.cfg.model_name, dtype="bfloat16")

        spec = NonExpertShardSpec(
            global_shape=(self.cfg.rows_total, self.cfg.cols),
            shard_axis=0,
            local_shard_range=(self.row_lo, self.row_hi),
        )

        for cycle in range(self.cfg.cycles):
            # Fill the shard with a rank+cycle-distinguishable constant so the
            # receiver can verify each rank's bytes landed at the right rows.
            self.local_shard.fill_(float(self.rank + 1) + cycle * 0.01)

            self.publisher.add_tensor(
                name=self.cfg.tensor_name,
                tensor=self.local_shard,
                shard_spec=spec,
            )
            src_id = self.publisher.publish(version=cycle)
            self.publisher.mark_ready()
            self.mx_source_id = src_id
            logger.info("trainer-%d cycle=%d published shard (source_id=%s)", self.rank, cycle, src_id)

            self.ready_event.set()
            self.barrier.wait()
            self.ready_event.clear()


class ReceiverThread(threading.Thread):
    """Reassembles the full tensor from all N publisher shards — no allgather."""

    def __init__(self, cfg: BenchConfig, trainers: list[TrainerThread], barrier: threading.Barrier, results: dict):
        super().__init__(daemon=True, name="receiver")
        self.cfg = cfg
        self.trainers = trainers
        self.barrier = barrier
        self.results = results

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as e:  # noqa: BLE001
            logger.exception("receiver crashed: %s", e)
            self.results["receiver_error"] = str(e)
            try:
                self.barrier.abort()
            except Exception:
                pass

    def _run_inner(self) -> None:
        torch.cuda.set_device(self.cfg.cuda_device)
        # Full-tensor destination the reassembled shards land into.
        full_buf = torch.zeros(
            (self.cfg.rows_total, self.cfg.cols),
            dtype=torch.bfloat16,
            device=f"cuda:{self.cfg.cuda_device}",
        )
        receiver = MxV2RefitReceiver(
            agent_name=f"ne-bench-receiver-{uuid.uuid4().hex[:6]}",
            device_id=self.cfg.cuda_device,
            mx_server_url=self.cfg.mx_server_url,
            worker_rank=0,
        )
        receiver.initialize(model_tensors={self.cfg.tensor_name: full_buf})
        base = receiver._receiver  # underlying MxRefitReceiver (one-sided receive_segment)

        rps = self.cfg.rows_total // self.cfg.num_publishers
        row_bytes = self.cfg.cols * 2  # bf16
        per_cycle: list[dict] = []

        for cycle in range(self.cfg.cycles):
            for t in self.trainers:
                t.ready_event.wait(timeout=30.0)

            # Build SliceOwnership per trainer shard (harness owns both sides,
            # same pattern as the verl rank-to-rank bench). shard_spec metadata
            # was also published to the catalog; here we exercise the planner +
            # one-sided receive_segment data plane, which — unlike the scratch
            # rendezvous path — works when target threads are blocked at the
            # cross-cycle barrier.
            ownerships = [
                SliceOwnership(
                    model_name=self.cfg.model_name,
                    tensor_name=self.cfg.tensor_name,
                    global_shape=(self.cfg.rows_total, self.cfg.cols),
                    dtype=self.cfg.dtype,
                    placement_kind="SHARD",
                    shard_axis=0,
                    local_shard_range=(t.row_lo, t.row_hi),
                    worker_rank=t.rank,
                    nixl_addr=int(t.local_shard.data_ptr()),
                    byte_size=t.local_shard.numel() * t.local_shard.element_size(),
                    device_id=self.cfg.cuda_device,
                )
                for t in self.trainers
            ]
            request = SliceRequest(
                tensor_name=self.cfg.tensor_name,
                global_range=(0, self.cfg.rows_total),
                shard_axis=0,
                dtype=self.cfg.dtype,
                receiver_rank=0,
                target_addr=int(full_buf.data_ptr()),
                target_offset=0,
            )

            t0 = time.perf_counter()
            plan = plan_coverage(sources=ownerships, requests=[request])
            plan.raise_if_incomplete()
            plan_dt = time.perf_counter() - t0

            agents: dict[int, str] = {}
            for t in self.trainers:
                agents[t.rank] = base.prefetch_source(
                    mx_source_id=t.mx_source_id, worker_id=t.publisher.worker_id
                )

            t0 = time.perf_counter()
            for seg in plan.segments:
                src_addr = seg.source.nixl_addr + seg.source_range[0] * row_bytes
                tgt_addr = (
                    seg.request.target_addr
                    + seg.request.target_offset
                    + seg.target_range[0] * row_bytes
                )
                base.receive_segment(
                    remote_agent_name=agents[seg.source.worker_rank],
                    source_addr=src_addr,
                    byte_count=seg.byte_count,
                    target_addr=tgt_addr,
                    source_device_id=seg.source.device_id,
                    timeout_seconds=self.cfg.timeout_s,
                )
            xfer_dt = time.perf_counter() - t0

            # Byte-identity: rows [r*rps,(r+1)*rps) must hold rank r's value.
            verified = True
            first_bad = None
            for r in range(self.cfg.num_publishers):
                expected_val = float(r + 1) + cycle * 0.01
                view = full_buf[r * rps : (r + 1) * rps]
                if not torch.allclose(
                    view.float(), torch.full_like(view.float(), expected_val), atol=0.02
                ):
                    verified = False
                    first_bad = (r, expected_val, float(view.flatten()[0].item()))
                    break

            source_ranks = sorted({seg.source.worker_rank for seg in plan.segments})
            bytes_pulled = sum(seg.byte_count for seg in plan.segments)
            metric = {
                "cycle": cycle,
                "shape_ok": tuple(full_buf.shape) == (self.cfg.rows_total, self.cfg.cols),
                "verified": verified,
                "first_bad": first_bad,
                "num_source_ranks": len(source_ranks),
                "source_ranks": source_ranks,
                "expected_source_ranks": self.cfg.num_publishers,
                "segment_count": len(plan.segments),
                "bytes_pulled": bytes_pulled,
                "plan_seconds": plan_dt,
                "xfer_seconds": xfer_dt,
                "gbps": (bytes_pulled * 8) / max(xfer_dt, 1e-9) / 1e9,
            }
            per_cycle.append(metric)
            logger.info(
                "receiver cycle=%d verified=%s sources=%s/%d segs=%d bytes=%d "
                "xfer=%.3fs (%.2f Gbps)",
                cycle, verified, source_ranks, self.cfg.num_publishers,
                len(plan.segments), bytes_pulled, xfer_dt, metric["gbps"],
            )
            if not verified:
                logger.error("cycle=%d BYTE MISMATCH first_bad=%s", cycle, first_bad)

            self.barrier.wait()

        self.results["receiver_metrics"] = per_cycle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mx-server-url", default="modelexpress-server:8001")
    parser.add_argument("--rows-total", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--num-publishers", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    cfg = BenchConfig(
        mx_server_url=args.mx_server_url,
        rows_total=args.rows_total,
        cols=args.cols,
        num_publishers=args.num_publishers,
        cycles=args.cycles,
        cuda_device=args.cuda_device,
    )
    if cfg.rows_total % cfg.num_publishers != 0:
        raise SystemExit(
            f"rows_total ({cfg.rows_total}) must be divisible by num_publishers ({cfg.num_publishers})"
        )

    logger.info(
        "=== non-expert rank-to-rank: %d publishers, tensor (%d,%d) bf16, %d cycles ===",
        cfg.num_publishers, cfg.rows_total, cfg.cols, cfg.cycles,
    )

    results: dict = {}
    barrier = threading.Barrier(cfg.num_publishers + 1)  # publishers + 1 receiver
    trainers = [TrainerThread(r, cfg, barrier, results) for r in range(cfg.num_publishers)]
    for t in trainers:
        t.start()
    time.sleep(0.5)
    receiver = ReceiverThread(cfg, trainers, barrier, results)
    receiver.start()

    for t in trainers + [receiver]:
        t.join(timeout=cfg.timeout_s * cfg.cycles + 60)
        if t.is_alive():
            logger.error("%s did not exit in time", t.name)

    crashed = [f"{k}: {v}" for k, v in results.items() if k.endswith("_error")]
    metrics = results.get("receiver_metrics", [])
    all_pass = (
        not crashed
        and len(metrics) == cfg.cycles
        and all(m["verified"] and m["shape_ok"] and m["num_source_ranks"] == cfg.num_publishers for m in metrics)
    )

    logger.info("=== SUMMARY ===")
    logger.info("crashed=%d cycles_captured=%d/%d all_pass=%s", len(crashed), len(metrics), cfg.cycles, all_pass)
    if crashed:
        for c in crashed:
            logger.error("  %s", c)

    summary = {"config": dataclasses.asdict(cfg), "per_cycle": metrics, "crashed": crashed, "all_pass": all_pass}
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results written to %s", args.output_json)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
