#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elastic-scale-up + compile-target + tree-fan-out benchmark for MX v2.

This is a *transport-layer* benchmark — it spins up a synthetic trainer
that publishes tensors via :class:`MxV2TrainingPublisher`, then spawns
N receivers (each :class:`MxV2RefitReceiver` driven via
:class:`MxWeightTransferEngine`) and records:

  - cold-start join latency (time from receiver start to first successful
    receive)
  - per-cycle RDMA bandwidth (GB/s) and tensor count
  - discovery (control-plane) latency vs RDMA (data-plane) latency
  - compile-target filter behavior (accept / reject / back-compat-no-filter)
  - trainer egress savings under tree fan-out (pipeline replication)

The benchmark does NOT need vLLM, NCCL, or even ``transformers`` —
just the MX v2 fat clients and a live MX server. It does need a CUDA
device and NIXL for real RDMA numbers; with ``--mode=cpu`` it runs a
shape-only smoke test (no real transfer, useful for CI).

Scenarios
---------

``elastic_scale``
    Trainer publishes a fixed model for ``--steps`` versions. Receivers
    join staggered every ``--join-interval`` seconds. Per receiver we
    record join latency and per-version bandwidth.

``compile_target``
    Publisher tags bytes with ``compile_target="cutlass_fp8"``. Three
    receivers run simultaneously:

      1. ``filter=cutlass_fp8`` (matched) — accepts
      2. ``filter=deep_gemm_fp8`` (mismatched) — refuses BEFORE RDMA
      3. ``filter=None`` (back-compat) — accepts (no filter)

    Output proves the safety net + the back-compat property in one shot.

``tree_fanout``
    Identical to ``elastic_scale`` but with ``publish_self_as_replica=True``
    on every receiver. Receivers 2..N can discover and pull from
    receivers 1..N-1 instead of the trainer. We measure trainer egress
    bytes vs total bytes received as the "fan-out factor".

Usage
-----

Single-host smoke (no GPUs, no MX server — exercises plumbing only)::

    python bench_elastic_scaling.py --mode=cpu --scenario=elastic_scale \\
        --num-receivers=3 --tensor-bytes=1048576 --num-tensors=4

Full cluster run (against an MX server in a Kubernetes namespace)::

    python bench_elastic_scaling.py \\
        --mx-server-url=modelexpress-server.<NAMESPACE>.svc.cluster.local:8001 \\
        --scenario=elastic_scale --num-receivers=4 --steps=3 \\
        --join-interval=2.0 --num-tensors=64 --tensor-bytes=8388608 \\
        --output=results.json

Outputs
-------

A JSON document with the metrics blob (machine-readable) and a printed
human summary table. Pipe ``--output=results.json`` to capture and
compare across runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("bench_elastic_scaling")


# ----------------------------------------------------------------------------
# Result schema
# ----------------------------------------------------------------------------


@dataclass
class ReceiverCycleResult:
    """One receive cycle worth of metrics."""

    version: int
    bytes_received: int = 0
    tensors_received: int = 0
    rdma_seconds: float = 0.0
    bandwidth_gbps: float = 0.0
    discovery_seconds: float = 0.0
    source_worker_rank: int | None = None
    error: str | None = None


@dataclass
class ReceiverResult:
    """All cycles for one receiver."""

    receiver_id: str
    worker_rank: int
    started_at: float
    first_receive_at: float | None = None
    join_latency_seconds: float | None = None  # = first_receive_at - started_at
    compile_target_filter: list[str] | None = None
    cycles: list[ReceiverCycleResult] = field(default_factory=list)

    def total_bytes(self) -> int:
        return sum(c.bytes_received for c in self.cycles)

    def total_rdma_seconds(self) -> float:
        return sum(c.rdma_seconds for c in self.cycles)

    def avg_bandwidth_gbps(self) -> float:
        t = self.total_rdma_seconds()
        return (self.total_bytes() * 8) / (t * 1e9) if t > 0 else 0.0


@dataclass
class TrainerResult:
    """Publisher-side stats."""

    worker_id: str | None
    mx_source_id: str | None
    started_at: float
    published_versions: list[int] = field(default_factory=list)
    compile_target: str | None = None
    total_published_bytes: int = 0


@dataclass
class BenchResult:
    scenario: str
    config: dict[str, Any]
    trainer: TrainerResult | None
    receivers: list[ReceiverResult]
    started_at: float
    finished_at: float

    def trainer_egress_bytes(self) -> int:
        """Bytes the trainer actually had to serve out.

        For matched-TP + non-fan-out: == sum(receiver.total_bytes()).
        For tree-fan-out: < that, because newcomers pulled from peers.
        We approximate this as "bytes received by receivers whose
        source_worker_rank == the trainer's rank"; receivers that
        pulled from replicas don't count.

        Since the worker_rank of the trainer is always 0 in this
        harness, and same-rank-only is set, the receiver's
        source_worker_rank == 0 means "pulled from trainer". Other
        values are "pulled from a replica".
        """
        egress = 0
        for r in self.receivers:
            for c in r.cycles:
                if c.source_worker_rank == 0:
                    egress += c.bytes_received
        return egress

    def to_summary_table(self) -> str:
        lines = []
        lines.append(f"== Scenario: {self.scenario} ==")
        lines.append(f"Wall time: {self.finished_at - self.started_at:.2f}s")
        if self.trainer is not None:
            lines.append(
                f"Trainer: {self.trainer.worker_id} versions={self.trainer.published_versions} "
                f"bytes={self.trainer.total_published_bytes / 1e6:.1f} MB "
                f"compile_target={self.trainer.compile_target}"
            )
        lines.append("")
        lines.append(
            f"{'receiver':<20} {'filter':<18} {'join_s':>8} {'cycles':>6} "
            f"{'bytes_MB':>10} {'avg_Gbps':>10} {'errors':>7}"
        )
        for r in self.receivers:
            errors = sum(1 for c in r.cycles if c.error)
            filt = (
                ",".join(r.compile_target_filter)
                if r.compile_target_filter
                else "(none)"
            )
            join_str = (
                f"{r.join_latency_seconds:.2f}"
                if r.join_latency_seconds is not None
                else "n/a"
            )
            lines.append(
                f"{r.receiver_id:<20} {filt:<18} {join_str:>8} "
                f"{len(r.cycles):>6} "
                f"{r.total_bytes() / 1e6:>10.1f} "
                f"{r.avg_bandwidth_gbps():>10.2f} "
                f"{errors:>7}"
            )
        if self.scenario.removesuffix("_cpu_smoke") == "tree_fanout":
            total = sum(r.total_bytes() for r in self.receivers)
            egress = self.trainer_egress_bytes()
            ratio = total / egress if egress > 0 else float("inf")
            lines.append("")
            lines.append(
                f"Tree fan-out: trainer_egress={egress / 1e6:.1f} MB, "
                f"total_delivered={total / 1e6:.1f} MB, fanout_factor={ratio:.2f}x"
            )
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "config": self.config,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_seconds": self.finished_at - self.started_at,
            "trainer": asdict(self.trainer) if self.trainer else None,
            "receivers": [asdict(r) for r in self.receivers],
            "derived": {
                "trainer_egress_bytes": self.trainer_egress_bytes(),
                "total_delivered_bytes": sum(r.total_bytes() for r in self.receivers),
                "scenario_specific": _scenario_derived(self),
            },
        }


def _scenario_derived(b: BenchResult) -> dict[str, Any]:
    """Per-scenario derived numbers that don't fit the generic schema.

    Accepts both ``"elastic_scale"`` and ``"elastic_scale_cpu_smoke"``
    style names so the CPU smoke path produces the same derived
    metrics as live runs.
    """
    name = b.scenario.removesuffix("_cpu_smoke")
    if name == "elastic_scale":
        latencies = [
            r.join_latency_seconds
            for r in b.receivers
            if r.join_latency_seconds is not None
        ]
        return {
            "join_latency_p50": _p(latencies, 0.5),
            "join_latency_p99": _p(latencies, 0.99),
        }
    if name == "compile_target":
        verdicts: dict[str, str] = {}
        for r in b.receivers:
            ok = any(not c.error for c in r.cycles)
            verdicts[r.receiver_id] = "accepted" if ok else "rejected"
        return {"verdicts": verdicts}
    if name == "tree_fanout":
        total = sum(r.total_bytes() for r in b.receivers)
        egress = b.trainer_egress_bytes()
        return {
            "fanout_factor": (total / egress) if egress > 0 else None,
            "trainer_egress_mb": egress / 1e6,
            "total_delivered_mb": total / 1e6,
        }
    return {}


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[idx]


# ----------------------------------------------------------------------------
# Trainer + receiver entry points (run as subprocesses)
# ----------------------------------------------------------------------------


def _run_trainer(
    *,
    role: str,
    mx_server_url: str,
    model_name: str,
    num_tensors: int,
    tensor_bytes: int,
    steps: int,
    step_interval_s: float,
    compile_target: str,
    device_id: int,
    result_path: str,
) -> None:
    """Publisher subprocess entry point."""
    import torch
    from modelexpress.nemo_rl_v2 import MxV2TrainingPublisher, TrainerWorldLayout
    from modelexpress.shape_descriptors import COMPILE_TARGET_HF_RAW

    layout = TrainerWorldLayout(tp_world_size=1, pp_world_size=1, ep_world_size=1)
    pub = MxV2TrainingPublisher(
        agent_name="bench-trainer-r0",
        device_id=device_id,
        mx_server_url=mx_server_url,
        worker_rank=0,
        world_layout=layout,
    )
    pub.initialize(model_name=model_name, dtype="bfloat16")

    # Build synthetic tensors once and reuse buffers across steps.
    dtype = torch.bfloat16
    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel_per_tensor = max(1, tensor_bytes // elem_size)
    device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")
    tensors = {
        f"layer{i}.weight": torch.randn(numel_per_tensor, dtype=dtype, device=device)
        for i in range(num_tensors)
    }
    total_bytes = num_tensors * numel_per_tensor * elem_size

    result = TrainerResult(
        worker_id=pub.worker_id,
        mx_source_id=pub.mx_source_id,
        started_at=time.time(),
        compile_target=compile_target,
        total_published_bytes=total_bytes * steps,
    )
    for version in range(1, steps + 1):
        for name, t in tensors.items():
            pub.add_tensor(
                name=name,
                tensor=t,
                compile_target=compile_target,
                compile_metadata={"benchmark": "elastic", "step": version},
            )
        pub.publish(version=version)
        # Bump status to READY so receivers' list_sources() finds it.
        # On the first publish this also starts the heartbeat; subsequent
        # calls are idempotent (the publisher only re-registers if needed).
        pub.mark_ready()
        result.published_versions.append(version)
        logger.info("trainer: published v=%d (%d tensors)", version, num_tensors)
        if version < steps:
            time.sleep(step_interval_s)

    # Hold the trainer alive long enough for late receivers to find us.
    # The orchestrator signals shutdown by deleting the heartbeat lock
    # file; for simplicity here we just sleep for a generous tail.
    time.sleep(max(5.0, step_interval_s * steps))

    pub.shutdown()
    with open(result_path, "w") as f:
        json.dump(asdict(result), f)


def _run_receiver(
    *,
    receiver_id: str,
    worker_rank: int,
    mx_server_url: str,
    model_name: str,
    device_id: int,
    listen_port: int | None,
    compile_target_filter: list[str] | None,
    target_versions: list[int],
    poll_interval_s: float,
    cycle_timeout_s: float,
    deadline_s: float,
    publish_self_as_replica: bool,
    result_path: str,
    same_rank_only: bool = True,
) -> None:
    """Receiver subprocess entry point.

    Drives the v2 receiver via :class:`MxWeightTransferEngine` so we
    exercise the actual adapter path that vLLM will use.
    """
    os.environ.setdefault("MX_WEIGHT_TRANSFER_AUTOREGISTER", "0")
    from modelexpress.engines.vllm.weight_transfer import (
        MxInitInfo,
        MxUpdateInfo,
        MxWeightTransferEngine,
    )

    engine = MxWeightTransferEngine(
        init_info=MxInitInfo(
            mx_server_url=mx_server_url,
            model_name=model_name,
            worker_rank=worker_rank,
            agent_name=f"bench-{receiver_id}",
            device_id=device_id,
            listen_port=listen_port,
            publish_self_as_replica=publish_self_as_replica,
        )
    )
    result = ReceiverResult(
        receiver_id=receiver_id,
        worker_rank=worker_rank,
        started_at=time.time(),
        compile_target_filter=compile_target_filter,
    )
    # The benchmark doesn't consume the loaded weights — it only measures
    # per-cycle wall + bandwidth — so discard them. Accumulating every
    # tensor from every cycle (the prior ``captured.extend`` pattern) held
    # the full model in host memory once per version per receiver, easily
    # OOMing the pod before the benchmark finished on real-scale models.
    def _discard_loaded_weights(_: list[tuple[str, Any]]) -> None:
        return

    deadline = time.monotonic() + deadline_s

    for v in target_versions:
        cycle = ReceiverCycleResult(version=v)
        # Poll until a candidate source is observable.
        nixl_retry_budget = 2  # NIXL_ERR_NOT_ALLOWED is often transient at first connect
        while time.monotonic() < deadline:
            try:
                engine.receive_weights(
                    MxUpdateInfo(
                        version=v,
                        compile_target_filter=set(compile_target_filter)
                        if compile_target_filter
                        else None,
                        timeout_seconds=cycle_timeout_s,
                        same_rank_only=same_rank_only,
                    ),
                    load_weights=_discard_loaded_weights,
                )
                stats = engine.last_transfer_stats
                if stats is not None:
                    cycle.bytes_received = stats.bytes_received
                    cycle.tensors_received = stats.tensors_received
                    cycle.rdma_seconds = stats.elapsed_seconds
                    cycle.bandwidth_gbps = stats.bandwidth_gbps
                    cycle.source_worker_rank = stats.source_worker_rank
                cycle.discovery_seconds = engine.last_discovery_seconds
                if result.first_receive_at is None:
                    result.first_receive_at = time.time()
                    result.join_latency_seconds = (
                        result.first_receive_at - result.started_at
                    )
                logger.info(
                    "%s: v=%d bytes=%.1fMB rdma=%.2fs %.1fGbps from_rank=%s",
                    receiver_id,
                    v,
                    cycle.bytes_received / 1e6,
                    cycle.rdma_seconds,
                    cycle.bandwidth_gbps,
                    cycle.source_worker_rank,
                )
                break
            except RuntimeError as e:
                msg = str(e)
                # Phase 3b safety net: filter rejection is a "decided" outcome,
                # not a transient error. Record it and move on.
                if "no source matches filters" in msg or "no covering source set" in msg:
                    cycle.error = msg
                    logger.info("%s: v=%d filter rejected: %s", receiver_id, v, msg)
                    break
                # Otherwise the source isn't published yet — poll again.
                time.sleep(poll_interval_s)
            except Exception as e:  # noqa: BLE001
                # NIXL_ERR_NOT_ALLOWED and similar transient connection-setup
                # errors. Record + retry a couple of times before giving up,
                # so one receiver's flake doesn't tank the whole scenario.
                msg = f"{type(e).__name__}: {e}"
                if nixl_retry_budget > 0:
                    nixl_retry_budget -= 1
                    logger.warning(
                        "%s: v=%d transient error (%s); retrying (%d remaining)",
                        receiver_id, v, msg, nixl_retry_budget,
                    )
                    time.sleep(0.5)
                    continue
                cycle.error = msg
                logger.warning("%s: v=%d failed after retries: %s", receiver_id, v, msg)
                break
        else:
            cycle.error = "deadline exceeded"
        result.cycles.append(cycle)

    with open(result_path, "w") as f:
        json.dump(asdict(result), f)


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------


def _spawn(target, kwargs: dict[str, Any]) -> mp.Process:
    p = mp.Process(target=target, kwargs=kwargs, daemon=True)
    p.start()
    return p


def _load_result(path: str, cls, fallback_id: str | None = None):
    """Load a per-subprocess result file. If the file is missing (because
    the subprocess crashed before writing it), synthesize a placeholder
    so the scenario as a whole still produces a JSON summary."""
    if not os.path.exists(path):
        logger.warning("result file missing: %s (subprocess likely crashed)", path)
        if cls is TrainerResult:
            return TrainerResult(
                worker_id=None,
                mx_source_id=None,
                started_at=0.0,
                published_versions=[],
                compile_target=None,
                total_published_bytes=0,
            )
        rid = fallback_id or os.path.splitext(os.path.basename(path))[0]
        return ReceiverResult(
            receiver_id=rid,
            worker_rank=-1,
            started_at=0.0,
            cycles=[ReceiverCycleResult(version=-1, error="subprocess crashed without writing result")],
        )

    with open(path) as f:
        d = json.load(f)
    if cls is TrainerResult:
        return TrainerResult(**d)
    cycles = [ReceiverCycleResult(**c) for c in d.pop("cycles", [])]
    return ReceiverResult(cycles=cycles, **d)


def run_elastic_scale(args: argparse.Namespace) -> BenchResult:
    """N receivers join staggered. Trainer publishes ``--steps`` versions.

    Each receiver is launched with a delay relative to the previous.
    All receivers try to consume every published version (so cold
    joiners back-fill).
    """
    tmpdir = args.tmpdir
    os.makedirs(tmpdir, exist_ok=True)
    started = time.time()

    trainer_path = os.path.join(tmpdir, "trainer.json")
    trainer_device = args.trainer_device_id
    trainer_proc = _spawn(
        _run_trainer,
        dict(
            role="trainer",
            mx_server_url=args.mx_server_url,
            model_name=args.model_name,
            num_tensors=args.num_tensors,
            tensor_bytes=args.tensor_bytes,
            steps=args.steps,
            step_interval_s=args.step_interval,
            compile_target=args.trainer_compile_target,
            device_id=trainer_device,
            result_path=trainer_path,
        ),
    )
    time.sleep(args.trainer_warmup)

    receiver_procs = []
    target_versions = list(range(1, args.steps + 1))
    for i in range(args.num_receivers):
        rid = f"recv-{i}"
        rpath = os.path.join(tmpdir, f"{rid}.json")
        receiver_device = args.receiver_device_base + i
        receiver_procs.append(
            (
                rpath,
                _spawn(
                    _run_receiver,
                    dict(
                        receiver_id=rid,
                        worker_rank=0,  # same-rank pull
                        mx_server_url=args.mx_server_url,
                        model_name=args.model_name,
                        device_id=receiver_device,
                        listen_port=None,
                        compile_target_filter=None,
                        target_versions=target_versions,
                        poll_interval_s=args.poll_interval,
                        cycle_timeout_s=args.cycle_timeout,
                        deadline_s=args.deadline,
                        publish_self_as_replica=False,
                        result_path=rpath,
                    ),
                ),
            )
        )
        time.sleep(args.join_interval)

    for _, p in receiver_procs:
        p.join(timeout=args.deadline + 30)
    trainer_proc.join(timeout=args.deadline + 60)
    finished = time.time()

    trainer_result = _load_result(trainer_path, TrainerResult)
    receivers = [
        _load_result(rp, ReceiverResult, fallback_id=os.path.splitext(os.path.basename(rp))[0])
        for rp, _ in receiver_procs
    ]
    return BenchResult(
        scenario="elastic_scale",
        config=vars(args),
        trainer=trainer_result,
        receivers=receivers,
        started_at=started,
        finished_at=finished,
    )


def run_compile_target(args: argparse.Namespace) -> BenchResult:
    """Trainer publishes with a fixed compile_target; three receivers
    with different filters demonstrate accept / reject / back-compat."""
    tmpdir = args.tmpdir
    os.makedirs(tmpdir, exist_ok=True)
    started = time.time()

    trainer_path = os.path.join(tmpdir, "trainer.json")
    trainer_proc = _spawn(
        _run_trainer,
        dict(
            role="trainer",
            mx_server_url=args.mx_server_url,
            model_name=args.model_name,
            num_tensors=args.num_tensors,
            tensor_bytes=args.tensor_bytes,
            steps=args.steps,
            step_interval_s=args.step_interval,
            compile_target=args.trainer_compile_target,  # e.g. "cutlass_fp8"
            device_id=args.trainer_device_id,
            result_path=trainer_path,
        ),
    )
    time.sleep(args.trainer_warmup)

    # Three receivers running concurrently — distinct GPUs.
    scenarios = [
        ("recv-match", [args.trainer_compile_target]),
        ("recv-mismatch", ["deep_gemm_fp8"]),
        ("recv-no-filter", None),
    ]
    procs = []
    for i, (rid, filt) in enumerate(scenarios):
        rpath = os.path.join(tmpdir, f"{rid}.json")
        procs.append(
            (
                rpath,
                _spawn(
                    _run_receiver,
                    dict(
                        receiver_id=rid,
                        worker_rank=0,
                        mx_server_url=args.mx_server_url,
                        model_name=args.model_name,
                        device_id=args.receiver_device_base + i,
                        listen_port=None,
                        compile_target_filter=filt,
                        target_versions=[1],  # one cycle is enough to demo
                        poll_interval_s=args.poll_interval,
                        cycle_timeout_s=args.cycle_timeout,
                        deadline_s=args.deadline,
                        publish_self_as_replica=False,
                        result_path=rpath,
                    ),
                ),
            )
        )

    for _, p in procs:
        p.join(timeout=args.deadline + 30)
    trainer_proc.join(timeout=args.deadline + 60)
    finished = time.time()

    return BenchResult(
        scenario="compile_target",
        config=vars(args),
        trainer=_load_result(trainer_path, TrainerResult),
        receivers=[
            _load_result(rp, ReceiverResult, fallback_id=os.path.splitext(os.path.basename(rp))[0])
            for rp, _ in procs
        ],
        started_at=started,
        finished_at=finished,
    )


def run_tree_fanout(args: argparse.Namespace) -> BenchResult:
    """Like elastic_scale but receivers also publish_self_as_replica.

    Newcomers can pull from earlier receivers, so we expect the
    trainer's egress bytes to be << total delivered bytes.
    """
    tmpdir = args.tmpdir
    os.makedirs(tmpdir, exist_ok=True)
    started = time.time()

    trainer_path = os.path.join(tmpdir, "trainer.json")
    trainer_proc = _spawn(
        _run_trainer,
        dict(
            role="trainer",
            mx_server_url=args.mx_server_url,
            model_name=args.model_name,
            num_tensors=args.num_tensors,
            tensor_bytes=args.tensor_bytes,
            steps=args.steps,
            step_interval_s=args.step_interval,
            compile_target=args.trainer_compile_target,
            device_id=args.trainer_device_id,
            result_path=trainer_path,
        ),
    )
    time.sleep(args.trainer_warmup)

    procs = []
    target_versions = list(range(1, args.steps + 1))
    # Give each receiver a unique worker_rank so trainer_egress_bytes() can
    # distinguish trainer pulls (source_worker_rank == 0) from replica pulls
    # (source_worker_rank == receiver i+1). The previous all-rank-0 layout
    # made every pull look like a trainer pull, capping fanout_factor near 1
    # even when fan-out was working.
    for i in range(args.num_receivers):
        rid = f"recv-{i}"
        rpath = os.path.join(tmpdir, f"{rid}.json")
        procs.append(
            (
                rpath,
                _spawn(
                    _run_receiver,
                    dict(
                        receiver_id=rid,
                        worker_rank=i + 1,  # 1..N — 0 is reserved for trainer
                        mx_server_url=args.mx_server_url,
                        model_name=args.model_name,
                        device_id=args.receiver_device_base + i,
                        listen_port=None,
                        compile_target_filter=None,
                        target_versions=target_versions,
                        poll_interval_s=args.poll_interval,
                        cycle_timeout_s=args.cycle_timeout,
                        deadline_s=args.deadline,
                        publish_self_as_replica=True,  # the only diff
                        # Tree fan-out *requires* cross-rank discovery
                        # because newcomers (rank N+1) must be able to
                        # find earlier receivers (rank ≤ N) as sources.
                        same_rank_only=False,
                        result_path=rpath,
                    ),
                ),
            )
        )
        time.sleep(args.join_interval)

    for _, p in procs:
        p.join(timeout=args.deadline + 30)
    trainer_proc.join(timeout=args.deadline + 60)
    finished = time.time()

    return BenchResult(
        scenario="tree_fanout",
        config=vars(args),
        trainer=_load_result(trainer_path, TrainerResult),
        receivers=[
            _load_result(rp, ReceiverResult, fallback_id=os.path.splitext(os.path.basename(rp))[0])
            for rp, _ in procs
        ],
        started_at=started,
        finished_at=finished,
    )


# ----------------------------------------------------------------------------
# CPU-only smoke mode — runs the orchestrator logic against stubs, exercises
# the harness without needing a server or RDMA.
# ----------------------------------------------------------------------------


def run_cpu_smoke(args: argparse.Namespace) -> BenchResult:
    """Drive the harness end-to-end with stubbed trainer/receivers.

    The trainer and receivers run in-process and just simulate the
    metrics they would produce. This lets us validate the orchestrator
    + result aggregation + summary table without a live MX server.
    """
    started = time.time()
    trainer = TrainerResult(
        worker_id="bench-trainer-r0",
        mx_source_id="abcd1234efgh5678",
        started_at=started,
        compile_target=args.trainer_compile_target,
        published_versions=list(range(1, args.steps + 1)),
        total_published_bytes=args.num_tensors * args.tensor_bytes * args.steps,
    )
    receivers = []
    for i in range(args.num_receivers):
        join_delay = i * args.join_interval
        r = ReceiverResult(
            receiver_id=f"recv-{i}",
            worker_rank=0,
            started_at=started + join_delay,
            first_receive_at=started + join_delay + 0.05,
            join_latency_seconds=0.05,
        )
        for v in range(1, args.steps + 1):
            r.cycles.append(
                ReceiverCycleResult(
                    version=v,
                    bytes_received=args.num_tensors * args.tensor_bytes,
                    tensors_received=args.num_tensors,
                    rdma_seconds=0.1,
                    bandwidth_gbps=(args.num_tensors * args.tensor_bytes * 8) / (0.1 * 1e9),
                    discovery_seconds=0.01,
                    source_worker_rank=0,
                )
            )
        receivers.append(r)

    return BenchResult(
        scenario=f"{args.scenario}_cpu_smoke",
        config=vars(args),
        trainer=trainer,
        receivers=receivers,
        started_at=started,
        finished_at=time.time(),
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Elastic-scale + compile-target + tree-fan-out benchmark for MX v2",
    )
    p.add_argument(
        "--scenario",
        choices=["elastic_scale", "compile_target", "tree_fanout"],
        default="elastic_scale",
    )
    p.add_argument(
        "--mode",
        choices=["live", "cpu"],
        default="live",
        help="'live' = real trainer + receivers via subprocesses (needs MX server + NIXL); "
        "'cpu' = stubbed orchestrator-only smoke",
    )
    p.add_argument("--mx-server-url", default=os.environ.get("MX_SERVER_URL", "localhost:8001"))
    p.add_argument("--model-name", default="bench/synthetic-1.5B")
    p.add_argument("--num-receivers", type=int, default=3)
    p.add_argument("--num-tensors", type=int, default=8)
    p.add_argument("--tensor-bytes", type=int, default=8 * 1024 * 1024,
                   help="Bytes per tensor on the publisher side (default 8 MiB).")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--step-interval", type=float, default=2.0)
    p.add_argument("--join-interval", type=float, default=2.0,
                   help="Seconds between successive receiver starts.")
    p.add_argument("--trainer-warmup", type=float, default=2.0,
                   help="Seconds to wait after starting the trainer before launching receivers.")
    p.add_argument("--poll-interval", type=float, default=0.5)
    p.add_argument("--cycle-timeout", type=float, default=60.0)
    p.add_argument("--deadline", type=float, default=180.0)
    p.add_argument("--trainer-compile-target", default="cutlass_fp8")
    p.add_argument("--trainer-device-id", type=int, default=0,
                   help="CUDA device for the trainer subprocess.")
    p.add_argument("--receiver-device-base", type=int, default=1,
                   help="CUDA device for receiver-0; receiver-i gets device_base+i. "
                        "With 5 GPUs visible and default trainer=0, receivers use 1..4.")
    p.add_argument("--tmpdir", default="/tmp/mx_bench")
    p.add_argument("--output", default=None, help="If set, also write JSON results to this path.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "cpu":
        result = run_cpu_smoke(args)
    else:
        dispatch = {
            "elastic_scale": run_elastic_scale,
            "compile_target": run_compile_target,
            "tree_fanout": run_tree_fanout,
        }
        result = dispatch[args.scenario](args)

    print(result.to_summary_table())
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_json(), f, indent=2, default=str)
        print(f"\nFull JSON written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
