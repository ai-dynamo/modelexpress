#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end NIXL+MX v2 demo for MoE-style weight refit.

Spawns 4 processes via torch.multiprocessing, one per CUDA device.
Each process plays both 'trainer-rank-N' (publishes a fake MoE weight
shard for that rank's experts) and 'inference-rank-N' (discovers + pulls
its same-rank trainer's shard via NIXL RDMA).

What this exercises end-to-end:
  - MxV2TrainingPublisher: NIXL register, publish_metadata with v2 markers
    (mx_v2=1, role=trainer, worker_rank=N, training_step=K), shape_registry
    JSON, expert ownership IDs, agent_name fallback for old-server compat.
  - MxV2RefitReceiver: discover_v2_sources with same_rank_only filter,
    freshest-per-rank dedup, pick_best_source with MoE expert coverage,
    receive_from (real RDMA WRITE).
  - Heartbeat: HeartbeatThread keeps source alive on MX server.
  - Two refit cycles to demonstrate version progression.

Run inside the trainer pod where the MX server is reachable as
'modelexpress-server.kavin.svc.cluster.local:8001' and NIXL is configured.

Expected output (key lines):
  [trainer R0] published v=0 mx_source_id=...
  [inference R0] picked source role=trainer src_rank=0 v=0
  [inference R0] received tensor 'experts.0.w1' shape=...
  [trainer R0] published v=1 mx_source_id=...
  [inference R0] picked freshest v=1
"""
from __future__ import annotations

import os
import sys
import time
import logging
import torch
import torch.multiprocessing as mp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("v2-demo")

MX_URL = os.environ.get("MX_URL", "modelexpress-server.kavin.svc.cluster.local:8001")
MODEL_NAME = os.environ.get("MODEL_NAME", "v2-demo/MoE-fake")
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "4"))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", "8"))   # 8 experts per "layer"
HIDDEN = int(os.environ.get("HIDDEN", "256"))
N_REFIT_CYCLES = int(os.environ.get("N_REFIT_CYCLES", "2"))

# FORCE_RDMA=1 disables UCX's intra-node `cuda_ipc` fast path so the demo
# exercises the same `rc_mlx5` (or `cuda_copy` over RDMA NIC) descriptor-list
# validation path that real cross-node transfers do. Without this, intra-node
# loopback runs through `cuda_ipc` which silently tolerates malformed
# descriptor entries — e.g. the v2 `__mx_v2_meta__` sidecar (addr=0, size=0)
# bug that MX PR #295 (commit 53c69ec) fixed. Set FORCE_RDMA=1 on every
# pre-deploy run so cross-host descriptor-list bugs surface in loopback.
if os.environ.get("FORCE_RDMA") == "1":
    # rc_mlx5 = RDMA over RoCE/IB; cuda_copy = staged GPU↔NIC via host bounce.
    # Both run UCX's strict prep_xfer_dlist validation. self+sm kept so the
    # ZMQ/gRPC control plane still works.
    os.environ["UCX_TLS"] = os.environ.get("UCX_TLS", "self,sm,rc_mlx5,cuda_copy,tcp")
    log.info(
        "FORCE_RDMA=1: UCX_TLS=%s (cuda_ipc disabled to exercise descriptor validation)",
        os.environ["UCX_TLS"],
    )


def trainer_publish(rank: int, version: int, layout, mx_url: str):
    """Run as the trainer side: publish a moe-flavored shard for our rank."""
    from modelexpress import MxV2TrainingPublisher

    # MoE expert layout: each rank owns NUM_EXPERTS / WORLD_SIZE experts.
    chunk = NUM_EXPERTS // WORLD_SIZE
    owned = set(range(rank * chunk, (rank + 1) * chunk))
    log.info(f"[trainer R{rank}] starts; owns experts {sorted(owned)}; v={version}")

    pub = MxV2TrainingPublisher(
        agent_name=f"v2demo-trainer-r{rank}-v{version}",
        device_id=rank,
        mx_server_url=mx_url,
        worker_rank=rank,
        world_layout=layout,
        heartbeat=False,  # short-lived demo; skip heartbeat
    )
    pub.initialize(model_name=MODEL_NAME, dtype="bfloat16")

    # Fake MoE expert tensor: leading axis is the expert dim (= owned chunk).
    # Each rank's local shard holds (chunk, HIDDEN, HIDDEN).
    # Use exact-in-bfloat16 sentinel values: multiples of 8 are exact for
    # magnitudes up to 2^15. Encode sentinel as ((rank+1) * 8 + version * 64)
    # so distinct (rank, version) pairs always have distinct values.
    sentinel = (rank + 1) * 8 + version * 64
    with torch.cuda.device(rank):
        moe_w = torch.randn(chunk, HIDDEN, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{rank}")
        moe_w[0, 0, 0] = float(sentinel)
        # Plus a non-expert tensor (replicated across ranks).
        ln_w = torch.ones(HIDDEN, dtype=torch.bfloat16, device=f"cuda:{rank}")
        ln_w[0] = float(sentinel)

    pub.add_tensor(
        name="model.layers.0.experts.weight",
        tensor=moe_w,
        is_expert=True,
        expert_axis=0,
        owned_expert_ids=owned,
    )
    pub.add_tensor(name="model.layers.0.layer_norm.weight", tensor=ln_w)

    mx_source_id = pub.publish(version=version)
    pub.mark_ready()
    log.info(
        f"[trainer R{rank}] published v={version} mx_source_id={mx_source_id} "
        f"sentinel_target={sentinel} got={moe_w[0, 0, 0].item():.0f}"
    )
    # Hand back the publisher so the worker can call shutdown after its peer reads.
    return pub, moe_w, ln_w


def inference_receive(rank: int, version: int, mx_url: str):
    """Run as the inference side: discover + pull our same-rank trainer."""
    from modelexpress import MxV2RefitReceiver

    chunk = NUM_EXPERTS // WORLD_SIZE
    log.info(f"[inference R{rank}] starts; expects experts of size {chunk}; v={version}")

    rec = MxV2RefitReceiver(
        agent_name=f"v2demo-inference-r{rank}-v{version}",
        device_id=rank,
        mx_server_url=mx_url,
        worker_rank=rank,
    )

    # Pre-allocate receive buffers matching the trainer's shape.
    with torch.cuda.device(rank):
        recv_moe = torch.zeros(
            chunk, HIDDEN, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{rank}"
        )
        recv_ln = torch.zeros(HIDDEN, dtype=torch.bfloat16, device=f"cuda:{rank}")
    rec.initialize(
        model_tensors={
            "model.layers.0.experts.weight": recv_moe,
            "model.layers.0.layer_norm.weight": recv_ln,
        }
    )

    # Discover same-rank source, with v2-only filter. Poll for up to 30 s
    # to handle propagation delays.
    deadline = time.perf_counter() + 30.0
    candidates = []
    while time.perf_counter() < deadline:
        candidates = rec.discover_v2_sources(
            model_name=MODEL_NAME,
            min_version=version,
            same_rank_only=True,
            include_replicas=True,
        )
        if candidates:
            break
        time.sleep(0.5)

    if not candidates:
        log.error(f"[inference R{rank}] no v2 source found (timeout)")
        return False

    chosen = rec.pick_best_source(candidates)
    log.info(
        f"[inference R{rank}] picked source role={chosen.role} "
        f"src_rank={chosen.worker_rank} v={chosen.ref.training_step} "
        f"updated_at={chosen.updated_at}"
    )

    bytes_received = 0
    t0 = time.perf_counter()
    for name, tensor in rec.receive_from(chosen, timeout_seconds=60.0):
        bytes_received += tensor.numel() * tensor.element_size()
        log.info(
            f"[inference R{rank}] received '{name}' shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype}"
        )
    elapsed = time.perf_counter() - t0
    bw_mbps = bytes_received / 1e6 / elapsed if elapsed > 0 else 0.0

    # Verify fingerprints match.
    expected_sentinel = (rank + 1) * 8 + version * 64
    actual_moe = recv_moe[0, 0, 0].item()
    actual_ln = recv_ln[0].item()
    log.info(
        f"[inference R{rank}] {bytes_received/1e6:.2f} MB in {elapsed*1000:.0f} ms "
        f"({bw_mbps:.0f} MB/s); moe[0,0,0]={actual_moe:.0f} ln[0]={actual_ln:.0f} "
        f"expected={expected_sentinel}"
    )
    ok = (
        abs(actual_moe - expected_sentinel) < 0.5
        and abs(actual_ln - expected_sentinel) < 0.5
    )
    log.info(f"[inference R{rank}] correctness: {'OK' if ok else 'FAIL'}")

    # Tree fan-out: republish self as inference_replica
    rec.publish_self_as_source(version=version, model_name=MODEL_NAME)

    return ok


def per_rank_main(rank: int, return_dict):
    """Entry point for each spawned process — plays both trainer & inference."""
    from modelexpress import TrainerWorldLayout

    layout = TrainerWorldLayout(fsdp_world_size=WORLD_SIZE, ep_world_size=WORLD_SIZE)
    publishers = []
    all_ok = True

    for cycle in range(N_REFIT_CYCLES):
        version = cycle  # 0, 1, ...
        log.info(f"=== R{rank} cycle {cycle} (version={version}) ===")

        pub, _moe, _ln = trainer_publish(rank, version, layout, MX_URL)
        publishers.append(pub)

        # Tiny barrier via wallclock — give all trainers ~2s to publish so
        # discover_v2_sources sees a coherent set.
        time.sleep(2.0)

        ok = inference_receive(rank, version, MX_URL)
        all_ok = all_ok and ok

        # Inter-cycle gap so version-N is observably newer than version-(N-1)
        time.sleep(2.0)

    # Drop publishers (releases NIXL agents)
    for p in publishers:
        try:
            p.shutdown()
        except Exception as e:
            log.warning(f"R{rank} shutdown: {e}")

    return_dict[rank] = all_ok
    log.info(f"=== R{rank} done; all_ok={all_ok} ===")


def main():
    log.info(f"=== v2 MoE E2E demo: WORLD_SIZE={WORLD_SIZE} NUM_EXPERTS={NUM_EXPERTS} ===")
    log.info(f"MX_URL={MX_URL} MODEL_NAME={MODEL_NAME} N_REFIT_CYCLES={N_REFIT_CYCLES}")

    if torch.cuda.device_count() < WORLD_SIZE:
        log.error(
            f"need {WORLD_SIZE} GPUs, only have {torch.cuda.device_count()}; aborting"
        )
        sys.exit(2)

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    procs = []
    for rank in range(WORLD_SIZE):
        p = mp.Process(target=per_rank_main, args=(rank, return_dict))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    log.info(f"=== summary: {dict(return_dict)} ===")
    if all(return_dict.values()) and len(return_dict) == WORLD_SIZE:
        log.info("=== ALL RANKS OK ===")
        sys.exit(0)
    else:
        log.error("=== SOME RANKS FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
