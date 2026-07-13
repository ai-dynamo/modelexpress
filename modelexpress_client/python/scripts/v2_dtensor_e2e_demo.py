#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end NIXL+MX v2 demo with REAL DTensors.

Sister to v2_moe_e2e_demo.py, but exercises the codepath that the NemoRL
DTensorPolicyWorker.stream_weights_via_mx uses in production: real DTensors
on a torch.distributed mesh, ``tensor.to_local()`` on the publisher,
``MxV2TrainingPublisher.add_tensor`` overriding ``global_shape`` from the
DTensor view, registry round-trip via the synthetic ``__mx_v2_meta__``
sidecar, and same-rank RDMA pulls.

What this validates that v2_moe_e2e_demo.py does NOT:
  * ``shape_descriptors.describe_tensor`` works on a real DTensor (not a
    fake stand-in object). Shard axis, local shard range, and the global
    shape inferred from ``tensor.shape × fsdp_world_size`` line up with
    the DTensor's actual placement.
  * The publisher's per-tensor ``global_shape`` override (set after
    ``add_tensor``) survives the JSON round-trip and is observable by the
    receiver via ``decode_registry``.
  * ``tensor.to_local()`` is in fact what gets NIXL-registered (no allgather
    happens). We assert the publisher's NIXL-registered tensor has the
    SHARD dim equal to ``global_dim / fsdp_world_size``.

Run inside any pod that has GPUs, NIXL, reachability to MX server, and
torch.distributed available.

  WORLD_SIZE=4 N_REFIT_CYCLES=2 python3 v2_dtensor_e2e_demo.py
"""
from __future__ import annotations

import logging
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Shard

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("v2-dt-demo")

MX_URL = os.environ.get("MX_URL", "modelexpress-server.kavin.svc.cluster.local:8001")
MODEL_NAME = os.environ.get("MODEL_NAME", "v2-dtensor-demo/Qwen3MoE-stub")
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "4"))
N_REFIT_CYCLES = int(os.environ.get("N_REFIT_CYCLES", "2"))

# Tensor sizes: produce a recognizably-sharded weight per rank.
HIDDEN = int(os.environ.get("HIDDEN", "1024"))
INTER = int(os.environ.get("INTER", "2048"))

# FORCE_RDMA=1 disables UCX's intra-node `cuda_ipc` fast path so the demo
# exercises the same `rc_mlx5` (or `cuda_copy` over RDMA NIC) descriptor-list
# validation path that real cross-node transfers do. Without this, intra-node
# loopback runs through `cuda_ipc` which silently tolerates malformed
# descriptor entries — e.g. the v2 `__mx_v2_meta__` sidecar (addr=0, size=0)
# bug that MX PR #295 (commit 53c69ec) fixed. Set FORCE_RDMA=1 on every
# pre-deploy run so cross-host descriptor-list bugs surface in loopback.
if os.environ.get("FORCE_RDMA") == "1":
    os.environ["UCX_TLS"] = os.environ.get("UCX_TLS", "self,sm,rc_mlx5,cuda_copy,tcp")
    log.info(
        "FORCE_RDMA=1: UCX_TLS=%s (cuda_ipc disabled to exercise descriptor validation)",
        os.environ["UCX_TLS"],
    )


def _setup_dist(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )
    torch.cuda.set_device(rank)


def _fingerprint(rank: int, version: int) -> int:
    """bf16-exact sentinel encoding (multiples of 8 are exact for |v| < 2**14)."""
    return (rank + 1) * 8 + version * 64


def trainer_publish_dt(rank: int, version: int, layout, mx_url: str, mesh):
    """Publish a sharded DTensor via the v2 publisher. Returns the publisher
    so the caller can keep its NIXL agent alive while inference pulls.
    """
    from modelexpress import MxV2TrainingPublisher

    log.info(f"[trainer R{rank}] publishing v={version}")

    pub = MxV2TrainingPublisher(
        agent_name=f"v2dt-trainer-r{rank}-v{version}",
        device_id=rank,
        mx_server_url=mx_url,
        worker_rank=rank,
        world_layout=layout,
        heartbeat=False,
    )
    pub.initialize(model_name=MODEL_NAME, dtype="bfloat16")

    sentinel = _fingerprint(rank, version)

    # Build a placeholder global tensor and distribute via FSDP-style Shard(0).
    # We deliberately seed the local view AFTER distribute_tensor so each rank
    # owns a recognizably-different sentinel (distribute_tensor only respects
    # rank-0's source, so seeding before would give every rank the same data).
    with torch.cuda.device(rank):
        global_placeholder = torch.zeros(
            (HIDDEN, INTER), dtype=torch.bfloat16, device=f"cuda:{rank}"
        )
    sharded_dt: DTensor = distribute_tensor(global_placeholder, mesh, [Shard(0)])

    local = sharded_dt.to_local()
    assert local.shape[0] == HIDDEN // WORLD_SIZE, (local.shape, HIDDEN, WORLD_SIZE)
    assert sharded_dt.shape[0] == HIDDEN, (sharded_dt.shape, HIDDEN)

    # Seed this rank's local shard with its own sentinel.
    local.fill_(float(sentinel))
    # bf16 round-trip self-check: multiples of 8 are exact.
    assert abs(local[0, 0].item() - sentinel) < 0.5, (local[0, 0].item(), sentinel)

    log.info(
        f"[trainer R{rank}] DTensor: global={tuple(sharded_dt.shape)} "
        f"local={tuple(local.shape)} sentinel={sentinel}"
    )

    # Add as a DTensor — describe_tensor reads .placements off the DTensor
    # and now correctly handles `tensor.shape == global` semantics.
    # global_shape, shard_axis, and local_shard_range are all computed
    # from the DTensor view; no manual override needed.
    pub.add_tensor(name="model.layers.0.qkv_proj.weight", tensor=sharded_dt)

    # NIXL-register the LOCAL shard, not the DTensor (which has no
    # data_ptr()). This mirrors what the NemoRL DTensorPolicyWorker does
    # after the tensor.to_local() call.
    pub._registered_tensors["model.layers.0.qkv_proj.weight"] = local.contiguous()

    mx_source_id = pub.publish(version=version)
    pub.mark_ready()
    log.info(
        f"[trainer R{rank}] published v={version} mx_source_id={mx_source_id}"
    )
    return pub, local


def inference_receive_dt(rank: int, version: int, mx_url: str) -> bool:
    """Pull our same-rank trainer's local shard, verify byte correctness AND
    that the registry exposes the GLOBAL shape (un-sharded)."""
    from modelexpress import MxV2RefitReceiver

    log.info(f"[inference R{rank}] starts; v>={version}")
    rec = MxV2RefitReceiver(
        agent_name=f"v2dt-inference-r{rank}-v{version}",
        device_id=rank,
        mx_server_url=mx_url,
        worker_rank=rank,
    )
    with torch.cuda.device(rank):
        recv_local = torch.zeros(HIDDEN // WORLD_SIZE, INTER,
                                  dtype=torch.bfloat16, device=f"cuda:{rank}")
    rec.initialize(model_tensors={"model.layers.0.qkv_proj.weight": recv_local})

    deadline = time.perf_counter() + 30.0
    candidates = []
    while time.perf_counter() < deadline:
        candidates = rec.discover_v2_sources(
            model_name=MODEL_NAME,
            min_version=version,
            same_rank_only=True,
            include_replicas=False,
        )
        if candidates:
            break
        time.sleep(0.5)

    if not candidates:
        log.error(f"[inference R{rank}] no v2 source found")
        return False

    chosen = rec.pick_best_source(candidates)
    log.info(
        f"[inference R{rank}] picked role={chosen.role} src_rank={chosen.worker_rank} "
        f"v={chosen.ref.training_step}"
    )

    # KEY ASSERTION: the registry the trainer published should expose the
    # GLOBAL shape, not the local shape. (The local shape is what NIXL
    # actually transferred; the global shape is the un-sharded view.)
    if chosen.registry is not None:
        for td in chosen.registry["tensors"]:
            if td.name == "model.layers.0.qkv_proj.weight":
                log.info(
                    f"[inference R{rank}] registry: global={td.global_shape} "
                    f"placement={td.placement_kind} shard_axis={td.shard_axis} "
                    f"local_range={td.local_shard_range}"
                )
                assert td.global_shape == (HIDDEN, INTER), (td.global_shape, HIDDEN, INTER)
                assert td.placement_kind == "SHARD"
                assert td.shard_axis == 0
                expected_lo = rank * (HIDDEN // WORLD_SIZE)
                expected_hi = expected_lo + (HIDDEN // WORLD_SIZE)
                assert td.local_shard_range == (expected_lo, expected_hi), (
                    td.local_shard_range, expected_lo, expected_hi
                )
                break
        else:
            log.warning(f"[inference R{rank}] tensor not in registry (sidecar may be missing)")
    else:
        log.warning(f"[inference R{rank}] registry missing on candidate (sidecar transport drop?)")

    bytes_received = 0
    t0 = time.perf_counter()
    for name, tensor in rec.receive_from(chosen, timeout_seconds=60.0):
        bytes_received += tensor.numel() * tensor.element_size()
        log.info(f"[inference R{rank}] received '{name}' shape={tuple(tensor.shape)}")
    elapsed = time.perf_counter() - t0
    bw_mbps = bytes_received / 1e6 / elapsed if elapsed > 0 else 0.0

    expected_value = _fingerprint(rank, version)
    actual_value = recv_local[0, 0].item()
    # Stronger check: every cell of the received local shard should equal sentinel
    # (the trainer filled the whole local view).
    elem_match = bool(torch.allclose(
        recv_local.float(), torch.full_like(recv_local, float(expected_value)).float(), atol=0.5
    ))
    log.info(
        f"[inference R{rank}] {bytes_received/1e6:.2f} MB in {elapsed*1000:.0f} ms "
        f"({bw_mbps:.0f} MB/s); local[0,0]={actual_value:.0f} expected={expected_value} "
        f"all_elem_match={elem_match}"
    )

    ok = abs(actual_value - expected_value) < 0.5 and elem_match
    log.info(f"[inference R{rank}] correctness: {'OK' if ok else 'FAIL'}")
    return ok


def per_rank_main(rank: int, return_dict):
    from modelexpress import TrainerWorldLayout

    _setup_dist(rank, WORLD_SIZE)
    mesh = init_device_mesh("cuda", (WORLD_SIZE,))
    layout = TrainerWorldLayout(fsdp_world_size=WORLD_SIZE)

    publishers = []
    all_ok = True

    for cycle in range(N_REFIT_CYCLES):
        version = cycle
        log.info(f"=== R{rank} cycle {cycle} (version={version}) ===")
        pub, _local = trainer_publish_dt(rank, version, layout, MX_URL, mesh)
        publishers.append(pub)

        dist.barrier()  # ensure all trainers published before inference polls
        time.sleep(1.0)

        ok = inference_receive_dt(rank, version, MX_URL)
        all_ok = all_ok and ok

        dist.barrier()
        time.sleep(1.0)

    for p in publishers:
        try:
            p.shutdown()
        except Exception as e:
            log.warning(f"R{rank} shutdown: {e}")

    return_dict[rank] = all_ok
    log.info(f"=== R{rank} done; all_ok={all_ok} ===")

    dist.destroy_process_group()


def main():
    log.info(f"=== v2 DTensor E2E: WORLD_SIZE={WORLD_SIZE} HIDDEN={HIDDEN} INTER={INTER} ===")
    log.info(f"MX_URL={MX_URL} MODEL_NAME={MODEL_NAME} N_REFIT_CYCLES={N_REFIT_CYCLES}")

    if torch.cuda.device_count() < WORLD_SIZE:
        log.error(f"need {WORLD_SIZE} GPUs, got {torch.cuda.device_count()}")
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
