#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sharded trainer publisher for the reshard-refit E2E (FSDP8 x EP8).

One process per trainer rank (1 GPU). Emulates an FSDP-8 + EP-8 trainer source:

  * expert tensors (``...experts.{E}....``) are EP-sharded: rank r owns experts
    ``[r*E/8 : (r+1)*E/8]`` (whole-tensor shard on the owner);
  * dense tensors are FSDP-row-sharded along dim0 across the 8 ranks when
    divisible; small/indivisible tensors (norms, 1-D) stay whole on rank 0.

Each rank registers only its owned sub-tensors with NIXL and publishes its shard
table; the TP-k receiver fans in the needed slices across all 8 ranks (the real
no-gather cross-rank reshard). Holds buffers until a stop-file appears.

Usage (per rank, 1 GPU): one invocation per local GPU, rank = node_rank*4 + local.
  MX_POOL_REG=1 python3 reshard_publisher_sharded.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --rank 0 --world-size 8 --device 0 \
    --mx-server modelexpress-server.kavin.svc.cluster.local:8001 \
    --listen-port 7200 --ready-file .../pub.r0.ready --stop-file .../pub.stop
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import socket
import time

_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


def _pack_into_arena(
    tensors: dict[str, "torch.Tensor"], *, alignment: int = 256
) -> tuple[dict[str, "torch.Tensor"], "torch.Tensor"]:
    """Copy owned shards into one aligned CUDA arena.

    NIXL can then register one allocation per trainer rank instead of thousands
    of independent cudaMalloc allocations. Tensor views retain their native
    dtype/shape and published addresses while sharing the arena's registration.
    """
    import torch

    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, tensor in tensors.items():
        cursor = (cursor + alignment - 1) // alignment * alignment
        nbytes = tensor.numel() * tensor.element_size()
        offsets[name] = (cursor, nbytes)
        cursor += nbytes
    arena = torch.empty(cursor, dtype=torch.uint8, device=next(iter(tensors.values())).device)
    packed = {}
    for name, tensor in tensors.items():
        offset, nbytes = offsets[name]
        destination = arena.narrow(0, offset, nbytes).view(tensor.dtype).view(tensor.shape)
        destination.copy_(tensor)
        packed[name] = destination
    return packed, arena


def _snapshot_dir(model_id: str) -> str:
    if os.path.isdir(model_id):
        return model_id if model_id.endswith("/") else model_id + "/"
    hub = os.path.join(
        os.environ["HF_HOME"], "hub",
        "models--" + model_id.replace("/", "--"), "snapshots",
    )
    return sorted(glob.glob(hub + "/*/"))[-1]


def _owned_shards(model_id, rank, world, device, wire_dtype):
    """Yield (name, gpu_tensor, full_shape, shard_offset) for this rank."""
    import torch
    from safetensors.torch import safe_open

    snap = _snapshot_dir(model_id)
    # discover expert count for EP sharding
    max_eid = -1
    files = sorted(glob.glob(snap + "*.safetensors"))
    for sh in files:
        with safe_open(sh, framework="pt", device="cpu") as f:
            for k in f.keys():
                m = _EXPERT_RE.search(k)
                if m:
                    max_eid = max(max_eid, int(m.group(1)))
    num_experts = max_eid + 1 if max_eid >= 0 else 0
    experts_per_rank = (num_experts // world) if num_experts else 0

    for sh in files:
        with safe_open(sh, framework="pt", device="cpu") as f:
            for k in f.keys():
                m = _EXPERT_RE.search(k)
                sl = f.get_slice(k)
                full_shape = tuple(sl.get_shape())
                if m:  # expert tensor -> EP shard (whole tensor on owner rank)
                    eid = int(m.group(1))
                    owner = eid // experts_per_rank if experts_per_rank else 0
                    if owner != rank:
                        continue
                    t = f.get_tensor(k)
                    if wire_dtype == "bf16" and t.dtype != torch.bfloat16:
                        t = t.to(torch.bfloat16)
                    yield k, t.to(device), full_shape, (0,) * t.ndim
                else:  # dense -> FSDP row shard along dim0 if divisible
                    d0 = full_shape[0]
                    if len(full_shape) >= 1 and d0 % world == 0:
                        step = d0 // world
                        lo = rank * step
                        t = sl[lo:lo + step]
                        if wire_dtype == "bf16" and t.dtype != torch.bfloat16:
                            t = t.to(torch.bfloat16)
                        off = (lo,) + (0,) * (len(full_shape) - 1)
                        yield k, t.contiguous().to(device), full_shape, off
                    else:  # indivisible/small -> whole on rank 0
                        if rank != 0:
                            continue
                        t = f.get_tensor(k)
                        if wire_dtype == "bf16" and t.dtype != torch.bfloat16:
                            t = t.to(torch.bfloat16)
                        yield k, t.to(device), full_shape, (0,) * t.ndim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--rendezvous-name",
        required=True,
        help="unique shared identity used by all publisher and receiver ranks",
    )
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--wire-dtype",
        choices=["bf16", "native"],
        default="bf16",
        help="bf16 load-time wire representation or native checkpoint dtype",
    )
    ap.add_argument("--mx-server", required=True)
    ap.add_argument("--listen-port", type=int, default=7200)
    ap.add_argument("--ready-file", required=True)
    ap.add_argument("--stop-file", required=True)
    args = ap.parse_args()

    import torch

    from modelexpress.client import MxClient
    from modelexpress.nixl_transfer import NixlTransferManager
    from modelexpress.reshard_refit.rendezvous import (
        MxReshardRendezvous,
        PublishedShard,
        PublishedTensor,
        wrap_rendezvous_blob,
    )

    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    print(f"[pub r{args.rank}] loading owned shards", flush=True)
    owned = {}
    meta = {}
    total = 0
    for name, t, full_shape, off in _owned_shards(
        args.model, args.rank, args.world_size, dev, args.wire_dtype
    ):
        owned[name] = t
        meta[name] = (full_shape, off, tuple(t.shape))
        total += t.numel() * t.element_size()
    owned, arena = _pack_into_arena(owned)
    print(f"[pub r{args.rank}] {len(owned)} shards, {total/1e9:.2f} GB on {dev}", flush=True)
    print(
        f"[pub r{args.rank}] packed into one {arena.numel()/1e9:.2f} GB CUDA arena",
        flush=True,
    )

    agent_name = f"trainer-{args.rank}"
    mgr = NixlTransferManager(
        agent_name=agent_name, device_id=args.device, listen_port=args.listen_port
    )
    mgr.initialize()
    agent_meta = mgr.register_tensors(owned)
    print(f"[pub r{args.rank}] registered with NIXL agent {agent_name}", flush=True)

    published = []
    for name, t in owned.items():
        full_shape, off, shp = meta[name]
        published.append(
            PublishedTensor(
                name=name,
                dtype=str(t.dtype),
                elsize=t.element_size(),
                full_shape=full_shape,
                shards=[
                    PublishedShard(
                        agent_name=agent_name,
                        device_id=args.device,
                        addr=t.data_ptr(),
                        shard_offset=off,
                        shape=shp,
                    )
                ],
            )
        )

    pod_ip = os.environ.get("POD_IP") or socket.gethostbyname(socket.gethostname())
    endpoint = f"{pod_ip}:{args.listen_port}"
    blob = wrap_rendezvous_blob(agent_meta, agent_name, endpoint, published)
    client = MxClient(server_url=args.mx_server)
    rdv = MxReshardRendezvous(
        client, role="trainer", rank=args.rank, model_name=args.rendezvous_name
    )
    src_id = rdv.publish(blob)
    print(f"[pub r{args.rank}] published: source_id={src_id} endpoint={endpoint} tensors={len(published)}", flush=True)

    os.makedirs(os.path.dirname(args.ready_file), exist_ok=True)
    with open(args.ready_file, "w") as f:
        f.write(f"{endpoint}\n{src_id}\n")
    print(f"[pub r{args.rank}] READY", flush=True)
    while not os.path.exists(args.stop_file):
        time.sleep(2.0)
    print(f"[pub r{args.rank}] stop-file seen; exiting", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
