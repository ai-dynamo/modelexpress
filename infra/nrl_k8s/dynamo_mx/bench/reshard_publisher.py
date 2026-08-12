#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Trainer-side publisher for the reshard-refit E2E benchmark.

Holds the model weights as NIXL-registered GPU buffers and publishes a shard
table (one full-tensor shard per source) under a ``role="trainer"`` rendezvous
identity, so a ``VllmReshardReceiver`` can discover, P2P-handshake, and RDMA-pull
its per-rank slices. Single trainer rank (TP1 source) -> TP-k inference reshard.

Keeps the process alive (buffers registered + NIXL listen thread up) until a
sentinel file appears, so the receiver can run many refit cycles against stable
addresses.

Usage (trainer pod, 1 GPU):
  MX_POOL_REG=1 python3 reshard_publisher.py \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --mx-server modelexpress-server.kavin.svc.cluster.local:8001 \
      --listen-port 7200 --ready-file /mnt/rl-workspace/kavink/reshard_e2e/pub.ready \
      --stop-file /mnt/rl-workspace/kavink/reshard_e2e/pub.stop
"""
from __future__ import annotations

import argparse
import glob
import os
import socket
import time


def _read_hf_weights(model_id: str):
    import torch
    from safetensors.torch import safe_open

    if os.path.isdir(model_id):
        snap = model_id
    else:
        hub = os.path.join(
            os.environ["HF_HOME"], "hub",
            "models--" + model_id.replace("/", "--"), "snapshots",
        )
        snap = sorted(glob.glob(hub + "/*/"))[-1]
    if not snap.endswith("/"):
        snap += "/"
    out = []
    for sh in sorted(glob.glob(snap + "*.safetensors")):
        with safe_open(sh, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if t.dtype != torch.bfloat16:
                    t = t.to(torch.bfloat16)
                out.append((k, t))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mx-server", required=True)
    ap.add_argument("--listen-port", type=int, default=7200)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
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

    dev = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    print(f"[pub] loading weights for {args.model}", flush=True)
    weights = _read_hf_weights(args.model)
    gpu = {}
    total = 0
    for name, t in weights:
        g = t.to(dev)
        gpu[name] = g
        total += g.numel() * g.element_size()
    print(f"[pub] {len(gpu)} tensors, {total/1e9:.1f} GB on {dev}", flush=True)

    agent_name = f"trainer-{args.rank}"
    mgr = NixlTransferManager(
        agent_name=agent_name, device_id=args.device, listen_port=args.listen_port
    )
    mgr.initialize()
    agent_meta = mgr.register_tensors(gpu)
    print(f"[pub] registered {len(gpu)} tensors with NIXL agent {agent_name}", flush=True)

    published = [
        PublishedTensor(
            name=name,
            dtype="torch.bfloat16",
            elsize=2,
            full_shape=tuple(t.shape),
            shards=[
                PublishedShard(
                    agent_name=agent_name,
                    device_id=args.device,
                    addr=t.data_ptr(),
                    shard_offset=(0,) * t.ndim,
                    shape=tuple(t.shape),
                )
            ],
        )
        for name, t in gpu.items()
    ]

    pod_ip = os.environ.get("POD_IP") or socket.gethostbyname(socket.gethostname())
    endpoint = f"{pod_ip}:{args.listen_port}"
    blob = wrap_rendezvous_blob(agent_meta, agent_name, endpoint, published)
    client = MxClient(server_url=args.mx_server)
    rdv = MxReshardRendezvous(
        client, role="trainer", rank=args.rank, model_name=args.model
    )
    src_id = rdv.publish(blob)
    print(f"[pub] published shard table: source_id={src_id} endpoint={endpoint}", flush=True)

    os.makedirs(os.path.dirname(args.ready_file), exist_ok=True)
    with open(args.ready_file, "w") as f:
        f.write(f"{endpoint}\n{src_id}\n")
    print("[pub] READY; holding buffers until stop-file appears", flush=True)
    while not os.path.exists(args.stop_file):
        time.sleep(2.0)
    print("[pub] stop-file seen; exiting", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
