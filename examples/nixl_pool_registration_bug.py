#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reproduction for NIXL pool registration bug.

Tests pool-based registration (registering contiguous memory pools)
with sub-region transfers (transferring individual tensor ranges within
pools). This matches the ModelExpress use case where we detect contiguous
GPU memory regions and register each pool as a single NIXL entry, then
transfer per-tensor ranges within those pools.

Usage (cross-node):
    # Node A: python repro.py --role source --mode tensor
    # Node B: python repro.py --role target --peer-ip <node_a_ip> --mode tensor

Modes:
    tensor: register each tensor individually (baseline, known to work)
    pool-tensor: detect pools, register each pool as a torch view (workaround)
    pool-tuple: detect pools, register as raw tuples
    pool-numpy: detect pools, register as numpy Nx3 array
"""

import argparse
import json
import socket
import struct
import time

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config

NIXL_PORT = 5580
COORD_PORT = 9998
NUM_TENSORS = 200
TENSOR_SIZE = 1024 * 1024  # 1MB each, ~200MB total


def make_tensors(device):
    """Create a list of contiguous tensors simulating model weights."""
    # Allocate one big block then split into views to guarantee contiguity
    big = torch.randn(NUM_TENSORS * TENSOR_SIZE // 4, dtype=torch.float32, device=f"cuda:{device}")
    tensors = []
    for i in range(NUM_TENSORS):
        start = i * (TENSOR_SIZE // 4)
        end = start + (TENSOR_SIZE // 4)
        tensors.append(big[start:end].clone())  # clone to get separate storage
    # Also keep some that are actually contiguous (from the same allocation)
    contiguous_block = torch.randn(50 * TENSOR_SIZE // 4, dtype=torch.float32, device=f"cuda:{device}")
    for i in range(50):
        start = i * (TENSOR_SIZE // 4)
        end = start + (TENSOR_SIZE // 4)
        t = contiguous_block[start:end]
        # These are views into the same storage, contiguous in memory
        tensors.append(t.contiguous())  # contiguous() to ensure they're standalone
    return tensors


def find_pools(tensors):
    """Detect contiguous memory pools from a list of tensors."""
    info = sorted(
        [(t.data_ptr(), t.numel() * t.element_size(), t) for t in tensors],
        key=lambda x: x[0],
    )
    pools = []
    pool_tensors = []
    cur_start = info[0][0]
    cur_end = cur_start + info[0][1]
    cur_group = [info[0]]

    for addr, size, t in info[1:]:
        if addr == cur_end:
            cur_end = addr + size
            cur_group.append((addr, size, t))
        else:
            pools.append((cur_start, cur_end - cur_start, cur_group))
            cur_start = addr
            cur_end = addr + size
            cur_group = [(addr, size, t)]
    pools.append((cur_start, cur_end - cur_start, cur_group))
    return pools


def register(agent, tensors, mode, device):
    """Register tensors with NIXL using the specified mode."""
    if mode == "tensor":
        agent.register_memory(tensors, backends=["UCX"])
        return

    pools = find_pools(tensors)
    print(f"  Found {len(pools)} pools from {len(tensors)} tensors")

    if mode == "pool-tuple":
        descs = [(start, size, device, "cuda") for start, size, _ in pools]
        agent.register_memory(descs, mem_type="cuda", backends=["UCX"])
    elif mode == "pool-numpy":
        arr = np.array([(start, size, device) for start, size, _ in pools], dtype=np.uint64)
        agent.register_memory(arr, mem_type="cuda", backends=["UCX"])
    elif mode == "pool-tensor":
        # Workaround: create torch views spanning each pool
        pool_views = []
        for start, size, group in pools:
            anchor = group[0][2]  # first tensor in pool
            byte_view = anchor.view(torch.uint8)
            pool_views.append(byte_view.as_strided([size], [1]))
        agent.register_memory(pool_views, backends=["UCX"])


def do_transfer(agent, remote_name, src_tensors, dst_tensors, device):
    """Transfer per-tensor ranges (sub-regions within registered pools)."""
    src_descs = [(t.data_ptr(), t.numel() * t.element_size(), device) for t in src_tensors]
    dst_descs = [(t.data_ptr(), t.numel() * t.element_size(), device) for t in dst_tensors]

    src_p = agent.prep_xfer_dlist(remote_name, src_descs, "cuda", ["UCX"])
    dst_p = agent.prep_xfer_dlist("", dst_descs, "cuda", ["UCX"])

    handle = agent.make_prepped_xfer("READ", dst_p, list(range(len(src_descs))),
                                     src_p, list(range(len(src_descs))), "", ["UCX"])
    agent.transfer(handle)

    for _ in range(30000):
        status = agent.check_xfer_state(handle)
        if status in ("DONE", "SUCCESS"):
            agent.release_xfer_handle(handle)
            torch.cuda.synchronize()
            return True, status
        if status in ("ERR", "ERROR", "FAIL"):
            agent.release_xfer_handle(handle)
            return False, status
        time.sleep(0.001)
    agent.release_xfer_handle(handle)
    return False, "TIMEOUT"


def run_source(mode):
    device = 0
    torch.cuda.set_device(device)
    tensors = make_tensors(device)
    total = sum(t.numel() * t.element_size() for t in tensors)
    print(f"Source: {len(tensors)} tensors, {total / 1e6:.1f} MB, mode={mode}")

    config = nixl_agent_config(backends=["UCX"], enable_listen_thread=True, listen_port=NIXL_PORT)
    agent = nixl_agent("source", config)
    register(agent, tensors, mode, device)
    print(f"Registered, NIXL listening on {NIXL_PORT}")

    # Coordination: send tensor addresses to target
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", COORD_PORT))
    srv.listen(1)
    print(f"Waiting for target on {COORD_PORT}...")

    conn, _ = srv.accept()
    addrs = [(t.data_ptr(), t.numel() * t.element_size()) for t in tensors]
    data = json.dumps(addrs).encode()
    conn.sendall(struct.pack("!I", len(data)) + data)
    print("Sent tensor info, waiting for target...")
    conn.recv(1)
    print("Done.")
    conn.close()
    srv.close()


def run_target(peer_ip, mode):
    device = 0
    torch.cuda.set_device(device)
    tensors = make_tensors(device)
    total = sum(t.numel() * t.element_size() for t in tensors)
    print(f"Target: {len(tensors)} tensors, {total / 1e6:.1f} MB, mode={mode}")

    agent = nixl_agent("target", nixl_agent_config(backends=["UCX"]))
    # Target always registers per-tensor (known to work)
    agent.register_memory(tensors, backends=["UCX"])

    # Get source tensor addresses
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((peer_ip, COORD_PORT))
    msg_len = struct.unpack("!I", sock.recv(4))[0]
    src_addrs = json.loads(sock.recv(msg_len).decode())
    print(f"Got {len(src_addrs)} source tensor addresses")

    # Fetch NIXL metadata
    print(f"Fetching NIXL metadata from {peer_ip}:{NIXL_PORT}...")
    agent.fetch_remote_metadata("source", peer_ip, NIXL_PORT)
    start = time.perf_counter()
    while not agent.check_remote_metadata("source"):
        if time.perf_counter() - start > 30:
            print(f"FAIL: timed out fetching remote metadata ({mode} on source)")
            sock.sendall(b"x")
            sock.close()
            return
        time.sleep(0.01)
    print(f"Metadata fetched in {time.perf_counter() - start:.3f}s")

    # Build source tensor list from addresses
    class FakeTensor:
        def __init__(self, addr, size, dev):
            self._addr = addr
            self._size = size
            self._dev = dev
        def data_ptr(self): return self._addr
        def numel(self): return self._size
        def element_size(self): return 1
        def get_device(self): return self._dev

    src_fake = [FakeTensor(a, s, device) for a, s in src_addrs]

    # Attempt transfer
    try:
        t0 = time.perf_counter()
        ok, status = do_transfer(agent, "source", src_fake, tensors, device)
        dt = time.perf_counter() - t0
        if ok:
            bw = total * 8 / dt / 1e9
            print(f"PASS: {total / 1e6:.1f} MB in {dt:.3f}s ({bw:.1f} Gbps) [{mode} on source]")
        else:
            print(f"FAIL: {status} [{mode} on source]")
    except Exception as e:
        print(f"FAIL: {e} [{mode} on source]")

    sock.sendall(b"x")
    sock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["source", "target"])
    p.add_argument("--peer-ip", help="Source IP (target only)")
    p.add_argument("--mode", required=True, choices=["tensor", "pool-tuple", "pool-numpy", "pool-tensor"])
    args = p.parse_args()

    if args.role == "source":
        run_source(args.mode)
    else:
        if not args.peer_ip:
            p.error("--peer-ip required")
        run_target(args.peer_ip, args.mode)
