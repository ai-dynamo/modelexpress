#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL registration count diagnostic.

Tests whether raw-address pool registration produces working RDMA transfers
compared to per-tensor registration. Uses a single persistent agent per run
to avoid UCX global state corruption from agent teardown/recreation.

Run once per mode (tensor or pool-numpy) as separate pod pairs.

Usage (cross-node, IB required):
    # Per-tensor baseline:
    source: python nixl_reg_count_diagnostic.py --role source --mode tensor
    target: python nixl_reg_count_diagnostic.py --role target --mode tensor --peer-ip <src_ip>

    # Pool registration test:
    source: python nixl_reg_count_diagnostic.py --role source --mode pool-numpy
    target: python nixl_reg_count_diagnostic.py --role target --mode pool-numpy --peer-ip <src_ip>
"""

import argparse
import json
import os
import socket
import struct
import time

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config

NIXL_PORT = 5580
COORD_PORT = 9998
TENSOR_SIZE = 1024 * 1024  # 1MB per tensor

XFER_COUNTS = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]
TOTAL_TENSORS = max(XFER_COUNTS)


def allocate_tensors(count, device):
    tensors = []
    for _ in range(count):
        t = torch.randn(TENSOR_SIZE // 4, dtype=torch.float32, device=f"cuda:{device}")
        tensors.append(t)
    return tensors


def find_pools(tensors):
    indexed = sorted(
        [(t.data_ptr(), t.numel() * t.element_size(), i, t) for i, t in enumerate(tensors)],
        key=lambda x: x[0],
    )
    pools = []
    cur_start = indexed[0][0]
    cur_end = cur_start + indexed[0][1]
    cur_indices = [indexed[0][2]]

    for addr, size, idx, t in indexed[1:]:
        if addr == cur_end:
            cur_end = addr + size
            cur_indices.append(idx)
        else:
            pools.append((cur_start, cur_end - cur_start, cur_indices))
            cur_start = addr
            cur_end = addr + size
            cur_indices = [idx]
    pools.append((cur_start, cur_end - cur_start, cur_indices))
    return pools


def coord_send(sock, data):
    encoded = json.dumps(data).encode()
    sock.sendall(struct.pack("!I", len(encoded)) + encoded)


def coord_recv(sock):
    raw_len = sock.recv(4)
    if len(raw_len) < 4:
        raise ConnectionError("Short read on length prefix")
    msg_len = struct.unpack("!I", raw_len)[0]
    chunks = []
    remaining = msg_len
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("Connection closed mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return json.loads(b"".join(chunks).decode())


def print_config(mode):
    print(f"=== NIXL Registration Count Diagnostic ({mode.upper()}) ===")
    for var in ["UCX_RCACHE_ENABLE", "UCX_LOG_LEVEL", "UCX_IB_REG_METHODS",
                "UCX_RCACHE_MAX_REGIONS", "UCX_TLS"]:
        val = os.environ.get(var, "(not set)")
        print(f"  {var}={val}")
    print()


def run_source(args):
    device = 0
    torch.cuda.set_device(device)
    mode = args.mode

    print_config(f"SOURCE / {mode}")

    print(f"Allocating {TOTAL_TENSORS} x {TENSOR_SIZE // 1024}KB tensors...")
    all_tensors = allocate_tensors(TOTAL_TENSORS, device)
    total_mb = TOTAL_TENSORS * TENSOR_SIZE / 1e6
    print(f"Allocated {total_mb:.0f} MB across {TOTAL_TENSORS} tensors")

    pools = find_pools(all_tensors)
    print(f"Detected {len(pools)} contiguous pools")

    all_addrs = [(t.data_ptr(), t.numel() * t.element_size()) for t in all_tensors]

    # Create agent and register memory based on mode
    agent = nixl_agent(f"src-{mode}", nixl_agent_config(
        backends=["UCX"], enable_listen_thread=True, listen_port=NIXL_PORT,
    ))

    if mode == "tensor":
        agent.register_memory(all_tensors, backends=["UCX"])
        print(f"Registered {TOTAL_TENSORS} tensors individually")
    elif mode == "tensor-tuple":
        # Same addresses as per-tensor, but via raw tuples instead of torch objects
        tuple_descs = [(t.data_ptr(), t.numel() * t.element_size(), device, "") for t in all_tensors]
        reg_descs = agent.get_reg_descs(tuple_descs, "cuda")
        agent.register_memory(reg_descs, backends=["UCX"])
        print(f"Registered {TOTAL_TENSORS} tensors individually via raw tuples")
    elif mode == "whole-alloc":
        # Register each cudaMalloc block as a WHOLE via raw tuples.
        # If the rcache already has the whole block, this should be a no-op
        # and produce correct rkeys since we match the rcache's natural granularity.
        from ctypes import c_uint64, c_size_t, byref, CDLL
        cuda = CDLL("libcuda.so")
        seen_allocs = {}
        for t in all_tensors:
            addr = t.data_ptr()
            base = c_uint64()
            alloc_size = c_size_t()
            ret = cuda.cuMemGetAddressRange_v2(byref(base), byref(alloc_size), c_uint64(addr))
            if ret != 0:
                continue
            alloc_base = base.value
            if alloc_base not in seen_allocs:
                seen_allocs[alloc_base] = alloc_size.value
        alloc_list = sorted(seen_allocs.items())
        tuple_descs = [(base, sz, device, "") for base, sz in alloc_list]
        reg_descs = agent.get_reg_descs(tuple_descs, "cuda")
        agent.register_memory(reg_descs, backends=["UCX"])
        print(f"Registered {len(alloc_list)} whole cudaMalloc blocks via tuples:")
        for base, sz in alloc_list:
            print(f"  0x{base:x} size={sz / (1024*1024):.1f}MB")
    elif mode == "multi-region":
        # Register N equal-sized regions from one big allocation, N from --regions arg.
        n_regions = args.regions
        total_bytes = TOTAL_TENSORS * TENSOR_SIZE
        big_tensor = torch.empty(total_bytes, dtype=torch.uint8, device=f"cuda:{device}")
        base_addr = big_tensor.data_ptr()
        region_size = total_bytes // n_regions
        all_tensors = []
        for i in range(TOTAL_TENSORS):
            offset = i * TENSOR_SIZE
            view = big_tensor[offset:offset + TENSOR_SIZE].view(torch.float32)
            all_tensors.append(view)
        for t in all_tensors:
            t.fill_(1.0)
        all_addrs = [(t.data_ptr(), t.numel() * t.element_size()) for t in all_tensors]
        tuple_descs = [(base_addr + i * region_size, region_size, device, "") for i in range(n_regions)]
        reg_descs = agent.get_reg_descs(tuple_descs, "cuda")
        agent.register_memory(reg_descs, backends=["UCX"])
        print(f"Registered {n_regions} regions x {region_size // (1024*1024)}MB from single cudaMalloc via tuples")
    elif mode == "pool-numpy":
        pool_descs = np.array(
            [(start, size, device) for start, size, _ in pools],
            dtype=np.uint64,
        )
        agent.register_memory(pool_descs, mem_type="cuda", backends=["UCX"])
        print(f"Registered {len(pools)} pools via numpy")
    elif mode == "pool-tuple":
        # vLLM-style: 4-element tuples with empty metaInfo string
        tuple_descs = [(start, size, device, "") for start, size, _ in pools]
        reg_descs = agent.get_reg_descs(tuple_descs, "cuda")
        agent.register_memory(reg_descs, backends=["UCX"])
        print(f"Registered {len(pools)} pools via 4-tuples (vLLM path)")
    elif mode == "single-alloc":
        # Hypothesis test: one big cudaMalloc, register via raw tuples.
        # If this works, the bug is about spanning multiple cudaMalloc blocks.
        total_bytes = TOTAL_TENSORS * TENSOR_SIZE
        big_tensor = torch.empty(total_bytes, dtype=torch.uint8, device=f"cuda:{device}")
        base_addr = big_tensor.data_ptr()
        # Create views into the big tensor to serve as "individual tensors"
        all_tensors = []
        for i in range(TOTAL_TENSORS):
            offset = i * TENSOR_SIZE
            view = big_tensor[offset:offset + TENSOR_SIZE].view(torch.float32)
            all_tensors.append(view)
        # Fill with data so transfers are meaningful
        for t in all_tensors:
            t.fill_(1.0)
        # Re-compute addresses from the views
        all_addrs = [(t.data_ptr(), t.numel() * t.element_size()) for t in all_tensors]
        # Register the single big allocation as one pool via tuples
        tuple_descs = [(base_addr, total_bytes, device, "")]
        reg_descs = agent.get_reg_descs(tuple_descs, "cuda")
        agent.register_memory(reg_descs, backends=["UCX"])
        print(f"Registered 1 pool ({total_bytes / 1e6:.0f} MB) from single cudaMalloc via tuples")
    elif mode == "pool-validated":
        # Register pools that are validated to be within single CUDA allocations.
        # Uses cuMemGetAddressRange to check each pool, splits pools that span allocations.
        from ctypes import c_uint64, c_size_t, byref, CDLL
        cuda = CDLL("libcuda.so")
        valid_pools = []
        split_count = 0
        for start, size, indices in pools:
            base = c_uint64()
            alloc_size = c_size_t()
            ret = cuda.cuMemGetAddressRange_v2(byref(base), byref(alloc_size), c_uint64(start))
            if ret != 0:
                print(f"  WARNING: cuMemGetAddressRange_v2 failed ({ret}) for pool at 0x{start:x}, skipping")
                continue
            alloc_base = base.value
            alloc_end = alloc_base + alloc_size.value
            pool_end = start + size
            if pool_end <= alloc_end:
                valid_pools.append((start, size, indices))
            else:
                # Pool spans allocation boundary - split at the boundary
                split_count += 1
                first_size = alloc_end - start
                second_start = alloc_end
                second_size = pool_end - alloc_end
                # Split indices based on address
                first_indices = [i for i in indices if all_tensors[i].data_ptr() < alloc_end]
                second_indices = [i for i in indices if all_tensors[i].data_ptr() >= alloc_end]
                if first_size > 0 and first_indices:
                    valid_pools.append((start, first_size, first_indices))
                if second_size > 0 and second_indices:
                    valid_pools.append((second_start, second_size, second_indices))
        print(f"Original: {len(pools)} pools, split {split_count} spanning allocations -> {len(valid_pools)} validated pools")
        tuple_descs = [(s, sz, device, "") for s, sz, _ in valid_pools]
        reg_descs = agent.get_reg_descs(tuple_descs, "cuda")
        agent.register_memory(reg_descs, backends=["UCX"])
        print(f"Registered {len(valid_pools)} allocation-aligned pools via tuples")

    # Wait for target
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", COORD_PORT))
    srv.listen(1)
    print(f"Waiting for target on port {COORD_PORT}...")
    conn, addr = srv.accept()
    print(f"Target connected from {addr[0]}")

    # Send setup info
    coord_send(conn, {
        "agent_name": f"src-{mode}",
        "nixl_port": NIXL_PORT,
        "all_addrs": all_addrs,
    })

    # Wait for metadata ack
    ack = coord_recv(conn)
    if ack.get("status") != "ready":
        print(f"Target not ready: {ack}")
        conn.close()
        srv.close()
        return

    print("Metadata exchange complete, running transfer sweep...\n")
    results = []

    for count in XFER_COUNTS:
        coord_send(conn, {"action": "xfer", "count": count})
        result = coord_recv(conn)
        status = result.get("status", "UNKNOWN")
        bw = result.get("bw_gbps", 0)
        detail = result.get("detail", "")
        tag = "PASS" if status == "PASS" else "FAIL"
        bw_str = f" ({bw:.1f} Gbps)" if bw > 0 else ""
        detail_str = f" [{detail}]" if detail else ""
        print(f"  {mode:12s} count={count:4d}  {tag}{bw_str}{detail_str}")
        results.append({"count": count, "status": tag, "bw_gbps": bw, "detail": detail})

    coord_send(conn, {"action": "done"})
    conn.close()
    srv.close()

    print(f"\n=== SUMMARY ({mode}) ===")
    print(f"{'Count':>6} {'Status':>6} {'BW (Gbps)':>10}")
    print("-" * 28)
    for r in results:
        bw_str = f"{r['bw_gbps']:.1f}" if r["bw_gbps"] > 0 else "-"
        print(f"{r['count']:>6} {r['status']:>6} {bw_str:>10}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] != "PASS")
    print(f"\n{passed} passed, {failed} failed")


def run_target(args):
    device = 0
    torch.cuda.set_device(device)
    mode = args.mode

    print_config(f"TARGET / {mode}")

    print(f"Allocating {TOTAL_TENSORS} target tensors...")
    target_tensors = allocate_tensors(TOTAL_TENSORS, device)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to source at {args.peer_ip}:{COORD_PORT}...")
    sock.connect((args.peer_ip, COORD_PORT))
    print("Connected.")

    # Get setup info
    setup = coord_recv(sock)
    agent_name = setup["agent_name"]
    nixl_port = setup["nixl_port"]
    src_addrs = setup["all_addrs"]

    # Target always registers per-tensor (known working)
    tgt_agent = nixl_agent(f"tgt-{mode}", nixl_agent_config(backends=["UCX"]))
    tgt_agent.register_memory(target_tensors, backends=["UCX"])
    print(f"Registered {TOTAL_TENSORS} target tensors")

    # Fetch source metadata
    tgt_agent.fetch_remote_metadata(agent_name, args.peer_ip, nixl_port)
    t0 = time.perf_counter()
    while not tgt_agent.check_remote_metadata(agent_name):
        if time.perf_counter() - t0 > 30:
            print("ERROR: metadata timeout after 30s")
            coord_send(sock, {"status": "FAIL", "detail": "metadata timeout"})
            sock.close()
            return
        time.sleep(0.01)
    md_time = time.perf_counter() - t0
    print(f"Metadata fetched in {md_time:.3f}s")
    coord_send(sock, {"status": "ready"})

    # Process transfer requests
    while True:
        msg = coord_recv(sock)
        if msg["action"] == "done":
            break

        count = msg["count"]
        addrs = src_addrs[:count]
        dst = target_tensors[:count]

        try:
            src_descs = [(addr, size, device) for addr, size in addrs]
            dst_descs = [(t.data_ptr(), t.numel() * t.element_size(), device) for t in dst]

            src_p = tgt_agent.prep_xfer_dlist(agent_name, src_descs, "cuda", ["UCX"])
            dst_p = tgt_agent.prep_xfer_dlist("", dst_descs, "cuda", ["UCX"])

            handle = tgt_agent.make_prepped_xfer(
                "READ", dst_p, list(range(count)),
                src_p, list(range(count)), "", ["UCX"]
            )

            total_bytes = sum(s for _, s in addrs)
            t0 = time.perf_counter()
            tgt_agent.transfer(handle)

            for _ in range(30000):
                status = tgt_agent.check_xfer_state(handle)
                if status in ("DONE", "SUCCESS"):
                    dt = time.perf_counter() - t0
                    bw = total_bytes * 8 / dt / 1e9
                    tgt_agent.release_xfer_handle(handle)
                    coord_send(sock, {
                        "status": "PASS",
                        "bw_gbps": round(bw, 1),
                        "detail": f"md={md_time:.3f}s xfer={dt:.3f}s",
                    })
                    break
                if status in ("ERR", "ERROR", "FAIL"):
                    tgt_agent.release_xfer_handle(handle)
                    coord_send(sock, {"status": "FAIL", "detail": f"xfer status: {status}"})
                    break
                time.sleep(0.001)
            else:
                tgt_agent.release_xfer_handle(handle)
                coord_send(sock, {"status": "FAIL", "detail": "xfer timeout (30s)"})

        except Exception as e:
            coord_send(sock, {"status": "FAIL", "detail": str(e)})

    sock.close()
    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NIXL registration count diagnostic")
    p.add_argument("--role", required=True, choices=["source", "target"])
    p.add_argument("--mode", required=True, choices=["tensor", "tensor-tuple", "pool-numpy", "pool-tuple", "single-alloc", "pool-validated", "whole-alloc", "multi-region"])
    p.add_argument("--peer-ip", help="Source IP (target only)")
    p.add_argument("--regions", type=int, default=2, help="Number of regions for multi-region mode")
    args = p.parse_args()

    if args.role == "source":
        run_source(args)
    else:
        if not args.peer_ip:
            p.error("--peer-ip required for target")
        run_target(args)
