#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end RDMA transfer test with VMM-compacted source memory.

Verifies that NIXL RDMA READ works correctly when the source side has
tensors packed into a single contiguous CUDA virtual memory range via
the VMM API (cuMemAddressReserve / cuMemCreate / cuMemMap).

Two roles:
  source  - Creates test tensors, compacts into VMM, registers with NIXL,
            waits for target to complete transfer.
  target  - Allocates per-tensor (normal cudaMalloc), fetches remote
            metadata via NIXL listen thread, reads via RDMA, verifies.

Usage:
    # On source node:
    python test_vmm_transfer.py --role source --listen-port 5555

    # On target node:
    python test_vmm_transfer.py --role target --peer-ip <source-ip> --peer-port 5555
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import struct
import sys
import threading
import time

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_TENSORS = 100
# Tensor sizes: mix of small and large, totaling roughly 1 GB.
# 80 tensors at 8 MB each (640 MB) + 20 tensors at 20 MB each (400 MB) = 1040 MB.
SMALL_COUNT = 80
SMALL_SIZE = 8 * 1024 * 1024  # 8 MB in bytes
LARGE_COUNT = 20
LARGE_SIZE = 20 * 1024 * 1024  # 20 MB in bytes

# Coordination TCP port for metadata exchange between source and target.
# The source opens a TCP listener on this port to hand off tensor descriptors
# and the NIXL agent name to the target.
COORD_PORT = 7777

# CUDA VMM constants
CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3

# ---------------------------------------------------------------------------
# CUDA VMM helpers (from vmm_compact.py, inlined for standalone use)
# ---------------------------------------------------------------------------


class _CUmemAllocationProp(ctypes.Structure):
    class _Location(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

    class _AllocFlags(ctypes.Structure):
        _fields_ = [
            ("compressionType", ctypes.c_ubyte),
            ("gpuDirectRDMACapable", ctypes.c_ubyte),
            ("usage", ctypes.c_ushort),
            ("reserved", ctypes.c_ubyte * 4),
        ]

    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    class _Location(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


def _check(ret: int, msg: str) -> None:
    if ret != 0:
        raise RuntimeError(f"CUDA VMM error ({ret}): {msg}")


def _get_cuda() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so")


def _get_granularity(cuda, device_id: int) -> int:
    prop = _CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    granularity = ctypes.c_size_t()
    _check(
        cuda.cuMemGetAllocationGranularity(
            ctypes.byref(granularity), ctypes.byref(prop), ctypes.c_ulonglong(0)
        ),
        "cuMemGetAllocationGranularity",
    )
    return granularity.value


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _tensor_from_pointer(
    data_ptr: int,
    num_elements: int,
    dtype: torch.dtype,
    device_index: int,
    storage_size_bytes: int,
) -> torch.Tensor:
    """Create a torch.Tensor aliasing a raw CUDA pointer."""
    device = torch.device("cuda", device_index)
    storage = torch._C._construct_storage_from_data_pointer(
        data_ptr, device, storage_size_bytes
    )
    t = torch.empty(0, dtype=dtype, device=device)
    t.set_(storage, 0, (num_elements,))
    return t


# ---------------------------------------------------------------------------
# Test tensor generation
# ---------------------------------------------------------------------------


def generate_test_tensors(
    device_id: int,
) -> dict[str, tuple[torch.Tensor, float]]:
    """Create test tensors with deterministic fill values.

    Returns {name: (tensor, fill_value)} so the target can verify.
    Each tensor is filled with a unique float derived from its index.
    """
    torch.cuda.set_device(device_id)
    tensors = {}
    for i in range(SMALL_COUNT):
        name = f"layer.{i}.weight"
        num_elements = SMALL_SIZE // 2  # float16 = 2 bytes
        t = torch.empty(num_elements, dtype=torch.float16, device=f"cuda:{device_id}")
        fill_val = float(i + 1)
        t.fill_(fill_val)
        tensors[name] = (t, fill_val)
    for i in range(LARGE_COUNT):
        name = f"layer.{SMALL_COUNT + i}.weight"
        num_elements = LARGE_SIZE // 2
        t = torch.empty(num_elements, dtype=torch.float16, device=f"cuda:{device_id}")
        fill_val = float(SMALL_COUNT + i + 1)
        t.fill_(fill_val)
        tensors[name] = (t, fill_val)
    torch.cuda.synchronize(device_id)
    return tensors


# ---------------------------------------------------------------------------
# VMM compaction (simplified standalone version)
# ---------------------------------------------------------------------------


def vmm_compact(
    tensors: dict[str, torch.Tensor], device_id: int
) -> tuple[int, int, dict[str, torch.Tensor], list]:
    """Compact tensors into a single VMM VA range.

    Returns (va_base, va_size, new_tensors, handles_to_keep_alive).
    """
    cuda = _get_cuda()
    granularity = _get_granularity(cuda, device_id)
    print(f"  VMM granularity: {granularity} bytes ({granularity // 1024} KB)")

    # Plan layout: pack tensors tightly, align total to granularity
    layout = {}
    current_offset = 0
    for name, t in tensors.items():
        data_bytes = t.numel() * t.element_size()
        layout[name] = (current_offset, data_bytes)
        current_offset += data_bytes
    total_va = _align_up(current_offset, granularity)
    total_data = sum(t.numel() * t.element_size() for t in tensors.values())

    print(
        f"  Layout: {len(tensors)} tensors, "
        f"{total_data / 1e6:.1f} MB data, "
        f"{total_va / 1e6:.1f} MB VA"
    )

    # Reserve VA range
    va_ptr = ctypes.c_uint64()
    _check(
        cuda.cuMemAddressReserve(
            ctypes.byref(va_ptr),
            ctypes.c_size_t(total_va),
            ctypes.c_size_t(granularity),
            ctypes.c_uint64(0),
            ctypes.c_ulonglong(0),
        ),
        f"cuMemAddressReserve({total_va / 1e6:.1f} MB)",
    )
    va_base = va_ptr.value
    print(f"  VA reserved at 0x{va_base:x}")

    # Allocate physical pages (single handle for the whole range)
    prop = _CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    prop.allocFlags.gpuDirectRDMACapable = 1

    handle = ctypes.c_uint64()
    _check(
        cuda.cuMemCreate(
            ctypes.byref(handle),
            ctypes.c_size_t(total_va),
            ctypes.byref(prop),
            ctypes.c_ulonglong(0),
        ),
        f"cuMemCreate({total_va / 1e6:.1f} MB)",
    )

    # Map physical pages into VA
    _check(
        cuda.cuMemMap(
            ctypes.c_uint64(va_base),
            ctypes.c_size_t(total_va),
            ctypes.c_size_t(0),
            handle,
            ctypes.c_ulonglong(0),
        ),
        "cuMemMap",
    )

    # Set access
    access = _CUmemAccessDesc()
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = device_id
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _check(
        cuda.cuMemSetAccess(
            ctypes.c_uint64(va_base),
            ctypes.c_size_t(total_va),
            ctypes.byref(access),
            ctypes.c_size_t(1),
        ),
        "cuMemSetAccess",
    )

    # Copy tensors into VMM range and create aliased tensors
    new_tensors = {}
    for name, t in tensors.items():
        offset, data_bytes = layout[name]
        dest_va = va_base + offset
        _check(
            cuda.cuMemcpyDtoD_v2(
                ctypes.c_uint64(dest_va),
                ctypes.c_uint64(t.data_ptr()),
                ctypes.c_size_t(data_bytes),
            ),
            f"cuMemcpyDtoD for '{name}'",
        )
        new_t = _tensor_from_pointer(
            data_ptr=dest_va,
            num_elements=t.numel(),
            dtype=t.dtype,
            device_index=device_id,
            storage_size_bytes=data_bytes,
        )
        new_tensors[name] = new_t

    torch.cuda.synchronize(device_id)
    print(f"  Compacted {len(new_tensors)} tensors into VMM range")
    return va_base, total_va, new_tensors, [handle]


# ---------------------------------------------------------------------------
# Coordination protocol (simple TCP)
# ---------------------------------------------------------------------------


def _send_json(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)


def _recv_json(sock: socket.socket) -> dict:
    raw_len = b""
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("Connection closed while reading length")
        raw_len += chunk
    length = struct.unpack("!I", raw_len)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("Connection closed while reading data")
        data += chunk
    return json.loads(data.decode())


# ---------------------------------------------------------------------------
# Source role
# ---------------------------------------------------------------------------


def run_source(device_id: int, listen_port: int, coord_port: int) -> int:
    timings = {}

    # Phase 1: Generate test tensors
    print(f"\n[source] Phase 1: Generating {NUM_TENSORS} test tensors")
    t0 = time.perf_counter()
    raw_tensors = generate_test_tensors(device_id)
    # Extract just the tensors (not fill values) for compaction
    tensors_only = {name: t for name, (t, _) in raw_tensors.items()}
    fill_values = {name: fv for name, (_, fv) in raw_tensors.items()}
    total_bytes = sum(t.numel() * t.element_size() for t in tensors_only.values())
    timings["generate"] = time.perf_counter() - t0
    print(
        f"  Generated {len(tensors_only)} tensors, "
        f"{total_bytes / 1e6:.1f} MB total ({timings['generate']:.3f}s)"
    )

    # Phase 2: VMM compaction
    print("\n[source] Phase 2: VMM compaction")
    t0 = time.perf_counter()
    va_base, va_size, vmm_tensors, _handles = vmm_compact(tensors_only, device_id)
    timings["vmm_compact"] = time.perf_counter() - t0
    print(
        f"  Compacted into VMM: base=0x{va_base:x}, size={va_size / 1e6:.1f} MB "
        f"({timings['vmm_compact']:.3f}s)"
    )

    # Free original tensors now that data lives in VMM
    del tensors_only
    del raw_tensors
    torch.cuda.empty_cache()

    # Verify VMM data integrity before RDMA
    print("\n[source] Phase 2b: Verifying VMM data integrity")
    for name, t in vmm_tensors.items():
        expected = fill_values[name]
        sample = t[:8].float().cpu()
        if not torch.allclose(sample, torch.full((8,), expected)):
            print(f"  [FAIL] VMM data integrity check failed for {name}")
            print(f"    expected={expected}, got={sample.tolist()}")
            return 1
    print(f"  All {len(vmm_tensors)} tensors verified in VMM")

    # Phase 3: NIXL registration
    print("\n[source] Phase 3: NIXL registration (VMM mode)")
    t0 = time.perf_counter()
    try:
        from nixl._api import nixl_agent as NixlAgent
        from nixl._api import nixl_agent_config
    except ImportError:
        print("  [FAIL] nixl not installed")
        return 1

    agent_name = f"vmm-source-{os.getpid()}"
    config = nixl_agent_config(
        backends=["UCX"],
        enable_listen_thread=True,
        listen_port=listen_port,
    )
    agent = NixlAgent(agent_name, config)

    # Register the single VMM range
    agent.register_memory(
        [(va_base, va_size, device_id, "")],
        mem_type="cuda",
        backends=["UCX"],
    )
    metadata = agent.get_agent_metadata()
    timings["nixl_register"] = time.perf_counter() - t0
    print(
        f"  Registered VMM range with NIXL: 1 region, {va_size / 1e6:.1f} MB, "
        f"metadata={len(metadata)} bytes ({timings['nixl_register']:.3f}s)"
    )

    # Phase 4: Build tensor descriptor manifest for the target
    tensor_manifest = []
    for name, t in vmm_tensors.items():
        tensor_manifest.append({
            "name": name,
            "addr": t.data_ptr(),
            "size": t.numel() * t.element_size(),
            "device_id": device_id,
            "dtype": str(t.dtype),
            "num_elements": t.numel(),
            "fill_value": fill_values[name],
        })

    # Phase 5: Wait for target to connect and send manifest
    print(f"\n[source] Phase 5: Waiting for target on TCP port {coord_port}")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", coord_port))
    server_sock.listen(1)
    print(f"  Listening on 0.0.0.0:{coord_port}")

    conn, addr = server_sock.accept()
    print(f"  Target connected from {addr}")

    # Send manifest to target
    _send_json(conn, {
        "agent_name": agent_name,
        "listen_port": listen_port,
        "tensors": tensor_manifest,
    })
    print(f"  Sent manifest ({len(tensor_manifest)} tensors)")

    # Wait for target to signal completion
    result = _recv_json(conn)
    conn.close()
    server_sock.close()

    # Summary
    print("\n" + "=" * 60)
    print("[source] TIMING SUMMARY")
    print("=" * 60)
    for phase, dur in timings.items():
        print(f"  {phase:20s}: {dur:.3f}s")
    print(f"  {'total':20s}: {sum(timings.values()):.3f}s")

    if result.get("success"):
        print("\n[source] Target reported SUCCESS")
        return 0
    else:
        print(f"\n[source] Target reported FAILURE: {result.get('error', 'unknown')}")
        return 1


# ---------------------------------------------------------------------------
# Target role
# ---------------------------------------------------------------------------


def run_target(device_id: int, peer_ip: str, peer_port: int, coord_port: int) -> int:
    timings = {}

    # Phase 1: Connect to source and receive manifest
    print(f"\n[target] Phase 1: Connecting to source at {peer_ip}:{coord_port}")
    t0 = time.perf_counter()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Retry connection for up to 60s (source may still be setting up)
    deadline = time.time() + 60
    while True:
        try:
            sock.connect((peer_ip, coord_port))
            break
        except ConnectionRefusedError:
            if time.time() > deadline:
                print("  [FAIL] Timed out connecting to source")
                return 1
            time.sleep(1)
    print(f"  Connected to source")

    manifest = _recv_json(sock)
    agent_name_remote = manifest["agent_name"]
    listen_port_remote = manifest["listen_port"]
    tensor_descs = manifest["tensors"]
    timings["connect"] = time.perf_counter() - t0
    print(
        f"  Received manifest: {len(tensor_descs)} tensors, "
        f"remote agent={agent_name_remote} ({timings['connect']:.3f}s)"
    )

    # Phase 2: Allocate local tensors (normal cudaMalloc, per-tensor)
    print(f"\n[target] Phase 2: Allocating {len(tensor_descs)} local tensors")
    t0 = time.perf_counter()
    torch.cuda.set_device(device_id)

    local_tensors = {}
    for desc in tensor_descs:
        name = desc["name"]
        num_elements = desc["num_elements"]
        dtype = getattr(torch, desc["dtype"].replace("torch.", ""))
        t = torch.zeros(num_elements, dtype=dtype, device=f"cuda:{device_id}")
        local_tensors[name] = t

    total_bytes = sum(d["size"] for d in tensor_descs)
    timings["allocate"] = time.perf_counter() - t0
    print(
        f"  Allocated {len(local_tensors)} tensors, "
        f"{total_bytes / 1e6:.1f} MB ({timings['allocate']:.3f}s)"
    )

    # Phase 3: NIXL setup and registration
    print("\n[target] Phase 3: NIXL registration (per-tensor)")
    t0 = time.perf_counter()
    try:
        from nixl._api import nixl_agent as NixlAgent
        from nixl._api import nixl_agent_config
    except ImportError:
        print("  [FAIL] nixl not installed")
        _send_json(sock, {"success": False, "error": "nixl not installed"})
        sock.close()
        return 1

    local_agent_name = f"vmm-target-{os.getpid()}"
    config = nixl_agent_config(backends=["UCX"])
    local_agent = NixlAgent(local_agent_name, config)

    # Register local tensors
    tensor_list = list(local_tensors.values())
    local_agent.register_memory(tensor_list, backends=["UCX"])
    timings["nixl_register"] = time.perf_counter() - t0
    print(
        f"  Registered {len(tensor_list)} tensors with NIXL "
        f"({timings['nixl_register']:.3f}s)"
    )

    # Phase 4: Fetch remote metadata via NIXL listen thread
    # NIXL's listen thread needs a numeric IP, not a DNS name (inet_pton).
    # Resolve the peer hostname to an IP address first.
    resolved_ip = socket.gethostbyname(peer_ip)
    print(f"\n[target] Phase 4: Fetching remote metadata from {resolved_ip}:{listen_port_remote} (resolved from {peer_ip})")
    t0 = time.perf_counter()
    local_agent.fetch_remote_metadata(agent_name_remote, resolved_ip, listen_port_remote)

    # Poll until remote metadata is loaded
    poll_start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - poll_start
        if elapsed > 30.0:
            msg = "Timed out fetching remote metadata"
            print(f"  [FAIL] {msg}")
            _send_json(sock, {"success": False, "error": msg})
            sock.close()
            return 1
        if local_agent.check_remote_metadata(agent_name_remote):
            break
        time.sleep(0.01)

    timings["fetch_metadata"] = time.perf_counter() - t0
    print(f"  Remote metadata loaded ({timings['fetch_metadata']:.3f}s)")

    # Phase 5: Execute RDMA READ transfer
    print(f"\n[target] Phase 5: RDMA READ transfer ({total_bytes / 1e6:.1f} MB)")
    t0 = time.perf_counter()

    # Build transfer descriptor lists
    remote_descs = []
    local_descs = []
    for desc in tensor_descs:
        name = desc["name"]
        local_t = local_tensors[name]
        remote_descs.append((desc["addr"], desc["size"], desc["device_id"]))
        local_descs.append(
            (local_t.data_ptr(), local_t.numel() * local_t.element_size(), device_id)
        )

    src_prepped = local_agent.prep_xfer_dlist(
        agent_name=agent_name_remote,
        xfer_list=remote_descs,
        mem_type="cuda",
        backends=["UCX"],
    )
    dst_prepped = local_agent.prep_xfer_dlist(
        agent_name="",
        xfer_list=local_descs,
        mem_type="cuda",
        backends=["UCX"],
    )
    indices = list(range(len(remote_descs)))

    handle = local_agent.make_prepped_xfer(
        operation="READ",
        local_xfer_side=dst_prepped,
        local_indices=indices,
        remote_xfer_side=src_prepped,
        remote_indices=indices,
        backends=["UCX"],
    )

    local_agent.transfer(handle)

    # Wait for completion
    xfer_start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - xfer_start
        if elapsed > 120.0:
            local_agent.release_xfer_handle(handle)
            msg = "Transfer timed out after 120s"
            print(f"  [FAIL] {msg}")
            _send_json(sock, {"success": False, "error": msg})
            sock.close()
            return 1
        status = local_agent.check_xfer_state(handle)
        if status in ("DONE", "SUCCESS"):
            local_agent.release_xfer_handle(handle)
            break
        if status in ("ERR", "ERROR", "FAIL"):
            local_agent.release_xfer_handle(handle)
            msg = f"Transfer failed with status {status}"
            print(f"  [FAIL] {msg}")
            _send_json(sock, {"success": False, "error": msg})
            sock.close()
            return 1
        time.sleep(0.001)

    torch.cuda.synchronize(device_id)
    timings["rdma_transfer"] = time.perf_counter() - t0
    bandwidth_gbps = (total_bytes * 8) / (timings["rdma_transfer"] * 1e9)
    print(
        f"  Transfer complete: {total_bytes / 1e6:.1f} MB in "
        f"{timings['rdma_transfer']:.3f}s ({bandwidth_gbps:.1f} Gbps)"
    )

    # Phase 6: Verify received data
    print(f"\n[target] Phase 6: Verifying {len(tensor_descs)} tensors")
    t0 = time.perf_counter()
    errors = []
    for desc in tensor_descs:
        name = desc["name"]
        expected_val = desc["fill_value"]
        local_t = local_tensors[name]
        # Check first 8 and last 8 elements
        first = local_t[:8].float().cpu()
        last = local_t[-8:].float().cpu()
        expected = torch.full((8,), expected_val)
        if not torch.allclose(first, expected):
            errors.append(
                f"{name}: first 8 mismatch, expected={expected_val}, got={first.tolist()}"
            )
        if not torch.allclose(last, expected):
            errors.append(
                f"{name}: last 8 mismatch, expected={expected_val}, got={last.tolist()}"
            )
    timings["verify"] = time.perf_counter() - t0

    if errors:
        print(f"  [FAIL] {len(errors)} verification errors:")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
        _send_json(sock, {"success": False, "error": f"{len(errors)} tensor mismatches"})
        sock.close()
        return 1

    print(f"  All {len(tensor_descs)} tensors verified ({timings['verify']:.3f}s)")

    # Signal success to source
    _send_json(sock, {"success": True})
    sock.close()

    # Summary
    print("\n" + "=" * 60)
    print("[target] TIMING SUMMARY")
    print("=" * 60)
    for phase, dur in timings.items():
        print(f"  {phase:20s}: {dur:.3f}s")
    print(f"  {'total':20s}: {sum(timings.values()):.3f}s")
    print(f"\n[target] PASS - all {len(tensor_descs)} tensors transferred and verified")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end RDMA transfer test with VMM-compacted source memory"
    )
    parser.add_argument(
        "--role",
        choices=["source", "target"],
        required=True,
        help="Role: source (VMM + NIXL serve) or target (RDMA READ + verify)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device ordinal (default: 0)",
    )
    parser.add_argument(
        "--peer-ip",
        type=str,
        default="",
        help="Source pod IP (required for target role)",
    )
    parser.add_argument(
        "--peer-port",
        type=int,
        default=5555,
        help="Source NIXL listen port (default: 5555)",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=5555,
        help="NIXL listen port for source role (default: 5555)",
    )
    parser.add_argument(
        "--coord-port",
        type=int,
        default=COORD_PORT,
        help=f"TCP coordination port for metadata exchange (default: {COORD_PORT})",
    )
    args = parser.parse_args()

    print(f"VMM RDMA Transfer Test - role={args.role}, device={args.device_id}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"  UCX_TLS={os.environ.get('UCX_TLS', 'not set')}")

    if args.role == "source":
        return run_source(args.device_id, args.listen_port, args.coord_port)
    elif args.role == "target":
        if not args.peer_ip:
            print("[FAIL] --peer-ip is required for target role")
            return 1
        return run_target(args.device_id, args.peer_ip, args.peer_port, args.coord_port)
    return 1


if __name__ == "__main__":
    sys.exit(main())
