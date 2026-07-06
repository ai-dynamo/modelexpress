# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UCX-specific workarounds for NIXL transfers.

This module collects helpers that apply only when the NIXL backend is UCX
(InfiniBand / RoCE / OPA / EFA RDMA traffic). The headline piece is the
per-rank NIC pinning logic that works around openucx/ucx#11259, where
UCX's lane scoring does not honor GPU<->NIC PCIe affinity for CUDA memory
paths and ends up picking cross-socket NICs on multi-NUMA hosts.

Public surface:
- ``apply_nic_pin_for_device(device_id)``: resolve ``MX_RDMA_NIC_PIN`` and
  set ``UCX_NET_DEVICES`` for the worker. Side-effecting; designed to be
  called once per worker before NIXL agent construction.
- ``probe_nic_pin_for_device(device_id, min_rate_gbps=None)``: pure
  topology probe; returns the NIC string this rank should pin to, or
  ``None``. Exposed for diagnostics and testing.

Everything else is private and may change without notice.
"""

from __future__ import annotations

import logging
import os
import re

import torch

from . import envs

logger = logging.getLogger("modelexpress.ucx_utils")


def _read_int_file(path: str) -> int | None:
    """Read a single int from a sysfs file. Returns None on any failure."""
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_str_file(path: str) -> str | None:
    """Read a string from a sysfs file. Returns None on any failure."""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except OSError:
        return None


def _parse_ib_rate_gbps(rate_str: str) -> float | None:
    """Parse an InfiniBand port rate string ('400 Gb/s (4X NDR)') -> 400.0."""
    if not rate_str:
        return None
    parts = rate_str.strip().split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def _gpu_pci_bdf(device_id: int) -> str | None:
    """Return the PCIe BDF ('0000:0f:00.0') for a CUDA visible device.

    Uses torch.cuda.get_device_properties; the CUDA runtime handles
    CUDA_VISIBLE_DEVICES filtering, so device_id here is the visible
    index that the worker drives.
    """
    try:
        props = torch.cuda.get_device_properties(device_id)
        domain = int(getattr(props, "pci_domain_id", 0))
        bus = int(props.pci_bus_id)
        dev = int(props.pci_device_id)
    except (AttributeError, RuntimeError, AssertionError, TypeError) as e:
        logger.warning(f"NIC pin probe: unable to read PCI BDF for device {device_id}: {e}")
        return None
    return f"{domain:04x}:{bus:02x}:{dev:02x}.0"


def _gpu_numa_node(device_id: int) -> int | None:
    """Read the NUMA node for a given CUDA visible device's GPU.

    Returns the numa_node int (which may be -1 on systems without NUMA),
    or None if the BDF or sysfs file isn't readable.
    """
    bdf = _gpu_pci_bdf(device_id)
    if bdf is None:
        return None
    return _read_int_file(f"/sys/bus/pci/devices/{bdf}/numa_node")


def _pci_path_components(bdf: str) -> list[str]:
    """Resolve a PCI BDF to its sysfs realpath and return the BDF chain.

    For a device at 0000:0f:00.0, the realpath of
    /sys/bus/pci/devices/0000:0f:00.0 typically looks like:
        /sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/0000:0f:00.0
    The returned list keeps only the BDF-shaped components, in order
    from closest-to-root to leaf. Common-prefix length between two such
    lists encodes PCIe affinity (longer prefix = same switch / bridge),
    which is exactly the metric nvidia-smi topo -m uses to label PIX /
    PXB / NODE / SYS connections.

    Returns [] on any read failure.
    """
    try:
        rp = os.path.realpath(f"/sys/bus/pci/devices/{bdf}")
    except OSError:
        return []
    bdf_re = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$")
    return [p for p in rp.split("/") if bdf_re.match(p)]


def _pci_common_depth(a: list[str], b: list[str]) -> int:
    """Length of the longest shared prefix between two PCIe path component lists.

    Higher values mean closer in the PCIe tree:
      - 4+ shared = PIX (single PCIe bridge), best
      - 2-3 shared = PXB / PHB (multiple bridges, same root port)
      - 1 shared = NODE (same root complex, different root ports)
      - 0 shared = SYS (different sockets, traffic crosses CPU UPI)
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _nic_pci_bdf(nic_name: str) -> str | None:
    """Return the PCI BDF for an InfiniBand NIC.

    Reads the symlink /sys/class/infiniband/<nic>/device which points at
    something like ../../../0000:10:00.0; returns the basename.
    """
    try:
        target = os.readlink(f"/sys/class/infiniband/{nic_name}/device")
    except OSError:
        return None
    return os.path.basename(target.rstrip("/"))


def _list_compute_ib_nics(
    min_rate_gbps: float | None = None,
) -> list[tuple[str, int, float, list[str]]]:
    """Enumerate IB-class NICs eligible for compute RDMA traffic.

    Probe surface is /sys/class/infiniband/, which is the kernel verbs
    API. This covers InfiniBand, RoCE, OPA, and AWS EFA - any fabric
    that exposes via ibv_*. Fabrics outside the verbs API (e.g. HPE
    Cray Slingshot's CXI driver) are not visible here; on those
    systems users should either leave MX_RDMA_NIC_PIN unset or set it
    to an explicit NIC list. If /sys/class/infiniband is missing or
    empty (most non-RDMA hosts) this returns []; the caller treats
    that as "skip pin" and leaves UCX selection alone.

    Filters out:
      - bonded interfaces (e.g. mlx5_bond_0): UCX cannot resolve them
        in containers and the AH lookup segfaults.
      - NICs without a /ports/1 directory.
      - NICs whose port-1 rate is below the effective threshold.
        If min_rate_gbps is None (default), the threshold is set to
        max(rate) over discovered NICs - this strips any side-fabric
        NICs (management, storage) running slower than the compute
        fabric without hardcoding a number. If min_rate_gbps is set,
        it is used as an absolute lower bound (overrides the max-rate
        autodetect; useful for clusters with mixed-tier compute
        fabrics where you do want to keep multiple rates).

    Returns a list of (nic_name, numa_node, rate_gbps, pci_path)
    sorted alphabetically by NIC name. The PCIe path is the BDF chain
    from /sys realpath; pair-wise common-prefix depth between a GPU's
    path and a NIC's path encodes affinity (PIX > PXB > NODE > SYS)
    and is the actual selection signal in probe_nic_pin_for_device().
    NIC name ordering only affects the final lex tiebreak.

    NUMA is read from /sys/class/infiniband/<nic>/device/numa_node and
    is kept on the tuple for diagnostic logging only; -1 if the
    kernel reports unknown.
    """
    base = "/sys/class/infiniband"
    if not os.path.isdir(base):
        return []

    try:
        names = sorted(os.listdir(base))
    except OSError:
        return []

    candidates: list[tuple[str, float, str]] = []
    for name in names:
        if "bond" in name:
            continue
        port_dir = f"{base}/{name}/ports/1"
        if not os.path.isdir(port_dir):
            continue
        rate_str = _read_str_file(f"{port_dir}/rate")
        rate = _parse_ib_rate_gbps(rate_str) if rate_str else None
        if rate is None:
            continue
        candidates.append((name, rate, port_dir))

    if not candidates:
        return []

    if min_rate_gbps is None:
        threshold = max(r for _, r, _ in candidates)
    else:
        threshold = min_rate_gbps

    out: list[tuple[str, int, float, list[str]]] = []
    for name, rate, _port_dir in candidates:
        if rate < threshold:
            continue
        numa = _read_int_file(f"{base}/{name}/device/numa_node")
        if numa is None:
            numa = -1
        bdf = _nic_pci_bdf(name)
        path = _pci_path_components(bdf) if bdf else []
        out.append((name, numa, rate, path))
    return out


def probe_nic_pin_for_device(
    device_id: int, min_rate_gbps: float | None = None
) -> str | None:
    """Probe topology and choose a UCX_NET_DEVICES value for a given GPU.

    Selection signal is PCIe sysfs path distance: each device's
    /sys/bus/pci/devices/<bdf> realpath exposes the full bus tree, and
    the longest common BDF prefix between a GPU's path and a NIC's
    path encodes affinity (PIX > PXB > NODE > SYS, the same metric
    nvidia-smi topo -m reports). NIC names and GPU indices stop
    mattering for correctness; they only affect the final lex tiebreak.

    Strategy:
      1. Enumerate compute-fabric IB-class NICs (verbs-API: IB / RoCE
         / OPA / EFA), with their PCIe paths. Rate filtering: by
         default, keep only NICs at max(rate) across the discovered
         set so side-fabric NICs (management, storage) at a lower
         rate are stripped. min_rate_gbps overrides this with an
         explicit absolute lower bound.
      2. Discover every visible CUDA device's PCIe path so this rank
         computes the same global GPU->NIC assignment that every
         other rank computes from the same /sys snapshot. No
         coordination.
      3. Greedy assignment in visible-index order. Each GPU picks the
         NIC with highest (score, fewest-prior-assignments,
         lex-smallest name) - score dominates, then load balancing
         across reuse, then determinism. Reuse is allowed when GPU
         count exceeds NIC count, with cycle counts kept balanced.
      4. Returns this rank's assignment as 'NICNAME:1', or None if no
         compute device is reachable.
    """
    nics = _list_compute_ib_nics(min_rate_gbps)
    if not nics:
        rate_desc = (
            "max-rate auto-detect"
            if min_rate_gbps is None
            else f"rate >= {min_rate_gbps} Gb/s"
        )
        logger.warning(
            f"MX_RDMA_NIC_PIN auto-probe: no compute IB-class NICs found "
            f"under /sys/class/infiniband ({rate_desc}); skipping pin. "
            f"This is expected on hosts without IB / RoCE / OPA / EFA, "
            f"or on fabrics outside the kernel verbs API (e.g. Slingshot)."
        )
        return None

    try:
        num_gpus = torch.cuda.device_count()
    except Exception:
        num_gpus = 0

    gpu_paths: dict[int, list[str]] = {}
    gpu_numa: dict[int, int] = {}
    for gi in range(num_gpus):
        bdf = _gpu_pci_bdf(gi)
        if bdf is None:
            continue
        gpu_paths[gi] = _pci_path_components(bdf)
        numa = _read_int_file(f"/sys/bus/pci/devices/{bdf}/numa_node")
        gpu_numa[gi] = numa if numa is not None else -1

    if device_id not in gpu_paths:
        logger.warning(
            f"MX_RDMA_NIC_PIN auto-probe: GPU {device_id} not found among "
            f"visible CUDA devices ({sorted(gpu_paths.keys())}); skipping pin"
        )
        return None

    # Greedy assignment in visible-index order. Each GPU picks the NIC
    # with the highest PCIe-affinity score; ties broken by fewest prior
    # assignments (load balancing across reuse), then lex-smallest NIC
    # name (determinism so every rank computes the same map).
    #
    # Note: greedy-by-index is not globally optimal. On an asymmetric
    # topology where two GPUs both score equally on the same best NIC
    # but each has a distinct second-best, the lower-index GPU wins the
    # shared best and the higher-index GPU may end up on a worse NIC
    # than a Hungarian-style global assignment would give it. In
    # practice real GPU clusters are symmetric within a NUMA (n GPUs +
    # n PIX-affined NICs on the same root complex), so each GPU's PIX
    # NIC is unique and the greedy result equals the optimal. If a
    # future topology breaks this assumption, replace with a Hungarian
    # solve over the (gpu, nic) score matrix - same inputs, just a
    # better assignment policy. Don't try to "fix" it by reshuffling
    # the iteration order; that just changes which rank is the loser.
    assigned_count: dict[str, int] = {n[0]: 0 for n in nics}
    assignments: dict[int, tuple[str, int]] = {}
    for gi in sorted(gpu_paths.keys()):
        gpu_path = gpu_paths[gi]
        ranked: list[tuple[int, int, str]] = []
        for nic_name, _nic_numa, _nic_rate, nic_path in nics:
            score = _pci_common_depth(gpu_path, nic_path)
            ranked.append((-score, assigned_count[nic_name], nic_name))
        ranked.sort()
        chosen_name = ranked[0][2]
        chosen_score = -ranked[0][0]
        assignments[gi] = (chosen_name, chosen_score)
        assigned_count[chosen_name] += 1

    chosen_name, chosen_score = assignments[device_id]
    nic_numa_map = {n[0]: n[1] for n in nics}
    nic_rate_map = {n[0]: n[2] for n in nics}
    same_numa_nics = [
        n[0] for n in nics if n[1] == gpu_numa.get(device_id, -2) and n[1] >= 0
    ]
    full_map = {gi: a[0] for gi, a in sorted(assignments.items())}
    cross_socket = (
        gpu_numa.get(device_id, -1) >= 0
        and nic_numa_map.get(chosen_name, -1) >= 0
        and gpu_numa[device_id] != nic_numa_map[chosen_name]
    )
    if cross_socket:
        logger.warning(
            f"MX_RDMA_NIC_PIN auto-probe: GPU {device_id} -> {chosen_name}:1 "
            f"is CROSS-SOCKET (GPU NUMA {gpu_numa[device_id]}, NIC NUMA "
            f"{nic_numa_map[chosen_name]}); single-flow bandwidth will be "
            f"capped by UPI / Infinity Fabric. PCIe common-depth {chosen_score}, "
            f"same-NUMA NICs available: {same_numa_nics}, full GPU->NIC map: "
            f"{full_map}"
        )
    else:
        logger.info(
            f"MX_RDMA_NIC_PIN auto-probe: GPU {device_id} -> {chosen_name}:1 "
            f"(PCIe common-depth {chosen_score}; GPU NUMA "
            f"{gpu_numa.get(device_id)}, NIC NUMA {nic_numa_map.get(chosen_name)}, "
            f"NIC rate {nic_rate_map.get(chosen_name)} Gb/s; "
            f"same-NUMA NICs: {same_numa_nics}; full GPU->NIC map: {full_map})"
        )
    return f"{chosen_name}:1"


def _resolve_nic_pin(device_id: int) -> str | None:
    """Resolve MX_RDMA_NIC_PIN env var into a UCX_NET_DEVICES value.

    Modes:
      - unset / "off" / "0" / "false" / "no": returns None (no pinning).
      - "stripe": enumerates all compute-rate NICs and returns a
        comma-separated list — every rank sees every NIC, so UCX can
        stripe RMA traffic across all of them. On multi-NIC nodes
        this can lift the per-receiver bandwidth roof from single-NIC
        (~400 Gbps ideal, ~316 Gbps observed on GB200) toward N *
        single-NIC. Also bumps UCX_MAX_RMA_RAILS to len(NICs) so UCX
        actually uses the rails rather than picking one.
      - explicit comma-separated list: indexed by device_id, like the
        original hardcoded shape. Useful for unusual topologies where
        the auto-probe heuristic doesn't fit (e.g. fabrics outside the
        kernel verbs API).
      - any other truthy value (e.g. "auto", "1", "true", "yes", "on"):
        runs probe_nic_pin_for_device(). Rate filtering defaults to
        max-rate auto-detect (keep only NICs at the fastest rate
        present, strips slower side-fabric NICs without hardcoding a
        number). MX_RDMA_NIC_PIN_MIN_RATE_GBPS overrides with an
        explicit absolute lower bound when needed.
    """
    raw = envs.MX_RDMA_NIC_PIN
    if raw == "" or raw.lower() in ("off", "0", "false", "no"):
        return None

    if raw.lower() == "stripe":
        # Enumerate ALL compute-rate NICs and hand them all to UCX.
        # Rate filter defaults to max-rate auto-detect so slower
        # side-fabric NICs (mgmt / storage) are stripped, matching
        # auto mode's filter.
        raw_min = os.environ.get("MX_RDMA_NIC_PIN_MIN_RATE_GBPS")
        try:
            min_rate = float(raw_min) if raw_min else None
        except ValueError:
            min_rate = None
        nics = _list_compute_ib_nics(min_rate_gbps=min_rate)
        if not nics:
            logger.warning(
                "MX_RDMA_NIC_PIN=stripe: no compute-rate NICs found; "
                "skipping pin"
            )
            return None
        # Format each NIC as ``<name>:1`` to match UCX's device
        # syntax. UCX picks the port from :1 (all IB NICs have their
        # single port at index 1).
        pinned = ",".join(f"{name}:1" for name, _numa, _rate, _path in nics)
        # Bump UCX_MAX_RMA_RAILS so UCX actually multiplexes RMA
        # traffic across the pinned NICs. Default is 1 (single rail);
        # setting to len(nics) lets the striping happen. Callers can
        # pre-set UCX_MAX_RMA_RAILS to override.
        if "UCX_MAX_RMA_RAILS" not in os.environ:
            os.environ["UCX_MAX_RMA_RAILS"] = str(len(nics))
            logger.info(
                f"MX_RDMA_NIC_PIN=stripe: set UCX_MAX_RMA_RAILS="
                f"{len(nics)} (was unset)"
            )
        logger.info(
            f"MX_RDMA_NIC_PIN=stripe: pinning device {device_id} to "
            f"{len(nics)} NICs -> {pinned}"
        )
        return pinned

    if "," in raw:
        nic_list = [n.strip() for n in raw.split(",") if n.strip()]
        if 0 <= device_id < len(nic_list):
            pinned = nic_list[device_id]
            logger.info(
                f"MX_RDMA_NIC_PIN explicit list: device {device_id} -> {pinned}"
            )
            return pinned
        logger.warning(
            f"MX_RDMA_NIC_PIN explicit list: device_id {device_id} out of "
            f"range for list of length {len(nic_list)}; skipping pin"
        )
        return None

    raw_min = envs.MX_RDMA_NIC_PIN_MIN_RATE_GBPS
    if raw_min is None or raw_min.strip() == "":
        min_rate = None
    else:
        try:
            min_rate = float(raw_min)
        except ValueError:
            logger.warning(
                f"MX_RDMA_NIC_PIN_MIN_RATE_GBPS={raw_min!r} not a float; "
                f"falling back to max-rate auto-detect"
            )
            min_rate = None
    return probe_nic_pin_for_device(device_id, min_rate_gbps=min_rate)


def apply_nic_pin_for_device(device_id: int) -> None:
    """Resolve MX_RDMA_NIC_PIN and apply it to UCX_NET_DEVICES.

    Set permanently for the worker's lifetime (no restore in finally) so
    any subsequently-created UCP contexts also use the pinned NIC. No-op
    when MX_RDMA_NIC_PIN is unset / "off" / "0" / "false" / "no", which
    is the default. Designed to be called once per worker before NIXL
    agent construction.

    See module docstring for the full semantics, including the explicit
    NIC-list override and the rate-filter env var.
    """
    pinned = _resolve_nic_pin(device_id)
    if pinned:
        prev = envs.UCX_NET_DEVICES
        os.environ["UCX_NET_DEVICES"] = pinned
        logger.info(
            f"NIXL NIC pin: device {device_id} -> "
            f"UCX_NET_DEVICES={pinned} (was: {prev})"
        )
