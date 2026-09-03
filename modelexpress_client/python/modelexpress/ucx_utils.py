# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UCX-specific workarounds for NIXL transfers.

This module collects helpers that apply only when the NIXL backend is UCX
(InfiniBand / RoCE / OPA / EFA RDMA traffic). The headline piece is the
per-rank NIC pinning logic, which addresses two distinct problems.

The first is openucx/ucx#11259, where UCX's lane scoring does not honor
GPU<->NIC PCIe affinity for CUDA memory paths and ends up picking
cross-socket NICs on multi-NUMA hosts.

The second is why the ranking leads with load balancing, and it is the one
that has actually been measured to cost throughput here. UCX chooses a NIC
per process with no knowledge of what sibling ranks on the same host chose.
When a pod is under-provisioned - fewer usable rails than GPUs, or rails
allocated on the wrong socket - every rank independently makes the same
locally-correct choice and they all land on one adapter. Measured on such a
pod: four concurrent readers sharing a rail ran at 1.5 GB/s each, 6.1 GB/s
aggregate; spread one per rail they ran at 6.75 GB/s each, 27.0 GB/s
aggregate. Aggregate with four readers was *lower* than one reader alone, so
throughput was being destroyed rather than divided.

Note that this second case is not UCX misbehaving. Given a pod holding
exactly one rail on the GPUs' own NUMA node, converging every rank onto it is
the correct answer to the question UCX is asking. It is the wrong answer to
the question the host is posing, and only a component that can see all the
ranks at once can tell the difference - which is what the global assignment
in ``probe_nic_pin_for_device`` is for.

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


_NVIDIA_PCI_VENDOR = "0x10de"


def _host_gpu_bdfs() -> list[str]:
    """Every NVIDIA GPU on the host, whether or not this process can see it.

    The assignment below spreads GPUs across rails by counting how many GPUs it
    has already placed on each one, which only spreads anything if every rank
    counts the same GPUs. ``torch.cuda.device_count()`` cannot supply that: under
    a per-rank ``CUDA_VISIBLE_DEVICES`` mask - how NeMo-RL and prime-RL both
    launch - each worker sees exactly one device, restarts the tally from zero,
    and picks the same best rail as every one of its peers. That is the rail
    convergence this module exists to prevent, arrived at by the code meant to
    prevent it.

    Enumerating PCI instead sidesteps it without any cross-rank coordination:
    sysfs is not filtered by CUDA visibility, so each rank derives the same
    host-wide map from the same snapshot and then reads off only its own row.

    Returns [] when the listing is unreadable, which leaves the caller on
    visible devices alone - degraded, and logged as such.
    """
    try:
        entries = sorted(os.listdir("/sys/bus/pci/devices"))
    except OSError:
        return []

    out: list[str] = []
    for bdf in entries:
        base = f"/sys/bus/pci/devices/{bdf}"
        vendor = _read_str_file(f"{base}/vendor")
        pci_class = _read_str_file(f"{base}/class")
        if vendor is None or pci_class is None:
            continue
        # Class 0x03xxxx is "display controller"; NVIDIA GPUs enumerate as
        # 0x030000 (VGA) or 0x030200 (3D controller), and matching the prefix
        # covers both without pinning the check to one of them.
        if vendor.lower() == _NVIDIA_PCI_VENDOR and pci_class.lower().startswith(
            "0x03"
        ):
            out.append(bdf)
    return out


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
      - 1 shared = same root port
      - 0 shared = NODE or SYS, indistinguishable here

    That last line is the one to remember. ``_pci_path_components`` keeps
    only BDF-shaped components, and the root complex appears in the
    realpath as ``pci0000:97``, which is not BDF-shaped and so is dropped.
    Two devices under the same root complex but different root ports
    therefore share no retained component and score 0, exactly like two
    devices on opposite sockets. Anything needing to tell NODE from SYS
    must read numa_node instead; this metric cannot, and the caller's
    cross-socket term exists for that reason.
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


def _nic_has_accessible_verbs_device(nic_name: str) -> bool:
    """Return whether the NIC has a verbs device exposed in this container."""
    verbs_dir = f"/sys/class/infiniband/{nic_name}/device/infiniband_verbs"
    try:
        verbs = os.listdir(verbs_dir)
    except OSError:
        return False
    return any(
        name.startswith("uverbs")
        and os.path.exists(f"/dev/infiniband/{name}")
        for name in verbs
    )


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
      - NICs whose verbs device is not exposed in the container. Kubernetes
        RDMA device plugins commonly expose one ``uverbs`` device while the
        host's complete InfiniBand sysfs remains visible.
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
        if not _nic_has_accessible_verbs_device(name):
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
      2. Discover the PCIe path of every GPU *on the host* - read from
         PCI, not from CUDA, so a per-rank CUDA_VISIBLE_DEVICES mask
         cannot shrink the set - so this rank computes the same global
         GPU->NIC assignment that every other rank computes from the
         same /sys snapshot. No coordination. See ``_host_gpu_bdfs``
         for why the visible set is the wrong input.
      3. Greedy assignment, best-affinity-first. Each GPU picks the NIC
         with lowest (prior-assignments, -score, cross-socket,
         lex-smallest name) - rail distinctness dominates, then PCIe
         depth, then NUMA locality, then determinism. Reuse is allowed
         when GPU count exceeds NIC count, with cycle counts kept
         balanced.
      4. Returns this rank's assignment as 'NICNAME:1', or None if no
         compute device is reachable.

    Distinctness ranking first is a measured decision. On a pod whose four
    GPUs shared one same-socket rail, four concurrent readers ran at
    1.5 GB/s each (6.1 GB/s aggregate); spread one-per-rail, with three of
    the four crossing sockets, they ran at 6.75 GB/s each (27.0 GB/s
    aggregate). Sharing a rail cost 4.45x. Crossing a socket cost nothing
    measurable: the one reader on its PCIe-affine rail got 6.747 GB/s
    against 6.745 for the three cross-socket ones.

    Two things this ordering is *not*. It is not a fix for a shipped bug:
    the previous key was ``(-score, count, name)``, and traced against that
    pod it gives three distinct rails with one doubled, because almost
    every GPU-NIC pair there scores 0 and the count term does break those
    ties. It is a fix for an unshipped one: an intermediate revision of
    this function inserted the cross-socket term *above* count, giving
    ``(-score, cross_socket, count, name)``, and that key hands the lone
    same-socket rail to all four GPUs in turn - a same-socket rail beats
    every free cross-socket one on every pass, and count never gets a say.
    That is the 1.5 GB/s configuration, and it was written while diagnosing
    the very collapse it would have caused.

    Hence the cross-socket term is kept but ranked below distinctness, so
    it is honoured only where free. Note the throughput figures above were
    taken against a single publisher rail, so cross-socket cost is bounded
    by that ceiling rather than shown to be zero in general - it is known
    to be small relative to sharing, which is all this ordering needs.
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

    # The rank's own GPU is identified by PCI address rather than by visible
    # index, because the index means nothing outside this process: under a
    # per-rank mask every worker drives "GPU 0" and they are four different
    # cards. The address is what the host-wide map below is keyed on.
    own_bdf = _gpu_pci_bdf(device_id)
    if own_bdf is None:
        logger.warning(
            f"MX_RDMA_NIC_PIN auto-probe: no readable PCI address for CUDA "
            f"device {device_id}; skipping pin"
        )
        return None

    try:
        num_visible = torch.cuda.device_count()
    except Exception:
        num_visible = 0
    visible_bdfs = {
        bdf for bdf in (_gpu_pci_bdf(gi) for gi in range(num_visible)) if bdf
    }
    visible_bdfs.add(own_bdf)

    # Union rather than replacement. PCI is the authority on what exists, but a
    # GPU this process is actually driving belongs in the map even if the
    # listing missed it, and on a host where the listing is unreadable this
    # degrades to exactly the old visible-device behaviour.
    host_bdfs = _host_gpu_bdfs()
    if not host_bdfs:
        logger.warning(
            "MX_RDMA_NIC_PIN auto-probe: could not enumerate host GPUs from "
            "/sys/bus/pci/devices; falling back to this process's visible "
            "devices, which may assign the same rail as a peer rank on this host"
        )
    gpu_bdfs = sorted(visible_bdfs | set(host_bdfs))

    if len(gpu_bdfs) < 2 and len(nics) > 1:
        # One GPU in the PCI listing too, so either the host genuinely has one
        # or the container is device-isolated and there is nothing to spread
        # across. Indistinguishable from in here, and only the second case is a
        # problem, so say so rather than report a one-entry map as a success.
        logger.warning(
            f"MX_RDMA_NIC_PIN auto-probe: only one GPU ({own_bdf}) is visible in "
            f"the PCI listing while {len(nics)} rails are available, so rails "
            f"cannot be spread. If peer ranks share this host under device "
            f"isolation, set UCX_NET_DEVICES or MX_RDMA_NIC_PIN explicitly to "
            f"keep them off one rail."
        )

    gpu_paths: dict[str, list[str]] = {}
    gpu_numa: dict[str, int] = {}
    for bdf in gpu_bdfs:
        gpu_paths[bdf] = _pci_path_components(bdf)
        numa = _read_int_file(f"/sys/bus/pci/devices/{bdf}/numa_node")
        gpu_numa[bdf] = numa if numa is not None else -1

    # Greedy assignment. Each GPU picks the least-assigned NIC, then the
    # closest of those, then a same-socket one, then lex-smallest name for
    # determinism so every rank computes the same map with no coordination.
    #
    # GPUs are visited best-affinity-first rather than in index order,
    # because with distinctness ranked first the visit order decides who
    # wins a contested rail. On the measured pod only GPU 3 had any PCIe
    # affinity to the single same-socket rail, and in index order GPU 0
    # took that rail on a zero-depth tie, pushing GPU 3 onto a cross-socket
    # one. Both orders give four distinct rails and so both capture the
    # 4.45x, but visiting the GPU that has a real affinity first also
    # honours it. Ties fall back to index, keeping the result deterministic.
    #
    # Greedy is still not globally optimal - a Hungarian solve over the
    # (gpu, nic) score matrix would be, on the same inputs - but the
    # remaining gap is now a question of which GPU gets which distinct
    # rail, not whether rails are shared, and sharing was the term worth
    # 4.45x.
    assigned_count: dict[str, int] = {n[0]: 0 for n in nics}
    assignments: dict[str, tuple[str, int]] = {}

    def _best_score(bdf: str) -> int:
        return max(
            (_pci_common_depth(gpu_paths[bdf], nic_path) for *_, nic_path in nics),
            default=0,
        )

    # Ties break on PCI address rather than visible index for the same reason
    # the map is keyed on it: the index is not comparable across ranks, so
    # ordering by it would let two ranks walk the same GPUs in different orders
    # and reach different assignments from identical inputs.
    visit_order = sorted(gpu_paths.keys(), key=lambda bdf: (-_best_score(bdf), bdf))
    for gpu_bdf in visit_order:
        gpu_path = gpu_paths[gpu_bdf]
        this_gpu_numa = gpu_numa.get(gpu_bdf, -1)
        ranked: list[tuple[int, int, int, str]] = []
        for nic_name, nic_numa, _nic_rate, nic_path in nics:
            score = _pci_common_depth(gpu_path, nic_path)
            # NUMA locality ranks BELOW load balancing. PCIe common depth
            # separates PIX from everything else but cannot separate NODE from
            # SYS - see _pci_common_depth, the root complex is filtered out of
            # the path - so without this term a same-socket rail ties with a
            # cross-socket one at depth 0 and the tiebreak falls through to name
            # order, leaving the local rail unused. It stays for that reason.
            # What it must not do is outrank distinctness: measured, sharing a
            # rail costs 4.45x and crossing a socket costs ~0, and ranking it
            # higher forces every GPU onto a lone same-socket rail in turn. An
            # intermediate revision did exactly that; see the function docstring.
            cross_socket = (
                this_gpu_numa >= 0 and nic_numa >= 0 and this_gpu_numa != nic_numa
            )
            ranked.append(
                (assigned_count[nic_name], -score, 1 if cross_socket else 0, nic_name)
            )
        ranked.sort()
        # Index 1 is -score; index 0 is the assignment count the sort now leads
        # with. Reading the wrong slot here silently mislabels every diagnostic
        # log line as PCIe common-depth 0.
        chosen_name = ranked[0][3]
        chosen_score = -ranked[0][1]
        assignments[gpu_bdf] = (chosen_name, chosen_score)
        assigned_count[chosen_name] += 1

    chosen_name, chosen_score = assignments[own_bdf]
    nic_numa_map = {n[0]: n[1] for n in nics}
    nic_rate_map = {n[0]: n[2] for n in nics}
    same_numa_nics = [
        n[0] for n in nics if n[1] == gpu_numa.get(own_bdf, -2) and n[1] >= 0
    ]
    full_map = {bdf: a[0] for bdf, a in sorted(assignments.items())}
    cross_socket = (
        gpu_numa.get(own_bdf, -1) >= 0
        and nic_numa_map.get(chosen_name, -1) >= 0
        and gpu_numa[own_bdf] != nic_numa_map[chosen_name]
    )
    if cross_socket:
        logger.warning(
            f"MX_RDMA_NIC_PIN auto-probe: GPU {device_id} ({own_bdf}) -> "
            f"{chosen_name}:1 "
            f"is CROSS-SOCKET (GPU NUMA {gpu_numa[own_bdf]}, NIC NUMA "
            f"{nic_numa_map[chosen_name]}); single-flow bandwidth will be "
            f"capped by UPI / Infinity Fabric. PCIe common-depth {chosen_score}, "
            f"same-NUMA NICs available: {same_numa_nics}, full GPU->NIC map: "
            f"{full_map}"
        )
    else:
        logger.info(
            f"MX_RDMA_NIC_PIN auto-probe: GPU {device_id} ({own_bdf}) -> "
            f"{chosen_name}:1 "
            f"(PCIe common-depth {chosen_score}; GPU NUMA "
            f"{gpu_numa.get(own_bdf)}, NIC NUMA {nic_numa_map.get(chosen_name)}, "
            f"NIC rate {nic_rate_map.get(chosen_name)} Gb/s; "
            f"same-NUMA NICs: {same_numa_nics}; full GPU->NIC map: {full_map})"
        )
    return f"{chosen_name}:1"


def _resolve_nic_pin(device_id: int) -> str | None:
    """Resolve MX_RDMA_NIC_PIN env var into a UCX_NET_DEVICES value.

    Modes:
      - unset / "off" / "0" / "false" / "no": returns None (no pinning).
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
        min_rate: float | None = None
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
