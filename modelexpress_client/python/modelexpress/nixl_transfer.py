# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL Transfer Manager for GPU-to-GPU weight transfers.

This module provides the NixlTransferManager class that handles all NIXL-related
operations including agent creation, tensor registration, and RDMA transfers.

Each vLLM worker creates its own NixlTransferManager instance to manage
a single NIXL agent for that worker's GPU.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from .types import TensorDescriptor

logger = logging.getLogger("modelexpress.nixl_transfer")

NIXL_AVAILABLE = False
NixlAgent = None
nixl_agent_config = None
try:
    from nixl._api import nixl_agent as NixlAgent
    from nixl._api import nixl_agent_config
    NIXL_AVAILABLE = True
except ImportError:
    pass


def is_nixl_available() -> bool:
    """Check if NIXL is available."""
    return NIXL_AVAILABLE


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
    import os
    import re

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
    import os

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
    import os

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
    import os

    raw = os.environ.get("MX_RDMA_NIC_PIN", "").strip()
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

    raw_min = os.environ.get("MX_RDMA_NIC_PIN_MIN_RATE_GBPS")
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


class NixlTransferManager:
    """
    Manages a single NIXL agent and RDMA transfers for GPU tensors.

    Each vLLM worker creates its own instance of this class to handle:
    - Creating and managing a NIXL agent for the worker's GPU
    - Registering tensors with NIXL for RDMA access
    - Executing transfers to receive weights from remote sources

    Args:
        agent_name: Name for the NIXL agent
        device_id: GPU device ID for this worker
    """

    def __init__(self, agent_name: str, device_id: int, listen_port: int | None = None):
        self._agent_name = agent_name
        self._device_id = device_id
        self._listen_port = listen_port

        self._agent: Any = None
        self._metadata: bytes = b""
        self._tensor_descriptors: list[TensorDescriptor] = []
        self._tensors: dict[str, torch.Tensor] = {}
        self._registered_regions: list[tuple[int, int]] | None = None

    @property
    def agent_name(self) -> str:
        """Get NIXL agent name."""
        return self._agent_name

    @property
    def nixl_metadata(self) -> bytes:
        """Get NIXL metadata for this agent."""
        return self._metadata

    @property
    def tensor_descriptors(self) -> list[TensorDescriptor]:
        """Get tensor descriptors for registered tensors."""
        return self._tensor_descriptors

    def initialize(self) -> None:
        """Initialize the NIXL agent.

        Temporarily overrides UCX_TLS to allow NIXL's UCX context to
        auto-detect RoCE/IB transports, even if the global UCX_TLS is
        restricted to TCP (e.g., for MPI). Restores the original value
        after agent creation.

        If MX_RDMA_NIC_PIN is set to a truthy value (e.g. "auto", "1",
        "true"), runs a sysfs topology probe to pick a NUMA-local IB NIC
        for this worker and sets UCX_NET_DEVICES to it. If the env var
        is set to an explicit comma-separated NIC list, that list is
        indexed by self._device_id directly (legacy / override path for
        unusual topologies). Default (unset, "off", "0", "false", "no")
        is no pinning, matching pre-feature behavior. The override is
        permanent for the worker's lifetime (not restored in finally)
        so any subsequent UCP contexts created in the worker also pin.
        See _resolve_nic_pin() and probe_nic_pin_for_device() for
        full semantics.
        """
        import os

        if not NIXL_AVAILABLE:
            raise RuntimeError("NIXL is not available")

        if self._agent is not None:
            return

        torch.cuda.set_device(self._device_id)

        # Let UCX auto-detect transports (RoCE, TCP, etc).
        # OMPI_MCA_pml=ob1 keeps MPI on TCP independently.
        # Only override UCX_TLS if explicitly set to "tcp" (legacy compat).
        saved_ucx_tls = os.environ.get("UCX_TLS")
        nixl_ucx_tls = os.environ.get("NIXL_UCX_TLS")
        if nixl_ucx_tls:
            os.environ["UCX_TLS"] = nixl_ucx_tls
            logger.info(f"NIXL UCX_TLS override: {nixl_ucx_tls} (was: {saved_ucx_tls})")
        elif saved_ucx_tls == "tcp":
            os.environ.pop("UCX_TLS", None)
            logger.info("NIXL: removed UCX_TLS=tcp for auto-detection")

        # Optional per-rank NIC pinning, set permanently for the worker's
        # lifetime (no restore in finally) so any subsequent UCP contexts
        # also use the pinned NIC. See _resolve_nic_pin() for env semantics.
        pinned = _resolve_nic_pin(self._device_id)
        if pinned:
            prev = os.environ.get("UCX_NET_DEVICES")
            os.environ["UCX_NET_DEVICES"] = pinned
            logger.info(
                f"NIXL NIC pin: device {self._device_id} -> "
                f"UCX_NET_DEVICES={pinned} (was: {prev})"
            )

        try:
            if self._listen_port is not None and nixl_agent_config:
                config = nixl_agent_config(
                    backends=["UCX"],
                    enable_listen_thread=True,
                    listen_port=self._listen_port,
                )
                logger.info(
                    f"NIXL listen thread enabled on port {self._listen_port}"
                )
            elif nixl_agent_config:
                config = nixl_agent_config(backends=["UCX"])
            else:
                config = None
            self._agent = NixlAgent(self._agent_name, config)
            logger.info(f"NIXL agent '{self._agent_name}' created on device {self._device_id}")
        finally:
            if saved_ucx_tls is not None:
                os.environ["UCX_TLS"] = saved_ucx_tls
            elif "UCX_TLS" in os.environ:
                os.environ.pop("UCX_TLS")

    def register_tensors(self, tensors: dict[str, torch.Tensor]) -> bytes:
        """
        Register tensors with NIXL for RDMA access.

        CRITICAL: We must ensure self._tensors contains the SAME tensor objects
        that are registered with NIXL, so receive_from_source uses correct memory.

        If MX_CONTIGUOUS_REG=1, detects and registers contiguous memory regions
        as single blocks, reducing descriptor overhead significantly.

        Args:
            tensors: Dictionary of tensor name -> tensor

        Returns:
            NIXL metadata bytes for this agent
        """
        import os

        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        # CRITICAL: Do NOT call .contiguous() here!
        # The tensors must be the exact same objects as param.data in vLLM,
        # otherwise RDMA writes to copies and vLLM uses originals = garbage.
        self._tensors = tensors
        tensor_descriptors = []

        for name, tensor in tensors.items():
            if not tensor.is_contiguous():
                raise RuntimeError(
                    f"Tensor '{name}' is not contiguous. "
                    "Non-contiguous tensors cannot be used for RDMA transfers."
                )
            tensor_descriptors.append(TensorDescriptor(
                name=name,
                addr=tensor.data_ptr(),
                size=tensor.numel() * tensor.element_size(),
                device_id=self._device_id,
                dtype=str(tensor.dtype),
            ))

        self._tensor_descriptors = tensor_descriptors

        # Check if contiguous region registration is enabled
        use_contiguous = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"

        if use_contiguous:
            # Register contiguous memory regions as single blocks
            regions = self._find_contiguous_regions(tensor_descriptors)
            logger.info(
                f"[Contiguous Registration] Found {len(regions)} contiguous regions "
                f"from {len(tensor_descriptors)} tensors "
                f"({(1 - len(regions)/len(tensor_descriptors))*100:.1f}% reduction)"
            )

            # Register regions using raw address tuples
            # Format: (addr, size, device_id, mem_type) - 4-tuple required by NIXL API
            region_tuples = [(r[0], r[1], self._device_id, "cuda") for r in regions]
            self._agent.register_memory(region_tuples, mem_type="cuda", backends=["UCX"])
            self._registered_regions = regions
            logger.info(f"Registered {len(regions)} contiguous regions with NIXL")
            # Debug: Log first few registered region addresses
            if len(regions) > 0:
                logger.info(f"[Contiguous Registration] DEBUG - First 3 regions: {[(hex(r[0]), r[1]) for r in regions[:3]]}")
        else:
            # Traditional: register individual tensors
            tensor_list = list(tensors.values())
            self._agent.register_memory(tensor_list, backends=["UCX"])
            self._registered_regions = None
            logger.info(f"Registered {len(tensor_list)} individual tensors with NIXL")

        self._metadata = self._agent.get_agent_metadata()
        return self._metadata

    def get_registered_descriptors(self) -> list[TensorDescriptor]:
        """
        Get the descriptors that were actually registered with NIXL.

        When MX_CONTIGUOUS_REG=1, returns contiguous region descriptors.
        Otherwise, returns individual tensor descriptors.

        This is important for publishing to the server - we must publish
        what was actually registered, not the original tensors.
        """
        if self._registered_regions is not None:
            # Return region descriptors with synthetic names
            return [
                TensorDescriptor(
                    name=f"__region_{i}__",
                    addr=addr,
                    size=size,
                    device_id=self._device_id,
                    dtype="contiguous_region",
                )
                for i, (addr, size) in enumerate(self._registered_regions)
            ]
        else:
            # Return original tensor descriptors
            return self._tensor_descriptors

    def _find_contiguous_regions(
        self, descriptors: list[TensorDescriptor]
    ) -> list[tuple[int, int]]:
        """
        Find contiguous memory regions from tensor descriptors.

        Sorts tensors by address and merges adjacent ones into larger regions.
        This reduces the number of NIXL registrations significantly.

        Args:
            descriptors: List of tensor descriptors

        Returns:
            List of (start_addr, total_size) tuples for contiguous regions
        """
        if not descriptors:
            return []

        # Sort by address
        sorted_descs = sorted(descriptors, key=lambda d: d.addr)

        regions = []
        current_start = sorted_descs[0].addr
        current_end = current_start + sorted_descs[0].size

        for desc in sorted_descs[1:]:
            if desc.addr == current_end:
                # Contiguous - extend region
                current_end = desc.addr + desc.size
            else:
                # Gap - save current region and start new one
                regions.append((current_start, current_end - current_start))
                current_start = desc.addr
                current_end = desc.addr + desc.size

        # Don't forget the last region
        regions.append((current_start, current_end - current_start))

        return regions

    def fetch_remote_and_wait(
        self,
        remote_agent_name: str,
        ip: str,
        port: int,
        timeout_seconds: float = 120.0,
    ) -> None:
        """Fetch remote NIXL agent metadata via the P2P listen thread.

        Initiates an async fetch and polls until the remote agent's metadata
        is loaded locally. Used in P2P mode instead of add_remote_agent().
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        logger.info(
            f"Fetching remote metadata from {remote_agent_name} at {ip}:{port}"
        )
        self._agent.fetch_remote_metadata(remote_agent_name, ip, port)

        start = time.perf_counter()
        while True:
            if time.perf_counter() - start >= timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for remote metadata from "
                    f"{remote_agent_name} at {ip}:{port}"
                )
            if self._agent.check_remote_metadata(remote_agent_name):
                logger.info(
                    f"Remote metadata loaded for {remote_agent_name} "
                    f"({time.perf_counter() - start:.2f}s)"
                )
                return
            time.sleep(0.01)

    def receive_from_source(
        self,
        source_metadata: bytes,
        source_tensors: list[TensorDescriptor],
        timeout_seconds: float | None = None,
        coalesce_transfers: bool = False,
        remote_agent_name: str | None = None,
    ) -> tuple[int, int, float]:
        """
        Receive weights from a remote source via NIXL RDMA.

        Args:
            source_metadata: NIXL metadata from the source agent (unused if remote_agent_name set)
            source_tensors: Tensor descriptors from the source
            timeout_seconds: Maximum time to wait for transfer (None for no timeout)
            coalesce_transfers: If True, coalesce contiguous memory regions (optimization)
            remote_agent_name: If set, use this pre-loaded agent (P2P mode) instead of
                calling add_remote_agent with source_metadata (centralized mode)

        Returns:
            Tuple of (total_bytes, total_tensors, duration)
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        start_time = time.perf_counter()
        torch.cuda.set_device(self._device_id)

        if remote_agent_name is None:
            add_start = time.perf_counter()
            remote_agent_name = self._agent.add_remote_agent(source_metadata)
            add_time = time.perf_counter() - add_start
            logger.info(
                f"[TIMING] add_remote_agent: {add_time:.3f}s "
                f"(agent={remote_agent_name}, blob={len(source_metadata)} bytes)"
            )
        else:
            logger.info(f"Using pre-loaded remote agent {remote_agent_name}")

        # Check if source is sending region descriptors (MX_CONTIGUOUS_REG=1 on source)
        is_region_transfer = (
            len(source_tensors) > 0 and
            source_tensors[0].name.startswith("__region_")
        )

        if is_region_transfer:
            # REGION-BASED TRANSFER: Source registered contiguous regions
            # We must also have registered regions and match by index
            if self._registered_regions is None:
                logger.error("Source sent region descriptors but we didn't register regions!")
                logger.error("Set MX_CONTIGUOUS_REG=1 on target to enable region transfer")
                raise RuntimeError("Region transfer mismatch: target must also use MX_CONTIGUOUS_REG=1")

            logger.info(f"Region-based transfer: {len(source_tensors)} source regions -> {len(self._registered_regions)} local regions")

            # Validate region counts match
            if len(source_tensors) != len(self._registered_regions):
                logger.warning(
                    f"Region count mismatch: source has {len(source_tensors)}, "
                    f"local has {len(self._registered_regions)}. Proceeding with min."
                )

            # Build transfer lists by region index
            remote_descs = []
            local_descs = []  # Will be (addr, size, device_id) tuples
            total_bytes = 0
            matched_count = min(len(source_tensors), len(self._registered_regions))

            for i in range(matched_count):
                src_region = source_tensors[i]
                local_addr, local_size = self._registered_regions[i]

                # Verify sizes match (regions should be same size)
                if src_region.size != local_size:
                    logger.warning(f"Region {i} size mismatch: source={src_region.size}, local={local_size}")

                remote_descs.append((src_region.addr, src_region.size, src_region.device_id))
                local_descs.append((local_addr, local_size, self._device_id))
                total_bytes += src_region.size

            matched_tensors = matched_count
            use_raw_descriptors = True
            coalesced_count = matched_count

            logger.info(f"[Region Transfer] Matched {matched_count} regions, {total_bytes / 1e9:.2f} GB")

            # Debug: Log first few region addresses for comparison
            if matched_count > 0:
                logger.info(f"[Region Transfer] DEBUG - First 3 source regions: {[(hex(r[0]), r[1]) for r in remote_descs[:3]]}")
                logger.info(f"[Region Transfer] DEBUG - First 3 local regions: {[(hex(r[0]), r[1]) for r in local_descs[:3]]}")

        else:
            # TENSOR-BASED TRANSFER: Match by tensor name (baseline)
            remote_descs = []
            local_tensor_list = []
            total_bytes = 0
            matched_tensors = 0

            for src_tensor in source_tensors:
                if src_tensor.name not in self._tensors:
                    continue
                local_tensor = self._tensors[src_tensor.name]
                remote_descs.append((src_tensor.addr, src_tensor.size, src_tensor.device_id))
                local_tensor_list.append(local_tensor)
                total_bytes += src_tensor.size
                matched_tensors += 1

            if not remote_descs:
                logger.warning("No matching tensors found for transfer")
                return 0, 0, 0.0

            # For tensor-based, we might still coalesce if enabled
            local_descs = local_tensor_list
            use_raw_descriptors = False
            coalesced_count = matched_tensors

        # OPTIMIZATION: Coalesce contiguous memory regions to reduce descriptor overhead
        # Skip if we're doing region-based transfer (already optimized at registration time)
        if is_region_transfer:
            # Region transfer already has optimal descriptors, skip coalescing
            logger.info(f"[Region Transfer] Skipping coalesce - already optimized with {coalesced_count} regions")
        elif coalesce_transfers:
            logger.info(f"[Coalesce] Starting coalescing of {len(remote_descs)} descriptors...")
            remote_descs, local_descs, coalesced_count = self._coalesce_transfers(
                remote_descs, local_tensor_list
            )
            reduction_pct = (1 - coalesced_count / matched_tensors) * 100 if matched_tensors > 0 else 0
            logger.info(
                f"[Coalesce] Reduced {matched_tensors} descriptors -> {coalesced_count} regions "
                f"({reduction_pct:.1f}% reduction)"
            )
            # local_descs are now (addr, size, device_id) tuples, not tensors
            use_raw_descriptors = True
        else:
            logger.info(f"[Coalesce] DISABLED - transferring {matched_tensors} individual tensors")
            # Fall back to tensor list
            local_descs = local_tensor_list
            use_raw_descriptors = False
            coalesced_count = matched_tensors

        # Prepare transfer
        src_prepped = self._agent.prep_xfer_dlist(
            agent_name=remote_agent_name,
            xfer_list=remote_descs,
            mem_type="cuda",
            backends=["UCX"],
        )

        if use_raw_descriptors:
            # Use raw address descriptors for coalesced regions
            dst_prepped = self._agent.prep_xfer_dlist(
                agent_name="",
                xfer_list=local_descs,
                mem_type="cuda",
                backends=["UCX"],
            )
        else:
            # Use tensor objects
            dst_prepped = self._agent.prep_xfer_dlist(
                agent_name="",
                xfer_list=local_descs,
                mem_type="cuda",
                backends=["UCX"],
            )

        indices = list(range(len(remote_descs)))

        # Execute transfer
        handle = self._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=dst_prepped,
            local_indices=indices,
            remote_xfer_side=src_prepped,
            remote_indices=indices,
            backends=["UCX"],
        )
        self._agent.transfer(handle)

        # Wait for completion
        start_wait = time.perf_counter()
        while True:
            if timeout_seconds is not None and time.perf_counter() - start_wait >= timeout_seconds:
                self._agent.release_xfer_handle(handle)
                raise TimeoutError("Transfer timed out")

            status = self._agent.check_xfer_state(handle)
            if status in ("DONE", "SUCCESS"):
                self._agent.release_xfer_handle(handle)
                break
            if status in ("ERR", "ERROR", "FAIL"):
                self._agent.release_xfer_handle(handle)
                raise RuntimeError(f"Transfer failed with status {status}")
            time.sleep(0.001)

        # CRITICAL: Synchronize CUDA to ensure RDMA writes are visible
        # GPUDirect RDMA writes bypass CUDA streams, so we must sync
        torch.cuda.synchronize(self._device_id)

        duration = time.perf_counter() - start_time
        bandwidth_gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0.0

        if coalesce_transfers and coalesced_count < matched_tensors:
            logger.info(
                f"Transfer complete: {matched_tensors} tensors ({coalesced_count} regions), "
                f"{total_bytes / 1e9:.2f} GB in {duration:.2f}s "
                f"({bandwidth_gbps:.1f} Gbps)"
            )
        else:
            logger.info(
                f"Transfer complete: {matched_tensors} tensors, "
                f"{total_bytes / 1e9:.2f} GB in {duration:.2f}s "
                f"({bandwidth_gbps:.1f} Gbps)"
            )

        return total_bytes, matched_tensors, duration

    def _coalesce_transfers(
        self,
        remote_descs: list[tuple[int, int, int]],
        local_tensors: list[torch.Tensor],
    ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], int]:
        """
        Coalesce contiguous memory regions into larger transfer blocks.

        Model weights are often allocated contiguously in memory. By detecting
        adjacent regions and merging them, we reduce RDMA descriptor overhead
        from 1327 descriptors to potentially dozens.

        NIXL's prep_xfer_dlist accepts both tensor objects AND raw (addr, size, device_id)
        tuples. We use raw tuples for both sides to enable true coalescing.

        Args:
            remote_descs: List of (addr, size, device_id) tuples
            local_tensors: List of local tensors

        Returns:
            Tuple of (coalesced_remote_descs, coalesced_local_descs, count)
            Note: local_descs are now tuples, not tensors!
        """
        if len(remote_descs) <= 1:
            # Convert single tensor to descriptor
            if local_tensors:
                t = local_tensors[0]
                local_descs = [(t.data_ptr(), t.numel() * t.element_size(), self._device_id)]
            else:
                local_descs = []
            return remote_descs, local_descs, len(remote_descs)

        # Build indexed list with local tensor info
        # (remote_desc, local_addr, local_size)
        indexed = []
        for remote, local in zip(remote_descs, local_tensors, strict=True):
            local_addr = local.data_ptr()
            local_size = local.numel() * local.element_size()
            indexed.append((remote, local_addr, local_size))

        # Sort by remote address to find contiguous regions
        indexed.sort(key=lambda x: x[0][0])

        # Coalesce contiguous regions
        coalesced_remote = []
        coalesced_local = []

        i = 0
        while i < len(indexed):
            # Start a new region
            start_remote_addr = indexed[i][0][0]
            start_local_addr = indexed[i][1]
            current_remote_end = start_remote_addr + indexed[i][0][1]
            current_local_end = start_local_addr + indexed[i][2]
            device_id = indexed[i][0][2]

            # Try to extend by checking next tensors
            j = i + 1
            while j < len(indexed):
                next_remote_addr = indexed[j][0][0]
                next_remote_size = indexed[j][0][1]
                next_local_addr = indexed[j][1]
                next_local_size = indexed[j][2]
                next_device = indexed[j][0][2]

                # Check if both remote AND local are contiguous
                # Strict check: no gaps allowed for RDMA correctness
                remote_contiguous = (next_remote_addr == current_remote_end)
                local_contiguous = (next_local_addr == current_local_end)
                same_device = (next_device == device_id)

                if remote_contiguous and local_contiguous and same_device:
                    # Extend region
                    current_remote_end = next_remote_addr + next_remote_size
                    current_local_end = next_local_addr + next_local_size
                    j += 1
                else:
                    break

            # Calculate total region sizes
            total_remote_size = current_remote_end - start_remote_addr
            total_local_size = current_local_end - start_local_addr

            # Add coalesced region descriptors
            coalesced_remote.append((start_remote_addr, total_remote_size, device_id))
            coalesced_local.append((start_local_addr, total_local_size, self._device_id))

            i = j

        # Log coalescing results
        original_count = len(remote_descs)
        coalesced_count = len(coalesced_remote)
        if coalesced_count < original_count:
            reduction_pct = 100 * (1 - coalesced_count / original_count)
            logger.info(
                f"Coalesced {original_count} tensors into {coalesced_count} regions "
                f"({reduction_pct:.1f}% reduction in descriptors)"
            )

        return coalesced_remote, coalesced_local, coalesced_count

    def is_healthy(self) -> bool:
        """Check if the NIXL agent is initialized and has registered metadata."""
        return self._agent is not None and len(self._metadata) > 0

    def shutdown(self) -> None:
        """Clean up NIXL resources."""
        self._agent = None
        self._metadata = b""
        self._tensor_descriptors.clear()
        self._tensors.clear()
        logger.info("NixlTransferManager shutdown complete")
