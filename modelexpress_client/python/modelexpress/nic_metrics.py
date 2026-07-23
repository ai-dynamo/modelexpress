# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-side RDMA NIC utilization sampling for load-aware source selection.

A source worker measures how busy its own RDMA NIC is and publishes the ratio
as ``source_load`` (see ``source_selection.LoadAwareSelector``). This is the
default ``source_load`` provider; ``make_source_load_provider`` is the seam
where an inference-runtime provider could replace it. The signal is
self-described source metadata -- the source reads its own InfiniBand port
counters -- so the MX server stays stateless: it only passes the number
through, it never computes or accumulates it.

Utilization is a rolling estimate: each call reads the port byte counters and
divides the delta since the previous call by the elapsed time and the link
capacity. Driving it from the publisher heartbeat means the sampling window is
the heartbeat interval, with no extra sleeps. Everything is best-effort: any
error (no RDMA device, unreadable sysfs, unparseable rate) yields ``0.0``, which
makes ``load_aware`` collapse to ``rendezvous_hash`` rather than fail.

Counter path: ``/sys/class/infiniband/<dev>/ports/<port>/counters/{port_xmit_data,
port_rcv_data}``. Per the IB spec these are in units of 4 octets, so bytes =
value * 4. Link capacity comes from ``.../ports/<port>/rate`` (e.g.
``"400 Gb/sec (4X HDR)"``).

Deployment caveat: with an SR-IOV Virtual Function (the common ``rdma/ib``
device-plugin setup in Kubernetes) the container sees only the VF, whose
``ports/<port>/`` sysfs exposes basic attributes (rate/state/gids) but *not*
the ``counters/`` (or ``hw_counters/``) statistics -- those live on the host
Physical Function and are not projected into the pod. There the sampler reads a
nonexistent path and yields ``0.0``, so this provider is inert and the runtime
provider (``MX_P2P_RUNTIME_METRICS_URL``, vLLM/SGLang/Dynamo serving load)
becomes the effective ``source_load`` signal. (A netlink-based reader --
``RDMA_NLDEV_CMD_STAT_GET`` -- would recover the counter in-VF; left as a
follow-up.)
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Callable, Optional

logger = logging.getLogger("modelexpress.nic_metrics")

_IB_SYSFS = "/sys/class/infiniband"
# IB port_xmit_data / port_rcv_data count 4-octet words (InfiniBand spec).
_COUNTER_WORD_BYTES = 4


def list_all_ib_devices() -> list[str]:
    """All InfiniBand/RoCE device names on this node, or [] if none/unreadable."""
    try:
        return sorted(os.listdir(_IB_SYSFS))
    except Exception:
        return []


def resolve_ib_device(device_id: int) -> Optional[str]:
    """Return the InfiniBand device name (e.g. ``"mlx5_3"``) for a GPU index.

    Reuses the same PCIe-affinity NIC assignment the transfer path pins to
    (``ucx_utils.probe_nic_pin_for_device`` returns ``"<nic>:1"``), so the
    counters we read belong to the NIC that actually carries transfers. Returns
    ``None`` if it cannot be resolved -- the caller then reports 0 utilization.
    """
    try:
        from . import ucx_utils

        pinned = ucx_utils.probe_nic_pin_for_device(device_id)
        if not pinned:
            return None
        # probe returns "<nic_name>:<port>"; we want the device name.
        return pinned.split(":", 1)[0] or None
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not resolve IB device for GPU %s: %s", device_id, e)
        return None


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _parse_rate_bytes_per_sec(rate_str: str) -> Optional[float]:
    """Parse an IB ``rate`` file (e.g. ``"400 Gb/sec (4X HDR)"``) to bytes/sec."""
    m = re.search(r"([0-9.]+)\s*Gb/sec", rate_str)
    if not m:
        return None
    gbps = float(m.group(1))
    return gbps * 1e9 / 8.0  # Gb/s -> bytes/s (full-duplex, per direction)


class NicUtilizationSampler:
    """Rolling RDMA NIC utilization sampler for one source worker.

    ``sample()`` returns the busier of the TX/RX directions as a fraction of
    link capacity in ``[0, 1]``, computed from the counter delta since the
    previous call. The first call establishes a baseline and returns 0.0.
    Constructed lazily and defensively: if the device or link rate cannot be
    read, ``sample()`` always returns 0.0.
    """

    def __init__(
        self,
        device: Optional[str],
        port: int = 1,
        *,
        _reader: Optional[Callable[[str], Optional[int]]] = None,
        _clock: Callable[[], float] = time.monotonic,
        _link_bytes_per_sec: Optional[float] = None,
    ) -> None:
        self._device = device
        self._port = port
        self._read_int = _reader or _read_int
        self._clock = _clock
        self._link_bps = _link_bytes_per_sec
        if self._link_bps is None and device is not None:
            self._link_bps = self._read_link_bps()
        self._last_t: Optional[float] = None
        self._last_bytes: Optional[tuple[int, int]] = None

    def _base(self) -> str:
        return f"{_IB_SYSFS}/{self._device}/ports/{self._port}"

    def _read_link_bps(self) -> Optional[float]:
        try:
            with open(f"{self._base()}/rate") as f:
                return _parse_rate_bytes_per_sec(f.read().strip())
        except Exception:
            return None

    def _read_bytes(self) -> Optional[tuple[int, int]]:
        base = f"{self._base()}/counters"
        tx = self._read_int(f"{base}/port_xmit_data")
        rx = self._read_int(f"{base}/port_rcv_data")
        if tx is None or rx is None:
            return None
        return tx * _COUNTER_WORD_BYTES, rx * _COUNTER_WORD_BYTES

    def sample(self) -> float:
        """Return current NIC utilization in ``[0, 1]`` (0.0 on any failure)."""
        if self._device is None or not self._link_bps:
            return 0.0
        now = self._clock()
        cur = self._read_bytes()
        if cur is None:
            return 0.0
        prev_t, prev = self._last_t, self._last_bytes
        self._last_t, self._last_bytes = now, cur
        if prev is None or prev_t is None:
            return 0.0  # first sample: establish baseline
        dt = now - prev_t
        if dt <= 0:
            return 0.0
        # Counters are monotonic; guard against wrap/reset with max(0, .).
        tx_rate = max(0, cur[0] - prev[0]) / dt
        rx_rate = max(0, cur[1] - prev[1]) / dt
        util = max(tx_rate, rx_rate) / self._link_bps
        return min(1.0, max(0.0, util))


class SourceLoadSampler:
    """Source-load provider backed by RDMA NIC utilization.

    The primary signal is the GPU-affine NIC -- the rail this worker actually
    transfers over -- so it reflects exactly the contention a puller would hit.
    If the affine device cannot be resolved, it falls back to the busiest of
    all the node's RDMA NICs. With no RDMA NIC at all it reports 0.0, so
    ``load_aware`` collapses to ``rendezvous_hash``.
    """

    def __init__(
        self,
        device_id: int,
        *,
        _resolver: Callable[[int], Optional[str]] = resolve_ib_device,
        _lister: Callable[[], list[str]] = list_all_ib_devices,
        _sampler_factory: Callable[[Optional[str]], "NicUtilizationSampler"] = (
            NicUtilizationSampler
        ),
    ) -> None:
        affine = _resolver(device_id)
        devices = [affine] if affine else _lister()
        self._samplers = [_sampler_factory(d) for d in devices if d]

    def sample(self) -> float:
        """Return this source's load in ``[0, 1]`` (max over sampled NICs)."""
        if not self._samplers:
            return 0.0
        return max((s.sample() for s in self._samplers), default=0.0)


def make_source_load_provider(device_id: int) -> Callable[[], float]:
    """Return a zero-arg provider of this source's load in ``[0, 1]``.

    The seam for the source-load signal. It always includes the physical
    RDMA-NIC-utilization provider (runtime-agnostic; effective wherever the
    NIC's port-counter sysfs is visible -- it no-ops to ``0.0`` on SR-IOV VF
    pods, see the module docstring). When ``MX_P2P_RUNTIME_METRICS_URL`` is set, it also reads the
    co-located inference runtime (vLLM/SGLang serving load) and reports the
    **max** of the two -- so selection reacts to the NIC being physically hot
    *and* to an imminent serving spike the counter has not seen yet. Neither
    path touches the server, proto, or selector; the wire contract is just the
    normalized ``source_load`` value. Any provider error degrades to 0.0.
    """
    from . import envs

    nic = SourceLoadSampler(device_id).sample
    url = envs.MX_P2P_RUNTIME_METRICS_URL
    if not url:
        return nic

    from .runtime_load import RuntimeLoadProvider

    runtime = RuntimeLoadProvider(url).sample

    def blended() -> float:
        return max(nic(), runtime())

    return blended
