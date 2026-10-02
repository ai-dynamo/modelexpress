# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-side RDMA NIC utilization sampling for load-aware source selection.

A source worker measures how busy its own RDMA NIC is and publishes the ratio
as ``source_load`` (see ``source_selection.LoadAwareSelector``). This is the
default ``source_load`` provider, and it is not the only one: setting
``MX_P2P_RUNTIME_METRICS_URL`` adds the co-located inference runtime's own
KV-cache utilization (``vllm:kv_cache_usage_perc``, ``sglang:token_usage``,
Dynamo's runtime-agnostic ``dynamo_component_gpu_cache_usage_percent``; see
``runtime_load.py``), and ``make_source_load_provider`` publishes the max of
the two. The serving signal is the more useful one on Kubernetes, where the
NIC counters are usually invisible -- see the SR-IOV caveat below. The signal is
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
Physical Function and are not projected into the pod.

The sampler therefore falls back to reading the same statistics over netlink
(``RDMA_NLDEV_CMD_STAT_GET``), which an unprivileged VF pod *can* do. That
recovers the counters but not, by default, the ones this module needs: mlx5's
default hw-counter set is requests and errors (``rx_write_requests``,
``rx_read_requests``, ...) with no byte totals. Byte totals exist only as
*optional* counters, and enabling them takes ``CAP_NET_ADMIN`` -- which a
worker pod does not hold, and which is a per-port setting shared with every
other tenant on the host. So the fallback yields a utilization figure only
where a cluster admin has already run, per device and port:

    rdma statistic set link <dev>/<port> optional-counters \
        rdma_tx_bytes,rdma_rx_bytes

Absent that, this provider still reports ``0.0`` on VF pods and the runtime
provider (``MX_P2P_RUNTIME_METRICS_URL``, vLLM/SGLang/Dynamo serving load) is
the effective ``source_load`` signal -- which is why that one, not this one,
is the signal to rely on under Kubernetes.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import struct
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


# NETLINK_RDMA constants, from <rdma/rdma_netlink.h>. An SR-IOV VF pod gets no
# `counters/` sysfs, but the same statistics are reachable over netlink, which
# needs no added capability to *read*.
_NETLINK_RDMA = 20
_NL_MSG_DEV_GET = (5 << 10) | 1  # RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, CMD_GET)
_NL_MSG_STAT_GET = (5 << 10) | 17  # RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, CMD_STAT_GET)
_NLA_DEV_INDEX = 1
_NLA_DEV_NAME = 2
_NLA_PORT_INDEX = 3
_NLA_HWCOUNTERS = 80
_NLA_HWCOUNTER_ENTRY = 81
_NLA_HWCOUNTER_ENTRY_NAME = 82
_NLA_HWCOUNTER_ENTRY_VALUE = 83
_NLM_F_REQUEST = 0x01
_NLM_F_ACK = 0x04
_NLM_F_DUMP = 0x300
_NLMSG_ERROR = 2
_NLMSG_DONE = 3

# mlx5 carries byte totals as *optional* counters: absent from the default set
# and enablable only with CAP_NET_ADMIN, which a worker pod does not hold. They
# appear here once a cluster admin turns them on, per port, on the host:
#   rdma statistic set link <dev>/<port> optional-counters \
#       rdma_tx_bytes,rdma_rx_bytes
# Until then this reader finds no byte counter and the caller degrades to 0.0.
_NL_TX_BYTES = "rdma_tx_bytes"
_NL_RX_BYTES = "rdma_rx_bytes"
_NL_TIMEOUT_SECS = 2.0


def _nl_align(n: int) -> int:
    return (n + 3) & ~3


def _nl_attr(attr_type: int, payload: bytes) -> bytes:
    length = 4 + len(payload)
    return (
        struct.pack("=HH", length, attr_type)
        + payload
        + b"\0" * (_nl_align(length) - length)
    )


def _nl_parse_attrs(buf: bytes) -> dict[int, list[bytes]]:
    out: dict[int, list[bytes]] = {}
    off = 0
    while off + 4 <= len(buf):
        alen, atype = struct.unpack_from("=HH", buf, off)
        if alen < 4 or off + alen > len(buf):
            break
        # Mask NLA_F_NESTED / NLA_F_NET_BYTEORDER out of the type.
        out.setdefault(atype & 0x3FFF, []).append(buf[off + 4 : off + alen])
        off += _nl_align(alen)
    return out


def _nl_messages(sock: "socket.socket"):
    """Yield (msg_type, payload) until NLMSG_DONE, an error, or a short read."""
    while True:
        data = sock.recv(65536)
        if not data:
            return
        off = 0
        while off + 16 <= len(data):
            mlen, mtype, _flags, _seq, _pid = struct.unpack_from("=IHHII", data, off)
            if mlen < 16 or off + mlen > len(data):
                return
            if mtype in (_NLMSG_DONE, _NLMSG_ERROR):
                return
            yield mtype, data[off + 16 : off + mlen]
            off += _nl_align(mlen)


def _nl_resolve_device_index(device: str) -> Optional[int]:
    """Map an IB device name to its netlink index.

    A VF pod has no ``/sys/class/infiniband/<dev>/index``, so the mapping has to
    come from the same netlink family.
    """
    try:
        with socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, _NETLINK_RDMA) as sock:
            sock.settimeout(_NL_TIMEOUT_SECS)
            sock.send(
                struct.pack(
                    "=IHHII", 16, _NL_MSG_DEV_GET, _NLM_F_REQUEST | _NLM_F_DUMP, 1, 0
                )
            )
            for _mtype, payload in _nl_messages(sock):
                attrs = _nl_parse_attrs(payload)
                name = attrs.get(_NLA_DEV_NAME, [b""])[0].rstrip(b"\0")
                index = attrs.get(_NLA_DEV_INDEX)
                if index and name.decode("utf-8", "replace") == device:
                    return int(struct.unpack("=I", index[0][:4])[0])
    except Exception:
        return None
    return None


def _nl_read_hw_counters(device: str, port: int) -> dict[str, int]:
    """Return this port's hw counters over netlink ({} on any failure)."""
    index = _nl_resolve_device_index(device)
    if index is None:
        return {}
    counters: dict[str, int] = {}
    try:
        with socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, _NETLINK_RDMA) as sock:
            sock.settimeout(_NL_TIMEOUT_SECS)
            body = _nl_attr(_NLA_DEV_INDEX, struct.pack("=I", index)) + _nl_attr(
                _NLA_PORT_INDEX, struct.pack("=I", port)
            )
            sock.send(
                struct.pack(
                    "=IHHII",
                    16 + len(body),
                    _NL_MSG_STAT_GET,
                    _NLM_F_REQUEST | _NLM_F_ACK,
                    1,
                    0,
                )
                + body
            )
            for _mtype, payload in _nl_messages(sock):
                for nest in _nl_parse_attrs(payload).get(_NLA_HWCOUNTERS, []):
                    entries = _nl_parse_attrs(nest).get(_NLA_HWCOUNTER_ENTRY, [])
                    for entry in entries:
                        fields = _nl_parse_attrs(entry)
                        name = (
                            fields.get(_NLA_HWCOUNTER_ENTRY_NAME, [b""])[0]
                            .rstrip(b"\0")
                            .decode("utf-8", "replace")
                        )
                        raw = fields.get(_NLA_HWCOUNTER_ENTRY_VALUE, [b""])[0]
                        if name and len(raw) >= 8:
                            counters[name] = int(struct.unpack("=Q", raw[:8])[0])
    except Exception:
        return {}
    return counters


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
        _nl_counters: Callable[[str, int], dict[str, int]] = _nl_read_hw_counters,
    ) -> None:
        self._device = device
        self._port = port
        self._read_int = _reader or _read_int
        self._clock = _clock
        self._nl_counters = _nl_counters
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
        if tx is not None and rx is not None:
            return tx * _COUNTER_WORD_BYTES, rx * _COUNTER_WORD_BYTES
        return self._read_bytes_netlink()

    def _read_bytes_netlink(self) -> Optional[tuple[int, int]]:
        """VF fallback for when ``counters/`` sysfs is not projected into the pod.

        Values are already byte totals, so unlike the sysfs counters they are not
        scaled by ``_COUNTER_WORD_BYTES``. Returns None unless an admin enabled
        the optional byte counters, which is the common case.
        """
        if self._device is None:
            return None
        counters = self._nl_counters(self._device, self._port)
        tx = counters.get(_NL_TX_BYTES)
        rx = counters.get(_NL_RX_BYTES)
        if tx is None or rx is None:
            return None
        return tx, rx

    def sample(self) -> Optional[float]:
        """Return current NIC utilization in ``[0, 1]``, or ``None`` with no reading.

        ``None`` is the honest answer whenever there is no counter to read (no
        device, unreadable link rate, VF pod without the optional byte counters)
        or no interval to divide by yet; publishing it as 0.0 would make the
        source look idle to every puller. Only a measured zero delta is 0.0.
        """
        if self._device is None or not self._link_bps:
            return None
        now = self._clock()
        cur = self._read_bytes()
        if cur is None:
            return None
        prev_t, prev = self._last_t, self._last_bytes
        self._last_t, self._last_bytes = now, cur
        if prev is None or prev_t is None:
            return None  # first sample: establish baseline
        dt = now - prev_t
        if dt <= 0:
            return None
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

    def sample(self) -> Optional[float]:
        """Return this source's load in ``[0, 1]``: max over NICs with a reading.

        ``None`` when no NIC produced one, so a fleet of VF pods does not all
        report 0.0 and look idle.
        """
        readings = [v for v in (s.sample() for s in self._samplers) if v is not None]
        return max(readings) if readings else None


def make_source_load_provider(device_id: int) -> Callable[[], Optional[float]]:
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

    def blended() -> Optional[float]:
        # Max over the providers that have a reading; None only when neither
        # does, so one silent provider cannot drag a real reading down to 0.0.
        readings = [v for v in (nic(), runtime()) if v is not None]
        return max(readings) if readings else None

    return blended
