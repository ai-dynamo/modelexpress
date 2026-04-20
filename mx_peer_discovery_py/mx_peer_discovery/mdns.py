# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multicast DNS service discovery.

Thin wrapper around the :mod:`zeroconf` library that provides a single
peer-oriented abstraction: register our own service once, continuously
browse for peers of the same service type, and emit a callback for each
resolved peer.

Default service type: ``_mx-peer._tcp.local.`` (note the trailing dot -
zeroconf requires it).

This implementation uses standard RFC 6763 SRV+A records for host and
port, and TXT purely for user metadata, matching the companion Rust
implementation (which wraps :crate:`mdns-sd`). The two halves interoperate
on the same LAN.
"""

import asyncio
import logging
import random
import socket
import string
from collections.abc import Callable
from dataclasses import dataclass, field

from zeroconf import ServiceInfo, ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

log = logging.getLogger(__name__)

DEFAULT_SERVICE_TYPE = "_mx-peer._tcp.local."

OnResolved = Callable[[str, int, list[str], dict[str, str]], None]


@dataclass
class Config:
    """Configuration for :class:`MdnsDiscovery`.

    Mirrors the Rust ``Config`` struct field-for-field so cross-language
    callers describe their intent the same way.
    """

    hostname: str
    """Hostname advertised in the SRV record (typically ``"<instance>.local."``)."""

    ip: str
    """Single IP address advertised in the A record."""

    port: int
    """Port advertised in the SRV record."""

    on_resolved: OnResolved
    """Invoked for each resolved peer (excluding ourselves).

    Arguments: ``(instance_name, port, addresses, txt_properties)``.
    """

    txt: dict[str, str] = field(default_factory=dict)
    """Free-form metadata advertised in TXT records. Keys and values are
    limited to 255 bytes combined by the mDNS spec."""

    service_type: str = DEFAULT_SERVICE_TYPE
    """Service type (must end with ``.local.``)."""

    instance_name: str | None = None
    """Instance name (the label prefix on the service-type FQDN). Defaults
    to a random alphanumeric string when ``None``."""


class MdnsDiscovery:
    """A live mDNS presence: advertises one service and browses for peers.

    Call :meth:`start` to register and begin discovery, :meth:`stop` to
    shut down. Instances are not reusable across stop/start cycles.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._instance_name = config.instance_name or _random_instance_name()
        self._fullname = f"{self._instance_name}.{config.service_type}"
        self._zc: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None

    @property
    def instance_name(self) -> str:
        """The resolved instance label (random if not explicitly provided)."""
        return self._instance_name

    async def start(self) -> None:
        """Register our service and start browsing for peers."""
        if self._zc is not None:
            raise RuntimeError("MdnsDiscovery already started")

        self._zc = AsyncZeroconf()

        info = ServiceInfo(
            type_=self._config.service_type,
            name=self._fullname,
            addresses=[socket.inet_aton(self._config.ip)],
            port=self._config.port,
            properties=self._encoded_properties(),
            server=self._config.hostname,
        )
        await self._zc.async_register_service(info)

        self._browser = AsyncServiceBrowser(
            self._zc.zeroconf,
            self._config.service_type,
            handlers=[self._on_state_change],
        )

    async def stop(self) -> None:
        """Unregister our service and close the zeroconf handle."""
        if self._browser is not None:
            await self._browser.async_cancel()
            self._browser = None
        if self._zc is not None:
            await self._zc.async_close()
            self._zc = None

    def _encoded_properties(self) -> dict[bytes, bytes]:
        """Zeroconf wants bytes on the wire, so encode up front."""
        return {
            k.encode("utf-8"): v.encode("utf-8") for k, v in self._config.txt.items()
        }

    def _on_state_change(
        self,
        zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        """ServiceBrowser handler. Dispatches resolution on a background task."""
        if state_change != ServiceStateChange.Added:
            return
        if name == self._fullname:
            return  # self-filter

        asyncio.ensure_future(self._resolve_and_dispatch(service_type, name))

    async def _resolve_and_dispatch(self, service_type: str, name: str) -> None:
        """Resolve a discovered service and invoke the user callback."""
        if self._zc is None:
            return

        info = AsyncServiceInfo(service_type, name)
        resolved = await info.async_request(self._zc.zeroconf, timeout=3000)
        if not resolved:
            log.debug(f"mdns: resolution timed out for {name}")
            return

        addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
        port = info.port or 0
        txt = _decode_properties(info.properties or {})
        instance_label = _instance_label_from_fullname(name, service_type) or name

        try:
            self._config.on_resolved(instance_label, port, addresses, txt)
        except Exception as e:
            log.warning(f"on_resolved callback raised: {e}")


def _decode_properties(props: dict[bytes, bytes | None]) -> dict[str, str]:
    """Decode zeroconf's bytes-keyed property map back into a string dict.

    Zero-length values and None-valued keys are preserved as empty strings.
    Non-UTF-8 bytes are replaced rather than raising: mDNS TXT values are
    untrusted input.
    """
    result: dict[str, str] = {}
    for raw_key, raw_val in props.items():
        key = raw_key.decode("utf-8", errors="replace")
        val = raw_val.decode("utf-8", errors="replace") if raw_val else ""
        result[key] = val
    return result


def _instance_label_from_fullname(fullname: str, service_type: str) -> str | None:
    """Strip the service-type suffix to recover the bare instance label.

    Returns ``None`` when ``fullname`` does not end with the service type.
    """
    suffix = f".{service_type}"
    if fullname.endswith(suffix):
        return fullname[: -len(suffix)]
    return None


def _random_instance_name() -> str:
    """Generate a 32-character lowercase alphanumeric instance name."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=32))
