# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes Lease-based bootstrap coordination for a Kademlia DHT.

A fixed set of ``coordination.k8s.io/v1`` Lease objects lets DHT participants
self-organize a stable, pre-converged bootstrap anchor quorum with no dedicated
seed pods and no central server. Each node maps to exactly one slot by the low
bits of its ``peer_id`` and contests only that slot's Lease. Winners become
"anchors" and publish their dial multiaddr into the Lease annotations; every
node reads all slots to assemble the bootstrap peer set.

The apiserver is reached in-cluster using only the Python standard library
(``urllib`` + ``ssl`` + ``json``) so no third-party Kubernetes client or async
HTTP dependency is required. Blocking HTTP calls are executed off the event loop
via ``asyncio.to_thread`` so the class presents an async API. TLS always
verifies the apiserver against the mounted service-account CA bundle; the
service-account token is re-read from disk on each request (tokens rotate).
"""

import asyncio
import datetime
import json
import logging
import os
import ssl
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

# Default in-cluster service-account mount paths.
DEFAULT_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
DEFAULT_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
DEFAULT_NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

# Annotation keys used to publish anchor state into a Lease object.
ANNOTATION_MULTIADDR = "mx-dht/multiaddr"
ANNOTATION_CONVERGED = "mx-dht/converged"

# HTTP timeout (seconds) for a single apiserver request.
REQUEST_TIMEOUT = 10.0


def _rfc3339_microtime() -> str:
    """Return the current UTC time as a k8s ``MicroTime`` (RFC3339 microseconds).

    Example: ``2026-06-30T01:00:00.000000Z``. The Kubernetes ``MicroTime`` type
    expects microsecond precision and a ``Z`` (UTC) suffix.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _parse_rfc3339(value: str) -> datetime.datetime | None:
    """Parse a k8s RFC3339 / MicroTime timestamp into an aware datetime.

    Returns ``None`` if the value is missing or unparseable. Accepts both the
    ``Z`` suffix and explicit offsets, with or without fractional seconds.
    """
    if not value:
        return None
    text = value.strip()
    # datetime.fromisoformat handles "+00:00" but not a bare "Z" before 3.11.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        log.warning("Failed to parse RFC3339 timestamp: %r", value)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


class LeaseCoordinator:
    """Coordinate DHT bootstrap anchors via a fixed set of K8s Lease objects.

    A node maps to one slot (``slot_for``) and contests that slot's Lease
    (``claim``). Holders periodically ``renew`` and publish their dial
    ``multiaddr`` plus a ``converged`` flag into the Lease annotations. Any node
    can ``read_all`` slots to assemble the bootstrap peer set.
    """

    def __init__(
        self,
        *,
        name_prefix: str = "mx-dht-anchor",
        num_slots: int = 8,
        ttl_seconds: int = 15,
        namespace: str | None = None,
        apiserver: str | None = None,
        token_path: str = DEFAULT_TOKEN_PATH,
        ca_path: str = DEFAULT_CA_PATH,
        namespace_path: str = DEFAULT_NAMESPACE_PATH,
    ) -> None:
        if num_slots <= 0 or (num_slots & (num_slots - 1)) != 0:
            raise ValueError(f"num_slots must be a positive power of two, got {num_slots}")

        self.name_prefix = name_prefix
        self.num_slots = num_slots
        self.ttl_seconds = ttl_seconds
        self.token_path = token_path
        self.ca_path = ca_path
        self.namespace_path = namespace_path

        self.namespace = namespace if namespace is not None else self._read_namespace()
        self.apiserver = apiserver if apiserver is not None else self._default_apiserver()

    # -- Setup helpers --------------------------------------------------------

    def _read_namespace(self) -> str:
        """Read the pod namespace from the mounted service-account file."""
        try:
            with open(self.namespace_path, encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError as exc:
            raise ValueError(
                f"Could not read namespace from {self.namespace_path}: {exc}. "
                "Pass namespace=... explicitly when running outside a cluster."
            ) from exc

    @staticmethod
    def _default_apiserver() -> str:
        """Return the in-cluster apiserver base URL.

        Prefers the ``KUBERNETES_SERVICE_HOST`` / ``KUBERNETES_SERVICE_PORT_HTTPS``
        environment variables (injected into every pod), falling back to the
        cluster DNS name ``https://kubernetes.default.svc``.
        """
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS")
        if host:
            # Bracket IPv6 literals for URL correctness.
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            return f"https://{host}:{port}" if port else f"https://{host}"
        return "https://kubernetes.default.svc"

    def _read_token(self) -> str:
        """Re-read the service-account bearer token from disk.

        Tokens are re-read on every request because Kubernetes rotates
        projected service-account tokens in place.
        """
        with open(self.token_path, encoding="utf-8") as handle:
            return handle.read().strip()

    def _ssl_context(self) -> ssl.SSLContext:
        """Build a verifying TLS context pinned to the apiserver CA bundle."""
        return ssl.create_default_context(cafile=self.ca_path)

    # -- Naming ---------------------------------------------------------------

    def slot_for(self, peer_id: bytes) -> int:
        """Return the slot index a peer maps to from the low bits of its id."""
        if not peer_id:
            raise ValueError("peer_id must be non-empty")
        return peer_id[-1] & (self.num_slots - 1)

    def lease_name(self, slot: int) -> str:
        """Return the Lease object name for a slot."""
        return f"{self.name_prefix}-{slot}"

    def _lease_url(self, slot: int) -> str:
        """Return the apiserver URL for a specific slot's Lease object."""
        return (
            f"{self.apiserver}/apis/coordination.k8s.io/v1/namespaces/"
            f"{self.namespace}/leases/{self.lease_name(slot)}"
        )

    def _collection_url(self) -> str:
        """Return the apiserver URL for the Lease collection (POST create)."""
        return (
            f"{self.apiserver}/apis/coordination.k8s.io/v1/namespaces/"
            f"{self.namespace}/leases"
        )

    # -- HTTP -----------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        body: dict | None = None,
    ) -> tuple[int, dict | None]:
        """Perform one apiserver request (blocking).

        Returns ``(status_code, parsed_json_or_None)``. HTTP error responses
        (4xx/5xx) are returned as ``(status, parsed_body)`` rather than raised,
        so callers can branch on status without exception handling for the
        common 404/409 flows. Network / TLS / parse failures return
        ``(0, None)`` so a transient apiserver error never crashes the caller.
        """
        token = self._read_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        context = self._ssl_context()
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT, context=context) as response:
                status = response.status
                raw = response.read()
                return status, self._decode_json(raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read() if hasattr(exc, "read") else b""
            return exc.code, self._decode_json(raw)
        except (urllib.error.URLError, ssl.SSLError, OSError) as exc:
            log.warning("apiserver request failed (%s %s): %s", method, url, exc)
            return 0, None

    @staticmethod
    def _decode_json(raw: bytes | None) -> dict | None:
        """Decode a JSON response body, returning ``None`` on empty/invalid."""
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None

    async def _arequest(
        self,
        method: str,
        url: str,
        body: dict | None = None,
    ) -> tuple[int, dict | None]:
        """Async wrapper running the blocking request off the event loop."""
        return await asyncio.to_thread(self._request, method, url, body)

    # -- Lease object shaping -------------------------------------------------

    def _build_lease(
        self,
        slot: int,
        holder_identity: str,
        multiaddr: str,
        converged: bool,
        acquire_time: str,
        renew_time: str,
        lease_transitions: int,
        resource_version: str | None = None,
    ) -> dict:
        """Construct a Lease object body for create/update."""
        metadata: dict = {
            "name": self.lease_name(slot),
            "namespace": self.namespace,
            "annotations": {
                ANNOTATION_MULTIADDR: multiaddr,
                ANNOTATION_CONVERGED: "true" if converged else "false",
            },
        }
        if resource_version is not None:
            metadata["resourceVersion"] = resource_version
        return {
            "apiVersion": "coordination.k8s.io/v1",
            "kind": "Lease",
            "metadata": metadata,
            "spec": {
                "holderIdentity": holder_identity,
                "leaseDurationSeconds": self.ttl_seconds,
                "acquireTime": acquire_time,
                "renewTime": renew_time,
                "leaseTransitions": lease_transitions,
            },
        }

    def _is_expired(self, spec: dict, now: datetime.datetime) -> bool:
        """Return True if the Lease described by ``spec`` has expired."""
        renew_time = _parse_rfc3339(spec.get("renewTime", ""))
        if renew_time is None:
            # No renewTime -> treat as expired / claimable.
            return True
        duration = spec.get("leaseDurationSeconds", self.ttl_seconds)
        try:
            duration = int(duration)
        except (TypeError, ValueError):
            duration = self.ttl_seconds
        return (now - renew_time).total_seconds() > duration

    # -- Public API -----------------------------------------------------------

    async def claim(self, slot: int, holder_identity: str, multiaddr: str) -> bool:
        """Contest a slot's Lease; return True iff this node holds it afterward.

        Standard Lease leader-election:
        - Missing (404): create the Lease; win on 201. A create race (409) falls
          through to the update path.
        - Present but expired, or already held by me: PUT-update to me using the
          object's resourceVersion for optimistic concurrency. A 409 means
          someone else won the race -> False.
        - Present, not expired, held by someone else: -> False.
        """
        status, obj = await self._arequest("GET", self._lease_url(slot))

        if status == 404:
            created = await self._create(slot, holder_identity, multiaddr)
            if created is not None:
                return created
            # 409 create race: re-read and fall through to the update path.
            status, obj = await self._arequest("GET", self._lease_url(slot))

        if status != 200 or obj is None:
            log.warning("claim(slot=%d): unexpected GET status %d", slot, status)
            return False

        spec = obj.get("spec", {}) or {}
        current_holder = spec.get("holderIdentity")
        now = datetime.datetime.now(datetime.timezone.utc)
        expired = self._is_expired(spec, now)

        if current_holder == holder_identity:
            return await self._update(slot, obj, holder_identity, multiaddr, converged=False)
        if expired:
            return await self._update(slot, obj, holder_identity, multiaddr, converged=False)

        # Held by another live holder.
        return False

    async def _create(self, slot: int, holder_identity: str, multiaddr: str) -> bool | None:
        """POST-create a slot's Lease.

        Returns True on 201, False on non-201/non-409, and ``None`` on 409
        (AlreadyExists) to signal the caller to fall through to the update path.
        """
        now = _rfc3339_microtime()
        body = self._build_lease(
            slot,
            holder_identity,
            multiaddr,
            converged=False,
            acquire_time=now,
            renew_time=now,
            lease_transitions=0,
        )
        status, _ = await self._arequest("POST", self._collection_url(), body)
        if status == 201:
            log.info("claim(slot=%d): created Lease, now anchor", slot)
            return True
        if status == 409:
            log.debug("claim(slot=%d): create raced (409), retrying via update", slot)
            return None
        log.warning("claim(slot=%d): create failed with status %d", slot, status)
        return False

    async def _update(
        self,
        slot: int,
        obj: dict,
        holder_identity: str,
        multiaddr: str,
        converged: bool,
    ) -> bool:
        """PUT-update a slot's Lease to this holder using optimistic concurrency.

        Increments ``leaseTransitions`` only when the holder actually changes.
        Returns True on 200, False on 409 (lost the race) or other error.
        """
        metadata = obj.get("metadata", {}) or {}
        spec = obj.get("spec", {}) or {}
        resource_version = metadata.get("resourceVersion")
        previous_holder = spec.get("holderIdentity")

        try:
            prior_transitions = int(spec.get("leaseTransitions", 0) or 0)
        except (TypeError, ValueError):
            prior_transitions = 0
        transitions = prior_transitions
        if previous_holder != holder_identity:
            transitions = prior_transitions + 1

        # Preserve the original acquireTime unless the holder changed.
        acquire_time = spec.get("acquireTime") or _rfc3339_microtime()
        if previous_holder != holder_identity:
            acquire_time = _rfc3339_microtime()
        renew_time = _rfc3339_microtime()

        body = self._build_lease(
            slot,
            holder_identity,
            multiaddr,
            converged=converged,
            acquire_time=acquire_time,
            renew_time=renew_time,
            lease_transitions=transitions,
            resource_version=resource_version,
        )
        status, _ = await self._arequest("PUT", self._lease_url(slot), body)
        if status == 200:
            return True
        if status == 409:
            log.debug("update(slot=%d): lost race (409 conflict)", slot)
            return False
        log.warning("update(slot=%d): PUT failed with status %d", slot, status)
        return False

    async def renew(
        self,
        slot: int,
        holder_identity: str,
        multiaddr: str,
        converged: bool,
    ) -> bool:
        """Renew this node's hold on a slot and publish anchor state.

        Returns False (and logs a warning) if the slot is no longer held by this
        node (holder changed) or on a 409 conflict.
        """
        status, obj = await self._arequest("GET", self._lease_url(slot))
        if status != 200 or obj is None:
            log.warning("renew(slot=%d): GET returned status %d", slot, status)
            return False

        spec = obj.get("spec", {}) or {}
        current_holder = spec.get("holderIdentity")
        if current_holder != holder_identity:
            log.warning(
                "renew(slot=%d): holder changed (%r != %r); no longer anchor",
                slot,
                current_holder,
                holder_identity,
            )
            return False

        return await self._update(slot, obj, holder_identity, multiaddr, converged=converged)

    async def read_all(self) -> dict[int, dict]:
        """Read every slot; return live (existing, non-expired) slots.

        Returns ``{slot: {"holder": str, "multiaddr": str, "converged": bool,
        "expired": bool}}``. Missing (404) or expired slots are omitted.
        """
        result: dict[int, dict] = {}
        now = datetime.datetime.now(datetime.timezone.utc)
        for slot in range(self.num_slots):
            status, obj = await self._arequest("GET", self._lease_url(slot))
            if status != 200 or obj is None:
                continue
            spec = obj.get("spec", {}) or {}
            metadata = obj.get("metadata", {}) or {}
            annotations = metadata.get("annotations", {}) or {}
            expired = self._is_expired(spec, now)
            if expired:
                continue
            result[slot] = {
                "holder": spec.get("holderIdentity", ""),
                "multiaddr": annotations.get(ANNOTATION_MULTIADDR, ""),
                "converged": annotations.get(ANNOTATION_CONVERGED, "false") == "true",
                "expired": expired,
            }
        return result

    async def anchor_multiaddrs(self) -> list[str]:
        """Return the non-empty anchor multiaddrs (the bootstrap peer set)."""
        slots = await self.read_all()
        return [info["multiaddr"] for info in slots.values() if info.get("multiaddr")]

    async def all_slots_converged(self) -> bool:
        """True iff every slot is currently held, non-expired, and converged."""
        slots = await self.read_all()
        if len(slots) != self.num_slots:
            return False
        return all(info.get("converged") is True for info in slots.values())

    async def wait_all_converged(self, timeout: float, poll_interval: float = 1.0) -> bool:
        """Poll ``all_slots_converged`` until True or ``timeout``.

        Returns the final observed convergence state.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            if await self.all_slots_converged():
                return True
            if loop.time() >= deadline:
                return False
            remaining = deadline - loop.time()
            await asyncio.sleep(min(poll_interval, max(0.0, remaining)))
