# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for K8s Lease-based DHT bootstrap coordination.

The HTTP layer is mocked by monkeypatching ``LeaseCoordinator._request`` so no
real apiserver, TLS context, or service-account files are needed. Covers:
- slot_for low-bit masking and lease_name formatting
- claim: create-on-404, acquire-when-expired, lose-when-held-by-other,
  renew-when-held-by-me, 409-on-PUT loses, create-409 race falls through
- read_all omits missing and expired slots
- all_slots_converged only when every slot is held + converged
- wait_all_converged timeout path
- power-of-two num_slots validation
"""

import datetime
import logging

import pytest

from kademlite.k8s_lease import (
    ANNOTATION_CONVERGED,
    ANNOTATION_MULTIADDR,
    LeaseCoordinator,
    _parse_rfc3339,
    _rfc3339_microtime,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# -- Helpers ------------------------------------------------------------------


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _micro(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def make_coordinator(**kwargs) -> LeaseCoordinator:
    """Construct a LeaseCoordinator without touching the filesystem."""
    defaults = dict(
        namespace="test-ns",
        apiserver="https://apiserver.test",
    )
    defaults.update(kwargs)
    return LeaseCoordinator(**defaults)


def lease_obj(
    slot: int,
    coord: LeaseCoordinator,
    holder: str,
    *,
    renew_time: str,
    multiaddr: str = "",
    converged: bool = False,
    lease_duration: int | None = None,
    resource_version: str = "100",
    lease_transitions: int = 0,
) -> dict:
    """Build a Lease object body as the apiserver would return it."""
    return {
        "apiVersion": "coordination.k8s.io/v1",
        "kind": "Lease",
        "metadata": {
            "name": coord.lease_name(slot),
            "namespace": coord.namespace,
            "resourceVersion": resource_version,
            "annotations": {
                ANNOTATION_MULTIADDR: multiaddr,
                ANNOTATION_CONVERGED: "true" if converged else "false",
            },
        },
        "spec": {
            "holderIdentity": holder,
            "leaseDurationSeconds": lease_duration if lease_duration is not None else coord.ttl_seconds,
            "acquireTime": renew_time,
            "renewTime": renew_time,
            "leaseTransitions": lease_transitions,
        },
    }


class FakeApi:
    """Records requests and replays scripted responses.

    ``handler`` is a callable ``(method, url, body) -> (status, obj)``. This is
    installed as the coordinator's ``_request`` (the sync method that
    ``_arequest`` dispatches to a thread), so both sync and async paths use it.
    """

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, method, url, body=None):
        self.calls.append((method, url, body))
        return self.handler(method, url, body)


def install(coord: LeaseCoordinator, handler) -> FakeApi:
    fake = FakeApi(handler)
    coord._request = fake  # type: ignore[method-assign]
    return fake


# -- Timestamp helpers --------------------------------------------------------


def test_rfc3339_microtime_roundtrip():
    ts = _rfc3339_microtime()
    assert ts.endswith("Z")
    parsed = _parse_rfc3339(ts)
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_parse_rfc3339_none_on_junk():
    assert _parse_rfc3339("") is None
    assert _parse_rfc3339("not-a-time") is None


# -- Construction / validation ------------------------------------------------


def test_num_slots_must_be_power_of_two():
    for bad in (0, 3, 5, 6, 7, 9, -8):
        with pytest.raises(ValueError):
            make_coordinator(num_slots=bad)


def test_num_slots_power_of_two_ok():
    for good in (1, 2, 4, 8, 16, 32):
        coord = make_coordinator(num_slots=good)
        assert coord.num_slots == good


def test_default_apiserver_from_env(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT_HTTPS", "6443")
    coord = LeaseCoordinator(namespace="ns")
    assert coord.apiserver == "https://10.0.0.1:6443"


def test_default_apiserver_fallback(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_PORT_HTTPS", raising=False)
    coord = LeaseCoordinator(namespace="ns")
    assert coord.apiserver == "https://kubernetes.default.svc"


# -- slot_for / lease_name ----------------------------------------------------


def test_slot_for_masking():
    coord = make_coordinator(num_slots=8)
    # 8 slots -> mask 0b111 -> last byte low 3 bits.
    assert coord.slot_for(b"\x00") == 0
    assert coord.slot_for(b"\x07") == 7
    assert coord.slot_for(b"\x08") == 0
    assert coord.slot_for(b"\xff") == 7
    assert coord.slot_for(b"\xab\xcd\x05") == 5


def test_slot_for_masking_16():
    coord = make_coordinator(num_slots=16)
    assert coord.slot_for(b"\xff") == 15
    assert coord.slot_for(b"\x10") == 0
    assert coord.slot_for(b"\x1a") == 10


def test_slot_for_empty_raises():
    coord = make_coordinator()
    with pytest.raises(ValueError):
        coord.slot_for(b"")


def test_lease_name():
    coord = make_coordinator(name_prefix="mx-dht-anchor")
    assert coord.lease_name(0) == "mx-dht-anchor-0"
    assert coord.lease_name(7) == "mx-dht-anchor-7"


# -- claim --------------------------------------------------------------------


async def test_claim_on_404_creates():
    coord = make_coordinator()

    def handler(method, url, body):
        if method == "GET":
            return 404, None
        if method == "POST":
            assert body["spec"]["holderIdentity"] == "me"
            assert body["metadata"]["annotations"][ANNOTATION_MULTIADDR] == "/ip4/1.2.3.4/tcp/5"
            return 201, body
        raise AssertionError(f"unexpected {method}")

    fake = install(coord, handler)
    won = await coord.claim(3, "me", "/ip4/1.2.3.4/tcp/5")
    assert won is True
    methods = [c[0] for c in fake.calls]
    assert methods == ["GET", "POST"]


async def test_claim_when_expired_acquires():
    coord = make_coordinator(ttl_seconds=15)
    old = _micro(_now() - datetime.timedelta(seconds=120))

    def handler(method, url, body):
        if method == "GET":
            return 200, lease_obj(2, coord, "old-holder", renew_time=old, resource_version="42")
        if method == "PUT":
            assert body["spec"]["holderIdentity"] == "me"
            # Holder changed -> transitions incremented.
            assert body["spec"]["leaseTransitions"] == 1
            assert body["metadata"]["resourceVersion"] == "42"
            return 200, body
        raise AssertionError(f"unexpected {method}")

    install(coord, handler)
    won = await coord.claim(2, "me", "/ip4/9.9.9.9/tcp/1")
    assert won is True


async def test_claim_when_held_by_other_loses():
    coord = make_coordinator(ttl_seconds=15)
    fresh = _micro(_now())

    def handler(method, url, body):
        if method == "GET":
            return 200, lease_obj(1, coord, "someone-else", renew_time=fresh)
        raise AssertionError(f"should not write: {method}")

    fake = install(coord, handler)
    won = await coord.claim(1, "me", "/ip4/1.1.1.1/tcp/1")
    assert won is False
    assert [c[0] for c in fake.calls] == ["GET"]


async def test_claim_when_held_by_me_renews():
    coord = make_coordinator(ttl_seconds=15)
    fresh = _micro(_now())

    def handler(method, url, body):
        if method == "GET":
            return 200, lease_obj(
                0, coord, "me", renew_time=fresh, resource_version="7", lease_transitions=3
            )
        if method == "PUT":
            assert body["spec"]["holderIdentity"] == "me"
            # Same holder -> transitions preserved (not incremented).
            assert body["spec"]["leaseTransitions"] == 3
            return 200, body
        raise AssertionError(f"unexpected {method}")

    install(coord, handler)
    won = await coord.claim(0, "me", "/ip4/2.2.2.2/tcp/2")
    assert won is True


async def test_claim_409_on_put_loses():
    coord = make_coordinator(ttl_seconds=15)
    old = _micro(_now() - datetime.timedelta(seconds=120))

    def handler(method, url, body):
        if method == "GET":
            return 200, lease_obj(4, coord, "old", renew_time=old)
        if method == "PUT":
            return 409, {"reason": "Conflict"}
        raise AssertionError(f"unexpected {method}")

    install(coord, handler)
    won = await coord.claim(4, "me", "/ip4/3.3.3.3/tcp/3")
    assert won is False


async def test_claim_create_409_race_falls_through_to_update():
    coord = make_coordinator(ttl_seconds=15)
    old = _micro(_now() - datetime.timedelta(seconds=120))
    state = {"created": False}

    def handler(method, url, body):
        if method == "GET":
            if not state["created"]:
                return 404, None
            # After the create race, the object now exists (expired holder).
            return 200, lease_obj(5, coord, "racer", renew_time=old, resource_version="55")
        if method == "POST":
            state["created"] = True
            return 409, {"reason": "AlreadyExists"}
        if method == "PUT":
            assert body["metadata"]["resourceVersion"] == "55"
            return 200, body
        raise AssertionError(f"unexpected {method}")

    fake = install(coord, handler)
    won = await coord.claim(5, "me", "/ip4/4.4.4.4/tcp/4")
    assert won is True
    assert [c[0] for c in fake.calls] == ["GET", "POST", "GET", "PUT"]


async def test_claim_get_transient_error_returns_false():
    coord = make_coordinator()

    def handler(method, url, body):
        # 0 status == network/TLS failure surfaced by _request.
        return 0, None

    install(coord, handler)
    won = await coord.claim(0, "me", "/ip4/1/tcp/1")
    assert won is False


# -- renew --------------------------------------------------------------------


async def test_renew_success():
    coord = make_coordinator(ttl_seconds=15)
    fresh = _micro(_now())

    def handler(method, url, body):
        if method == "GET":
            return 200, lease_obj(0, coord, "me", renew_time=fresh, resource_version="9")
        if method == "PUT":
            assert body["metadata"]["annotations"][ANNOTATION_CONVERGED] == "true"
            assert body["metadata"]["annotations"][ANNOTATION_MULTIADDR] == "/ip4/5/tcp/5"
            return 200, body
        raise AssertionError(f"unexpected {method}")

    install(coord, handler)
    ok = await coord.renew(0, "me", "/ip4/5/tcp/5", converged=True)
    assert ok is True


async def test_renew_holder_changed_fails():
    coord = make_coordinator(ttl_seconds=15)
    fresh = _micro(_now())

    def handler(method, url, body):
        if method == "GET":
            return 200, lease_obj(0, coord, "someone-else", renew_time=fresh)
        raise AssertionError(f"should not write: {method}")

    install(coord, handler)
    ok = await coord.renew(0, "me", "/ip4/5/tcp/5", converged=True)
    assert ok is False


# -- read_all -----------------------------------------------------------------


async def test_read_all_omits_expired_and_missing():
    coord = make_coordinator(num_slots=4, ttl_seconds=15)
    fresh = _micro(_now())
    old = _micro(_now() - datetime.timedelta(seconds=120))

    def handler(method, url, body):
        assert method == "GET"
        if url.endswith("-0"):
            return 200, lease_obj(0, coord, "h0", renew_time=fresh, multiaddr="/ip4/10/tcp/1", converged=True)
        if url.endswith("-1"):
            return 404, None  # missing -> omitted
        if url.endswith("-2"):
            return 200, lease_obj(2, coord, "h2", renew_time=old, multiaddr="/ip4/20/tcp/2")  # expired -> omitted
        if url.endswith("-3"):
            return 200, lease_obj(3, coord, "h3", renew_time=fresh, multiaddr="/ip4/30/tcp/3", converged=False)
        raise AssertionError(url)

    install(coord, handler)
    result = await coord.read_all()
    assert set(result.keys()) == {0, 3}
    assert result[0]["holder"] == "h0"
    assert result[0]["multiaddr"] == "/ip4/10/tcp/1"
    assert result[0]["converged"] is True
    assert result[3]["converged"] is False
    assert result[0]["expired"] is False


async def test_anchor_multiaddrs_non_empty_only():
    coord = make_coordinator(num_slots=2, ttl_seconds=15)
    fresh = _micro(_now())

    def handler(method, url, body):
        if url.endswith("-0"):
            return 200, lease_obj(0, coord, "h0", renew_time=fresh, multiaddr="/ip4/10/tcp/1")
        if url.endswith("-1"):
            return 200, lease_obj(1, coord, "h1", renew_time=fresh, multiaddr="")  # empty -> filtered
        raise AssertionError(url)

    install(coord, handler)
    addrs = await coord.anchor_multiaddrs()
    assert addrs == ["/ip4/10/tcp/1"]


# -- convergence --------------------------------------------------------------


async def test_all_slots_converged_true_when_all_held_and_converged():
    coord = make_coordinator(num_slots=2, ttl_seconds=15)
    fresh = _micro(_now())

    def handler(method, url, body):
        slot = 0 if url.endswith("-0") else 1
        return 200, lease_obj(slot, coord, f"h{slot}", renew_time=fresh, multiaddr=f"/a/{slot}", converged=True)

    install(coord, handler)
    assert await coord.all_slots_converged() is True


async def test_all_slots_converged_false_when_one_not_converged():
    coord = make_coordinator(num_slots=2, ttl_seconds=15)
    fresh = _micro(_now())

    def handler(method, url, body):
        if url.endswith("-0"):
            return 200, lease_obj(0, coord, "h0", renew_time=fresh, converged=True)
        return 200, lease_obj(1, coord, "h1", renew_time=fresh, converged=False)

    install(coord, handler)
    assert await coord.all_slots_converged() is False


async def test_all_slots_converged_false_when_slot_missing():
    coord = make_coordinator(num_slots=2, ttl_seconds=15)
    fresh = _micro(_now())

    def handler(method, url, body):
        if url.endswith("-0"):
            return 200, lease_obj(0, coord, "h0", renew_time=fresh, converged=True)
        return 404, None

    install(coord, handler)
    assert await coord.all_slots_converged() is False


async def test_wait_all_converged_timeout():
    coord = make_coordinator(num_slots=1, ttl_seconds=15)
    fresh = _micro(_now())

    def handler(method, url, body):
        # Held but never converged -> never satisfies the predicate.
        return 200, lease_obj(0, coord, "h0", renew_time=fresh, converged=False)

    install(coord, handler)
    ok = await coord.wait_all_converged(timeout=0.3, poll_interval=0.05)
    assert ok is False


async def test_wait_all_converged_success():
    coord = make_coordinator(num_slots=1, ttl_seconds=15)
    fresh = _micro(_now())
    state = {"polls": 0}

    def handler(method, url, body):
        state["polls"] += 1
        converged = state["polls"] >= 2
        return 200, lease_obj(0, coord, "h0", renew_time=fresh, converged=converged)

    install(coord, handler)
    ok = await coord.wait_all_converged(timeout=2.0, poll_interval=0.05)
    assert ok is True
