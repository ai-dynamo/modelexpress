# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in smoke check for a caller-provided Mooncake artifact cluster."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from modelexpress.metadata import mooncake_artifact_store as mc


# Representative chunk sizes observed while publishing vLLM artifacts.  The
# small smoke payload above does not exercise the same native RDMA path or
# timeout behaviour as real cache contents.
_ARTIFACT_PAYLOAD_SIZES = (
    10 * 1024,
    7_178_240,
    10_332_160,
    18_247_680,
)


if os.getenv("MX_RUN_MOONCAKE_INTEGRATION") != "1":
    pytest.skip(
        "set MX_RUN_MOONCAKE_INTEGRATION=1 to use a preconfigured Mooncake store",
        allow_module_level=True,
    )


@pytest.mark.slow
def test_real_mooncake_store_put_get_and_missing_key():
    prefix = f"modelexpress-test/{uuid4().hex}"
    payload = b"modelexpress-mooncake-smoke"
    key = f"{prefix}/present"
    with mc._store_session() as store:
        try:
            assert store.put_bytes(key, payload) == 0
            assert store.get_bytes(key, expected_size=len(payload)) == payload
            assert (
                store.get_bytes(f"{prefix}/missing", expected_size=len(payload))
                is None
            )
        finally:
            mc._remove_object_with_retry(store, key)


@pytest.mark.slow
def test_real_mooncake_store_one_object_with_different_buffer_layouts():
    """Publish and fetch one object with different source/destination slices."""
    key = f"modelexpress-test/{uuid4().hex}/multi-buffer"
    source = tuple(mc._bytes_to_tensor(part) for part in (b"abc", b"defgh", b"ijk"))
    with mc._store_session() as store:
        try:
            assert store.put_buffers(key, source, soft_pin=True) == 0
            fetched = store.get_buffers(key, [4, 7])
            assert fetched is not None
            assert b"".join(mc._tensor_to_bytes(part) for part in fetched) == (
                b"abcdefghijk"
            )
        finally:
            mc._remove_object_with_retry(store, key)


@pytest.mark.slow
@pytest.mark.parametrize("payload_size", _ARTIFACT_PAYLOAD_SIZES)
def test_real_mooncake_store_artifact_sized_put_get(payload_size):
    """Exercise the production envelope + payload multi-buffer RDMA shape."""
    key = f"modelexpress-test/{uuid4().hex}/artifact-{payload_size}"
    envelope_size = mc._ARTIFACT_ENVELOPE_BYTES
    envelope = b"m" * envelope_size
    pattern = b"modelexpress-mooncake-artifact-integration\0"
    payload = (pattern * ((payload_size // len(pattern)) + 1))[:payload_size]
    source = (mc._bytes_to_tensor(envelope), mc._bytes_to_tensor(payload))

    with mc._store_session() as store:
        try:
            assert store.put_buffers(key, source, soft_pin=True) == 0
            fetched = store.get_buffers(key, [envelope_size, payload_size])
            assert fetched is not None
            assert mc._tensor_to_bytes(fetched[0]) == envelope
            assert mc._tensor_to_bytes(fetched[1]) == payload
        finally:
            # Soft-pinned objects may briefly return OBJECT_HAS_LEASE.  Match
            # the production cleanup behaviour instead of treating that as a
            # transfer failure.
            mc._remove_object_with_retry(store, key)


@pytest.mark.slow
def test_real_mooncake_store_keeps_first_value_for_duplicate_key():
    key = f"modelexpress-test/{uuid4().hex}/first-writer-wins"
    first = b"first-value"
    second = b"a-different-second-value"
    with mc._store_session() as store:
        try:
            assert store.put_bytes(key, first) == 0
            # Mooncake maps OBJECT_ALREADY_EXISTS to success for batch put.
            assert store.put_bytes(key, second) == 0
            assert store.get_bytes(key) == first
        finally:
            mc._remove_object_with_retry(store, key)


@pytest.mark.slow
def test_real_mooncake_store_reclaims_expired_repair_marker(monkeypatch):
    cache_key = f"modelexpress-test/{uuid4().hex}/artifact"
    repair_key = mc._repair_key(cache_key)
    created_at = 1000.0
    monkeypatch.setattr(mc.time, "time", lambda: created_at)
    stale_lock = mc._encode_repair_lock("dead-owner")
    monkeypatch.setattr(
        mc.time,
        "time",
        lambda: created_at + mc._REPAIR_LOCK_STALE_SECONDS + 1,
    )
    token = "replacement-owner"
    replacement_lock = mc._encode_repair_lock(token)

    with mc._store_session() as store:
        try:
            assert store.put_bytes(repair_key, stale_lock, soft_pin=True) == 0
            assert mc._acquire_repair_lock(
                store,
                repair_key,
                replacement_lock,
                token=token,
                transfer_name="integration-test",
                cache_key=cache_key,
            )
            observed = store.get_bytes(repair_key)
            decoded = mc._decode_repair_lock(observed) if observed is not None else None
            assert decoded is not None
            assert decoded.token == token
        finally:
            mc._remove_object_with_retry(store, repair_key)


@pytest.mark.slow
def test_real_mooncake_store_created_on_main_thread_and_used_by_publisher_thread():
    """Match the worker-query then PublisherThread native-store lifecycle."""
    prefix = f"modelexpress-test/{uuid4().hex}"
    missing_key = f"{prefix}/query-miss"
    artifact_key = f"{prefix}/publisher-artifact"
    payload_size = 10_332_160
    pattern = b"modelexpress-mooncake-cross-thread\0"
    payload = (pattern * ((payload_size // len(pattern)) + 1))[:payload_size]

    # Artifact lookup initializes the process-local store on the worker's main
    # thread before the asynchronous publisher is started.
    with mc._store_session() as store:
        assert store.get_bytes(missing_key, expected_size=1) is None

    def publish_from_background_thread():
        with mc._store_session() as store:
            try:
                put_result = store.put_bytes(artifact_key, payload)
                fetched = (
                    store.get_bytes(artifact_key, expected_size=payload_size)
                    if put_result == 0
                    else None
                )
                return put_result, fetched
            finally:
                mc._remove_object_with_retry(store, artifact_key)

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="PublisherThread",
    ) as pool:
        put_result, fetched = pool.submit(publish_from_background_thread).result()

    assert put_result == 0
    assert fetched == payload
