# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
import tracemalloc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest
from botocore.exceptions import ClientError

from modelexpress_rl.s3 import S3Client


class _Backend:
    def __init__(self, data):
        self.data = data
        self.etag = '"original"'
        self.requests = []
        self.failures = Counter()
        self.short_reads = Counter()
        self.active = 0
        self.peak_active = 0
        self.closed = 0
        self.lock = threading.Lock()
        self.delay = 0
        self.change = False
        self.ignore_range = False
        self.wrong_range = False
        self.gate = None

    def head_object(self, **request):
        return {"ContentLength": len(self.data), "ETag": self.etag}

    def get_object(self, **request):
        with self.lock:
            self.requests.append(request)
        if self.change:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "GetObject")
        start, end = 0, len(self.data)
        if "Range" in request and not self.ignore_range:
            start, last = map(int, request["Range"].removeprefix("bytes=").split("-"))
            end = last + 1
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        backend = self

        class Body:
            offset = start

            def readinto(self, target):
                if backend.gate is not None:
                    backend.gate.wait(timeout=5)
                time.sleep(backend.delay)
                with backend.lock:
                    if backend.failures[start]:
                        backend.failures[start] -= 1
                        raise OSError("interrupted response")
                    if backend.short_reads[start]:
                        backend.short_reads[start] -= 1
                        return 0
                count = min(len(target), end - self.offset)
                target[:count] = memoryview(backend.data)[
                    self.offset : self.offset + count
                ]
                self.offset += count
                return count

            def read(self, size):
                return backend.data[self.offset : min(self.offset + size, end)]

            def close(self):
                with backend.lock:
                    backend.active -= 1
                    backend.closed += 1

        response = {"Body": Body(), "ContentLength": end - start, "ETag": self.etag}
        if "Range" in request and not self.ignore_range:
            response["ContentRange"] = f"bytes {start}-{end - 1}/{len(self.data)}"
        if self.wrong_range:
            response["ContentRange"] = "bytes 1-4/10"
        return response


def _client(monkeypatch, backend, *, workers=2, threshold=16, range_bytes=8):
    import boto3

    monkeypatch.setenv("MX_S3_DOWNLOAD_BACKEND", "ranged")
    monkeypatch.setenv("MX_REFIT_DOWNLOAD_WORKERS", str(workers))
    monkeypatch.setenv("MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES", str(threshold))
    monkeypatch.setenv("MX_S3_DOWNLOAD_RANGE_BYTES", str(range_bytes))
    monkeypatch.setenv("MX_S3_MAX_ATTEMPTS", "3")
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: backend)
    return S3Client()


@pytest.mark.parametrize("size", [0, 4, 12, 16, 35])
def test_ranged_download_and_threshold(monkeypatch, size):
    backend = _Backend(bytes(range(size)))
    client = _client(monkeypatch, backend)
    try:
        data = client.get("s3://weights/checkpoint")
    finally:
        client.close()
    assert data == backend.data and isinstance(data, bytes)
    assert backend.closed == len(backend.requests)
    assert all(r["IfMatch"] == backend.etag for r in backend.requests)
    if size < 16:
        assert len(backend.requests) == bool(size)
        assert all("Range" not in r for r in backend.requests)
    else:
        assert sorted(r["Range"] for r in backend.requests) == sorted(
            f"bytes={start}-{min(start + 8, size) - 1}" for start in range(0, size, 8)
        )


@pytest.mark.parametrize("start", [0, 8])
@pytest.mark.parametrize("failure", ["failures", "short_reads"])
def test_retries_first_and_later_body_reads(monkeypatch, start, failure):
    backend = _Backend(b"x" * 24)
    getattr(backend, failure)[start] = 1
    client = _client(monkeypatch, backend)
    try:
        assert client.get("s3://weights/checkpoint") == backend.data
    finally:
        client.close()
    assert (
        Counter(r["Range"] for r in backend.requests)[f"bytes={start}-{start + 7}"] == 2
    )
    assert backend.closed == len(backend.requests)


def test_small_object_body_retries_are_bounded(monkeypatch):
    backend = _Backend(b"x" * 12)
    backend.failures[0] = 5
    client = _client(monkeypatch, backend)
    try:
        with pytest.raises(OSError, match="interrupted"):
            client.get("s3://weights/checkpoint")
    finally:
        client.close()
    assert len(backend.requests) == backend.closed == 3


def test_changed_object_is_not_retried(monkeypatch):
    backend = _Backend(b"x" * 12)
    backend.change = True
    client = _client(monkeypatch, backend)
    try:
        with pytest.raises(ClientError, match="PreconditionFailed"):
            client.get("s3://weights/checkpoint")
    finally:
        client.close()
    assert len(backend.requests) == 1


@pytest.mark.parametrize("problem", ["ignore_range", "wrong_range"])
def test_invalid_range_response_fails_and_closes_body(monkeypatch, problem):
    backend = _Backend(b"x" * 24)
    setattr(backend, problem, True)
    client = _client(monkeypatch, backend)
    try:
        with pytest.raises(ValueError, match="ContentRange"):
            client.get("s3://weights/checkpoint")
    finally:
        client.close()
    assert backend.closed == len(backend.requests)
    assert len({r["Range"] for r in backend.requests}) == len(backend.requests)


def test_concurrent_objects_and_retries_share_get_budget(monkeypatch):
    backend = _Backend(b"x" * 35)
    backend.failures[0] = 1
    backend.short_reads[0] = 1
    backend.delay = 0.005
    backend.gate = threading.Barrier(2)
    client = _client(monkeypatch, backend, workers=2)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(client.get, [f"s3://weights/{i}" for i in range(4)]))
        assert values == [backend.data] * 4
    finally:
        client.close()
    assert backend.peak_active == 2
    assert backend.active == 0


def test_ranged_download_avoids_a_second_full_object_allocation(monkeypatch):
    backend = _Backend(b"x" * (16 * 1024**2))
    client = _client(monkeypatch, backend, range_bytes=1024**2)
    tracemalloc.start()
    try:
        data = client.get("s3://weights/checkpoint")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        client.close()
    assert data == backend.data and isinstance(data, bytes)
    assert peak < len(data) * 1.5


def test_older_sdk_streaming_body_without_readinto(monkeypatch):
    import io

    backend = _Backend(b"x" * 35)
    original = backend.get_object

    def get_object(**request):
        response = original(**request)
        body = response["Body"]
        data = bytearray(response["ContentLength"])
        body.readinto(data)
        body.close()

        class LegacyBody:
            def __init__(self):
                self.stream = io.BytesIO(data)

            def read(self, size):
                return self.stream.read(size)

            def close(self):
                self.stream.close()

        response["Body"] = LegacyBody()
        return response

    backend.get_object = get_object
    client = _client(monkeypatch, backend)
    try:
        assert client.get("s3://weights/checkpoint") == backend.data
    finally:
        client.close()


def test_partial_body_reads_fill_the_entire_range(monkeypatch):
    backend = _Backend(b"x" * 35)
    original = backend.get_object

    def get_object(**request):
        response = original(**request)
        readinto = response["Body"].readinto
        response["Body"].readinto = lambda target: readinto(target[:3])
        return response

    backend.get_object = get_object
    client = _client(monkeypatch, backend)
    try:
        assert client.get("s3://weights/checkpoint") == backend.data
    finally:
        client.close()
    assert len(backend.requests) == 5


@pytest.mark.parametrize("corrupt", [False, True])
def test_sdk_body_checksum_is_validated_at_eof(monkeypatch, corrupt):
    import base64
    import io

    from botocore.exceptions import FlexibleChecksumError
    from botocore.httpchecksum import Crc32Checksum, StreamingChecksumBody

    backend = _Backend(b"checkpoint")
    original = backend.get_object
    checksum = Crc32Checksum()
    checksum.update(b"corrupted" if corrupt else backend.data)
    expected = base64.b64encode(checksum.digest()).decode()

    def get_object(**request):
        response = original(**request)
        response["Body"].close()
        response["Body"] = StreamingChecksumBody(
            io.BytesIO(backend.data), len(backend.data), Crc32Checksum(), expected
        )
        return response

    backend.get_object = get_object
    client = _client(monkeypatch, backend)
    try:
        if corrupt:
            with pytest.raises(FlexibleChecksumError):
                client.get("s3://weights/checkpoint")
        else:
            assert client.get("s3://weights/checkpoint") == backend.data
    finally:
        client.close()
    assert len(backend.requests) == 1
