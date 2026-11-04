# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Barrier, Lock
from unittest.mock import Mock, create_autospec
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree

import pytest

pytest.importorskip("azure.storage.blob")

from azure.core.exceptions import HttpResponseError
from azure.storage.blob import BlobServiceClient
import requests
from urllib3.response import HTTPResponse

import modelexpress_rl.azure as azure_module
from modelexpress_rl.azure import AzureBlobClient, ImmutableAzureConflict


class _BlobHTTP:
    """In-memory Blob REST responses behind the real SDK pipeline."""

    def __init__(self):
        self.data = None
        self.blocks = {}
        self.calls = []
        self.lock = Lock()
        self.commit_barrier = None
        self.fail_stage_once = False
        self.fail_commit_once = False
        self.fail_response_once = False
        self.error_code = None

    @staticmethod
    def response(method, status, data=b"", error=None):
        response = requests.Response()
        response.status_code = status
        response.headers.update({"Content-Length": str(len(data)), "ETag": '"etag"'})
        if method == "GET" and status == 206:
            response.headers["Content-Range"] = f"bytes 0-{len(data) - 1}/{len(data)}"
        if error:
            response.headers["x-ms-error-code"] = error
        response.raw = HTTPResponse(
            body=BytesIO(data), headers=response.headers, preload_content=False,
            decode_content=False, request_method=method,
        )
        return response

    def request(self, method, url, **kwargs):
        query = parse_qs(urlsplit(url).query)
        operation = query.get("comp", [None])[0]
        headers = kwargs.get("headers", {})
        body = kwargs.get("data", b"")
        if hasattr(body, "read"):
            body = body.read()
        if method == "PUT" and operation != "block" and self.commit_barrier:
            self.commit_barrier.wait(timeout=10)
        with self.lock:
            self.calls.append((method, operation, dict(headers)))
            if self.error_code:
                return self.response(method, 403, error=self.error_code)
            if method == "HEAD":
                assert self.data is not None
                response = self.response(method, 200)
                response.headers["Content-Length"] = str(len(self.data))
                return response
            if method == "GET":
                assert self.data is not None
                if not self.data:
                    if headers.get("x-ms-range"):
                        return self.response(method, 416, error="InvalidRange")
                    return self.response(method, 200)
                return self.response(method, 206, self.data)
            assert method == "PUT"
            if operation == "block":
                if self.fail_stage_once and self.blocks:
                    self.fail_stage_once = False
                    return self.response(method, 500, error="InternalError")
                self.blocks[query["blockid"][0]] = bytes(body)
                return self.response(method, 201)
            assert headers.get("If-None-Match") == "*"
            if self.data is not None:
                return self.response(method, 412, error="ConditionNotMet")
            if self.fail_commit_once:
                self.fail_commit_once = False
                return self.response(method, 500, error="InternalError")
            if operation == "blocklist":
                block_list = ElementTree.fromstring(body)
                assert all(block.tag == "Latest" for block in block_list)
                self.data = b"".join(self.blocks[block.text] for block in block_list)
            else:
                assert operation is None
                self.data = bytes(body)
            if self.fail_response_once:
                self.fail_response_once = False
                return self.response(method, 500, error="InternalError")
            return self.response(method, 201)


@pytest.fixture
def writer(monkeypatch):
    monkeypatch.setenv(
        "AZURE_STORAGE_CONNECTION_STRING",
        "BlobEndpoint=https://testaccount.blob.core.windows.net;"
        "SharedAccessSignature=sig=test",
    )
    # Exercise the real block protocol with tiny payloads, without large allocations.
    monkeypatch.setattr(azure_module, "_MAX_SINGLE_PUT_BYTES", 8)
    monkeypatch.setattr(azure_module, "_BLOCK_BYTES", 4)
    create = BlobServiceClient.from_connection_string
    monkeypatch.setattr(
        BlobServiceClient, "from_connection_string",
        lambda value, **kwargs: create(value, **{**kwargs, "retry_total": 0}),
    )
    backend = _BlobHTTP()
    monkeypatch.setattr(
        requests.Session, "request",
        create_autospec(
            requests.Session.request,
            side_effect=lambda _session, method, url, **kwargs: backend.request(
                method, url, **kwargs
            ),
        ),
    )
    client = AzureBlobClient()
    close = Mock(wraps=client._client.close)
    monkeypatch.setattr(client._client, "close", close)
    try:
        yield client, backend
    finally:
        client.close()
        close.assert_called_once()


@pytest.mark.parametrize("data", [b"", b"small", b"large payload"])
def test_immutable_create_identical_retry_and_conflicting_retry(writer, data):
    client, backend = writer
    uri = "az://models/checkpoint"
    client.put(uri=uri, data=data)
    assert backend.data == data
    client.put(uri=uri, data=data)
    conflict_start = len(backend.calls)
    with pytest.raises(ImmutableAzureConflict):
        client.put(uri=uri, data=data + b"!")
    assert backend.data == data
    assert not any(
        method == "GET" for method, _, _ in backend.calls[conflict_start:]
    )
    assert all(method != "DELETE" for method, _, _ in backend.calls)


@pytest.mark.parametrize("same_bytes", [True, False])
@pytest.mark.parametrize("large", [False, True], ids=["single-put", "block-upload"])
def test_concurrent_creates_preserve_one_complete_object(writer, same_bytes, large):
    client, backend = writer
    backend.commit_barrier = Barrier(2)
    first = b"a" * (12 if large else 4)
    second = first if same_bytes else b"b" * len(first)

    def put(data):
        try:
            client.put(uri="az://models/checkpoint", data=data)
            return data
        except ImmutableAzureConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(put, [first, second]))
    successes = [result for result in results if result is not None]
    assert len(successes) == (2 if same_bytes else 1)
    assert all(data == backend.data for data in successes)
    if large:
        assert len(backend.blocks) == 6


@pytest.mark.parametrize("failure", ["stage", "commit"])
def test_failed_block_upload_is_not_visible_and_retry_uses_new_blocks(writer, failure):
    client, backend = writer
    setattr(backend, f"fail_{failure}_once", True)
    with pytest.raises(HttpResponseError):
        client.put(uri="az://models/checkpoint", data=b"large payload")
    assert backend.data is None
    old_blocks = set(backend.blocks)
    client.put(uri="az://models/checkpoint", data=b"large payload")
    assert backend.data == b"large payload"
    assert len(set(backend.blocks) - old_blocks) == 4
    # Blob has no per-upload abort API. Never delete a possibly concurrent winner.
    assert all(method != "DELETE" for method, _, _ in backend.calls)


@pytest.mark.parametrize("data", [b"small", b"large payload"])
def test_retry_accepts_committed_bytes_after_lost_response(writer, data):
    client, backend = writer
    backend.fail_response_once = True
    with pytest.raises(HttpResponseError):
        client.put(uri="az://models/checkpoint", data=data)
    assert backend.data == data
    client.put(uri="az://models/checkpoint", data=data)
    assert backend.data == data


def test_unrelated_storage_failure_is_not_an_identical_retry(writer):
    client, backend = writer
    backend.error_code = "AuthorizationPermissionMismatch"
    with pytest.raises(HttpResponseError):
        client.put(uri="az://models/checkpoint", data=b"small")
    assert backend.data is None
    assert not any(method == "GET" for method, _, _ in backend.calls)


def test_block_limit_fails_before_upload(writer, monkeypatch):
    client, backend = writer
    monkeypatch.setattr(azure_module, "_MAX_BLOCKS", 2)
    with pytest.raises(ValueError, match="50,000 blocks"):
        client.put(uri="az://models/checkpoint", data=b"large payload")
    assert backend.calls == []
