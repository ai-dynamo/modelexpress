# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from unittest.mock import Mock

import pytest

pytest.importorskip("azure.storage.blob")
pytest.importorskip("azure.identity")

import azure.identity
import requests
from azure.core.credentials import AccessToken
from azure.core.exceptions import HttpResponseError
from azure.core.pipeline.transport import HttpTransport
from urllib3.response import HTTPResponse

from modelexpress_rl import envs
from modelexpress_rl.azure import AzureBlobReader


DEFAULTS = {
    "MX_AZURE_DOWNLOAD_CONCURRENCY": 1,
    "MX_AZURE_DOWNLOAD_RANGE_BYTES": 4 * 1024**2,
    "MX_AZURE_DOWNLOAD_RANGE_THRESHOLD_BYTES": 32 * 1024**2,
    "MX_AZURE_MAX_POOL_CONNECTIONS": 32,
    "MX_AZURE_MAX_ATTEMPTS": 4,
    "MX_AZURE_CONNECTION_TIMEOUT_SECONDS": 20,
    "MX_AZURE_READ_TIMEOUT_SECONDS": 60,
}


@pytest.fixture(params=["connection-string", "identity"])
def authentication(monkeypatch, request):
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://account.blob.core.windows.net")
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    if request.param == "connection-string":
        monkeypatch.setenv(
            "AZURE_STORAGE_CONNECTION_STRING",
            "BlobEndpoint=https://account.blob.core.windows.net;SharedAccessSignature=sig=test",
        )
    credential = Mock(spec=["get_token", "close"])
    credential.get_token.return_value = AccessToken("synthetic-token", 9999999999)
    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", lambda: credential)
    return request.param, credential


def test_transfer_defaults_are_independent_of_shard_workers(authentication, monkeypatch):
    monkeypatch.setenv("MX_REFIT_DOWNLOAD_WORKERS", "7")
    monkeypatch.setenv("MX_S3_DOWNLOAD_WORKERS", "9")
    for name, value in DEFAULTS.items():
        assert getattr(envs, name) == value
    reader = AzureBlobReader()
    session = reader._transport.session
    close = Mock(wraps=session.close)
    monkeypatch.setattr(session, "close", close)
    try:
        assert reader._download_concurrency == 1
        assert reader._client._config.max_single_get_size == 32 * 1024**2
        assert reader._client._config.max_chunk_get_size == 4 * 1024**2
        assert session.get_adapter("https://")._pool_maxsize == 32
    finally:
        reader.close()
    close.assert_called_once()
    if authentication[0] == "identity":
        authentication[1].close.assert_called_once()


@pytest.mark.parametrize("recover", [True, False])
def test_sdk_uses_ranges_attempts_timeouts_and_pool_settings(
    authentication, monkeypatch, recover
):
    values = dict(zip(DEFAULTS, [2, 2, 4, 7, 2, 11, 13], strict=True))
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))
    calls = []

    def request(session, method, url, **kwargs):
        assert kwargs["timeout"] == (11, 13)
        assert session.get_adapter(url)._pool_maxsize == 7
        assert session.get_adapter(url).max_retries.total == 0
        interval = kwargs["headers"]["x-ms-range"].removeprefix("bytes=")
        begin, end = map(int, interval.split("-"))
        calls.append((begin, end))
        response = requests.Response()
        response.status_code = 503 if len(calls) == 1 or not recover else 206
        body = b"abcdefgh"[begin:end + 1]
        response.headers.update({
            "Content-Length": str(len(body)), "Content-Range": f"bytes {begin}-{end}/8",
            "ETag": '"immutable"',
        })
        response.raw = HTTPResponse(
            body=BytesIO(body), headers=response.headers, preload_content=False,
            decode_content=False, request_method=method,
        )
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    monkeypatch.setattr(HttpTransport, "sleep", lambda *args: None)
    reader = AzureBlobReader()
    try:
        assert reader._download_concurrency == 2
        if recover:
            assert reader.get("az://models/shard") == b"abcdefgh"
            assert sorted(calls[2:]) == [(4, 5), (6, 7)]
        else:
            with pytest.raises(HttpResponseError):
                reader.get("az://models/shard")
            assert len(calls) == 2
        assert calls[:2] == [(0, 3), (0, 3)]
    finally:
        reader.close()


@pytest.mark.parametrize("name", DEFAULTS)
@pytest.mark.parametrize("value", ["", "bad", "0", "-1"])
def test_invalid_transfer_setting_fails_before_client_creation(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    create = Mock(side_effect=AssertionError("storage client must not be constructed"))
    monkeypatch.setattr(AzureBlobReader, "_create_client", create)
    with pytest.raises(ValueError, match=name):
        AzureBlobReader()
    create.assert_not_called()


def test_client_construction_failure_closes_transport_and_credential(
    authentication, monkeypatch
):
    def fail(*args, **kwargs):
        raise RuntimeError("client construction failed")

    close = Mock()
    monkeypatch.setattr(requests.Session, "close", close)
    monkeypatch.setattr(azure.storage.blob.BlobServiceClient, "__init__", fail)
    with pytest.raises(RuntimeError, match="client construction failed"):
        AzureBlobReader()
    close.assert_called_once()
    if authentication[0] == "identity":
        authentication[1].close.assert_called_once()
