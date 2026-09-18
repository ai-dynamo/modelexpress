# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side TLS: channel selection from the address scheme and CA env, and a
real handshake against a grpc server holding a throwaway certificate."""

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest

from modelexpress import client as client_mod
from modelexpress.client import MxClient, _tls_requested, open_channel
from modelexpress.model_client import ModelCacheClient


def _self_signed(tmp_path):
    """Write a self-signed cert for 127.0.0.1 with the openssl CLI."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI not available")
    cert = tmp_path / "tls.crt"
    key = tmp_path / "tls.key"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert), "-days", "1",
            "-subj", "/CN=127.0.0.1",
            "-addext", "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


class _Handler(grpc.GenericRpcHandler):
    def __init__(self):
        self.calls = 0

    def service(self, handler_call_details):
        def handler(request, context):
            self.calls += 1
            return b""

        return grpc.unary_unary_rpc_method_handler(handler)


def _serve(handler, credentials=None):
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    server.add_generic_rpc_handlers((handler,))
    if credentials is None:
        port = server.add_insecure_port("127.0.0.1:0")
    else:
        port = server.add_secure_port("127.0.0.1:0", credentials)
    server.start()
    return server, port


def _call(channel):
    channel.unary_unary("/test.Svc/Ping")(b"", timeout=5)


def test_tls_requested_from_https_scheme(monkeypatch):
    monkeypatch.delenv("MODEL_EXPRESS_TLS_CA_FILE", raising=False)
    monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
    monkeypatch.delenv("MX_SERVER_ADDRESS", raising=False)
    assert _tls_requested("https://mx:8001") is True
    assert _tls_requested("http://mx:8001") is False
    assert _tls_requested("mx:8001") is False
    assert _tls_requested(None) is False


def test_tls_requested_from_env_address(monkeypatch):
    monkeypatch.delenv("MODEL_EXPRESS_TLS_CA_FILE", raising=False)
    monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
    monkeypatch.setenv("MX_SERVER_ADDRESS", "https://mx:8001")
    assert _tls_requested(None) is True
    monkeypatch.setenv("MODEL_EXPRESS_URL", "http://legacy:8001")
    assert _tls_requested(None) is False


def test_ca_file_env_turns_tls_on_for_bare_addresses(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_EXPRESS_TLS_CA_FILE", str(tmp_path / "ca.crt"))
    assert _tls_requested("mx:8001") is True


def test_clients_record_tls_and_strip_the_scheme(monkeypatch):
    monkeypatch.delenv("MODEL_EXPRESS_TLS_CA_FILE", raising=False)
    mx = MxClient(server_url="https://mx:8001")
    assert mx.server_url == "mx:8001"
    assert mx._tls is True
    model = ModelCacheClient(server_url="http://mx:8001")
    assert model.server_url == "mx:8001"
    assert model._tls is False


def test_open_channel_insecure_when_tls_off(monkeypatch):
    monkeypatch.delenv("MODEL_EXPRESS_TLS_CA_FILE", raising=False)
    handler = _Handler()
    server, port = _serve(handler)
    try:
        channel = open_channel(f"127.0.0.1:{port}", tls=False, options=[])
        _call(channel)
    finally:
        server.stop(0)
    assert handler.calls == 1


def test_secure_channel_handshakes_with_configured_ca(monkeypatch, tmp_path):
    cert, key = _self_signed(tmp_path)
    monkeypatch.setenv("MODEL_EXPRESS_TLS_CA_FILE", str(cert))
    credentials = grpc.ssl_server_credentials([(key.read_bytes(), cert.read_bytes())])
    handler = _Handler()
    server, port = _serve(handler, credentials)
    try:
        channel = open_channel(f"127.0.0.1:{port}", tls=True, options=[])
        _call(channel)
    finally:
        server.stop(0)
    assert handler.calls == 1


def test_secure_channel_rejects_untrusted_server(monkeypatch, tmp_path):
    cert, key = _self_signed(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_cert, _ = _self_signed(other_dir)
    monkeypatch.setenv("MODEL_EXPRESS_TLS_CA_FILE", str(other_cert))
    credentials = grpc.ssl_server_credentials([(key.read_bytes(), cert.read_bytes())])
    handler = _Handler()
    server, port = _serve(handler, credentials)
    try:
        channel = open_channel(f"127.0.0.1:{port}", tls=True, options=[])
        with pytest.raises(grpc.RpcError) as exc_info:
            _call(channel)
    finally:
        server.stop(0)
    assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE
    assert handler.calls == 0


def test_plaintext_channel_cannot_reach_tls_server(monkeypatch, tmp_path):
    cert, key = _self_signed(tmp_path)
    credentials = grpc.ssl_server_credentials([(key.read_bytes(), cert.read_bytes())])
    handler = _Handler()
    server, port = _serve(handler, credentials)
    try:
        channel = open_channel(f"127.0.0.1:{port}", tls=False, options=[])
        with pytest.raises(grpc.RpcError) as exc_info:
            _call(channel)
    finally:
        server.stop(0)
    assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE
    assert handler.calls == 0


def test_missing_ca_file_raises_clearly(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_EXPRESS_TLS_CA_FILE", str(tmp_path / "absent.crt"))
    with pytest.raises(FileNotFoundError):
        client_mod._channel_credentials()
