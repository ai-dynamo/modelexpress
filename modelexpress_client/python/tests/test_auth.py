# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bearer-token interceptor tests: real temp-file token reads against an in-process server."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest

from modelexpress.auth import TokenProvider, with_auth

_METHOD = "/test.Echo/Call"


class _CaptureHandler(grpc.GenericRpcHandler):
    """Generic handler that records invocation metadata for any method."""

    def __init__(self):
        self.last_metadata: list[tuple[str, str]] | None = None

    def service(self, handler_call_details):
        def handler(request, context):
            self.last_metadata = list(context.invocation_metadata())
            return b"ok"

        return grpc.unary_unary_rpc_method_handler(handler)


def _serve(handler: _CaptureHandler) -> tuple[grpc.Server, int]:
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    server.add_generic_rpc_handlers((handler,))
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


def _call(channel: grpc.Channel) -> None:
    rpc = channel.unary_unary(_METHOD)
    rpc(b"hi", timeout=5)


def test_token_provider_reads_file(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("abc.def")
    provider = TokenProvider(path=str(token_file))
    assert provider.get() == "abc.def"


def test_token_provider_refreshes_on_change(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("tok1")
    # ttl=0 forces a re-read every call so we can observe rotation.
    provider = TokenProvider(path=str(token_file), ttl_seconds=0.0)
    assert provider.get() == "tok1"

    token_file.write_text("tok2")
    later = os.stat(str(token_file)).st_mtime + 10
    os.utime(str(token_file), (later, later))
    assert provider.get() == "tok2"


def test_token_provider_missing_file_returns_none_and_warns_once(tmp_path, caplog):
    provider = TokenProvider(path=str(tmp_path / "absent"))
    with caplog.at_level(logging.WARNING, logger="modelexpress.auth"):
        assert provider.get() is None
        assert provider.get() is None
    warnings = [r for r in caplog.records if r.name == "modelexpress.auth"]
    assert len(warnings) == 1


def test_interceptor_attaches_bearer_when_token_present(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("secret-token")
    provider = TokenProvider(path=str(token_file))

    handler = _CaptureHandler()
    server, port = _serve(handler)
    try:
        channel = with_auth(grpc.insecure_channel(f"127.0.0.1:{port}"), provider)
        _call(channel)
    finally:
        server.stop(0)

    assert handler.last_metadata is not None
    metadata = dict(handler.last_metadata)
    assert metadata.get("authorization") == "Bearer secret-token"


def test_interceptor_omits_header_when_token_absent(tmp_path):
    provider = TokenProvider(path=str(tmp_path / "absent"))

    handler = _CaptureHandler()
    server, port = _serve(handler)
    try:
        channel = with_auth(grpc.insecure_channel(f"127.0.0.1:{port}"), provider)
        _call(channel)
    finally:
        server.stop(0)

    assert handler.last_metadata is not None
    metadata = dict(handler.last_metadata)
    assert "authorization" not in metadata


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
