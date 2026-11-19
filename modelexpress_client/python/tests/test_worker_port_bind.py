# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The worker gRPC server must fail loudly, with the MX setting, when its port is taken."""

import socket
from unittest.mock import MagicMock, patch

import grpc
import pytest

from modelexpress import p2p_pb2
from modelexpress.metadata import publish as publish_mod
from modelexpress.metadata.worker_server import WorkerGrpcServer


@pytest.fixture
def taken_port():
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.bind(("::", 0))
    sock.listen()
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def _server(port: int) -> WorkerGrpcServer:
    return WorkerGrpcServer(tensor_protos=[], mx_source_id=None, port=port)


def test_start_raises_when_port_is_taken(taken_port):
    server = _server(taken_port)

    with pytest.raises(RuntimeError, match=f"failed to bind \\[::\\]:{taken_port}"):
        server.start()

    assert server.port is None


def test_start_raises_when_another_worker_server_holds_the_port():
    first = _server(0)
    port = first.start()
    try:
        with pytest.raises(RuntimeError, match=f"failed to bind \\[::\\]:{port}"):
            _server(port).start()
    finally:
        first.stop(grace=None)


def test_restart_on_same_port_after_stop():
    first = _server(0)
    port = first.start()
    first.stop(grace=None)

    second = _server(port)
    try:
        assert second.start() == port
    finally:
        second.stop(grace=None)


def test_start_raises_when_grpc_returns_port_zero():
    real_server = grpc.server

    def fake_server(*args, **kwargs):
        server = real_server(*args, **kwargs)
        server.add_insecure_port = MagicMock(return_value=0)
        return server

    with patch("modelexpress.metadata.worker_server.grpc.server", fake_server):
        with pytest.raises(RuntimeError, match="got port 0"):
            _server(6555).start()


def test_publish_names_worker_grpc_port_setting(taken_port, monkeypatch):
    base_port = taken_port - 3
    monkeypatch.setenv("MX_WORKER_GRPC_PORT", str(base_port))
    monkeypatch.setenv("MX_P2P_METADATA", "1")
    monkeypatch.setenv("MX_WORKER_HOST", "127.0.0.1")
    monkeypatch.setattr(publish_mod, "_is_p2p_metadata_enabled", lambda client: True)
    nixl_manager = MagicMock()
    nixl_manager._listen_port = 5555

    with pytest.raises(RuntimeError) as exc_info:
        publish_mod.publish_metadata_and_ready(
            MagicMock(),
            nixl_manager,
            {},
            worker_rank=3,
            device_id=3,
            identity=p2p_pb2.SourceIdentity(model_name="m"),
            worker_id="w",
        )

    message = str(exc_info.value)
    assert f"{taken_port} = MX_WORKER_GRPC_PORT ({base_port}) + device_id (3)" in message
    assert "another service" in message
    assert 3 not in publish_mod._worker_servers
