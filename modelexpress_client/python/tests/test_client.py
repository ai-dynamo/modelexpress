# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from modelexpress.client import MxClient, _get_server_url


def test_get_server_url_defaults_to_insecure_localhost():
    address, use_tls = _get_server_url()
    assert address == "localhost:8001"
    assert use_tls is False


def test_get_server_url_parses_https_as_tls():
    address, use_tls = _get_server_url("https://mx.example.com:8443")
    assert address == "mx.example.com:8443"
    assert use_tls is True


def test_mx_client_uses_secure_channel_for_https():
    client = MxClient(server_url="https://mx.example.com:8443")

    with patch("modelexpress.client.grpc.ssl_channel_credentials", return_value="creds") as creds, \
         patch("modelexpress.client.grpc.secure_channel", return_value="secure-channel") as secure, \
         patch("modelexpress.client.p2p_pb2_grpc.P2pServiceStub", return_value="stub") as stub:
        assert client.stub == "stub"
        creds.assert_called_once_with()
        secure.assert_called_once_with(
            "mx.example.com:8443",
            "creds",
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ],
        )
        stub.assert_called_once_with("secure-channel")


def test_mx_client_uses_insecure_channel_for_plaintext():
    client = MxClient(server_url="http://mx.example.com:8001")

    with patch("modelexpress.client.grpc.insecure_channel", return_value="insecure-channel") as insecure, \
         patch("modelexpress.client.p2p_pb2_grpc.P2pServiceStub", return_value="stub") as stub:
        assert client.stub == "stub"
        insecure.assert_called_once_with(
            "mx.example.com:8001",
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ],
        )
        stub.assert_called_once_with("insecure-channel")
