# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent import futures

import grpc
import pytest
from google.protobuf import empty_pb2, wrappers_pb2

from modelexpress_rl.inference.engines.vllm.startup_probe import (
    reconcile_serving_version,
)


def _start_control_server(*, initial="default", accept_update=True):
    """Start a test server implementing vLLM's native Control methods."""
    state = {"version": initial, "updates": []}

    def update(request, _context):
        """Record and optionally apply a serving-version update."""
        state["updates"].append(request.value)
        if accept_update:
            state["version"] = request.value
        return empty_pb2.Empty()

    def get(_request, _context):
        """Return the server's current serving version."""
        return wrappers_pb2.StringValue(value=state["version"])

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "vllm.Control",
                {
                    "UpdateWeightVersion": grpc.unary_unary_rpc_method_handler(
                        update,
                        request_deserializer=wrappers_pb2.StringValue.FromString,
                        response_serializer=empty_pb2.Empty.SerializeToString,
                    ),
                    "GetWeightVersion": grpc.unary_unary_rpc_method_handler(
                        get,
                        request_deserializer=empty_pb2.Empty.FromString,
                        response_serializer=wrappers_pb2.StringValue.SerializeToString,
                    ),
                },
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}", state


def test_reconcile_serving_version_updates_then_reads_vllm_control():
    """Update and read back the desired version through vLLM Control."""
    server, address, state = _start_control_server()
    try:
        observed = reconcile_serving_version(
            address=address,
            desired_version_uid="version-7",
        )
    finally:
        server.stop(grace=None).wait()

    assert observed == "version-7"
    assert state["updates"] == ["version-7"]


def test_reconcile_serving_version_rejects_observed_mismatch():
    """Reject a serving version that differs after reconciliation."""
    server, address, _state = _start_control_server(accept_update=False)
    try:
        with pytest.raises(RuntimeError, match="observed='default'"):
            reconcile_serving_version(
                address=address,
                desired_version_uid="version-7",
            )
    finally:
        server.stop(grace=None).wait()


def test_reconcile_without_desired_version_only_checks_control_readiness():
    """Check Control readiness without updating when no version is desired."""
    server, address, state = _start_control_server(initial="default")
    try:
        observed = reconcile_serving_version(
            address=address,
            desired_version_uid=None,
        )
    finally:
        server.stop(grace=None).wait()

    assert observed == "default"
    assert state["updates"] == []
