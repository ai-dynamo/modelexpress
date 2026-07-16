# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated gRPC client for ephemeral M2N bootstrap records."""

from __future__ import annotations

import grpc

from .. import m2n_bootstrap_pb2
from .. import m2n_bootstrap_pb2_grpc
from ..client import _get_server_url


class MxM2nBootstrapClient:
    """Thin client kept separate from model-source metadata clients."""

    def __init__(self, server_url: str | None = None, rpc_timeout_s: float = 30.0):
        if rpc_timeout_s <= 0:
            raise ValueError("rpc_timeout_s must be positive")
        self.server_url = _get_server_url(server_url)
        self.rpc_timeout_s = rpc_timeout_s
        self._channel: grpc.Channel | None = None
        self._stub: m2n_bootstrap_pb2_grpc.M2nBootstrapServiceStub | None = None

    @property
    def stub(self) -> m2n_bootstrap_pb2_grpc.M2nBootstrapServiceStub:
        if self._channel is None:
            self._channel = grpc.insecure_channel(self.server_url)
            self._stub = m2n_bootstrap_pb2_grpc.M2nBootstrapServiceStub(self._channel)
        assert self._stub is not None
        return self._stub

    def _rpc_timeout(self, timeout_s: float | None) -> float:
        if timeout_s is None:
            return self.rpc_timeout_s
        if timeout_s <= 0:
            raise ValueError("RPC timeout must be positive")
        return min(timeout_s, self.rpc_timeout_s)

    def publish_bootstrap(
        self,
        request: m2n_bootstrap_pb2.PublishM2nBootstrapRequest,
        *,
        timeout_s: float | None = None,
    ) -> m2n_bootstrap_pb2.M2nBootstrapRecord:
        response = self.stub.PublishBootstrap(
            request, timeout=self._rpc_timeout(timeout_s)
        )
        if not response.success or not response.HasField("record"):
            raise RuntimeError(f"PublishBootstrap failed: {response.message}")
        return response.record

    def get_bootstrap(
        self,
        key: m2n_bootstrap_pb2.M2nBootstrapKey,
        *,
        timeout_s: float | None = None,
    ) -> m2n_bootstrap_pb2.M2nBootstrapRecord | None:
        response = self.stub.GetBootstrap(
            m2n_bootstrap_pb2.GetM2nBootstrapRequest(key=key),
            timeout=self._rpc_timeout(timeout_s),
        )
        if not response.found:
            return None
        if not response.HasField("record"):
            raise RuntimeError("GetBootstrap returned found=true without a record")
        return response.record

    def abort_bootstrap(
        self,
        key: m2n_bootstrap_pb2.M2nBootstrapKey,
        *,
        requested_by: str,
        reason: str,
        timeout_s: float | None = None,
    ) -> m2n_bootstrap_pb2.M2nBootstrapRecord:
        response = self.stub.AbortBootstrap(
            m2n_bootstrap_pb2.AbortM2nBootstrapRequest(
                key=key,
                requested_by=requested_by,
                reason=reason,
            ),
            timeout=self._rpc_timeout(timeout_s),
        )
        if not response.success or not response.HasField("record"):
            raise RuntimeError(f"AbortBootstrap failed: {response.message}")
        return response.record

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
