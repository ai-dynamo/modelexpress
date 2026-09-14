# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct client for vLLM's native serving-version Control service."""

from __future__ import annotations

import grpc
from google.protobuf import empty_pb2, wrappers_pb2

_CONTROL_SERVICE = "/vllm.Control"


class VllmControlClient:
    """Call the weight-version methods exposed by vLLM's Rust gRPC server."""

    def __init__(self, address: str, *, timeout_seconds: float = 3.0) -> None:
        """Connect to the native vLLM Control service at ``address``."""
        self._timeout_seconds = timeout_seconds
        self._channel = grpc.insecure_channel(address)
        # vLLM's request/response messages are either empty or one string at
        # field 1, which are wire-compatible with these standard protobuf types.
        self._update_weight_version = self._channel.unary_unary(
            f"{_CONTROL_SERVICE}/UpdateWeightVersion",
            request_serializer=wrappers_pb2.StringValue.SerializeToString,
            response_deserializer=empty_pb2.Empty.FromString,
        )
        self._get_weight_version = self._channel.unary_unary(
            f"{_CONTROL_SERVICE}/GetWeightVersion",
            request_serializer=empty_pb2.Empty.SerializeToString,
            response_deserializer=wrappers_pb2.StringValue.FromString,
        )

    def update_weight_version(self, version_uid: str) -> None:
        """Set the serving weight version reported by vLLM."""
        self._update_weight_version(
            wrappers_pb2.StringValue(value=version_uid),
            timeout=self._timeout_seconds,
        )

    def get_weight_version(self) -> str:
        """Return the serving weight version reported by vLLM."""
        response = self._get_weight_version(
            empty_pb2.Empty(),
            timeout=self._timeout_seconds,
        )
        return response.value

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        self._channel.close()

    def __enter__(self) -> VllmControlClient:
        """Return this client for use as a context manager."""
        return self

    def __exit__(self, *_args) -> None:
        """Close the client when leaving its context."""
        self.close()


__all__ = ["VllmControlClient"]
