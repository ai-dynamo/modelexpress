# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bearer-token auth for ModelExpress gRPC clients; a no-op when no token file is present."""

import logging
import os
import threading
import time

import grpc

logger = logging.getLogger("modelexpress.auth")

DEFAULT_TOKEN_PATH = "/var/run/secrets/tokens/modelexpress"
ENV_TOKEN_PATH = "MX_AUTH_TOKEN_PATH"
ENV_TOKEN_TTL = "MX_AUTH_TOKEN_TTL_SECONDS"
DEFAULT_TTL_SECONDS = 60.0


class TokenProvider:
    """Caches a projected ServiceAccount token, re-reading on TTL expiry or mtime change."""

    def __init__(self, path: str | None = None, ttl_seconds: float | None = None):
        self._path = path or os.environ.get(ENV_TOKEN_PATH, DEFAULT_TOKEN_PATH)
        if ttl_seconds is None:
            ttl_seconds = float(os.environ.get(ENV_TOKEN_TTL, DEFAULT_TTL_SECONDS))
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._cached: str | None = None
        self._cached_at: float = 0.0
        self._cached_mtime: float = 0.0
        self._warned_missing = False

    def get(self) -> str | None:
        """Return the current token, or None when the token file is absent."""
        now = time.monotonic()
        with self._lock:
            if self._cached is not None and (now - self._cached_at) < self._ttl:
                return self._cached
            try:
                mtime = os.stat(self._path).st_mtime
                if self._cached is not None and mtime == self._cached_mtime:
                    self._cached_at = now
                    return self._cached
                with open(self._path, encoding="utf-8") as handle:
                    token = handle.read().strip()
                self._cached = token or None
                self._cached_at = now
                self._cached_mtime = mtime
                self._warned_missing = False
                return self._cached
            except FileNotFoundError:
                if not self._warned_missing:
                    logger.warning(
                        "Auth token file %s not found; RPCs sent without a bearer token "
                        "(server must have auth off).",
                        self._path,
                    )
                    self._warned_missing = True
                self._cached = None
                return None
            except OSError as exc:
                if not self._warned_missing:
                    logger.warning(
                        "Auth token file %s could not be read (%s); RPCs sent without a "
                        "bearer token.",
                        self._path,
                        exc,
                    )
                    self._warned_missing = True
                self._cached = None
                return None


class _BearerInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    def __init__(self, provider: TokenProvider):
        self._provider = provider

    def _with_auth(self, client_call_details):
        token = self._provider.get()
        if not token:
            return client_call_details
        metadata = list(client_call_details.metadata or [])
        metadata.append(("authorization", f"Bearer {token}"))
        return _ClientCallDetails(
            client_call_details.method,
            client_call_details.timeout,
            metadata,
            client_call_details.credentials,
            client_call_details.wait_for_ready,
            client_call_details.compression,
        )

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return continuation(self._with_auth(client_call_details), request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return continuation(self._with_auth(client_call_details), request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        return continuation(self._with_auth(client_call_details), request_iterator)

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        return continuation(self._with_auth(client_call_details), request_iterator)


class _ClientCallDetails(grpc.ClientCallDetails):
    """Concrete carrier; the gRPC namedtuple is not public API."""

    def __init__(self, method, timeout, metadata, credentials, wait_for_ready, compression):
        self.method = method
        self.timeout = timeout
        self.metadata = metadata
        self.credentials = credentials
        self.wait_for_ready = wait_for_ready
        self.compression = compression


_SHARED_PROVIDER: TokenProvider | None = None
_SHARED_LOCK = threading.Lock()


def shared_provider() -> TokenProvider:
    """Process-wide token provider, created on first use."""
    global _SHARED_PROVIDER
    with _SHARED_LOCK:
        if _SHARED_PROVIDER is None:
            _SHARED_PROVIDER = TokenProvider()
        return _SHARED_PROVIDER


def with_auth(channel: grpc.Channel, provider: TokenProvider | None = None) -> grpc.Channel:
    """Wrap a channel so every RPC carries the bearer token (no-op when absent)."""
    return grpc.intercept_channel(
        channel, _BearerInterceptor(provider or shared_provider())
    )
