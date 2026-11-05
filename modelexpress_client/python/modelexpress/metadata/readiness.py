# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health endpoints derived from an engine's resolved HTTP server arguments."""


def health_url_from_server_args(server_args: object) -> str | None:
    """Format an HTTP health URL from host, port and TLS settings."""
    port = getattr(server_args, "port", None)
    if type(port) is not int or not 0 < port < 65536:
        return None
    host = getattr(server_args, "host", None)
    if host is not None and not isinstance(host, str):
        return None
    if host is None or host in {"", "0.0.0.0"}:
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    use_ssl = bool(
        getattr(server_args, "ssl_keyfile", None)
        and getattr(server_args, "ssl_certfile", None)
    )
    scheme = "https" if use_ssl else "http"
    return f"{scheme}://{host}:{port}/health"
