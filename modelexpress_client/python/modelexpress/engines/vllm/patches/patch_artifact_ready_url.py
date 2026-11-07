# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carry the resolved HTTP endpoint into vLLM's model-loading workers."""

from functools import wraps
from urllib.parse import urlsplit

from ....metadata.readiness import health_url_from_server_args


def patch_artifact_ready_url() -> bool:
    from vllm.engine.arg_utils import EngineArgs

    original_from_cli_args = EngineArgs.from_cli_args.__func__
    if getattr(original_from_cli_args, "__modelexpress_patched__", False):
        return False
    original_create_config = EngineArgs.create_engine_config

    @wraps(original_from_cli_args)
    def from_cli_args(cls, args):
        engine_args = original_from_cli_args(cls, args)
        # UDS uses a Unix socket; gRPC mode does not expose HTTP on --port.
        if getattr(args, "uds", None) or getattr(args, "grpc", False):
            url = None
        else:
            url = health_url_from_server_args(args)
        if url and (
            getattr(args, "headless", False)
            or getattr(args, "api_server_count", None) == 0
        ):
            # Headless nodes do not bind --host. Keep the inferred URL local
            # so the shared resolver redirects it to the engine's head.
            parsed = urlsplit(url)
            url = parsed._replace(netloc=f"127.0.0.1:{parsed.port}").geturl()
        engine_args._mx_artifact_ready_url = url
        return engine_args

    @wraps(original_create_config)
    def create_engine_config(self, *args, **kwargs):
        config = original_create_config(self, *args, **kwargs)
        url = getattr(self, "_mx_artifact_ready_url", None)
        if isinstance(url, str) and url:
            # An ordinary attribute survives worker pickling without entering
            # additional_config or the engine's compilation-cache hash.
            config.load_config._mx_artifact_ready_url = url
        return config

    from_cli_args.__modelexpress_patched__ = True
    EngineArgs.from_cli_args = classmethod(from_cli_args)
    EngineArgs.create_engine_config = create_engine_config
    return True
