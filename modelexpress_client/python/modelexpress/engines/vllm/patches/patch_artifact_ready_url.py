# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carry the resolved HTTP endpoint into vLLM's model-loading workers."""

import logging
from functools import wraps
from inspect import getattr_static, iscoroutinefunction, isfunction
from urllib.parse import urlsplit

from .... import envs
from ....metadata.readiness import health_url_from_server_args

logger = logging.getLogger(__name__)
_MARKER = "__modelexpress_artifact_ready_url__"


def _needs_discovery(config) -> bool:
    return (
        envs.MX_ARTIFACT_TRANSFER
        and not envs.MX_ARTIFACT_READY_URL.strip()
        and getattr(config, "load_format", None) in ("modelexpress", "mx")
    )


def patch_artifact_ready_url() -> bool:
    try:
        from vllm.engine.arg_utils import EngineArgs
    except ImportError:
        logger.warning(
            "vLLM readiness discovery unavailable; use MX_ARTIFACT_READY_URL"
        )
        return False

    cli_method = getattr_static(EngineArgs, "from_cli_args", None)
    original_from_cli_args = getattr(cli_method, "__func__", None)
    if (
        not isinstance(cli_method, classmethod)
        or not isfunction(original_from_cli_args)
        or iscoroutinefunction(original_from_cli_args)
    ):
        logger.warning("Unsupported vLLM readiness hook; use MX_ARTIFACT_READY_URL")
        return False
    if getattr(original_from_cli_args, _MARKER, False):
        return False

    @wraps(original_from_cli_args)
    def from_cli_args(cls, *args, **kwargs):
        engine_args = original_from_cli_args(cls, *args, **kwargs)
        if not _needs_discovery(engine_args):
            return engine_args
        namespace = args[0] if args else kwargs.get("args")
        if namespace is None:
            logger.warning(
                "No parsed args for vLLM readiness; use MX_ARTIFACT_READY_URL"
            )
            return engine_args
        # UDS and gRPC mode do not expose HTTP on --port.
        url = (
            None
            if getattr(namespace, "uds", None) or getattr(namespace, "grpc", False)
            else health_url_from_server_args(namespace)
        )
        if url and (
            getattr(namespace, "headless", False)
            or getattr(namespace, "api_server_count", None) == 0
        ):
            parsed = urlsplit(url)
            url = parsed._replace(netloc=f"127.0.0.1:{parsed.port}").geturl()
        if url:
            options = dict(getattr(engine_args, "model_loader_extra_config", None) or {})
            options["modelexpress_artifact_ready_url"] = url
            try:
                engine_args.model_loader_extra_config = options
            except (AttributeError, TypeError):
                logger.warning(
                    "Cannot propagate vLLM readiness URL; use MX_ARTIFACT_READY_URL"
                )
        return engine_args

    setattr(from_cli_args, _MARKER, True)
    EngineArgs.from_cli_args = classmethod(from_cli_args)
    return True
