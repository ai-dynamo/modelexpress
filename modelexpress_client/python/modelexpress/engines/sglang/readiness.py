# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read SGLang's resolved HTTP listener in a model worker."""

import os
from dataclasses import dataclass
from importlib import import_module
from urllib.parse import urlsplit

from ...metadata.readiness import health_url_from_server_args


@dataclass(frozen=True)
class SglangServerReadiness:
    url: str | None = None
    local_dummy: bool = False


def _uses_dummy_listener(server_args: object, module: object) -> bool | None:
    try:
        node_rank = getattr(server_args, "node_rank", None)
        if node_rank is None:
            node_rank = module.get_parallel().node_rank
        if type(node_rank) is not int or node_rank < 0:
            return None
        if node_rank == 0:
            return False
        if os.environ.get("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            return None
        try:
            rust = import_module("sglang.srt.environ").envs.SGLANG_RUST_SERVER.get()
        except (ImportError, AttributeError):
            rust = False
        if not rust:
            return True
        # Rust can own the base port on only a subset of follower nodes.
        # Reuse the engine's placement decision rather than duplicating it.
        engine = import_module("sglang.srt.entrypoints.engine")
        return not engine.node_hosts_rust_server()
    except (ImportError, AttributeError, RuntimeError, ValueError):
        return None


def sglang_server_readiness() -> SglangServerReadiness:
    for module_name, getter_name in (
        ("sglang.srt.runtime_context", "get_serving"),
        ("sglang.srt.server_args", "get_global_server_args"),
    ):
        try:
            module = import_module(module_name)
            server_args = getattr(module, getter_name)()
            url = health_url_from_server_args(server_args)
        except (ImportError, AttributeError, RuntimeError, ValueError):
            continue
        if url is not None:
            dummy = _uses_dummy_listener(server_args, module)
            if dummy is None:
                continue
            if dummy:
                url = urlsplit(url)._replace(scheme="http").geturl()
            return SglangServerReadiness(url, local_dummy=dummy)
    return SglangServerReadiness()
