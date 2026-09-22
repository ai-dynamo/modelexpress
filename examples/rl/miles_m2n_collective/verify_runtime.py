# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail fast unless the image contains the real MILES collective runtime."""

from __future__ import annotations

import ast
import ctypes
import importlib
import os
from importlib.metadata import entry_points
from pathlib import Path

MINIMUM_NCCL_VERSION = 23007
PLUGIN_NAME = "modelexpress_miles_collective"
MEGATRON_BRIDGE_SOURCE = Path("/opt/megatron-bridge/src").resolve()
AUTO_BRIDGE_SOURCE = (
    MEGATRON_BRIDGE_SOURCE
    / "megatron"
    / "bridge"
    / "models"
    / "conversion"
    / "auto_bridge.py"
)


def _nccl_version() -> int:
    library = ctypes.CDLL(os.environ["SGLANG_NCCL_SO_PATH"])
    version = ctypes.c_int()
    result = library.ncclGetVersion(ctypes.byref(version))
    if result != 0:
        raise RuntimeError(f"ncclGetVersion failed with result {result}")
    return version.value


def _verify_megatron_bridge() -> None:
    if not AUTO_BRIDGE_SOURCE.is_file():
        raise RuntimeError(
            f"Megatron-Bridge overlay is missing AutoBridge source: {AUTO_BRIDGE_SOURCE}"
        )
    module = ast.parse(AUTO_BRIDGE_SOURCE.read_text(), filename=str(AUTO_BRIDGE_SOURCE))
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != "AutoBridge":
            continue
        for child in node.body:
            if (
                not isinstance(child, ast.FunctionDef)
                or child.name != "export_hf_weights"
            ):
                continue
            keyword_only = {argument.arg for argument in child.args.kwonlyargs}
            if "yield_pp_local" in keyword_only:
                return
    raise RuntimeError(
        "Megatron-Bridge AutoBridge.export_hf_weights lacks the keyword-only "
        "yield_pp_local contract"
    )


def main() -> None:
    for module in ("miles", "nccl.m2n", "modelexpress_rl"):
        importlib.import_module(module)
    _verify_megatron_bridge()

    if os.environ.get("NCCL_CUMEM_ENABLE") != "1":
        raise RuntimeError(
            "MILES M2N requires NCCL_CUMEM_ENABLE=1 for symmetric-memory "
            "device communicators"
        )

    plugins = {entry.name for entry in entry_points(group="sglang.srt.plugins")}
    if PLUGIN_NAME not in plugins:
        raise RuntimeError(f"SGLang plugin {PLUGIN_NAME!r} is not installed")

    nccl_version = _nccl_version()
    if nccl_version < MINIMUM_NCCL_VERSION:
        raise RuntimeError(
            f"NCCL {nccl_version} is loaded; M2N requires at least {MINIMUM_NCCL_VERSION}"
        )

    print(
        "MILES M2N runtime ready: "
        f"NCCL={nccl_version} plugin={PLUGIN_NAME} "
        f"NCCL_CUMEM_ENABLE={os.environ['NCCL_CUMEM_ENABLE']} "
        f"LD_PRELOAD={os.environ.get('LD_PRELOAD')}"
    )


if __name__ == "__main__":
    main()
