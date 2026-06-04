# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang receiver-owned tensor refit smoke wrapper.

This module is intentionally narrower than a live SGLang engine integration. It
lets SGLang-owned or SGLang-shaped ``torch.nn.Module`` objects use the same
receiver-side slice request, multi-source planner, install, checksum/allclose,
and restore path as the live vLLM smoke helper. A real SGLang engine-owned
post-load refit artifact is still a Level-4 gap until a GPU runtime process is
schedulable and wired to this helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .refit_vllm_receiver_smoke import run_receiver_refit_on_module


def detect_sglang_version() -> tuple[str, bool]:
    """Return the importable SGLang version without requiring tests to import it."""

    try:
        import sglang
    except Exception:
        return "unavailable", False
    return getattr(sglang, "__version__", "unknown"), True


def run_sglang_receiver_refit_on_module(
    module: torch.nn.Module,
    *,
    model_name: str,
    model_version: str,
    module_path: str,
    sglang_version: str = "",
    sglang_imported: bool | None = None,
    real_runtime_engine_used: bool = False,
    preferred_tensor_name: str = "",
    artifact_path: str | Path | None = None,
    mode: str = "sglang-receiver-owned-tensor-smoke",
    model_path: str = "",
) -> dict[str, Any]:
    """Run the generic receiver-owned tensor smoke with SGLang metadata."""

    if sglang_imported is None or not sglang_version:
        detected_version, detected_imported = detect_sglang_version()
        if not sglang_version:
            sglang_version = detected_version
        if sglang_imported is None:
            sglang_imported = detected_imported

    return run_receiver_refit_on_module(
        module,
        model_name=model_name,
        model_version=model_version,
        module_path=module_path,
        runtime_framework="sglang",
        framework_version=sglang_version,
        runtime_imported=bool(sglang_imported),
        real_runtime_engine_used=real_runtime_engine_used,
        preferred_tensor_name=preferred_tensor_name,
        artifact_path=artifact_path,
        mode=mode,
        model_path=model_path,
    )
