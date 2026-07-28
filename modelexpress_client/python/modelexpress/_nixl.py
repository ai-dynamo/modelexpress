# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL Python package compatibility."""

from __future__ import annotations

import importlib
from types import ModuleType


def load_nixl_api() -> ModuleType | None:
    """Load the API from a CUDA-specific wheel or the NIXL meta-package."""
    for package in ("nixl_cu13", "nixl_cu12", "nixl"):
        module_name = f"{package}._api"
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name not in {package, module_name}:
                raise
            continue
    return None
