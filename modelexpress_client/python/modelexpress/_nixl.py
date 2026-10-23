# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL Python package compatibility."""

from __future__ import annotations

import importlib
from types import ModuleType


def load_nixl_api() -> ModuleType | None:
    """Prefer the NIXL wheel matching PyTorch's CUDA build.

    Both CUDA wheels can be installed; importing another major can load
    incompatible UCX libraries before NIXL initializes.
    """
    from torch.version import cuda

    packages = (
        (f"nixl_cu{cuda.split('.', 1)[0]}", "nixl")
        if cuda else ("nixl", "nixl_cu13", "nixl_cu12")
    )
    for package in packages:
        module_name = f"{package}._api"
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name not in {package, module_name}:
                raise
            continue
    return None
