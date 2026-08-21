# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL M2N collective reshard transport (optional ``[nccl-m2n]`` extra)."""

from .executor import NcclM2nExecutor, ReshardParam, build_reshard_params, run_reshard
from .runtime import _M2nPPGroupSpec, _M2nRuntime

__all__ = [
    "NcclM2nExecutor",
    "ReshardParam",
    "_M2nPPGroupSpec",
    "_M2nRuntime",
    "build_reshard_params",
    "run_reshard",
]
