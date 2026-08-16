# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL M2N collective reshard transport (optional ``[nccl-m2n]`` extra)."""

from .executor import NcclM2nExecutor, ReshardParam, build_reshard_params, run_reshard
from .layout import LayoutShard, LayoutTable, LayoutTensor
from .mesh import REPLICATE, Mesh, build_tp_meshes, shard_dim_from_layout, tile_shape
from .runtime import _M2nLaneSpec, _M2nRuntime

__all__ = [
    "LayoutShard",
    "LayoutTable",
    "LayoutTensor",
    "Mesh",
    "NcclM2nExecutor",
    "REPLICATE",
    "ReshardParam",
    "_M2nLaneSpec",
    "_M2nRuntime",
    "build_reshard_params",
    "build_tp_meshes",
    "run_reshard",
    "shard_dim_from_layout",
    "tile_shape",
]
