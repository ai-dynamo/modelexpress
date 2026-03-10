# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang adapter for ModelExpress TransferPlanner.

Extracts TargetParamInfo from a SGLang model by inspecting module types
(ColumnParallelLinear, RowParallelLinear) and parameter attributes
(output_dim, input_dim, output_sizes).

Usage::

    from modelexpress.adapters.sglang import extract_target_params
    from modelexpress.transfer_planner import TransferPlanner

    target_params = extract_target_params(model, tp_rank, tp_size)
    planner = TransferPlanner()
    ops, fixups = planner.compute_plan(source_index, target_params, tp_rank, tp_size)
"""

from __future__ import annotations

import logging

from ..transfer_planner import TargetParamInfo

logger = logging.getLogger("modelexpress.adapters.sglang")


def _is_column_parallel(module) -> bool:
    try:
        from sglang.srt.layers.linear import ColumnParallelLinear
        return isinstance(module, ColumnParallelLinear)
    except ImportError:
        return False


def _is_row_parallel(module) -> bool:
    try:
        from sglang.srt.layers.linear import RowParallelLinear
        return isinstance(module, RowParallelLinear)
    except ImportError:
        return False


def extract_target_params(
    model,
    tp_rank: int,
    tp_size: int,
) -> list[TargetParamInfo]:
    """Extract target parameter info from a SGLang model.

    Inspects each parameter's parent module to determine sharding strategy.

    Args:
        model: SGLang nn.Module with loaded (possibly dummy) weights.
        tp_rank: This worker's tensor-parallel rank.
        tp_size: Total tensor-parallel world size.

    Returns:
        List of TargetParamInfo for all model parameters.
    """
    param_to_module: dict[str, object] = {}
    for module_name, module in model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            param_to_module[full_name] = module

    result: list[TargetParamInfo] = []
    for name, param in model.named_parameters():
        module = param_to_module.get(name)

        output_dim = getattr(param, "output_dim", None)
        input_dim = getattr(param, "input_dim", None)

        is_row_par = module is not None and _is_row_parallel(module)
        is_col_par = module is not None and _is_column_parallel(module)

        # Infer shard_dim: FP8 quant sets both output_dim=0 and input_dim=1
        # on ALL weights, so we must check module type.
        if is_row_par and input_dim is not None and name.endswith(".weight"):
            shard_dim = input_dim
        elif output_dim is not None:
            shard_dim = output_dim
        elif input_dim is not None:
            shard_dim = input_dim
        elif is_col_par and (
            name.endswith(".weight") or name.endswith(".weight_scale")
        ):
            shard_dim = 0
        elif name.endswith(".weight") and is_row_par:
            shard_dim = 1
        else:
            shard_dim = -1

        effective_tp = 1
        shard_index = 0
        if shard_dim >= 0 and module is not None:
            effective_tp = getattr(module, "tp_size", tp_size)
            shard_index = getattr(module, "tp_rank", tp_rank)

        output_sizes = getattr(module, "output_sizes", None) if module else None

        result.append(TargetParamInfo(
            name=name,
            data_ptr=param.data_ptr(),
            numel=param.numel(),
            element_size=param.element_size(),
            shape=list(param.shape),
            dtype=param.dtype,
            device=param.device,
            param=param,
            shard_dim=shard_dim,
            shard_index=shard_index,
            effective_tp=effective_tp,
            output_sizes=list(output_sizes) if output_sizes is not None else None,
            is_contiguous=param.is_contiguous(),
        ))

    return result
