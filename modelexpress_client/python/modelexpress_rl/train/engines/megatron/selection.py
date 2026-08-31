# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate Megatron-Bridge conversion tasks into MX publication specs."""

from __future__ import annotations

from typing import Any

from .aliases import MegatronTensorSpec


def _inherits(mapping: object, class_name: str) -> bool:
    return any(base.__name__ == class_name for base in type(mapping).__mro__)


def model_express_role_for_mapping(
    *, mapping: Any, megatron_module: Any, tensor_ndim: int
) -> str:
    """Return the MX role for one supported Megatron-Bridge mapping."""
    mapping_name = type(mapping).__name__
    is_expert = bool(getattr(mapping, "is_expert", False))
    if _inherits(mapping, "QKVMapping"):
        if is_expert:
            raise NotImplementedError(
                "ModelExpress does not support expert QKV mapping"
            )
        if tensor_ndim not in {1, 2}:
            raise NotImplementedError(
                "ModelExpress QKV publication supports only weights and biases"
            )
        return "qkv_column"
    if _inherits(mapping, "GatedMLPMapping"):
        return "expert_column" if is_expert else "gated_mlp_column"
    if _inherits(mapping, "ColumnParallelMapping"):
        return "expert_column" if is_expert else "column"
    if _inherits(mapping, "RowParallelMapping"):
        if tensor_ndim <= 1:
            return "replicated"
        return "expert_row" if is_expert else "row"
    if _inherits(mapping, "ReplicatedMapping") or _inherits(mapping, "DirectMapping"):
        return "replicated"
    if mapping_name == "AutoMapping":
        if getattr(mapping, "permute_dims", None) is not None:
            raise NotImplementedError(
                "ModelExpress publication does not yet support AutoMapping "
                "dimension permutation"
            )
        detected = mapping._detect_parallelism_type(megatron_module)
        if detected == "column":
            return "expert_column" if is_expert else "column"
        if detected == "row":
            if tensor_ndim <= 1:
                return "replicated"
            return "expert_row" if is_expert else "row"
        if detected == "replicated":
            return "replicated"
        raise ValueError(f"AutoMapping returned unsupported parallelism {detected!r}")
    raise NotImplementedError(
        f"ModelExpress publication does not support Megatron mapping {mapping_name}"
    )


def _qkv_descriptor_extras(
    *, transformer_config: Any, tensor_parallel_size: int, global_rows: int
) -> dict[str, str]:
    num_heads = int(transformer_config.num_attention_heads)
    num_kv_heads = int(transformer_config.num_query_groups)
    head_dim = int(transformer_config.kv_channels)
    if (
        tensor_parallel_size < 1
        or num_heads < 1
        or num_kv_heads < 1
        or head_dim < 1
        or num_heads % num_kv_heads
    ):
        raise ValueError("invalid global Q/KV geometry for ModelExpress")
    expected_rows = (num_heads + 2 * num_kv_heads) * head_dim
    if global_rows != expected_rows:
        raise ValueError(
            f"fused QKV rows {global_rows} disagree with global head geometry "
            f"{expected_rows}"
        )
    extras = {
        "qkv_interleave": "by_head",
        "head_dim": str(head_dim),
        "num_heads": str(num_heads),
        "num_kv_heads": str(num_kv_heads),
    }
    if (
        num_heads % tensor_parallel_size == 0
        and num_kv_heads % tensor_parallel_size == 0
    ):
        extras["num_heads_local"] = str(num_heads // tensor_parallel_size)
        extras["num_kv_heads_local"] = str(num_kv_heads // tensor_parallel_size)
    return extras


def build_megatron_tensor_specs(
    *,
    conversion_tasks: list[Any],
    transformer_config: Any,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    expert_tensor_parallel_size: int,
    expert_tensor_parallel_rank: int,
) -> list[MegatronTensorSpec]:
    """Translate local Megatron-Bridge conversion tasks into MX tensor specs."""
    for name, size, rank in (
        ("tensor", tensor_parallel_size, tensor_parallel_rank),
        ("expert tensor", expert_tensor_parallel_size, expert_tensor_parallel_rank),
    ):
        if size < 1 or not 0 <= rank < size:
            raise ValueError(f"invalid {name} parallel rank {rank} for size {size}")
    specs = []
    for task in conversion_tasks:
        tensor = task.param_weight
        if tensor is None:
            continue
        role = model_express_role_for_mapping(
            mapping=task.mapping,
            megatron_module=task.megatron_module,
            tensor_ndim=tensor.ndim,
        )
        hf_param = task.mapping.hf_param
        is_expert = bool(getattr(task.mapping, "is_expert", False))
        parallel_size = (
            expert_tensor_parallel_size if is_expert else tensor_parallel_size
        )
        parallel_rank = (
            expert_tensor_parallel_rank if is_expert else tensor_parallel_rank
        )
        extras: dict[str, str] = {}
        if role == "qkv_column":
            hf_names = tuple(hf_param[key] for key in ("q", "k", "v"))
            extras = _qkv_descriptor_extras(
                transformer_config=transformer_config,
                tensor_parallel_size=tensor_parallel_size,
                global_rows=int(tensor.shape[0]) * tensor_parallel_size,
            )
            shard_axis = 0
        elif role in {"gated_mlp_column", "expert_column"} and _inherits(
            task.mapping, "GatedMLPMapping"
        ):
            hf_names = (hf_param["gate"], hf_param["up"])
            extras = {"gated_mlp_order": "gate_then_up"}
            shard_axis = 0
        elif role in {"column", "expert_column"}:
            hf_names = (str(hf_param),)
            shard_axis = 0
        elif role in {"row", "expert_row"}:
            hf_names = (str(hf_param),)
            shard_axis = 1
        else:
            hf_names = (str(hf_param),)
            shard_axis = None

        if role == "replicated":
            if tensor_parallel_rank != 0:
                continue
            global_shape = tuple(int(dim) for dim in tensor.shape)
            shard_range = None
            placement = "REPLICATE"
        else:
            assert shard_axis is not None
            local_extent = int(tensor.shape[shard_axis])
            global_shape_list = [int(dim) for dim in tensor.shape]
            global_shape_list[shard_axis] *= parallel_size
            global_shape = tuple(global_shape_list)
            shard_range = (
                parallel_rank * local_extent,
                (parallel_rank + 1) * local_extent,
            )
            placement = "SHARD"

        specs.append(
            MegatronTensorSpec(
                name=task.global_param_name,
                tensor=tensor.detach(),
                role=role,
                hf_names=hf_names,
                global_shape=global_shape,
                placement_kind=placement,
                shard_axis=shard_axis,
                local_shard_range=shard_range,
                extras=extras,
            )
        )
    if not specs:
        raise RuntimeError("this Megatron rank has no ModelExpress tensors")
    return specs


__all__ = ["build_megatron_tensor_specs", "model_express_role_for_mapping"]
