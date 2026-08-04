# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded HF-canonical capture from FSDP and DTensor models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .base import (
    BoundedBucketBuilder,
    CanonicalBucketConsumer,
    CanonicalTensorSpec,
    CollectiveDeadline,
    canonical_schema_plan,
    canonical_tensor_name,
    configured_rank,
    finish_collectives,
    synchronize_errors,
    synchronize_preflight,
)
from .canonical import (
    DEFAULT_CANONICAL_FORMAT_IDENTITY,
    CanonicalFormatIdentity,
    canonical_capture_units,
    validate_canonical_format_identity,
)


@dataclass(frozen=True)
class FsdpHfBucketConfig:
    """Dependencies and memory bound for one FSDP canonical export."""

    bucket_bytes: int = 256 * 1024 * 1024
    rank: Callable[[], int] | None = None
    state_dict_getter: Callable[[object], Mapping[str, object]] | None = None
    metadata_group: Any = None
    materializer_topology: Mapping[str, object] | None = None
    format_identity: CanonicalFormatIdentity = DEFAULT_CANONICAL_FORMAT_IDENTITY
    canonical_schema: Sequence[CanonicalTensorSpec] | None = None
    deadline_monotonic: float | None = None
    abort_collectives: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.bucket_bytes, int)
            or isinstance(self.bucket_bytes, bool)
            or self.bucket_bytes <= 0
        ):
            raise ValueError("bucket_bytes must be positive")
        if not isinstance(self.format_identity, CanonicalFormatIdentity):
            raise TypeError("format_identity must be CanonicalFormatIdentity")
        if self.materializer_topology is not None:
            if not isinstance(self.materializer_topology, Mapping):
                raise TypeError("materializer_topology must be a mapping")
            if any(
                not isinstance(name, str) or not name
                for name in self.materializer_topology
            ):
                raise ValueError(
                    "materializer_topology keys must be non-empty state-dict names"
                )


def _state_dict(model: object, config: FsdpHfBucketConfig) -> Mapping[str, object]:
    if config.state_dict_getter is not None:
        return config.state_dict_getter(model)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch distributed checkpoint state-dict support is required"
            ) from exc
        return get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=False, cpu_offload=False),
        )
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("FSDP source model must provide state_dict()")
    return state_dict()


def _sync_dtype(model: object, name: str, dtype: torch.dtype) -> torch.dtype:
    sync_dtypes = getattr(model, "_fsdp_sync_dtypes", None)
    if sync_dtypes is None:
        return dtype
    if not isinstance(sync_dtypes, Mapping):
        raise TypeError("model _fsdp_sync_dtypes must be a mapping")
    target = sync_dtypes.get(name, dtype)
    if not isinstance(target, torch.dtype):
        raise TypeError(f"FSDP sync dtype for {name!r} is not a torch.dtype")
    return target


_WEIGHT_BRIDGE_MODEL_TYPES = frozenset({"glm4_moe_lite", "qwen3_moe"})


def _model_type(model: object) -> str:
    value = getattr(getattr(model, "config", None), "model_type", "")
    if not isinstance(value, str):
        raise TypeError("FSDP model config.model_type must be a string")
    return value


def _needs_weight_bridge(model: object, name: str, tensor: torch.Tensor) -> bool:
    return (
        _model_type(model) in _WEIGHT_BRIDGE_MODEL_TYPES
        and tensor.dim() == 3
        and (
            name.endswith(".experts.gate_up_proj")
            or name.endswith(".experts.down_proj")
        )
    )


def _hf_tensors(
    model: object,
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
) -> list[tuple[str, torch.Tensor]]:
    if tensor.dtype is not target_dtype:
        tensor = tensor.to(dtype=target_dtype)
    if _needs_weight_bridge(model, name, tensor):
        try:
            from transformers.core_model_loading import revert_weight_conversion
        except ImportError as exc:
            raise RuntimeError(
                "Transformers WeightBridge conversion is required for "
                f"FSDP tensor {name!r}"
            ) from exc
        converted = revert_weight_conversion(model, {name: tensor})
        if not isinstance(converted, Mapping):
            raise TypeError("FSDP WeightBridge conversion must return a mapping")
    else:
        converted = {name: tensor}
    outputs = []
    for output_name, output in converted.items():
        canonical_name = canonical_tensor_name(output_name)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"HF save conversion output {canonical_name!r} is not a tensor"
            )
        outputs.append((canonical_name, output))
    return outputs


def _ordered_outputs(
    outputs: list[tuple[str, torch.Tensor]],
    positions: Mapping[str, int],
) -> list[tuple[str, torch.Tensor]]:
    try:
        return sorted(outputs, key=lambda item: positions[item[0]])
    except KeyError as exc:
        raise ValueError(
            f"FSDP conversion produced tensor outside canonical HF schema: {exc.args[0]!r}"
        ) from exc


def _materialize_tensor(value: object) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor, Replicate
    except ImportError:
        DTensor = ()
        Replicate = None
    if isinstance(value, DTensor):
        device_value = value.cuda()
        if torch.distributed.get_world_size() == 1:
            return device_value.full_tensor()
        assert Replicate is not None
        return device_value.redistribute(
            placements=[Replicate()] * device_value.device_mesh.ndim
        ).to_local()
    full_tensor = getattr(value, "full_tensor", None)
    if callable(full_tensor):
        return full_tensor()
    if not isinstance(value, torch.Tensor):
        raise TypeError("FSDP state value is not a tensor or DTensor")
    return value


def _stable_topology_descriptor(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        entries = []
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(
                    "materializer topology mapping keys must be non-empty strings"
                )
            entries.append((key, _stable_topology_descriptor(nested)))
        return tuple(sorted(entries))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_stable_topology_descriptor(nested) for nested in value)
    raise TypeError(
        "materializer topology must contain only stable scalar, mapping, and "
        "sequence values"
    )


def _rank_layout(value: object) -> int | tuple[object, ...]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ValueError("DTensor DeviceMesh rank layout must be non-empty")
        return tuple(_rank_layout(nested) for nested in value)
    raise TypeError("DTensor DeviceMesh must contain non-negative integer global ranks")


def _flatten_rank_layout(value: int | tuple[object, ...]) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    ranks = []
    for nested in value:
        if not isinstance(nested, (int, tuple)):
            raise TypeError("invalid DTensor DeviceMesh rank layout")
        ranks.extend(_flatten_rank_layout(nested))
    return tuple(ranks)


def _placement_descriptor(placement: object) -> tuple[object, ...]:
    try:
        from torch.distributed.tensor import Partial, Replicate, Shard
    except ImportError as exc:
        raise RuntimeError("PyTorch DTensor placements are required") from exc

    if type(placement) is Replicate:
        return ("replicate",)
    if type(placement) is Shard:
        dim = placement.dim
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("DTensor Shard placement dimension must be an integer")
        return ("shard", dim)
    if type(placement) is Partial:
        reduce_op = placement.reduce_op
        if not isinstance(reduce_op, str) or not reduce_op:
            raise TypeError("DTensor Partial placement reduction must be a string")
        return ("partial", reduce_op)
    raise TypeError(
        "unsupported DTensor placement type: "
        f"{type(placement).__module__}.{type(placement).__qualname__}"
    )


def _metadata_boundary_ranks(group: Any) -> frozenset[int]:
    if group is None:
        ranks = tuple(range(torch.distributed.get_world_size()))
    else:
        get_process_group_ranks = getattr(
            torch.distributed, "get_process_group_ranks", None
        )
        if not callable(get_process_group_ranks):
            raise RuntimeError(
                "PyTorch process-group rank enumeration is required to validate "
                "the FSDP metadata boundary"
            )
        ranks = tuple(get_process_group_ranks(group))
    if any(
        not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
        for rank in ranks
    ):
        raise TypeError("FSDP metadata boundary contains an invalid global rank")
    if len(set(ranks)) != len(ranks):
        raise ValueError("FSDP metadata boundary contains duplicate global ranks")
    return frozenset(ranks)


def _dtensor_descriptor(value: object, metadata_group: Any) -> tuple[object, ...]:
    mesh = value.device_mesh
    device_type = getattr(mesh, "device_type", None)
    if not isinstance(device_type, str) or not device_type:
        raise TypeError("DTensor DeviceMesh device_type must be a non-empty string")
    mesh_tensor = getattr(mesh, "mesh", None)
    if not isinstance(mesh_tensor, torch.Tensor):
        raise TypeError("DTensor DeviceMesh must expose its global-rank mesh tensor")
    rank_layout = _rank_layout(mesh_tensor.detach().cpu().tolist())
    mesh_ranks = _flatten_rank_layout(rank_layout)
    if len(set(mesh_ranks)) != len(mesh_ranks):
        raise ValueError("DTensor DeviceMesh contains duplicate global ranks")

    mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
    if mesh_dim_names is not None:
        mesh_dim_names = tuple(mesh_dim_names)
        if any(not isinstance(name, str) or not name for name in mesh_dim_names):
            raise TypeError("DTensor DeviceMesh dimension names must be strings")
    placements = tuple(_placement_descriptor(item) for item in value.placements)
    ndim = getattr(mesh, "ndim", mesh_tensor.ndim)
    if not isinstance(ndim, int) or isinstance(ndim, bool) or ndim != mesh_tensor.ndim:
        raise ValueError("DTensor DeviceMesh ndim disagrees with its rank layout")
    if mesh_dim_names is not None and len(mesh_dim_names) != ndim:
        raise ValueError("DTensor DeviceMesh dimension names do not match its ndim")
    if len(placements) != ndim:
        raise ValueError("DTensor placements do not match DeviceMesh ndim")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        boundary_ranks = _metadata_boundary_ranks(metadata_group)
        outside = sorted(set(mesh_ranks) - boundary_ranks)
        if outside:
            raise ValueError(
                f"DTensor DeviceMesh ranks {outside} are outside the MX metadata "
                "boundary"
            )
    return (
        "dtensor",
        device_type,
        rank_layout,
        mesh_dim_names,
        placements,
    )


def _materializer_descriptor(
    value: object,
    source_name: str,
    config: FsdpHfBucketConfig,
) -> tuple[object, ...]:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()
    if isinstance(value, DTensor):
        return _dtensor_descriptor(value, config.metadata_group)
    if isinstance(value, torch.Tensor):
        return ("tensor",)
    if callable(getattr(value, "full_tensor", None)):
        descriptor: tuple[object, ...] = (
            "full_tensor",
            f"{type(value).__module__}.{type(value).__qualname__}",
        )
        topology = config.materializer_topology
        if topology is not None and source_name in topology:
            return descriptor + (_stable_topology_descriptor(topology[source_name]),)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise TypeError(
                f"distributed FSDP materializer {source_name!r} requires an "
                "explicit stable topology descriptor"
            )
        return descriptor
    raise TypeError("FSDP state value is not a tensor or DTensor")


def _for_each_fsdp_hf_bucket(
    model: object,
    config: FsdpHfBucketConfig,
    consume_bucket: CanonicalBucketConsumer,
    deadline: CollectiveDeadline,
) -> None:
    """Materialize DTensors on every rank; emit bounded HF buckets on rank 0."""
    rank = 0
    ordered: list[
        tuple[str, object, torch.dtype, tuple[tuple[str, str, tuple[int, ...]], ...]]
    ] = []
    plan = None
    execution_plan = None
    capture_units = ()
    capture_unit_sizes = ()
    positions: dict[str, int] = {}
    local_error: str | None = None
    try:
        deadline.check("FSDP local preflight")
        validate_canonical_format_identity(config.format_identity)
        plan, planned_sizes = canonical_schema_plan(
            config.canonical_schema, config.bucket_bytes
        )
        positions = {entry[0]: index for index, entry in enumerate(plan)}
        capture_units, capture_unit_sizes = canonical_capture_units(
            planned_sizes,
            config.format_identity,
            config.bucket_bytes,
        )
        rank = configured_rank(config.rank, config.metadata_group)
    except Exception as exc:
        local_error = str(exc) or type(exc).__name__
    synchronize_preflight(
        local_error,
        plan,
        config.metadata_group,
        representation=config.format_identity,
    )
    deadline.check("FSDP local preflight completion")

    local_error = None
    try:
        state = _state_dict(model, config)
        if not isinstance(state, Mapping):
            raise TypeError("FSDP state_dict_getter must return a mapping")
        names: set[str] = set()
        source_groups = []
        for source_name, value in sorted(state.items(), key=lambda item: item[0]):
            full_tensor = getattr(value, "full_tensor", None)
            is_distributed = callable(full_tensor)
            if not is_distributed and not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"FSDP state value {source_name!r} is not a tensor or DTensor"
                )
            dtype = getattr(value, "dtype", None)
            shape = getattr(value, "shape", None)
            if not isinstance(dtype, torch.dtype) or shape is None:
                raise TypeError(
                    f"FSDP state value {source_name!r} has no tensor metadata"
                )
            target_dtype = _sync_dtype(model, source_name, dtype)
            source_shape = tuple(int(size) for size in shape)
            materializer = _materializer_descriptor(value, source_name, config)
            element_count = torch.Size(source_shape).numel()
            source_bytes = element_count * max(
                torch.empty((), dtype=dtype).element_size(),
                torch.empty((), dtype=target_dtype).element_size(),
            )
            if source_bytes > config.bucket_bytes:
                raise ValueError(
                    f"canonical tensor {source_name!r} size {source_bytes} exceeds "
                    f"bucket_bytes={config.bucket_bytes}"
                )
            metadata_tensor = torch.empty(source_shape, dtype=dtype, device="meta")
            converted = _ordered_outputs(
                _hf_tensors(model, source_name, metadata_tensor, target_dtype),
                positions,
            )
            converted_plan = tuple(
                (name, str(output.dtype), tuple(output.shape))
                for name, output in converted
            )
            converted_bytes = sum(
                output.numel() * output.element_size() for _name, output in converted
            )
            if converted_bytes > config.bucket_bytes:
                raise ValueError(
                    f"FSDP conversion group {source_name!r} output bytes "
                    f"{converted_bytes} exceed bucket_bytes={config.bucket_bytes}"
                )
            for (name, output), (_, output_dtype, output_shape) in zip(
                converted, converted_plan, strict=True
            ):
                output_bytes = output.numel() * output.element_size()
                if output_bytes > config.bucket_bytes:
                    raise ValueError(
                        f"canonical tensor {name!r} size {output_bytes} exceeds "
                        f"bucket_bytes={config.bucket_bytes}"
                    )
                if name in names:
                    raise ValueError(
                        f"canonical tensor {name!r} appeared more than once"
                    )
                names.add(name)
                expected = plan[positions[name]]
                if (name, output_dtype, output_shape) != expected[:3]:
                    raise ValueError(
                        f"FSDP tensor {name!r} differs from canonical HF schema"
                    )
            group_positions = tuple(positions[name] for name, *_rest in converted_plan)
            if group_positions != tuple(
                range(group_positions[0], group_positions[0] + len(group_positions))
            ):
                raise ValueError(
                    f"FSDP conversion outputs for {source_name!r} are not contiguous "
                    "in the canonical HF schema"
                )
            source_groups.append(
                (
                    group_positions[0],
                    source_name,
                    value,
                    target_dtype,
                    converted_plan,
                    (
                        source_name,
                        str(dtype),
                        source_shape,
                        str(target_dtype),
                        materializer,
                        converted_plan,
                    ),
                )
            )
        source_groups.sort(key=lambda item: item[0])
        if tuple(
            name
            for _position, _source, _value, _dtype, converted_plan, _entry in source_groups
            for name, _output_dtype, _shape in converted_plan
        ) != tuple(entry[0] for entry in plan):
            raise ValueError(
                "FSDP state does not provide complete canonical HF schema coverage"
            )
        ordered = [
            (source_name, value, target_dtype, converted_plan)
            for _position, source_name, value, target_dtype, converted_plan, _entry in source_groups
        ]
        execution_plan = tuple(entry for *_rest, entry in source_groups)
    except Exception as exc:
        local_error = str(exc) or type(exc).__name__
    synchronize_preflight(
        local_error,
        plan,
        config.metadata_group,
        representation=(config.format_identity, execution_plan),
    )
    deadline.check("FSDP execution preflight completion")

    builder = BoundedBucketBuilder(
        bucket_bytes=config.bucket_bytes,
        emit=rank == 0,
        consume_bucket=consume_bucket,
        capture_units=capture_units,
        capture_unit_sizes=capture_unit_sizes,
    )
    for source_name, value, target_dtype, converted_plan in ordered:
        deadline.check("FSDP tensor gather")
        tensor = None
        output = None
        outputs = None
        unit_error: str | None = None
        try:
            tensor = _materialize_tensor(value)
            deadline.check("FSDP tensor gather completion")
            outputs = _ordered_outputs(
                _hf_tensors(model, source_name, tensor, target_dtype), positions
            )
            actual_plan = tuple(
                (name, str(output.dtype), tuple(output.shape))
                for name, output in outputs
            )
            if actual_plan != converted_plan:
                raise ValueError(f"FSDP HF conversion plan changed for {source_name!r}")
            for name, output in outputs:
                builder.record(name, output)
            builder.flush_completed_units()
        except Exception as exc:
            unit_error = str(exc) or type(exc).__name__
            builder._record_error(exc)
        finally:
            output = None
            outputs = None
            tensor = None
        synchronize_errors(
            unit_error or builder.error,
            config.metadata_group,
        )
        deadline.check("FSDP tensor unit completion")
    builder.finish()
    synchronize_errors(
        builder.error,
        config.metadata_group,
        content_digest=builder.content_digest,
    )
    deadline.check("FSDP final error synchronization")
    finish_collectives(config.metadata_group)
    deadline.check("FSDP final barrier")


def for_each_fsdp_hf_bucket(
    model: object,
    config: FsdpHfBucketConfig,
    consume_bucket: CanonicalBucketConsumer,
) -> None:
    """Run one deadline-bounded FSDP capture on every trainer rank."""
    deadline = CollectiveDeadline(config.deadline_monotonic, config.abort_collectives)
    try:
        _for_each_fsdp_hf_bucket(model, config, consume_bucket, deadline)
    finally:
        deadline.close()
