# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded HF-canonical capture from Megatron models through Megatron-Bridge."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from .base import (
    BoundedBucketBuilder,
    CanonicalBucketConsumer,
    CanonicalSourceError,
    CanonicalTensorSpec,
    CollectiveDeadline,
    canonical_schema_plan,
    canonical_tensor_name,
    configured_rank,
    finish_collectives,
    synchronize_errors,
    synchronize_preflight,
    tensor_nbytes,
)
from .canonical import (
    DEFAULT_CANONICAL_FORMAT_IDENTITY,
    CanonicalFormatIdentity,
    canonical_capture_units,
    validate_canonical_format_identity,
)


@dataclass(frozen=True)
class MegatronBridgeHfBucketConfig:
    """Dependencies and memory bound for one Megatron canonical export."""

    bucket_bytes: int = 256 * 1024 * 1024
    hf_model_path: str | None = None
    bridge: Any = None
    rank: Callable[[], int] | None = None
    model_context: Callable[[], AbstractContextManager[Any]] | None = None
    weights_getter: Callable[[], Mapping[str, torch.Tensor]] | None = None
    metadata_group: Any = None
    routing_group: Any = None
    format_identity: CanonicalFormatIdentity = DEFAULT_CANONICAL_FORMAT_IDENTITY
    vocab_size: int | None = None
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
        if self.bridge is None and not self.hf_model_path:
            raise ValueError("bridge or hf_model_path is required")
        if not isinstance(self.format_identity, CanonicalFormatIdentity):
            raise TypeError("format_identity must be CanonicalFormatIdentity")
        if self.vocab_size is not None and (
            not isinstance(self.vocab_size, int)
            or isinstance(self.vocab_size, bool)
            or self.vocab_size <= 0
        ):
            raise ValueError("vocab_size must be a positive integer")


def _load_bridge(config: MegatronBridgeHfBucketConfig) -> Any:
    if config.bridge is not None:
        return config.bridge
    try:
        from megatron.bridge import AutoBridge
    except ImportError as exc:
        raise CanonicalSourceError(
            "Megatron-Bridge is required for Megatron canonical capture"
        ) from exc
    return AutoBridge.from_hf_pretrained(config.hf_model_path, trust_remote_code=True)


@contextmanager
def _patch_megatron_model(model: object) -> Iterator[None]:
    """Apply the bounded Bridge compatibility patch around conversion only."""
    from megatron.core.utils import unwrap_model

    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    unwrapped = unwrap_model(chunks)
    root = unwrapped[0] if isinstance(unwrapped, (list, tuple)) else unwrapped
    config = root.config
    added_share_flag = not hasattr(config, "share_embeddings_and_output_weights")
    if added_share_flag:
        config.share_embeddings_and_output_weights = (
            root.share_embeddings_and_output_weights
        )
    try:
        for chunk in chunks:
            modules = chunk.modules() if hasattr(chunk, "modules") else (chunk,)
            for module in modules:
                maintain = getattr(module, "_maintain_float32_expert_bias", None)
                if callable(maintain):
                    maintain()
        yield
    finally:
        if added_share_flag:
            delattr(config, "share_embeddings_and_output_weights")


def _replace_task_weight(task: Any, weight: torch.Tensor) -> Any:
    replacement = weight.detach().to(device=task.param_weight.device)
    if dataclasses.is_dataclass(task):
        return dataclasses.replace(task, param_weight=replacement)
    if hasattr(task, "_replace"):
        return task._replace(param_weight=replacement)
    raise TypeError(
        f"unsupported Megatron-Bridge conversion task {type(task).__name__}"
    )


def _conversion_tasks(
    bridge: Any,
    model: object,
    weights_getter: Callable[[], Mapping[str, torch.Tensor]] | None,
) -> tuple[list[Any], dict[int, torch.Tensor]]:
    getter = getattr(bridge, "get_conversion_tasks", None)
    if not callable(getter):
        raise TypeError("Megatron-Bridge must provide get_conversion_tasks()")
    tasks = list(getter(model))
    weights = weights_getter() if weights_getter is not None else None
    if weights is not None and not isinstance(weights, Mapping):
        raise TypeError("weights_getter must return a mapping")
    normalized: dict[str, torch.Tensor] = {}
    for source_name, weight in (weights or {}).items():
        if not isinstance(source_name, str):
            raise TypeError("authoritative model handle names must be strings")
        while source_name.startswith("module."):
            source_name = source_name[len("module.") :]
        if source_name in normalized:
            raise ValueError(f"authoritative model handle duplicates {source_name!r}")
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"authoritative weight {source_name!r} is not a tensor")
        normalized[source_name] = weight
    replacements = {}
    for ordinal, task in enumerate(tasks):
        if task is None:
            continue
        vp_stage = getattr(task, "vp_stage", None)
        if vp_stage is not None and not isinstance(vp_stage, int):
            raise TypeError("Megatron-Bridge task has an invalid vp_stage")
        if not isinstance(getattr(task, "param_name", None), str):
            raise TypeError("Megatron-Bridge task has an invalid param_name")
        global_param_name = getattr(task, "global_param_name", None)
        if global_param_name is not None and not isinstance(global_param_name, str):
            raise TypeError("Megatron-Bridge task has an invalid global_param_name")
        pp_rank = getattr(task, "pp_rank", None)
        if pp_rank is not None and not isinstance(pp_rank, int):
            raise TypeError("Megatron-Bridge task has an invalid pp_rank")
        param_weight = getattr(task, "param_weight", None)
        if param_weight is not None and not isinstance(param_weight, torch.Tensor):
            raise TypeError("Megatron-Bridge task param_weight is not a tensor")
        if param_weight is None:
            continue
        if weights is None:
            continue
        if vp_stage is None:
            raise TypeError("an owning Megatron-Bridge task must identify its vp_stage")
        key = f"vp_stages.{vp_stage}.{task.param_name}"
        if key in normalized:
            replacement = normalized[key]
            if tuple(replacement.shape) != tuple(param_weight.shape):
                raise ValueError(
                    f"authoritative replacement {key!r} shape differs from its "
                    "Megatron-Bridge task"
                )
            if replacement.dtype is not param_weight.dtype:
                raise ValueError(
                    f"authoritative replacement {key!r} dtype differs from its "
                    "Megatron-Bridge task"
                )
            if replacement.device.type != param_weight.device.type:
                raise ValueError(
                    f"authoritative replacement {key!r} device differs from its "
                    "Megatron-Bridge task"
                )
            replacements[ordinal] = replacement
    return tasks, replacements


def _postprocess_export(exported, vocab_size: int | None):
    name, tensor = exported[0], exported[1]
    native_name = exported[2] if len(exported) > 2 else None
    if isinstance(native_name, str):
        while native_name.startswith("module."):
            native_name = native_name[len("module.") :]
        if native_name in {
            "embedding.word_embeddings.weight",
            "output_layer.weight",
        }:
            if vocab_size is None:
                raise ValueError(
                    "vocab_size is required to remove Megatron vocabulary padding"
                )
            if not isinstance(tensor, torch.Tensor) or not tensor.shape:
                raise TypeError("padded Megatron vocabulary weight is not a matrix")
            if tensor.shape[0] < vocab_size:
                raise ValueError(
                    "Megatron vocabulary weight is smaller than vocab_size"
                )
            tensor = tensor[:vocab_size]
    return name, tensor


def _distributed_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _mapping_descriptor(task: Any) -> tuple[str | None, str | None]:
    mapping = getattr(task, "mapping", None)
    if mapping is None:
        return None, None
    mapping_type = f"{type(mapping).__module__}.{type(mapping).__qualname__}"
    if not bool(getattr(mapping, "is_grouped_export", False)):
        return mapping_type, None
    group_key = getattr(mapping, "group_key", None)
    if not isinstance(group_key, str) or not group_key:
        raise TypeError(
            "grouped Megatron-Bridge mappings must provide a non-empty group_key"
        )
    return mapping_type, group_key


def _hf_mapping_descriptor(task: Any) -> tuple[object, ...] | None:
    mapping = getattr(task, "mapping", None)
    if mapping is None:
        return None
    hf_param = getattr(mapping, "hf_param", None)
    if isinstance(hf_param, str):
        return ("name", canonical_tensor_name(hf_param))
    if isinstance(hf_param, Mapping) and hf_param:
        items = []
        for key, name in hf_param.items():
            if not isinstance(key, str) or not key:
                raise TypeError("Megatron-Bridge HF mapping keys must be strings")
            if not isinstance(name, str):
                raise TypeError("Megatron-Bridge HF mapping names must be strings")
            items.append((key, canonical_tensor_name(name)))
        return ("mapping", tuple(sorted(items)))
    if hf_param is not None:
        raise TypeError("Megatron-Bridge task has an invalid hf_param mapping")
    return None


def _collective_group_descriptor(
    role: str,
    group: Any,
    declared_size: object,
) -> tuple[str, str, tuple[int, ...]]:
    if group is None:
        ranks = tuple(range(torch.distributed.get_world_size()))
    else:
        get_ranks = getattr(torch.distributed, "get_process_group_ranks", None)
        if not callable(get_ranks):
            raise CanonicalSourceError(
                "PyTorch process-group rank discovery is required for "
                "Megatron-Bridge capture"
            )
        try:
            ranks = tuple(get_ranks(group))
        except Exception as exc:
            raise CanonicalSourceError(
                f"Megatron-Bridge {role} group is unreadable: {exc}"
            ) from exc
    if (
        not ranks
        or len(ranks) != len(set(ranks))
        or any(not isinstance(rank, int) or rank < 0 for rank in ranks)
    ):
        raise CanonicalSourceError(
            f"Megatron-Bridge {role} group has invalid global ranks"
        )
    if declared_size is not None and (
        not isinstance(declared_size, int)
        or isinstance(declared_size, bool)
        or declared_size != len(ranks)
    ):
        raise CanonicalSourceError(
            f"Megatron-Bridge {role} group size differs from its mapping"
        )
    try:
        backend = str(torch.distributed.get_backend(group)).lower()
    except (RuntimeError, ValueError) as exc:
        if group is not None:
            raise CanonicalSourceError(
                f"Megatron-Bridge {role} group has no backend: {exc}"
            ) from exc
        backend = "gloo"
    return role, backend, ranks


def _mapping_topology(task: Any) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    if not _distributed_is_initialized():
        return ()
    mapping = getattr(task, "mapping", None)
    is_expert = bool(getattr(mapping, "is_expert", False))
    pp_group = getattr(mapping, "pp_group", None)
    tp_group = getattr(mapping, "tp_group", None)
    topology = [
        _collective_group_descriptor(
            "pipeline", pp_group, getattr(mapping, "pp_size", None)
        ),
        _collective_group_descriptor(
            "expert_tensor" if is_expert else "tensor",
            tp_group,
            getattr(mapping, "tp_size", None),
        ),
    ]
    if is_expert:
        topology.append(
            _collective_group_descriptor(
                "expert",
                getattr(mapping, "ep_group", None),
                getattr(mapping, "ep_size", None),
            )
        )
    return tuple(topology)


def _planned_output_names(
    task: Any,
    group_key: str | None,
    positions: Mapping[str, int],
) -> tuple[str, ...]:
    export_hook = getattr(task, "export_hook", None)
    if export_hook is not None:
        raise TypeError(
            "Megatron-Bridge tasks with export_hook do not expose a stable "
            "canonical HF output plan"
        )

    mapping = getattr(task, "mapping", None)
    hf_param = getattr(mapping, "hf_param", None)
    if group_key is not None:
        candidates: tuple[object, ...] = (group_key,)
    elif isinstance(hf_param, str):
        candidates = (hf_param,)
    elif isinstance(hf_param, Mapping) and hf_param:
        candidates = tuple(hf_param.values())
    elif hf_param is None:
        candidates = (
            getattr(task, "global_param_name", None)
            or getattr(task, "param_name", None),
        )
    else:
        raise TypeError(
            "Megatron-Bridge task mapping has no stable canonical HF output plan"
        )

    if not candidates or any(not isinstance(name, str) for name in candidates):
        raise TypeError(
            "Megatron-Bridge task mapping has invalid canonical HF output names"
        )
    names = tuple(canonical_tensor_name(name) for name in candidates)
    if len(names) != len(set(names)):
        raise ValueError("Megatron-Bridge task output plan contains duplicate names")
    unknown = tuple(name for name in names if name not in positions)
    if unknown:
        raise ValueError(
            f"Megatron-Bridge task output plan is outside the canonical HF schema: "
            f"{unknown!r}"
        )
    return tuple(sorted(names, key=positions.__getitem__))


def _conversion_units(
    tasks: Sequence[Any],
    replacements: Mapping[int, torch.Tensor],
    bucket_bytes: int,
    canonical_plan: Sequence[tuple[str, str, tuple[int, ...], bool]],
    named_sizes: Sequence[tuple[str, int]],
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[Any, ...], ...],
    tuple[tuple[str, tuple[int, ...], str] | None, ...],
]:
    if not tasks:
        raise ValueError("Megatron-Bridge returned no conversion task schema")
    schedule = []
    ownership = []
    group_keys = []
    positions = {entry[0]: index for index, entry in enumerate(canonical_plan)}
    canonical_sizes = dict(named_sizes)
    task_outputs: list[tuple[str, ...]] = []
    for ordinal, task in enumerate(tasks):
        if task is None:
            schedule.append((ordinal, None, None, None, None, (), None))
            ownership.append((None, ()))
            group_keys.append(None)
            task_outputs.append(())
            continue
        weight = replacements.get(ordinal, task.param_weight)
        target_device = (
            task.param_weight.device.type
            if ordinal in replacements
            else weight.device.type
            if weight is not None
            else None
        )
        owner_metadata = (
            None
            if weight is None
            else (str(weight.dtype), tuple(weight.shape), target_device)
        )
        if weight is not None and tensor_nbytes(weight) > bucket_bytes:
            raise ValueError(
                f"Megatron conversion task {task.param_name!r} local tensor size "
                f"{tensor_nbytes(weight)} exceeds bucket_bytes={bucket_bytes}"
            )
        global_name = getattr(task, "global_param_name", task.param_name)
        mapping_type, group_key = _mapping_descriptor(task)
        mapping_semantics = _hf_mapping_descriptor(task)
        output_names = _planned_output_names(task, group_key, positions)
        weight_dtype = getattr(task, "weight_dtype", None)
        if weight_dtype is not None and not isinstance(weight_dtype, torch.dtype):
            raise TypeError("Megatron-Bridge task has an invalid weight_dtype")
        schedule.append(
            (
                ordinal,
                global_name,
                mapping_type,
                group_key,
                mapping_semantics,
                output_names,
                None if weight_dtype is None else str(weight_dtype),
            )
        )
        ownership.append((owner_metadata, _mapping_topology(task)))
        group_keys.append(group_key)
        task_outputs.append(output_names)

    grouped_indices: dict[str, list[int]] = {}
    for index, group_key in enumerate(group_keys):
        if group_key is not None:
            grouped_indices.setdefault(group_key, []).append(index)

    planned_units = []
    seen_groups: set[str] = set()
    for position, task in enumerate(tasks):
        if task is None:
            continue
        group_key = group_keys[position]
        if group_key is None:
            unit = (position,)
        else:
            if group_key in seen_groups:
                continue
            unit = tuple(grouped_indices[group_key])
            seen_groups.add(group_key)
        local_bytes = sum(
            tensor_nbytes(replacements.get(index, tasks[index].param_weight))
            for index in unit
            if replacements.get(index, tasks[index].param_weight) is not None
        )
        if local_bytes > bucket_bytes:
            raise ValueError(
                "Megatron conversion unit local tensor bytes "
                f"{local_bytes} exceed bucket_bytes={bucket_bytes}"
            )
        output_names = tuple(
            sorted(
                {name for task_index in unit for name in task_outputs[task_index]},
                key=positions.__getitem__,
            )
        )
        output_positions = tuple(positions[name] for name in output_names)
        if not output_positions or output_positions != tuple(
            range(output_positions[0], output_positions[0] + len(output_positions))
        ):
            raise ValueError(
                "Megatron conversion unit output is not contiguous in the "
                "canonical HF schema"
            )
        output_bytes = sum(canonical_sizes[name] for name in output_names)
        if output_bytes > bucket_bytes:
            raise ValueError(
                "Megatron conversion unit planned output bytes "
                f"{output_bytes} exceed bucket_bytes={bucket_bytes}"
            )
        planned_units.append((output_positions[0], unit, output_names))

    planned_units.sort(key=lambda item: item[0])
    flattened_outputs = tuple(
        name
        for _position, _unit, output_names in planned_units
        for name in output_names
    )
    expected_outputs = tuple(entry[0] for entry in canonical_plan)
    if flattened_outputs != expected_outputs:
        raise ValueError(
            "Megatron conversion tasks do not provide complete canonical HF "
            "schema coverage"
        )
    return (
        tuple(unit for _position, unit, _outputs in planned_units),
        tuple(schedule),
        tuple(ownership),
    )


def _synchronize_preflight(
    local_error,
    identity,
    canonical_plan,
    task_schedule,
    task_ownership,
    group,
):
    contribution = (
        local_error,
        identity,
        canonical_plan,
        task_schedule,
        task_ownership,
    )
    if not _distributed_is_initialized():
        if local_error is not None:
            raise CanonicalSourceError(local_error)
        gathered = [contribution]
    else:
        gathered = [None] * torch.distributed.get_world_size(group)
        try:
            torch.distributed.all_gather_object(gathered, contribution, group=group)
        except Exception as exc:
            raise CanonicalSourceError(
                f"Megatron canonical preflight synchronization failed: {exc}"
            ) from exc
    failures = [
        f"rank {rank}: {item[0]}"
        for rank, item in enumerate(gathered)
        if isinstance(item, tuple) and len(item) == 5 and item[0]
    ]
    if failures:
        raise CanonicalSourceError("; ".join(failures))
    if any(
        not isinstance(item, tuple)
        or len(item) != 5
        or item[1] != identity
        or item[2] != canonical_plan
        or item[3] != task_schedule
        for item in gathered
    ):
        raise CanonicalSourceError(
            "Megatron canonical schema or global conversion schedule differs "
            "across trainer ranks"
        )
    assert task_schedule is not None
    assert task_ownership is not None
    validate_topology = _distributed_is_initialized()
    boundary_ranks = tuple(_global_rank(group, rank) for rank in range(len(gathered)))
    boundary_rank_set = set(boundary_ranks)
    for task_index, descriptor in enumerate(task_schedule):
        ownership_entries = []
        for item in gathered:
            if (
                len(item[4]) != len(task_schedule)
                or not isinstance(item[4][task_index], tuple)
                or len(item[4][task_index]) != 2
            ):
                raise CanonicalSourceError(
                    "Megatron-Bridge task ownership or collective topology is malformed"
                )
            owner_metadata, topology = item[4][task_index]
            if not isinstance(topology, tuple):
                raise CanonicalSourceError(
                    "Megatron-Bridge task collective topology is malformed"
                )
            ownership_entries.append((owner_metadata, topology))
        if descriptor[1] is None:
            if any(
                owner_metadata is not None or topology
                for owner_metadata, topology in ownership_entries
            ):
                raise CanonicalSourceError(
                    "unmapped Megatron-Bridge task has an owning tensor"
                )
            continue
        owners = [
            (boundary_ranks[rank], owner_metadata)
            for rank, (owner_metadata, _topology) in enumerate(ownership_entries)
            if owner_metadata is not None
        ]
        if not owners:
            raise CanonicalSourceError(
                f"Megatron-Bridge task {descriptor[1]!r} has no owning rank"
            )
        if len({metadata for _rank, metadata in owners}) != 1:
            raise CanonicalSourceError(
                f"Megatron-Bridge task {descriptor[1]!r} owner metadata differs "
                "across trainer ranks"
            )
        if not validate_topology:
            continue

        reporters: dict[tuple[str, str, tuple[int, ...]], set[int]] = {}
        pipeline_groups: set[tuple[int, ...]] = set()
        for rank, (global_rank, (_metadata, topology)) in enumerate(
            zip(boundary_ranks, ownership_entries, strict=True)
        ):
            del rank
            roles: set[str] = set()
            for group_descriptor in topology:
                if (
                    not isinstance(group_descriptor, tuple)
                    or len(group_descriptor) != 3
                ):
                    raise CanonicalSourceError(
                        "Megatron-Bridge task collective topology is malformed"
                    )
                role, backend, ranks = group_descriptor
                if (
                    not isinstance(role, str)
                    or role in roles
                    or not isinstance(backend, str)
                    or not isinstance(ranks, tuple)
                    or global_rank not in ranks
                    or not set(ranks).issubset(boundary_rank_set)
                ):
                    raise CanonicalSourceError(
                        "Megatron-Bridge task collective topology is inconsistent"
                    )
                roles.add(role)
                reporters.setdefault(group_descriptor, set()).add(global_rank)
                if role == "pipeline":
                    pipeline_groups.add(ranks)
            if "pipeline" not in roles or not ({"tensor", "expert_tensor"} & roles):
                raise CanonicalSourceError(
                    "Megatron-Bridge task collective topology is incomplete"
                )
        if any(
            reporters[group_descriptor] != set(group_descriptor[2])
            for group_descriptor in reporters
        ):
            raise CanonicalSourceError(
                "Megatron-Bridge task collective topology differs across group members"
            )
        owner_ranks = {rank for rank, _metadata in owners}
        if any(not owner_ranks.intersection(ranks) for ranks in pipeline_groups):
            raise CanonicalSourceError(
                f"Megatron-Bridge task {descriptor[1]!r} has no owner in one "
                "pipeline lane"
            )


def _tensor_record(name: str, tensor: torch.Tensor) -> tuple[Any, ...]:
    cpu = tensor.detach().to(device="cpu").contiguous()
    data = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return (
        canonical_tensor_name(name),
        str(cpu.dtype),
        tuple(cpu.shape),
        f"sha256:{hashlib.sha256(data).hexdigest()}",
    )


def _global_rank(group: Any, group_rank: int) -> int:
    if group is None:
        return group_rank
    get_global_rank = getattr(torch.distributed, "get_global_rank", None)
    if not callable(get_global_rank):
        raise CanonicalSourceError(
            "PyTorch get_global_rank is required for Megatron peer tensor routing"
        )
    return get_global_rank(group, group_rank)


def _validated_routing_group(metadata_group: Any, routing_group: Any):
    if not _distributed_is_initialized():
        return routing_group, None
    resolved = routing_group if routing_group is not None else metadata_group
    try:
        backend = str(torch.distributed.get_backend(resolved)).lower()
    except (RuntimeError, ValueError) as exc:
        # Unit tests may emulate initialized collectives without constructing a
        # real default group. A real initialized group always has a backend.
        if resolved is not None:
            raise CanonicalSourceError(
                f"Megatron peer routing group is invalid: {exc}"
            ) from exc
        backend = "gloo"
    if backend != "gloo":
        raise CanonicalSourceError(
            "Megatron canonical peer routing requires a Gloo process group"
        )

    world_size = torch.distributed.get_world_size(metadata_group)
    metadata_ranks = tuple(
        _global_rank(metadata_group, rank) for rank in range(world_size)
    )
    if resolved is None:
        routing_ranks = tuple(range(torch.distributed.get_world_size()))
    else:
        get_ranks = getattr(torch.distributed, "get_process_group_ranks", None)
        if not callable(get_ranks):
            raise CanonicalSourceError(
                "PyTorch process-group rank discovery is required for Megatron "
                "peer routing"
            )
        try:
            routing_ranks = tuple(get_ranks(resolved))
        except Exception as exc:
            raise CanonicalSourceError(
                f"Megatron peer routing group is unreadable: {exc}"
            ) from exc
    if routing_ranks != metadata_ranks:
        raise CanonicalSourceError(
            "Megatron peer routing group must exactly match the canonical "
            "metadata boundary"
        )
    return resolved, (backend, routing_ranks)


def _synchronize_task_outputs(
    *,
    local_error: str | None,
    local_outputs: Sequence[tuple[str, torch.Tensor]],
    canonical_plan,
    next_position: int,
    rank: int,
    group: Any,
    routing_group: Any,
    builder: BoundedBucketBuilder,
) -> int:
    try:
        records = tuple(_tensor_record(name, tensor) for name, tensor in local_outputs)
    except Exception as exc:
        if local_error is None:
            local_error = str(exc) or type(exc).__name__
        records = ()
    record_names = tuple(record[0] for record in records)
    if len(record_names) != len(set(record_names)) and local_error is None:
        local_error = "canonical tensor appeared more than once in one Bridge task"
    contribution = (local_error, records)
    if not _distributed_is_initialized():
        if local_error is not None:
            raise CanonicalSourceError(local_error)
        gathered = [contribution]
    else:
        gathered = [None] * torch.distributed.get_world_size(group)
        try:
            torch.distributed.all_gather_object(gathered, contribution, group=group)
        except Exception as exc:
            raise CanonicalSourceError(
                f"Megatron task output synchronization failed: {exc}"
            ) from exc
    failures = [
        f"rank {owner}: {item[0]}"
        for owner, item in enumerate(gathered)
        if isinstance(item, tuple) and len(item) == 2 and item[0]
    ]
    if failures:
        raise CanonicalSourceError("; ".join(failures))

    positions = {entry[0]: index for index, entry in enumerate(canonical_plan)}
    owners: dict[str, list[tuple[int, tuple[Any, ...]]]] = {}
    for owner, item in enumerate(gathered):
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[1], tuple)
        ):
            raise CanonicalSourceError("Megatron task output contribution is malformed")
        for record in item[1]:
            if not isinstance(record, tuple) or len(record) != 4:
                raise CanonicalSourceError("Megatron task output metadata is malformed")
            owners.setdefault(record[0], []).append((owner, record))
    try:
        task_names = sorted(owners, key=positions.__getitem__)
    except KeyError as exc:
        raise CanonicalSourceError(
            f"Megatron export produced tensor outside canonical HF schema: {exc.args[0]!r}"
        ) from exc
    expected_positions = tuple(range(next_position, next_position + len(task_names)))
    if tuple(positions[name] for name in task_names) != expected_positions:
        raise CanonicalSourceError(
            "Megatron conversion task output is not the next complete canonical HF schema segment"
        )

    completion_error: str | None = None
    try:
        local_by_name = {name: tensor for name, tensor in local_outputs}
        for name in task_names:
            replicas = owners[name]
            expected = canonical_plan[positions[name]]
            metadata = {(record[1], record[2]) for _owner, record in replicas}
            digests = {record[3] for _owner, record in replicas}
            if metadata != {(expected[1], expected[2])}:
                raise CanonicalSourceError(
                    f"Megatron tensor {name!r} differs from canonical HF schema"
                )
            if len(digests) != 1:
                raise CanonicalSourceError(
                    f"canonical replica content differs for tensor {name!r}"
                )
            selected = min(owner for owner, _record in replicas)
            if selected == 0:
                if rank == 0:
                    builder.record(name, local_by_name[name])
                continue
            dtype = getattr(torch, expected[1].removeprefix("torch."), None)
            if not isinstance(dtype, torch.dtype):
                raise CanonicalSourceError(
                    f"canonical HF schema has unsupported dtype {expected[1]!r}"
                )
            if rank == selected:
                route_error: str | None = None
                try:
                    routed = local_by_name[name].detach().to(device="cpu").contiguous()
                except Exception as exc:
                    route_error = str(exc) or type(exc).__name__
                    routed = None
            else:
                route_error = None
                try:
                    routed = torch.empty(expected[2], dtype=dtype, device="cpu")
                except Exception as exc:
                    route_error = str(exc) or type(exc).__name__
                    routed = None
            synchronize_errors(route_error, group)
            assert routed is not None
            try:
                torch.distributed.broadcast(
                    routed,
                    src=_global_rank(group, selected),
                    group=routing_group,
                )
            except Exception as exc:
                raise CanonicalSourceError(
                    f"Megatron peer tensor routing failed for {name!r}: {exc}"
                ) from exc
            if rank == 0:
                builder.record(name, routed)
        builder.flush_completed_units()
        completion_error = builder.error
    except Exception as exc:
        completion_error = str(exc) or type(exc).__name__
    synchronize_errors(completion_error, group)
    return next_position + len(task_names)


def _for_each_megatron_hf_bucket(
    model: object,
    config: MegatronBridgeHfBucketConfig,
    consume_bucket: CanonicalBucketConsumer,
    deadline: CollectiveDeadline,
) -> None:
    """Drain one Bridge export on every rank; emit bounded HF buckets on rank 0."""
    rank = 0
    bridge = None
    context: AbstractContextManager[Any] | None = None
    context_entered = False
    tasks: list[Any] = []
    replacements: dict[int, torch.Tensor] = {}
    units: tuple[tuple[int, ...], ...] = ()
    plan = None
    task_schedule = None
    task_ownership = None
    routing_group = config.routing_group
    routing_descriptor = None
    capture_units = ()
    capture_unit_sizes = ()
    local_error: str | None = None
    try:
        deadline.check("Megatron local preflight")
        validate_canonical_format_identity(config.format_identity)
        plan, named_sizes = canonical_schema_plan(
            config.canonical_schema, config.bucket_bytes
        )
        capture_units, capture_unit_sizes = canonical_capture_units(
            named_sizes,
            config.format_identity,
            config.bucket_bytes,
        )
        rank = configured_rank(config.rank, config.metadata_group)
        routing_group, routing_descriptor = _validated_routing_group(
            config.metadata_group, config.routing_group
        )
        bridge = _load_bridge(config)
        context = (
            config.model_context()
            if config.model_context is not None
            else _patch_megatron_model(model)
        )
        context.__enter__()
        context_entered = True
    except Exception as exc:
        local_error = str(exc) or type(exc).__name__
    try:
        deadline.check("Megatron local preflight collective")
        synchronize_preflight(
            local_error,
            plan,
            config.metadata_group,
            representation=(config.format_identity, routing_descriptor),
        )
        deadline.check("Megatron local preflight completion")
    except Exception:
        if context_entered and context is not None:
            try:
                context.__exit__(None, None, None)
            except Exception:
                pass
        raise

    local_error = None
    try:
        tasks, replacements = _conversion_tasks(bridge, model, config.weights_getter)
        units, task_schedule, task_ownership = _conversion_units(
            tasks,
            replacements,
            config.bucket_bytes,
            plan,
            named_sizes,
        )
    except Exception as exc:
        local_error = str(exc) or type(exc).__name__
    try:
        deadline.check("Megatron task preflight collective")
        _synchronize_preflight(
            local_error,
            config.format_identity,
            plan,
            task_schedule,
            task_ownership,
            config.metadata_group,
        )
        deadline.check("Megatron task preflight completion")
    except Exception:
        if context_entered and context is not None:
            try:
                context.__exit__(None, None, None)
            except Exception:
                pass
        raise

    assert bridge is not None
    assert context is not None

    builder = BoundedBucketBuilder(
        bucket_bytes=config.bucket_bytes,
        emit=rank == 0,
        consume_bucket=consume_bucket,
        capture_units=capture_units if rank == 0 else None,
        capture_unit_sizes=capture_unit_sizes if rank == 0 else None,
    )
    execution_error: str | None = None
    next_position = 0
    for unit in units:
        execution_tasks = []
        staging_error: str | None = None
        try:
            deadline.check("Megatron conversion unit staging")
            for index in unit:
                task = tasks[index]
                replacement = replacements.get(index)
                execution_tasks.append(
                    _replace_task_weight(task, replacement)
                    if replacement is not None
                    else task
                )
        except Exception as exc:
            staging_error = str(exc) or type(exc).__name__
        try:
            synchronize_errors(staging_error, config.metadata_group)
            deadline.check("Megatron conversion unit staging completion")
        except Exception as exc:
            execution_error = str(exc) or type(exc).__name__
            break

        local_outputs: list[tuple[str, torch.Tensor]] = []
        local_task_error: str | None = None
        task_bytes = 0
        kwargs: dict[str, Any] = {
            "cpu": True,
            "show_progress": False,
            "merge_adapter_weights": False,
            "conversion_tasks": tuple(execution_tasks),
        }
        exported = None
        tensor = None
        try:
            for exported in bridge.export_hf_weights(model, **kwargs):
                name, tensor = _postprocess_export(exported, config.vocab_size)
                canonical_name = canonical_tensor_name(name)
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"Megatron-Bridge output {canonical_name!r} is not a tensor"
                    )
                task_bytes += tensor_nbytes(tensor)
                if task_bytes > config.bucket_bytes:
                    raise ValueError(
                        "Megatron conversion task materialized output bytes "
                        f"{task_bytes} above bucket_bytes={config.bucket_bytes}"
                    )
                local_outputs.append((canonical_name, tensor))
        except Exception as exc:
            local_task_error = str(exc) or type(exc).__name__
        try:
            next_position = _synchronize_task_outputs(
                local_error=local_task_error,
                local_outputs=local_outputs,
                canonical_plan=plan,
                next_position=next_position,
                rank=rank,
                group=config.metadata_group,
                routing_group=routing_group,
                builder=builder,
            )
            deadline.check("Megatron task collective completion")
        except Exception as exc:
            execution_error = str(exc) or type(exc).__name__
            break
        finally:
            local_outputs.clear()
            execution_tasks.clear()
            kwargs.clear()
            exported = None
            tensor = None
    context_error: str | None = None
    try:
        context.__exit__(None, None, None)
    except Exception as exc:
        context_error = str(exc) or type(exc).__name__
    builder.finish()
    local_result_error = execution_error or context_error or builder.error
    if local_result_error is None and next_position != len(plan):
        local_result_error = "Megatron-Bridge export did not provide complete canonical HF schema coverage"
    synchronize_errors(
        local_result_error,
        config.metadata_group,
    )
    deadline.check("Megatron final error synchronization")
    finish_collectives(config.metadata_group)
    deadline.check("Megatron final barrier")


def for_each_megatron_hf_bucket(
    model: object,
    config: MegatronBridgeHfBucketConfig,
    consume_bucket: CanonicalBucketConsumer,
) -> None:
    """Run one deadline-bounded Bridge capture on every trainer rank."""
    deadline = CollectiveDeadline(config.deadline_monotonic, config.abort_collectives)
    try:
        _for_each_megatron_hf_bucket(model, config, consume_bucket, deadline)
    finally:
        deadline.close()
