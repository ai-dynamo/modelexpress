# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded canonical capture through Megatron-Bridge's native export stream."""

from __future__ import annotations

import dataclasses
import hashlib
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .canonical import (
    CanonicalBucket,
    CanonicalError,
    CanonicalTensorSpec,
    canonical_tensor_name,
    dtype_name,
    tensor_bytes,
    tensor_from_bytes,
)

CanonicalBucketConsumer = Callable[[CanonicalBucket], None]


@dataclass(frozen=True)
class MegatronBridgeHfBucketConfig:
    """Megatron-Bridge dependencies and the bounded CPU bucket size."""

    bucket_bytes: int = 256 * 1024 * 1024
    hf_model_path: str | None = None
    bridge: Any = None
    canonical_schema: Sequence[CanonicalTensorSpec] | None = None
    spool_directory: str | Path | None = None
    rank: Callable[[], int] | None = None
    metadata_group: Any = None
    model_context: Callable[[], AbstractContextManager[Any]] | None = None
    weights_getter: Callable[[], Mapping[str, torch.Tensor]] | None = None
    vocab_size: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.bucket_bytes, int)
            or isinstance(self.bucket_bytes, bool)
            or self.bucket_bytes <= 0
        ):
            raise ValueError("bucket_bytes must be a positive integer")
        if self.bridge is None and not self.hf_model_path:
            raise ValueError("bridge or hf_model_path is required")
        if self.vocab_size is not None and (
            not isinstance(self.vocab_size, int)
            or isinstance(self.vocab_size, bool)
            or self.vocab_size <= 0
        ):
            raise ValueError("vocab_size must be a positive integer")

    def with_schema(
        self, schema: Sequence[CanonicalTensorSpec]
    ) -> MegatronBridgeHfBucketConfig:
        normalized = tuple(schema)
        if (
            self.canonical_schema is not None
            and tuple(self.canonical_schema) != normalized
        ):
            raise ValueError("Megatron capture schema differs from launch checkpoint")
        return replace(self, canonical_schema=normalized)


class _BucketBuilder:
    def __init__(self, maximum_bytes: int, consume: CanonicalBucketConsumer) -> None:
        self._maximum_bytes = maximum_bytes
        self._consume = consume
        self._items: list[tuple[str, torch.Tensor]] = []
        self._size = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        size = tensor.nbytes
        if size > self._maximum_bytes:
            raise CanonicalError(
                f"canonical tensor {name!r} exceeds bucket_bytes={self._maximum_bytes}"
            )
        if self._items and self._size + size > self._maximum_bytes:
            self.flush()
        self._items.append((name, tensor))
        self._size += size

    def flush(self) -> None:
        if not self._items:
            return
        items = tuple(self._items)
        try:
            self._consume(items)
        finally:
            self._items.clear()
            self._size = 0


def _load_bridge(config: MegatronBridgeHfBucketConfig):
    if config.bridge is not None:
        return config.bridge
    try:
        from megatron.bridge import AutoBridge
    except ImportError as exc:
        raise CanonicalError(
            "Megatron-Bridge is required for Megatron canonical capture"
        ) from exc
    return AutoBridge.from_hf_pretrained(
        config.hf_model_path,
        trust_remote_code=True,
    )


@contextmanager
def _patch_megatron_model(model: object):
    """Apply the small compatibility patch required by the pinned Bridge."""
    from megatron.core.utils import unwrap_model

    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    unwrapped = unwrap_model(chunks)
    root = unwrapped[0] if isinstance(unwrapped, (list, tuple)) else unwrapped
    model_config = root.config
    added_share_flag = not hasattr(model_config, "share_embeddings_and_output_weights")
    if added_share_flag:
        model_config.share_embeddings_and_output_weights = (
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
            delattr(model_config, "share_embeddings_and_output_weights")


def _model_context(
    model: object, config: MegatronBridgeHfBucketConfig
) -> AbstractContextManager[Any]:
    if config.model_context is not None:
        return config.model_context()
    try:
        import megatron.core.utils  # noqa: F401
    except ImportError:
        if config.bridge is not None:
            return nullcontext()
        raise
    return _patch_megatron_model(model)


def _replace_task_weight(task: object, weight: torch.Tensor) -> object:
    current = getattr(task, "param_weight", None)
    if not isinstance(current, torch.Tensor):
        raise CanonicalError("authoritative weight has no owning Bridge task")
    replacement = weight.detach().to(device=current.device)
    if dataclasses.is_dataclass(task):
        return replace(task, param_weight=replacement)
    if hasattr(task, "_replace"):
        return task._replace(param_weight=replacement)
    raise CanonicalError(
        f"unsupported Megatron-Bridge conversion task {type(task).__name__}"
    )


def _conversion_tasks(
    bridge: object,
    model: object,
    weights_getter: Callable[[], Mapping[str, torch.Tensor]] | None,
) -> list[object]:
    tasks = list(bridge.get_conversion_tasks(model))
    if weights_getter is None:
        return tasks
    weights = weights_getter()
    if not isinstance(weights, Mapping):
        raise CanonicalError("weights_getter must return a mapping")
    normalized: dict[str, torch.Tensor] = {}
    for source_name, weight in weights.items():
        if not isinstance(source_name, str):
            raise CanonicalError("authoritative model weight names must be strings")
        while source_name.startswith("module."):
            source_name = source_name.removeprefix("module.")
        if source_name in normalized:
            raise CanonicalError(
                f"authoritative model weights duplicate {source_name!r}"
            )
        if not isinstance(weight, torch.Tensor):
            raise CanonicalError(
                f"authoritative model weight {source_name!r} is not a tensor"
            )
        normalized[source_name] = weight

    converted = []
    for task in tasks:
        if task is None:
            converted.append(task)
            continue
        current = getattr(task, "param_weight", None)
        vp_stage = getattr(task, "vp_stage", None)
        name = getattr(task, "param_name", None)
        if current is None or not isinstance(name, str):
            converted.append(task)
            continue
        if not isinstance(current, torch.Tensor):
            raise CanonicalError("Megatron-Bridge task param_weight is not a tensor")
        if not isinstance(vp_stage, int) or isinstance(vp_stage, bool):
            raise CanonicalError("owning Megatron-Bridge task has no vp_stage")
        key = f"vp_stages.{vp_stage}.{name}"
        replacement = normalized.get(key)
        if replacement is None:
            converted.append(task)
            continue
        if tuple(replacement.shape) != tuple(current.shape):
            raise CanonicalError(
                f"authoritative replacement {key!r} changed Bridge task shape"
            )
        if replacement.dtype != current.dtype:
            raise CanonicalError(
                f"authoritative replacement {key!r} changed Bridge task dtype"
            )
        if replacement.device.type != current.device.type:
            raise CanonicalError(
                f"authoritative replacement {key!r} changed Bridge task device"
            )
        converted.append(_replace_task_weight(task, replacement))
    return converted


def _postprocess_export(item: object, vocab_size: int | None) -> tuple[object, object]:
    if not isinstance(item, tuple) or len(item) < 2:
        raise CanonicalError("Megatron-Bridge output is malformed")
    name, tensor = item[0], item[1]
    native_name = item[2] if len(item) > 2 else None
    if isinstance(native_name, str):
        while native_name.startswith("module."):
            native_name = native_name.removeprefix("module.")
        if native_name in {
            "embedding.word_embeddings.weight",
            "output_layer.weight",
        }:
            if vocab_size is None:
                raise CanonicalError(
                    "vocab_size is required to remove Megatron vocabulary padding"
                )
            if not isinstance(tensor, torch.Tensor) or not tensor.shape:
                raise CanonicalError(
                    "padded Megatron vocabulary weight is not a matrix"
                )
            if tensor.shape[0] < vocab_size:
                raise CanonicalError(
                    "Megatron vocabulary weight is smaller than vocab_size"
                )
            tensor = tensor[:vocab_size]
    return name, tensor


def _validate_tasks(tasks: Sequence[object]) -> None:
    for task in tasks:
        if task is None:
            continue
        mapping = getattr(task, "mapping", None)
        if mapping is None:
            continue
        hf_param = getattr(mapping, "hf_param", None)
        if isinstance(hf_param, dict):
            if not hf_param or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in hf_param.items()
            ):
                raise CanonicalError("Megatron-Bridge HF parameter mapping is invalid")
        elif hf_param is not None and not isinstance(hf_param, str):
            raise CanonicalError("Megatron-Bridge HF parameter mapping is invalid")
        if bool(getattr(mapping, "is_grouped_export", False)):
            key = getattr(mapping, "group_key", None)
            if not isinstance(key, str) or not key:
                raise CanonicalError(
                    "grouped Megatron-Bridge mapping has no stable group_key"
                )


def _stable_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((key, _stable_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_stable_value(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _task_plan(tasks: Sequence[object]) -> tuple[object, ...]:
    plan = []
    for ordinal, task in enumerate(tasks):
        if task is None:
            plan.append((ordinal, None))
            continue
        mapping = getattr(task, "mapping", None)
        mapping_type = (
            None
            if mapping is None
            else f"{type(mapping).__module__}.{type(mapping).__qualname__}"
        )
        plan.append(
            (
                ordinal,
                getattr(task, "global_param_name", getattr(task, "param_name", None)),
                mapping_type,
                _stable_value(getattr(mapping, "hf_param", None)),
                bool(getattr(mapping, "is_grouped_export", False)),
                getattr(mapping, "group_key", None),
            )
        )
    return tuple(plan)


def _rank(config: MegatronBridgeHfBucketConfig) -> int:
    if config.rank is not None:
        value = config.rank()
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        value = torch.distributed.get_rank(config.metadata_group)
    else:
        value = 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CanonicalError("Megatron capture rank is invalid")
    return value


def _synchronize_preflight(
    local_error: str | None,
    identity: object,
    group: Any,
    *,
    disagreement: str,
) -> str | None:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_error
    contribution = (local_error, identity)
    gathered = [None] * torch.distributed.get_world_size(group)
    try:
        torch.distributed.all_gather_object(gathered, contribution, group=group)
    except Exception as exc:
        return f"Megatron preflight synchronization failed: {exc}"
    failures = [
        f"rank {rank}: {item[0]}"
        for rank, item in enumerate(gathered)
        if isinstance(item, tuple) and len(item) == 2 and item[0]
    ]
    if failures:
        return "; ".join(failures)
    if any(item != contribution for item in gathered):
        return disagreement
    return None


def _synchronize(
    local_error: str | None,
    records: tuple[tuple[str, str, tuple[int, ...], str], ...],
    group: Any,
) -> str | None:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_error
    contribution = (local_error, records)
    gathered = [None] * torch.distributed.get_world_size(group)
    try:
        torch.distributed.all_gather_object(gathered, contribution, group=group)
    except Exception as exc:
        return f"Megatron capture result synchronization failed: {exc}"
    failures = [
        f"rank {rank}: {item[0]}"
        for rank, item in enumerate(gathered)
        if isinstance(item, tuple) and item[0]
    ]
    if failures:
        return "; ".join(failures)
    if any(item != gathered[0] for item in gathered[1:]):
        return "Megatron canonical HF content differs across trainer ranks"
    try:
        torch.distributed.barrier(group=group)
    except Exception as exc:
        return f"Megatron final capture barrier failed: {exc}"
    return None


def _synchronize_completion(local_error: str | None, group: Any) -> str | None:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_error
    gathered = [None] * torch.distributed.get_world_size(group)
    try:
        torch.distributed.all_gather_object(gathered, local_error, group=group)
    except Exception as exc:
        return f"Megatron completion synchronization failed: {exc}"
    failures = [
        f"rank {rank}: {detail}"
        for rank, detail in enumerate(gathered)
        if isinstance(detail, str) and detail
    ]
    return "; ".join(failures) if failures else None


def _capture_megatron_hf_buckets(
    model: object,
    config: MegatronBridgeHfBucketConfig,
    consume_bucket: CanonicalBucketConsumer,
    bridge: object,
    tasks: Sequence[object],
    rank: int,
    schema: Sequence[CanonicalTensorSpec],
    spool: Path,
) -> None:
    """Drain one native Bridge export and emit canonical buckets on rank zero.

    Megatron-Bridge receives its complete conversion-task list in one call, so
    grouped mappings are never split. Bridge outputs are spooled by name and
    drained in the launch checkpoint's canonical order. This permits a QKV
    conversion to emit non-adjacent HF names while keeping resident tensor
    memory bounded by the configured bucket size.
    """
    schema = tuple(schema)
    names = tuple(spec.name for spec in schema)
    expected = {spec.name: spec for spec in schema}
    records: dict[str, tuple[str, str, tuple[int, ...], str]] = {}
    seen: set[str] = set()
    local_error: str | None = None
    next_index = 0

    pending: dict[str, Path] = {}
    builder = _BucketBuilder(config.bucket_bytes, consume_bucket)

    def drain_ready() -> None:
        nonlocal next_index
        while next_index < len(schema):
            spec = schema[next_index]
            path = pending.get(spec.name)
            if path is None:
                return
            with path.open("rb") as handle:
                data = handle.read(spec.nbytes + 1)
            if len(data) != spec.nbytes:
                raise CanonicalError(
                    f"spooled Megatron tensor {spec.name!r} changed size"
                )
            builder.add(
                spec.name,
                tensor_from_bytes(data, spec.dtype, spec.shape).contiguous(),
            )
            path.unlink()
            del pending[spec.name]
            next_index += 1

    try:
        exported = bridge.export_hf_weights(
            model,
            cpu=True,
            show_progress=False,
            conversion_tasks=tasks,
            merge_adapter_weights=False,
        )
        for item in exported:
            try:
                raw_name, tensor = _postprocess_export(item, config.vocab_size)
                name = canonical_tensor_name(raw_name)
                if name in seen:
                    raise CanonicalError(
                        f"Megatron-Bridge duplicated canonical tensor {name!r}"
                    )
                seen.add(name)
                spec = expected.get(name)
                if spec is None:
                    raise CanonicalError(
                        f"Megatron-Bridge emitted tensor outside launch schema: {name!r}"
                    )
                if not isinstance(tensor, torch.Tensor):
                    raise CanonicalError(
                        f"Megatron-Bridge output {name!r} is not a tensor"
                    )
                if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
                    raise CanonicalError(
                        f"Megatron-Bridge output {name!r} differs from launch shape/dtype"
                    )
                data = tensor_bytes(tensor)
                if len(data) > config.bucket_bytes:
                    raise CanonicalError(
                        f"Megatron output {name!r} exceeds "
                        f"bucket_bytes={config.bucket_bytes}"
                    )
                digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
                records[name] = (name, dtype_name(spec.dtype), spec.shape, digest)
                if rank == 0:
                    destination = spool / f"{len(seen) - 1:08d}.tensor"
                    with destination.open("xb") as handle:
                        handle.write(data)
                    pending[name] = destination
                    if local_error is None:
                        drain_ready()
            except Exception as exc:
                if local_error is None:
                    local_error = str(exc) or type(exc).__name__
    except Exception as exc:
        if local_error is None:
            local_error = str(exc) or type(exc).__name__

    if local_error is None:
        missing = tuple(name for name in names if name not in seen)
        if missing:
            local_error = (
                "Megatron-Bridge did not provide complete canonical HF coverage: "
                f"missing {missing!r}"
            )
    if local_error is None and rank == 0:
        try:
            drain_ready()
            builder.flush()
            if next_index != len(schema) or pending:
                raise CanonicalError(
                    "Megatron-Bridge did not provide complete canonical HF coverage"
                )
        except Exception as exc:
            local_error = str(exc) or type(exc).__name__

    ordered_records = tuple(records[name] for name in names if name in records)
    synchronized_error = _synchronize(
        local_error,
        ordered_records,
        config.metadata_group,
    )
    if synchronized_error is not None:
        raise CanonicalError(synchronized_error)


def for_each_megatron_hf_bucket(
    model: object,
    config: MegatronBridgeHfBucketConfig,
    consume_bucket: CanonicalBucketConsumer,
) -> None:
    """Run one native Bridge capture on every rank and emit on rank zero."""
    stack = ExitStack()
    operation_error: str | None = None
    try:
        bridge = None
        rank = 0
        schema: tuple[CanonicalTensorSpec, ...] = ()
        spool = None
        setup_identity = None
        setup_error = None
        try:
            if not callable(consume_bucket):
                raise TypeError("consume_bucket must be callable")
            schema = tuple(config.canonical_schema or ())
            if not schema:
                raise CanonicalError("Megatron capture requires launch HF schema")
            names = tuple(spec.name for spec in schema)
            if len(names) != len(set(names)):
                raise CanonicalError("canonical HF schema contains duplicate names")
            rank = _rank(config)
            bridge = _load_bridge(config)
            stack.enter_context(_model_context(model, config))
            spool_parent = None
            if config.spool_directory is not None:
                spool_parent = Path(config.spool_directory).resolve()
                spool_parent.mkdir(parents=True, exist_ok=True)
            raw = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="mx-megatron-", dir=spool_parent)
            )
            spool = Path(raw)
            setup_identity = (
                config.bucket_bytes,
                config.vocab_size,
                tuple(
                    (spec.name, spec.shape, dtype_name(spec.dtype)) for spec in schema
                ),
                f"{type(bridge).__module__}.{type(bridge).__qualname__}",
            )
        except Exception as exc:
            setup_error = str(exc) or type(exc).__name__
        synchronized_error = _synchronize_preflight(
            setup_error,
            setup_identity,
            config.metadata_group,
            disagreement="Megatron capture setup differs across trainer ranks",
        )
        if synchronized_error is not None:
            raise CanonicalError(synchronized_error)

        if bridge is None or spool is None:
            raise CanonicalError("Megatron capture setup did not complete")
        tasks: list[object] = []
        task_identity = None
        task_error = None
        try:
            tasks = _conversion_tasks(bridge, model, config.weights_getter)
            _validate_tasks(tasks)
            task_identity = _task_plan(tasks)
        except Exception as exc:
            task_error = str(exc) or type(exc).__name__
        synchronized_error = _synchronize_preflight(
            task_error,
            task_identity,
            config.metadata_group,
            disagreement="Megatron conversion task plan differs across trainer ranks",
        )
        if synchronized_error is not None:
            raise CanonicalError(synchronized_error)
        _capture_megatron_hf_buckets(
            model,
            config,
            consume_bucket,
            bridge,
            tasks,
            rank,
            schema,
            spool,
        )
    except Exception as exc:
        operation_error = str(exc) or type(exc).__name__
    finally:
        try:
            stack.close()
        except Exception as exc:
            cleanup_error = str(exc) or type(exc).__name__
            operation_error = (
                cleanup_error
                if operation_error is None
                else f"{operation_error}; cleanup failed: {cleanup_error}"
            )

    synchronized_error = _synchronize_completion(
        operation_error,
        config.metadata_group,
    )
    if synchronized_error is not None:
        raise CanonicalError(synchronized_error)
