# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receiver-owned tensor refit smoke helpers.

The Level-4 goal is a real trainer-to-vLLM/SGLang refit path. This module keeps
the live vLLM entrypoint and exposes a framework-explicit module helper used by
SGLang-shaped tests: build receiver-side ``SliceRequest`` metadata from one
runtime-owned parameter tensor, plan two synthetic trainer-held source ranges,
install the planned payloads into that runtime-owned tensor, validate
checksum/allclose, and restore the original tensor.

It intentionally does not claim NIXL data-plane transfer or trainer integration;
those are covered by separate Level-2/3 synthetic proofs and remain future
Level-4 work when attached to real trainer/runtime-owned payloads.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import torch

from .resharding import SliceOwnership, plan_segments
from .resharding_receiver import (
    build_receiver_requests_from_runtime_tensors,
    install_segment_payloads_into_runtime_tensors,
)

DEFAULT_TENSOR_SUFFIXES = (
    "lm_head.weight",
    "embed_tokens.weight",
    "qkv_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_checksum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().sum().item())


def _tensor_close(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(left.detach().float(), right.detach().float()))


def _tensor_max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _write_artifact(payload: dict[str, Any], artifact_path: str | Path) -> None:
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return repr(value)


def _replacement_payload_like(tensor: torch.Tensor) -> torch.Tensor:
    values = torch.arange(tensor.numel(), device=tensor.device, dtype=torch.float32)
    values = ((values % 257) - 128) / 8
    return values.reshape(_shape(tensor)).to(dtype=tensor.dtype)


def _source_ownerships_for_tensor(
    *,
    tensor_name: str,
    tensor: torch.Tensor,
    model_name: str,
    model_version: str,
    runtime_framework: str = "vllm",
) -> list[SliceOwnership]:
    if tensor.ndim < 2:
        raise ValueError(
            f"{runtime_framework} receiver smoke needs a rank-2-or-higher tensor"
        )
    rows = int(tensor.shape[0])
    if rows < 2:
        raise ValueError(f"{runtime_framework} receiver smoke needs at least two rows")

    split = max(1, rows // 2)
    full_shape = _shape(tensor)
    trailing_ranges = tuple((0, int(dim)) for dim in full_shape[1:])
    dtype = _torch_dtype_name(tensor.dtype)
    mode_slug = runtime_framework.replace("_", "-")
    return [
        SliceOwnership(
            model_name=model_name,
            model_version=model_version,
            tensor_name=tensor_name,
            global_shape=full_shape,
            dtype=dtype,
            source_range=((0, split), *trailing_ranges),
            worker_id="trainer-rank0-worker",
            source_id="trainer-rank0",
            worker_rank=0,
            source_lease=f"trainer-rank0-live-{mode_slug}-receiver-smoke",
            nixl_descriptor_id=f"trainer-rank0-live-{mode_slug}-receiver-smoke",
            layout_tags={
                "trainer_layout": "synthetic-fsdp",
                "storage_layout": "row-major",
            },
            element_size_bytes=int(tensor.element_size()),
        ),
        SliceOwnership(
            model_name=model_name,
            model_version=model_version,
            tensor_name=tensor_name,
            global_shape=full_shape,
            dtype=dtype,
            source_range=((split, rows), *trailing_ranges),
            worker_id="trainer-rank1-worker",
            source_id="trainer-rank1",
            worker_rank=1,
            source_lease=f"trainer-rank1-live-{mode_slug}-receiver-smoke",
            nixl_descriptor_id=f"trainer-rank1-live-{mode_slug}-receiver-smoke",
            layout_tags={
                "trainer_layout": "synthetic-fsdp",
                "storage_layout": "row-major",
            },
            element_size_bytes=int(tensor.element_size()),
        ),
    ]


def _select_refit_tensor(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    preferred_name: str = "",
) -> tuple[str, torch.Tensor]:
    candidates = [
        (name, tensor)
        for name, tensor in named_tensors
        if tensor.ndim >= 2
        and int(tensor.shape[0]) >= 2
        and torch.is_floating_point(tensor)
    ]
    if not candidates:
        raise RuntimeError("no rank-2 floating vLLM-owned tensor is available")

    if preferred_name:
        for name, tensor in candidates:
            if name == preferred_name:
                return name, tensor
        raise RuntimeError(f"preferred vLLM tensor {preferred_name!r} was not found")

    for suffix in DEFAULT_TENSOR_SUFFIXES:
        for name, tensor in candidates:
            if name.endswith(suffix):
                return name, tensor
    return candidates[0]


def _module_has_parameters(module: torch.nn.Module) -> bool:
    return next(module.parameters(), None) is not None


def _known_module_paths(root: Any) -> Iterable[tuple[str, Any]]:
    paths = (
        ("llm.llm_engine.model_executor.driver_worker.model_runner.model",),
        ("llm.llm_engine.model_executor.driver_worker.worker.model_runner.model",),
        ("llm.llm_engine.engine_core.model_executor.driver_worker.model_runner.model",),
        ("llm.engine_core.model_executor.driver_worker.model_runner.model",),
        ("llm.model_executor.driver_worker.model_runner.model",),
        ("llm.model_runner.model",),
        ("llm.model",),
    )
    for (path,) in paths:
        current = root
        ok = True
        for attr in path.split(".")[1:]:
            try:
                current = getattr(current, attr)
            except Exception:
                ok = False
                break
        if ok:
            yield path, current


def _find_vllm_module(root: Any, *, max_depth: int = 5) -> tuple[str, torch.nn.Module]:
    for path, value in _known_module_paths(root):
        if isinstance(value, torch.nn.Module) and _module_has_parameters(value):
            return path, value

    queue: deque[tuple[str, Any, int]] = deque([("llm", root, 0)])
    seen: set[int] = set()
    while queue:
        path, value, depth = queue.popleft()
        if id(value) in seen:
            continue
        seen.add(id(value))

        if isinstance(value, torch.nn.Module) and _module_has_parameters(value):
            return path, value
        if depth >= max_depth:
            continue
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            continue
        if isinstance(value, dict):
            items = list(value.items())[:32]
            for key, item in items:
                queue.append((f"{path}.{key}", item, depth + 1))
            continue

        attr_names = [
            "llm_engine",
            "engine_core",
            "model_executor",
            "driver_worker",
            "worker",
            "model_runner",
            "runner",
            "executor",
            "model",
        ]
        if depth < 2:
            try:
                attr_names.extend(
                    name
                    for name in vars(value)
                    if not name.startswith("_") and name not in set(attr_names)
                )
            except TypeError:
                pass
        for attr in attr_names:
            try:
                item = getattr(value, attr)
            except Exception:
                continue
            if callable(item) and not isinstance(item, torch.nn.Module):
                continue
            queue.append((f"{path}.{attr}", item, depth + 1))

    raise RuntimeError(
        "could not find an in-process vLLM torch.nn.Module; rerun with "
        "--distributed-executor-backend=uni or use a vLLM build that exposes "
        "the driver worker model in process"
    )


def create_tiny_qwen2_checkpoint(path: str | Path) -> Path:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    checkpoint_path = Path(path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    Qwen2ForCausalLM(config).save_pretrained(checkpoint_path, safe_serialization=True)
    return checkpoint_path


def load_vllm_module(
    *,
    model_path: str,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    distributed_executor_backend: str,
) -> tuple[Any, str, torch.nn.Module]:
    from vllm import LLM

    llm = LLM(
        model=model_path,
        skip_tokenizer_init=True,
        dtype=dtype,
        enforce_eager=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        distributed_executor_backend=distributed_executor_backend,
    )
    module_path, module = _find_vllm_module(llm)
    return llm, module_path, module


def run_receiver_refit_on_module(
    module: torch.nn.Module,
    *,
    model_name: str,
    model_version: str,
    module_path: str,
    vllm_version: str = "",
    runtime_framework: str = "vllm",
    framework_version: str | None = None,
    runtime_imported: bool = False,
    real_runtime_engine_used: bool = False,
    preferred_tensor_name: str = "",
    artifact_path: str | Path | None = None,
    mode: str = "live-vllm-receiver-owned-tensor-smoke",
    model_path: str = "",
) -> dict[str, Any]:
    runtime_framework = runtime_framework.strip().lower()
    if not runtime_framework:
        raise ValueError("runtime_framework is required")
    if framework_version is None:
        framework_version = vllm_version if runtime_framework == "vllm" else ""
    if not framework_version:
        framework_version = "unknown"

    tensor_name, target_tensor = _select_refit_tensor(
        module.named_parameters(),
        preferred_name=preferred_tensor_name,
    )
    original = target_tensor.detach().clone()
    replacement = _replacement_payload_like(target_tensor)
    original_checksum = _tensor_checksum(target_tensor)
    replacement_checksum = _tensor_checksum(replacement)

    request_start = time.perf_counter()
    requests = build_receiver_requests_from_runtime_tensors(
        {tensor_name: target_tensor},
        model_name=model_name,
        model_version=model_version,
        runtime_framework=runtime_framework,
        target_id_prefix=runtime_framework,
        layout_tags_by_tensor={
            tensor_name: {
                "runtime_module_path": module_path,
                "runtime_tensor_name": tensor_name,
                f"{runtime_framework}_module_path": module_path,
                f"{runtime_framework}_tensor_name": tensor_name,
                "runtime_lifecycle": "post-load-refit-smoke",
            }
        },
    )
    request_duration_ms = (time.perf_counter() - request_start) * 1000

    planner_start = time.perf_counter()
    owners = _source_ownerships_for_tensor(
        tensor_name=tensor_name,
        tensor=target_tensor,
        model_name=model_name,
        model_version=model_version,
        runtime_framework=runtime_framework,
    )
    plans = plan_segments(owners, requests)
    planner_duration_ms = (time.perf_counter() - planner_start) * 1000

    target_key = requests[0].target_id or tensor_name
    payloads = [
        (
            plan,
            replacement[
                tuple(slice(start, end) for start, end in plan.source_range)
            ].clone(),
        )
        for plan in plans
    ]
    install_start = time.perf_counter()
    installed = install_segment_payloads_into_runtime_tensors(
        payloads,
        {target_key: target_tensor},
    )
    if target_tensor.is_cuda:
        torch.cuda.synchronize(target_tensor.device)
    install_duration_ms = (time.perf_counter() - install_start) * 1000

    allclose = _tensor_close(target_tensor, replacement)
    checksum = _tensor_checksum(target_tensor)
    checksum_matches = checksum == replacement_checksum
    max_abs_error = _tensor_max_abs_error(target_tensor, replacement)

    restore_start = time.perf_counter()
    with torch.no_grad():
        target_tensor.copy_(original)
    if target_tensor.is_cuda:
        torch.cuda.synchronize(target_tensor.device)
    restore_duration_ms = (time.perf_counter() - restore_start) * 1000
    restored_original = _tensor_close(target_tensor, original)

    proof = {
        f"{runtime_framework}_imported": bool(runtime_imported),
        f"{runtime_framework}_owned_target_tensor": True,
        f"receiver_installed_into_{runtime_framework}_owned_tensor": allclose,
        "runtime_imported": bool(runtime_imported),
        "runtime_owned_target_tensor": True,
        "receiver_request_from_runtime_owned_tensor": True,
        "target_slice_spans_multiple_trainers": len({plan.source_id for plan in plans})
        >= 2,
        "receiver_installed_into_runtime_owned_tensor": allclose,
        "checksum_gate": checksum_matches,
        "allclose": allclose,
        "restored_original_tensor": restored_original,
        "trainer_full_all_gather_used": False,
        "trainer_side_inference_layout_conversion_used": False,
        "host_side_torch_cat_used": False,
        "actual_nixl_reads_used": False,
        "synthetic_trainer_payloads_used": True,
        "real_trainer_process_used": False,
        "real_runtime_engine_used": bool(real_runtime_engine_used),
    }

    result = {
        "schema_version": 1,
        "result": (
            "pass" if allclose and checksum_matches and restored_original else "fail"
        ),
        "mode": mode,
        "runtime_framework": runtime_framework,
        "framework_version": framework_version,
        "vllm_version": (
            framework_version if runtime_framework == "vllm" else vllm_version
        ),
        "model_name": model_name,
        "model_version": model_version,
        "model_path": model_path,
        "module_class": type(module).__name__,
        "module_path": module_path,
        "target_tensor_name": tensor_name,
        "target_key": target_key,
        "target_shape": list(_shape(target_tensor)),
        "target_dtype": _torch_dtype_name(target_tensor.dtype),
        "target_device": str(target_tensor.device),
        "request": requests[0].to_dict(),
        "source_ownerships": [owner.to_dict() for owner in owners],
        "segment_plans": [plan.to_dict() for plan in plans],
        "installed_segments": [segment.__dict__ for segment in installed],
        "proof": proof,
        "validation": {
            "allclose": allclose,
            "checksum": checksum,
            "expected_checksum": replacement_checksum,
            "checksum_matches": checksum_matches,
            "max_abs_error": max_abs_error,
            "original_checksum": original_checksum,
            "restored_checksum": _tensor_checksum(target_tensor),
            "restored_original": restored_original,
        },
        "metrics": {
            "request_build_duration_ms": request_duration_ms,
            "planner_duration_ms": planner_duration_ms,
            "activation_install_duration_ms": install_duration_ms,
            "restore_duration_ms": restore_duration_ms,
            "trainer_to_inference_bytes": sum(plan.bytes for plan in plans),
            "segment_count": len(plans),
            "source_count_per_target_tensor": {
                tensor_name: len({plan.source_id for plan in plans})
            },
            "target_tensor_bytes": _nbytes(target_tensor),
        },
    }
    if runtime_framework != "vllm":
        result[f"{runtime_framework}_version"] = framework_version
    if artifact_path is not None:
        _write_artifact(result, artifact_path)
    if result["result"] != "pass":
        raise RuntimeError(
            f"{runtime_framework} receiver smoke failed: "
            f"{json.dumps(result, default=_json_default)}"
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-name", default="mx-live-vllm-receiver-smoke")
    parser.add_argument("--model-version", default="step-live-vllm-receiver")
    parser.add_argument("--preferred-tensor-name", default="")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.1)
    parser.add_argument("--distributed-executor-backend", default="uni")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import vllm

    model_path = args.model_path
    if not model_path:
        model_path = str(
            create_tiny_qwen2_checkpoint(
                Path(tempfile.mkdtemp(prefix="mx-vllm-receiver-model-"))
            )
        )

    _llm, module_path, module = load_vllm_module(
        model_path=model_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        distributed_executor_backend=args.distributed_executor_backend,
    )
    result = run_receiver_refit_on_module(
        module,
        model_name=args.model_name,
        model_version=args.model_version,
        module_path=module_path,
        vllm_version=getattr(vllm, "__version__", "unknown"),
        runtime_framework="vllm",
        framework_version=getattr(vllm, "__version__", "unknown"),
        runtime_imported=True,
        real_runtime_engine_used=True,
        preferred_tensor_name=args.preferred_tensor_name,
        artifact_path=args.artifact_path,
        model_path=model_path,
    )
    print(
        "MX_VLLM_RECEIVER_SMOKE_RESULT "
        + json.dumps(result, default=_json_default, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
