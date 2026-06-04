# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang receiver-owned tensor refit smoke wrapper.

This module exposes two SGLang Level-4 smoke paths:

* ``run_sglang_receiver_refit_on_module`` keeps the lightweight module-shaped
  helper used by unit tests and CPU receiver-install artifacts.
* ``run_sglang_receiver_refit_on_engine`` starts from a live ``sglang.Engine``
  weight snapshot, builds receiver-side ``SliceRequest`` metadata, assembles a
  planned multi-source replacement tensor, installs it through SGLang's native
  ``Engine.update_weights_from_tensor`` path, validates the engine-owned weight
  with ``Engine.get_weights_by_name``, and restores the original weight.

The live engine path intentionally still uses synthetic trainer payloads and
does not claim NIXL reads or real trainer integration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time
from typing import Any

import torch

from .refit_vllm_receiver_smoke import (
    DEFAULT_TENSOR_SUFFIXES,
    _nbytes,
    _replacement_payload_like,
    _shape,
    _source_ownerships_for_tensor,
    _tensor_checksum,
    _tensor_close,
    _tensor_max_abs_error,
    _torch_dtype_name,
    _write_artifact,
    run_receiver_refit_on_module,
)
from .resharding import plan_segments
from .resharding_receiver import (
    build_receiver_requests_from_runtime_tensors,
    install_segment_payloads_into_runtime_tensors,
)


def detect_sglang_version() -> tuple[str, bool]:
    """Return the importable SGLang version without requiring tests to import it."""

    try:
        import sglang
    except Exception:
        return "unavailable", False
    return getattr(sglang, "__version__", "unknown"), True


def create_tiny_llama_checkpoint(path: str | Path) -> Path:
    """Create a tiny Llama checkpoint compatible with SGLang weight probes."""

    from transformers import LlamaConfig, LlamaForCausalLM

    checkpoint_path = Path(path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).save_pretrained(checkpoint_path, safe_serialization=True)
    return checkpoint_path


def load_sglang_engine(
    *,
    model_path: str,
    dtype: str,
    context_length: int,
    mem_fraction_static: float,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    log_level: str = "info",
) -> Any:
    """Construct a live SGLang Engine for a tiny receiver smoke."""

    from sglang import Engine

    return Engine(
        model_path=model_path,
        skip_tokenizer_init=True,
        dtype=dtype,
        context_length=context_length,
        mem_fraction_static=mem_fraction_static,
        disable_cuda_graph=disable_cuda_graph,
        disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
        log_level=log_level,
    )


def _coerce_sglang_weight_to_tensor(value: Any, *, name: str) -> torch.Tensor:
    """Convert SGLang's get_weights_by_name result into a torch tensor."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
    elif isinstance(value, list):
        tensor = torch.tensor(value, dtype=torch.float32)
    else:
        raise RuntimeError(
            f"SGLang Engine.get_weights_by_name({name!r}) returned "
            f"{type(value).__name__}, expected tensor or list"
        )
    if tensor.ndim < 2 or int(tensor.shape[0]) < 2:
        raise RuntimeError(
            f"SGLang engine tensor {name!r} must be rank-2-or-higher with "
            f"at least two rows, got shape={tuple(tensor.shape)}"
        )
    if not torch.is_floating_point(tensor):
        raise RuntimeError(f"SGLang engine tensor {name!r} is not floating point")
    return tensor.contiguous()


def _select_sglang_engine_weight(
    engine: Any,
    *,
    preferred_tensor_name: str = "",
    truncate_size: int = 10_000_000,
) -> tuple[str, torch.Tensor]:
    candidates = [preferred_tensor_name] if preferred_tensor_name else []
    candidates.extend(
        name for name in DEFAULT_TENSOR_SUFFIXES if name not in set(candidates)
    )

    failures: list[str] = []
    for name in candidates:
        if not name:
            continue
        try:
            value = engine.get_weights_by_name(name, truncate_size=truncate_size)
            if value is None:
                failures.append(f"{name}: returned None")
                continue
            return name, _coerce_sglang_weight_to_tensor(value, name=name)
        except Exception as exc:  # pragma: no cover - exercised by live SGLang
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "could not fetch a usable SGLang engine weight; tried "
        f"{candidates!r}; failures={failures!r}"
    )


def _sglang_update_succeeded(update_result: Any) -> tuple[bool, str]:
    if isinstance(update_result, tuple) and update_result:
        message = update_result[1] if len(update_result) > 1 else ""
        return bool(update_result[0]), str(message)
    success = getattr(update_result, "success", None)
    if success is not None:
        return bool(success), str(getattr(update_result, "message", ""))
    return bool(update_result), str(update_result)


def run_sglang_receiver_refit_on_engine(
    engine: Any,
    *,
    model_name: str,
    model_version: str,
    sglang_version: str = "",
    preferred_tensor_name: str = "",
    artifact_path: str | Path | None = None,
    mode: str = "live-sglang-engine-owned-weight-smoke",
    model_path: str = "",
    engine_start_duration_ms: float | None = None,
) -> dict[str, Any]:
    """Install a planned receiver refit through a live SGLang Engine."""

    if not sglang_version:
        sglang_version, _ = detect_sglang_version()

    fetch_start = time.perf_counter()
    tensor_name, original = _select_sglang_engine_weight(
        engine,
        preferred_tensor_name=preferred_tensor_name,
    )
    initial_fetch_duration_ms = (time.perf_counter() - fetch_start) * 1000

    replacement = _replacement_payload_like(original)
    original_checksum = _tensor_checksum(original)
    replacement_checksum = _tensor_checksum(replacement)

    request_start = time.perf_counter()
    requests = build_receiver_requests_from_runtime_tensors(
        {tensor_name: original},
        model_name=model_name,
        model_version=model_version,
        runtime_framework="sglang",
        target_id_prefix="sglang",
        layout_tags_by_tensor={
            tensor_name: {
                "runtime_module_path": "sglang.Engine",
                "runtime_tensor_name": tensor_name,
                "sglang_module_path": "sglang.Engine",
                "sglang_tensor_name": tensor_name,
                "runtime_lifecycle": "engine-update-weights-refit-smoke",
            }
        },
    )
    request_duration_ms = (time.perf_counter() - request_start) * 1000

    planner_start = time.perf_counter()
    owners = _source_ownerships_for_tensor(
        tensor_name=tensor_name,
        tensor=original,
        model_name=model_name,
        model_version=model_version,
        runtime_framework="sglang",
    )
    plans = plan_segments(owners, requests)
    planner_duration_ms = (time.perf_counter() - planner_start) * 1000

    assembled = torch.empty_like(original)
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
    assemble_start = time.perf_counter()
    installed = install_segment_payloads_into_runtime_tensors(
        payloads,
        {target_key: assembled},
    )
    assemble_duration_ms = (time.perf_counter() - assemble_start) * 1000
    assembled_allclose = _tensor_close(assembled, replacement)

    update_start = time.perf_counter()
    update_result = engine.update_weights_from_tensor(
        [(tensor_name, assembled)],
        load_format="direct",
        flush_cache=True,
    )
    update_duration_ms = (time.perf_counter() - update_start) * 1000
    update_success, update_message = _sglang_update_succeeded(update_result)

    validate_fetch_start = time.perf_counter()
    updated = _coerce_sglang_weight_to_tensor(
        engine.get_weights_by_name(tensor_name, truncate_size=10_000_000),
        name=tensor_name,
    )
    validate_fetch_duration_ms = (time.perf_counter() - validate_fetch_start) * 1000
    allclose = _tensor_close(updated, replacement)
    checksum = _tensor_checksum(updated)
    checksum_matches = checksum == replacement_checksum
    max_abs_error = _tensor_max_abs_error(updated, replacement)

    restore_start = time.perf_counter()
    restore_result = engine.update_weights_from_tensor(
        [(tensor_name, original)],
        load_format="direct",
        flush_cache=True,
    )
    restore_duration_ms = (time.perf_counter() - restore_start) * 1000
    restore_success, restore_message = _sglang_update_succeeded(restore_result)
    restored = _coerce_sglang_weight_to_tensor(
        engine.get_weights_by_name(tensor_name, truncate_size=10_000_000),
        name=tensor_name,
    )
    restored_original = _tensor_close(restored, original)

    proof = {
        "sglang_imported": True,
        "sglang_engine_started": True,
        "sglang_engine_get_weights_by_name_used": True,
        "sglang_engine_update_weights_from_tensor_used": True,
        "sglang_engine_owned_target_tensor": True,
        "sglang_owned_target_tensor": True,
        "receiver_request_from_sglang_engine_weight": True,
        "receiver_request_from_runtime_owned_tensor": True,
        "receiver_segment_assembly_used": True,
        "receiver_installed_into_sglang_engine_owned_tensor": allclose,
        "receiver_installed_into_sglang_owned_tensor": allclose,
        "receiver_installed_into_runtime_owned_tensor": allclose,
        "runtime_imported": True,
        "runtime_owned_target_tensor": True,
        "target_slice_spans_multiple_trainers": len({plan.source_id for plan in plans})
        >= 2,
        "checksum_gate": checksum_matches,
        "allclose": allclose,
        "restored_original_tensor": restored_original,
        "trainer_full_all_gather_used": False,
        "trainer_side_inference_layout_conversion_used": False,
        "host_side_torch_cat_used": False,
        "actual_nixl_reads_used": False,
        "synthetic_trainer_payloads_used": True,
        "real_trainer_process_used": False,
        "real_runtime_engine_used": True,
    }

    result = {
        "schema_version": 1,
        "result": (
            "pass"
            if update_success
            and allclose
            and checksum_matches
            and restore_success
            and restored_original
            else "fail"
        ),
        "mode": mode,
        "runtime_framework": "sglang",
        "framework_version": sglang_version,
        "sglang_version": sglang_version,
        "model_name": model_name,
        "model_version": model_version,
        "model_path": model_path,
        "engine_class": type(engine).__name__,
        "module_class": "sglang.Engine",
        "module_path": "sglang.Engine",
        "target_tensor_name": tensor_name,
        "target_key": target_key,
        "target_shape": list(_shape(original)),
        "target_dtype": _torch_dtype_name(original.dtype),
        "target_device": str(original.device),
        "request": requests[0].to_dict(),
        "source_ownerships": [owner.to_dict() for owner in owners],
        "segment_plans": [plan.to_dict() for plan in plans],
        "installed_segments": [segment.__dict__ for segment in installed],
        "proof": proof,
        "validation": {
            "assembled_allclose": assembled_allclose,
            "allclose": allclose,
            "checksum": checksum,
            "expected_checksum": replacement_checksum,
            "checksum_matches": checksum_matches,
            "max_abs_error": max_abs_error,
            "original_checksum": original_checksum,
            "restored_checksum": _tensor_checksum(restored),
            "restored_original": restored_original,
            "update_success": update_success,
            "update_message": update_message,
            "restore_success": restore_success,
            "restore_message": restore_message,
        },
        "metrics": {
            "engine_start_duration_ms": engine_start_duration_ms,
            "initial_weight_fetch_duration_ms": initial_fetch_duration_ms,
            "request_build_duration_ms": request_duration_ms,
            "planner_duration_ms": planner_duration_ms,
            "segment_assembly_duration_ms": assemble_duration_ms,
            "activation_install_duration_ms": update_duration_ms,
            "validate_weight_fetch_duration_ms": validate_fetch_duration_ms,
            "restore_duration_ms": restore_duration_ms,
            "trainer_to_inference_bytes": sum(plan.bytes for plan in plans),
            "segment_count": len(plans),
            "source_count_per_target_tensor": {
                tensor_name: len({plan.source_id for plan in plans})
            },
            "target_tensor_bytes": _nbytes(original),
        },
    }
    if artifact_path is not None:
        _write_artifact(result, artifact_path)
    if result["result"] != "pass":
        raise RuntimeError(f"SGLang engine receiver smoke failed: {result!r}")
    return result


def run_sglang_receiver_refit_on_module(
    module: torch.nn.Module,
    *,
    model_name: str,
    model_version: str,
    module_path: str,
    sglang_version: str = "",
    sglang_imported: bool | None = None,
    real_runtime_engine_used: bool = False,
    preferred_tensor_name: str = "",
    artifact_path: str | Path | None = None,
    mode: str = "sglang-receiver-owned-tensor-smoke",
    model_path: str = "",
) -> dict[str, Any]:
    """Run the generic receiver-owned tensor smoke with SGLang metadata."""

    if sglang_imported is None or not sglang_version:
        detected_version, detected_imported = detect_sglang_version()
        if not sglang_version:
            sglang_version = detected_version
        if sglang_imported is None:
            sglang_imported = detected_imported

    return run_receiver_refit_on_module(
        module,
        model_name=model_name,
        model_version=model_version,
        module_path=module_path,
        runtime_framework="sglang",
        framework_version=sglang_version,
        runtime_imported=bool(sglang_imported),
        real_runtime_engine_used=real_runtime_engine_used,
        preferred_tensor_name=preferred_tensor_name,
        artifact_path=artifact_path,
        mode=mode,
        model_path=model_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-name", default="mx-live-sglang-receiver-smoke")
    parser.add_argument("--model-version", default="step-live-sglang-receiver")
    parser.add_argument("--preferred-tensor-name", default="lm_head.weight")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--mem-fraction-static", type=float, default=0.2)
    parser.add_argument("--enable-cuda-graph", action="store_true")
    parser.add_argument("--enable-piecewise-cuda-graph", action="store_true")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = args.model_path
    if not model_path:
        model_path = str(
            create_tiny_llama_checkpoint(
                Path(tempfile.mkdtemp(prefix="mx-sglang-tiny-llama-"))
            )
        )

    version, _ = detect_sglang_version()
    engine = None
    try:
        engine_start = time.perf_counter()
        engine = load_sglang_engine(
            model_path=model_path,
            dtype=args.dtype,
            context_length=args.context_length,
            mem_fraction_static=args.mem_fraction_static,
            disable_cuda_graph=not args.enable_cuda_graph,
            disable_piecewise_cuda_graph=not args.enable_piecewise_cuda_graph,
            log_level=args.log_level,
        )
        engine_start_duration_ms = (time.perf_counter() - engine_start) * 1000
        result = run_sglang_receiver_refit_on_engine(
            engine,
            model_name=args.model_name,
            model_version=args.model_version,
            sglang_version=version,
            preferred_tensor_name=args.preferred_tensor_name,
            artifact_path=args.artifact_path,
            model_path=model_path,
            engine_start_duration_ms=engine_start_duration_ms,
        )
        print(
            "MX_SGLANG_RECEIVER_SMOKE "
            f"result={result['result']} tensor={result['target_tensor_name']} "
            f"shape={result['target_shape']} allclose={result['validation']['allclose']} "
            f"checksum_matches={result['validation']['checksum_matches']}",
            flush=True,
        )
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
