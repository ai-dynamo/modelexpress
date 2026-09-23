# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime driver and receipt producer for the matched MILES benchmark."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import types
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

SCHEMA = "miles-m2n-benchmark-v1"
HEADLINE_START_BOUNDARY = "pre_weights_getter_post_begin_barrier"
HEADLINE_END_BOUNDARY = "post_transfer_trainer_barrier"
E2E_START_BOUNDARY = "post_pause_pre_update"
E2E_END_BOUNDARY = "all_rollout_ready"
EXTERNAL_PROTOCOL = (
    "modelexpress_rl.collective.integrations.miles_pr3304:build_protocol"
)
TRANSFER_MODES = frozenset(("broadcast", "external"))
_STATE = threading.local()


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} must be set")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_receipt(
    *,
    checkpoint_manifest_path: Path,
    checkpoint_revision: str,
    checkpoint_sha256: str,
    model_path: Path,
    model_id: str,
) -> None:
    resolved_model_path = model_path.resolve()
    try:
        resolved_manifest_path = checkpoint_manifest_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(
            f"checkpoint manifest is not materialized at {checkpoint_manifest_path}"
        ) from error
    if resolved_model_path not in resolved_manifest_path.parents:
        raise ValueError(
            f"checkpoint manifest must be inside model checkpoint {resolved_model_path}"
        )

    lines = checkpoint_manifest_path.read_text().splitlines()
    if not lines:
        raise ValueError("checkpoint manifest is empty")
    canonical = ("\n".join(sorted(lines)) + "\n").encode()
    observed_sha256 = hashlib.sha256(canonical).hexdigest()
    if observed_sha256 != checkpoint_sha256:
        raise ValueError(
            "checkpoint manifest SHA-256 mismatch: "
            f"expected {checkpoint_sha256}, observed {observed_sha256}"
        )

    entries: set[PurePosixPath] = set()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"invalid checkpoint manifest entry at line {line_number}")
        try:
            int(fields[0], 16)
        except ValueError as error:
            raise ValueError(
                f"invalid checkpoint digest at line {line_number}"
            ) from error
        relative = PurePosixPath(fields[1].removeprefix("./"))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe checkpoint manifest path at line {line_number}")
        if relative in entries:
            raise ValueError(f"duplicate checkpoint manifest path {relative}")
        entries.add(relative)
        entry_path = model_path / Path(*relative.parts)
        if not entry_path.is_file():
            raise ValueError(f"checkpoint manifest path is missing: {relative}")
        if resolved_model_path not in entry_path.resolve().parents:
            raise ValueError(
                f"checkpoint manifest path escapes the model tree: {relative}"
            )
        observed_entry_sha256 = _sha256(entry_path)
        if observed_entry_sha256 != fields[0]:
            raise ValueError(
                "checkpoint file SHA-256 mismatch for "
                f"{relative}: expected {fields[0]}, "
                f"observed {observed_entry_sha256}"
            )
    if PurePosixPath("config.json") not in entries:
        raise ValueError("checkpoint manifest does not cover config.json")

    if not (model_path / ".prewarm-complete").is_file():
        raise ValueError("checkpoint completion marker is missing")
    repository_path = model_path / "repository.txt"
    if not repository_path.is_file() or repository_path.read_text().strip() != model_id:
        raise ValueError("checkpoint repository receipt does not match model_id")
    revision_path = (
        model_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    )
    if not revision_path.is_file():
        raise ValueError("checkpoint revision receipt is missing")
    revision_lines = revision_path.read_text().splitlines()
    if not revision_lines:
        raise ValueError("checkpoint revision receipt is empty")
    observed_revision = revision_lines[0].strip()
    if observed_revision != checkpoint_revision:
        raise ValueError(
            "checkpoint revision mismatch: "
            f"expected {checkpoint_revision}, observed {observed_revision}"
        )


def validate_workload_files(
    *,
    checkpoint_manifest_path: Path,
    checkpoint_revision: str,
    checkpoint_sha256: str,
    model_path: Path,
    model_id: str,
    dataset_path: Path,
    dataset_revision: str,
    dataset_sha256: str,
) -> None:
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise ValueError(f"model checkpoint is not materialized at {model_path}")
    validate_checkpoint_receipt(
        checkpoint_manifest_path=checkpoint_manifest_path,
        checkpoint_revision=checkpoint_revision,
        checkpoint_sha256=checkpoint_sha256,
        model_path=model_path,
        model_id=model_id,
    )
    if not dataset_path.is_file():
        raise ValueError(f"dataset is not materialized at {dataset_path}")
    observed_sha256 = _sha256(dataset_path)
    if observed_sha256 != dataset_sha256:
        raise ValueError(
            f"dataset SHA-256 mismatch: expected {dataset_sha256}, "
            f"observed {observed_sha256}"
        )
    observed_revision = subprocess.run(
        ["git", "-C", str(dataset_path.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_revision != dataset_revision:
        raise ValueError(
            f"dataset revision mismatch: expected {dataset_revision}, "
            f"observed {observed_revision}"
        )


def validate_external_runtime_contract(
    *,
    get_protocol: Callable[[argparse.Namespace], Any] | None = None,
    protocol_type: type | None = None,
    module_factory: Callable[[argparse.Namespace], Any] | None = None,
) -> None:
    if get_protocol is None or protocol_type is None:
        from miles.backends.training_utils.weight_update.protocol import (
            WeightTransferProtocol,
            get_weight_transfer_protocol,
        )

        get_protocol = get_weight_transfer_protocol
        protocol_type = WeightTransferProtocol

    if module_factory is None:

        class ProbeProtocol(protocol_type):
            def connect(self, *args, **kwargs) -> None:
                self.is_sender = True

            def send_bucket(self, bucket) -> None:
                return None

        module_factory = ProbeProtocol

    module_name = "_miles_benchmark_normal_protocol_probe"
    probe_module = types.ModuleType(module_name)
    probe_module.build_protocol = module_factory
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = probe_module
    try:
        args = argparse.Namespace(
            colocate=False,
            update_weight_transfer_mode="external",
            update_weight_transfer_protocol=f"{module_name}:build_protocol",
        )
        try:
            protocol = get_protocol(args)
        except Exception as error:
            raise RuntimeError(
                "runtime MILES loader does not accept the normal "
                "WeightTransferProtocol lifecycle required by PR3304"
            ) from error
        if not isinstance(protocol, protocol_type) or getattr(
            protocol, "owns_update_round", False
        ):
            raise RuntimeError(
                "runtime MILES loader did not return a normal "
                "WeightTransferProtocol lifecycle"
            )
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def validate_external_protocol_factory(path: str = EXTERNAL_PROTOCOL) -> None:
    module_name, separator, attribute_name = path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise RuntimeError(f"invalid external protocol factory path {path!r}")
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory) or inspect.iscoroutinefunction(factory):
        raise RuntimeError(
            f"external protocol factory {path!r} must be a synchronous callable"
        )


def build_train_args(
    *,
    model_path: Path,
    dataset_path: Path,
    transfer_mode: str,
    num_rollouts: int,
    actor_gpus: int,
    pipeline_parallel_size: int,
    rollout_gpus: int,
    rollout_gpus_per_engine: int,
) -> str:
    if transfer_mode not in TRANSFER_MODES:
        raise ValueError(f"unsupported transfer mode {transfer_mode!r}")
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive")
    transfer_args = [f"--update-weight-transfer-mode {transfer_mode}"]
    if transfer_mode == "external":
        transfer_args.append(f"--update-weight-transfer-protocol {EXTERNAL_PROTOCOL}")
    arguments = (
        "--hf-checkpoint",
        str(model_path),
        "--prompt-data",
        str(dataset_path),
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--rm-type",
        "math",
        "--num-rollout",
        str(num_rollouts),
        "--rollout-batch-size",
        "8",
        "--n-samples-per-prompt",
        "4",
        "--rollout-max-response-len",
        "256",
        "--rollout-temperature",
        "0.8",
        "--global-batch-size",
        "32",
        "--advantage-estimator",
        "grpo",
        "--entropy-coef",
        "0.00",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        str(pipeline_parallel_size),
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        "9216",
        "--rollout-num-gpus",
        str(rollout_gpus),
        "--rollout-num-gpus-per-engine",
        str(rollout_gpus_per_engine),
        "--sglang-mem-fraction-static",
        "0.65",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend",
        "flash",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(actor_gpus),
        "--megatron-to-hf-mode",
        "bridge",
        "--update-weights-interval",
        "1",
        *shlex.split(" ".join(transfer_args)),
        "--skip-eval-before-train",
        "--ci-test",
    )
    return shlex.join(arguments)


def build_structured_trial(
    raw: Mapping[str, Any],
    *,
    mode: str,
    training_run_completed: bool,
) -> dict[str, Any]:
    trial = int(raw["trial"])
    bytes_total = int(raw["bytes_total"])
    bytes_m2n = int(raw["bytes_m2n"])
    bytes_residual = int(raw["bytes_residual"])
    correctness = {
        "all_names_installed": raw["all_names_installed"],
        "all_rollout_ready": bool(raw["all_rollout_ready"]),
        "digest_equal": raw["digest_verified"],
        "training_run_completed": training_run_completed,
        "version_agreement": bool(raw["version_agreement"]),
        "weight_checker_enabled": bool(raw["weight_checker_enabled"]),
    }
    return {
        "schema": SCHEMA,
        "mode": mode,
        "trial": trial,
        "step": trial,
        "cold": trial == 0,
        "headline_start_boundary": HEADLINE_START_BOUNDARY,
        "headline_end_boundary": HEADLINE_END_BOUNDARY,
        "update_weights_implementation_s": float(
            raw["update_weights_implementation_s"]
        ),
        "e2e_start_boundary": E2E_START_BOUNDARY,
        "e2e_end_boundary": E2E_END_BOUNDARY,
        "e2e_update_s": float(raw["e2e_update_s"]),
        "e2e_boundary_exact": True,
        "transport_s": float(raw["transport_s"]),
        "residual_s": float(raw["residual_s"]),
        "install_s": float(raw["install_s"]),
        "bytes_total": bytes_total,
        "bytes_m2n": bytes_m2n,
        "bytes_residual": bytes_residual,
        "coverage_fraction": (
            (bytes_m2n + bytes_residual) / bytes_total if bytes_total else 0.0
        ),
        "correctness": correctness,
        "excluded_names": list(raw.get("excluded_names", [])),
        "receipt_complete": all(value is True for value in correctness.values()),
        "status": "ok",
        "timing_source": "miles_runtime_timer_sink",
        "timing_valid": True,
        "weight_version": str(raw["weight_version"]),
    }


def weight_compare_succeeded(result: Any) -> bool:
    if not isinstance(result, (list, tuple)) or not result:
        return False
    for item in result:
        if not isinstance(item, Mapping) or item.get("success") is not True:
            return False
        if item.get("equal") is False:
            return False
    return True


class RuntimeReceiptState:
    """Collect one actor rank's real update activity outside measured timers."""

    def __init__(self, *, mode: str, output_path: Path) -> None:
        self.mode = mode
        self.output_path = output_path
        self.weight_version = 0
        self.names: dict[str, int] = {}
        self.m2n_names: dict[str, int] = {}
        self.residual_names: dict[str, int] = {}
        self.transport_s = 0.0
        self.residual_s = 0.0
        self.install_s = 0.0
        self.implementation_s: float | None = None
        self.implementation_started_at: float | None = None
        self.digest_verified = False
        self.excluded_names: list[str] = []

    def reset(self) -> None:
        self.names = {}
        self.m2n_names = {}
        self.residual_names = {}
        self.transport_s = 0.0
        self.residual_s = 0.0
        self.install_s = 0.0
        self.implementation_s = None
        self.implementation_started_at = None
        self.digest_verified = False
        self.excluded_names = []

    def record_bucket(self, bucket: list[tuple[str, Any]]) -> None:
        for name, tensor in bucket:
            size = int(tensor.numel()) * int(tensor.element_size())
            self.names[str(name)] = size
            self.residual_names[str(name)] = size

    def record_hybrid_plan(self, protocol: Any) -> None:
        projection = getattr(protocol, "_projection", None)
        if projection is None:
            raise RuntimeError(
                "ModelExpress hybrid protocol did not expose its projection"
            )
        names = {}
        for entry in projection.topology_plan.plan.bulk:
            elements = 1
            for dimension in entry.global_shape:
                elements *= int(dimension)
            names[str(entry.name)] = elements * _dtype_element_size(entry.dtype)
        if not names:
            raise RuntimeError("ModelExpress hybrid plan contains no routed tensors")
        self.m2n_names = names
        self.names.update(names)

    def raw_record(self, *, e2e_update_s: float) -> dict[str, Any]:
        if self.implementation_s is None:
            raise RuntimeError("missing update_weights_implementation timer")
        if self.implementation_started_at is None:
            raise RuntimeError("missing update_weights_implementation start time")
        total = sum(self.names.values())
        if total <= 0:
            raise RuntimeError("runtime receipt recorded no transferred tensor bytes")
        return {
            "trial": self.weight_version - 1,
            "weight_version": str(self.weight_version),
            "names": sorted(self.names),
            "name_sizes": dict(sorted(self.names.items())),
            "m2n_name_sizes": dict(sorted(self.m2n_names.items())),
            "residual_name_sizes": dict(sorted(self.residual_names.items())),
            "excluded_names": self.excluded_names,
            "bytes_total": total,
            "bytes_m2n": sum(self.m2n_names.values()),
            "bytes_residual": sum(self.residual_names.values()),
            "digest_verified": self.digest_verified,
            "transport_s": self.transport_s,
            "residual_s": self.residual_s,
            "install_s": self.install_s,
            "implementation_started_at_epoch_s": self.implementation_started_at,
            "update_weights_implementation_s": self.implementation_s,
            "e2e_update_s": e2e_update_s,
        }


def _dtype_element_size(dtype: Any) -> int:
    label = str(dtype)
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
    }
    name = aliases.get(label.removeprefix("torch."), label.removeprefix("torch."))
    sizes = {
        "bfloat16": 2,
        "bool": 1,
        "float16": 2,
        "float32": 4,
        "float64": 8,
        "float8_e4m3fn": 1,
        "float8_e4m3fnuz": 1,
        "float8_e5m2": 1,
        "float8_e5m2fnuz": 1,
        "int8": 1,
        "int16": 2,
        "int32": 4,
        "int64": 8,
        "uint8": 1,
        "uint16": 2,
        "uint32": 4,
        "uint64": 8,
    }
    if name not in sizes:
        raise RuntimeError(f"unsupported benchmark tensor dtype {label!r}")
    return sizes[name]


class _TimerSink:
    def __call__(self, name: str, started: float, ended: float) -> None:
        state = getattr(_STATE, "receipt", None)
        if state is None:
            return
        elapsed = ended - started
        if name == "update_weights_implementation":
            state.implementation_started_at = started
            state.implementation_s = elapsed
        elif name == "finalize_and_resume_engines":
            state.install_s = elapsed
        elif name == "update_weights":
            _write_distributed_raw_record(state, e2e_update_s=elapsed)


def _merge_name_sizes(
    records: list[Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for item in records:
        for name, size in item[field].items():
            integer_size = int(size)
            if name in sizes and sizes[name] != integer_size:
                raise RuntimeError(f"inconsistent runtime size for {name}")
            sizes[str(name)] = integer_size
    return sizes


def _combine_rank_records(
    records: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("runtime receipt gather returned no trainer records")
    combined = dict(records[0])
    if mode == "external":
        m2n_sizes = dict(records[0]["m2n_name_sizes"])
        if any(item["m2n_name_sizes"] != m2n_sizes for item in records[1:]):
            raise RuntimeError(
                "ModelExpress routed plan disagrees across trainer ranks"
            )
        residual_sizes = _merge_name_sizes(
            records,
            field="residual_name_sizes",
        )
    else:
        m2n_sizes = {}
        residual_sizes = _merge_name_sizes(records, field="residual_name_sizes")
    name_sizes = dict(m2n_sizes)
    for name, size in residual_sizes.items():
        if name in name_sizes:
            raise RuntimeError(
                f"tensor {name} appears in both routed and residual phases"
            )
        name_sizes[name] = size
    combined["names"] = sorted(name_sizes)
    combined["name_sizes"] = dict(sorted(name_sizes.items()))
    combined["m2n_name_sizes"] = dict(sorted(m2n_sizes.items()))
    combined["residual_name_sizes"] = dict(sorted(residual_sizes.items()))
    combined["bytes_total"] = sum(name_sizes.values())
    combined["bytes_m2n"] = sum(m2n_sizes.values())
    combined["bytes_residual"] = sum(residual_sizes.values())
    combined["digest_verified"] = all(bool(item["digest_verified"]) for item in records)
    combined["excluded_names"] = sorted(
        {str(name) for item in records for name in item.get("excluded_names", [])}
    )
    combined["transport_s"] = max(float(item["transport_s"]) for item in records)
    combined["residual_s"] = max(float(item["residual_s"]) for item in records)
    return combined


def _write_distributed_raw_record(
    state: RuntimeReceiptState,
    *,
    e2e_update_s: float,
) -> None:
    import torch.distributed as dist
    from miles.utils.distributed_utils import get_gloo_group

    local = state.raw_record(e2e_update_s=e2e_update_s)
    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local, group=get_gloo_group())
    if dist.get_rank() != 0:
        state.reset()
        return
    records = [item for item in gathered if item is not None]
    combined = _combine_rank_records(records, mode=state.mode)
    state.output_path.parent.mkdir(parents=True, exist_ok=True)
    with state.output_path.open("a") as stream:
        stream.write(json.dumps(combined, sort_keys=True, separators=(",", ":")) + "\n")
    state.reset()


def _instrument_protocol(protocol: Any, state: RuntimeReceiptState) -> None:
    if getattr(protocol, "owns_update_round", False):
        raise RuntimeError(
            "benchmark runtime requires the normal WeightTransferProtocol lifecycle"
        )
    if hasattr(protocol, "send_bucket"):
        original_send_bucket = protocol.send_bucket

        def send_bucket(this, bucket):
            state.record_bucket(bucket)
            started = time.monotonic()
            try:
                return original_send_bucket(bucket)
            finally:
                elapsed = time.monotonic() - started
                if state.mode == "external":
                    state.residual_s += elapsed
                else:
                    state.transport_s += elapsed

        protocol.send_bucket = types.MethodType(send_bucket, protocol)
    if state.mode == "external" and hasattr(protocol, "before_base_weights"):
        original_before_base_weights = protocol.before_base_weights

        def before_base_weights(this, weights):
            started = time.monotonic()
            try:
                result = original_before_base_weights(weights)
                if getattr(protocol, "_projection", None) is not None:
                    state.record_hybrid_plan(protocol)
                return result
            finally:
                state.transport_s += time.monotonic() - started

        protocol.before_base_weights = types.MethodType(
            before_base_weights,
            protocol,
        )


def install_miles_hooks() -> None:
    if os.environ.get("MILES_BENCH_ENABLE_RUNTIME_RECEIPTS") != "1":
        return
    from miles.backends.training_utils.weight_update.updater import WeightUpdater
    from miles.ray.rollout.inference_controller import InferenceController
    from miles.ray.train.group import TrainerController
    from miles.utils.timer import Timer

    if getattr(WeightUpdater, "_miles_benchmark_patched", False):
        return
    original_init = WeightUpdater.__init__
    original_update = WeightUpdater.update_weights
    original_controller_update = TrainerController.update_weights
    original_end_update = InferenceController.end_update_weights
    original_check_weights = InferenceController.check_weights
    mode = _required_environment("MILES_WEIGHT_TRANSFER_MODE")
    output_path = Path(_required_environment("MILES_BENCH_RAW_RECEIPTS"))
    e2e_output_path = Path(_required_environment("MILES_BENCH_E2E_RECEIPTS"))

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if os.environ.get("MILES_BENCH_REQUIRE_WEIGHT_CHECKER") == "1" and not getattr(
            self.args, "check_weight_update_equal", False
        ):
            raise RuntimeError(
                "accepted benchmark runs require the symmetric MILES "
                "weight-update checker"
            )
        state = RuntimeReceiptState(mode=mode, output_path=output_path)
        self._miles_benchmark_receipt = state
        _instrument_protocol(self.protocol, state)

    def patched_update(self, *args, **kwargs):
        state = self._miles_benchmark_receipt
        _STATE.receipt = state
        result = original_update(self, *args, **kwargs)
        state.weight_version = int(self.weight_version)
        return result

    async def patched_end_update(self, *args, **kwargs):
        result = await original_end_update(self, *args, **kwargs)
        self._miles_benchmark_rollouts_ready_at_epoch_s = time.time()
        return result

    async def patched_controller_update(self, *args, **kwargs):
        result = await original_controller_update(self, *args, **kwargs)
        ended = getattr(
            self._inference_controller,
            "_miles_benchmark_rollouts_ready_at_epoch_s",
            None,
        )
        if result is not None:
            if ended is None:
                raise RuntimeError(
                    "MILES controller did not expose all-rollout-ready time"
                )
            clients = [
                client
                for server in self._inference_controller.servers.values()
                if server.update_weights
                for client in server.api_clients
            ]
            versions = await asyncio.gather(
                *(client.get_weight_version() for client in clients)
            )
            version_agreement = bool(versions) and all(
                str(version) == str(result) for version in versions
            )
            checker_enabled = bool(
                getattr(self.args, "check_weight_update_equal", False)
            )
            self._miles_benchmark_latest_weight_version = str(result)
            e2e_output_path.parent.mkdir(parents=True, exist_ok=True)
            with e2e_output_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "all_names_installed": False,
                            "all_rollout_ready": True,
                            "all_rollout_ready_epoch_s": ended,
                            "digest_verified": False,
                            "record_type": "update",
                            "version_agreement": version_agreement,
                            "versions": [str(version) for version in versions],
                            "weight_checker_enabled": checker_enabled,
                            "weight_version": str(result),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return result

    async def patched_check_weights(self, *args, **kwargs):
        result = await original_check_weights(self, *args, **kwargs)
        action = kwargs.get("action", args[0] if args else None)
        if action != "compare":
            return result
        version = getattr(self, "_miles_benchmark_latest_weight_version", None)
        if version is None:
            raise RuntimeError(
                "weight comparison completed without a benchmark weight version"
            )
        skip_list = kwargs.get("skip_list")
        succeeded = weight_compare_succeeded(result)
        with e2e_output_path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "all_names_installed": succeeded and not skip_list,
                        "digest_verified": succeeded,
                        "record_type": "weight_compare",
                        "weight_version": version,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        return result

    WeightUpdater.__init__ = patched_init
    WeightUpdater.update_weights = patched_update
    InferenceController.end_update_weights = patched_end_update
    InferenceController.check_weights = patched_check_weights
    TrainerController.update_weights = patched_controller_update
    WeightUpdater._miles_benchmark_patched = True
    Timer().event_sinks.append(_TimerSink())


def finalize_receipts(
    raw_path: Path,
    *,
    e2e_path: Path,
    mode: str,
    expected_updates: int,
) -> list[dict[str, Any]]:
    raw_records = [
        json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()
    ]
    if len(raw_records) != expected_updates:
        raise RuntimeError(
            f"expected {expected_updates} runtime receipts, observed {len(raw_records)}"
        )
    e2e_records: dict[str, dict[str, Any]] = {}
    compare_records: dict[str, dict[str, Any]] = {}
    for line in e2e_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        version = str(item["weight_version"])
        if item.get("record_type") == "weight_compare":
            if version in compare_records:
                raise RuntimeError(
                    f"duplicate weight-compare receipt for version {version}"
                )
            compare_records[version] = item
            continue
        if version in e2e_records:
            raise RuntimeError(
                f"duplicate controller e2e receipt for version {version}"
            )
        e2e_records[version] = item
    if len(e2e_records) != expected_updates:
        raise RuntimeError(
            f"expected {expected_updates} controller e2e receipts, "
            f"observed {len(e2e_records)}"
        )
    for raw in raw_records:
        version = str(raw["weight_version"])
        if version not in e2e_records:
            raise RuntimeError(f"missing controller e2e receipt for version {version}")
        e2e = e2e_records[version]
        started = float(raw["implementation_started_at_epoch_s"])
        ended = float(e2e["all_rollout_ready_epoch_s"])
        if ended < started:
            raise RuntimeError(
                f"all-rollout-ready precedes post-pause start for version {version}"
            )
        raw["e2e_update_s"] = ended - started
        comparison = compare_records.get(version, {})
        if comparison and not bool(comparison.get("digest_verified")):
            raise RuntimeError(f"weight comparison failed for version {version}")
        raw["all_names_installed"] = comparison.get("all_names_installed")
        raw["all_rollout_ready"] = bool(e2e["all_rollout_ready"])
        raw["digest_verified"] = comparison.get("digest_verified")
        raw["version_agreement"] = bool(e2e["version_agreement"])
        raw["weight_checker_enabled"] = bool(e2e["weight_checker_enabled"])
    trials = [
        build_structured_trial(
            raw,
            mode=("native_broadcast" if mode == "broadcast" else "modelexpress_m2n"),
            training_run_completed=True,
        )
        for raw in raw_records
    ]
    for index, trial in enumerate(trials):
        if trial["trial"] != index or trial["step"] != index:
            raise RuntimeError("runtime receipt indices are not contiguous")
        print(json.dumps(trial, sort_keys=True, separators=(",", ":")), flush=True)
    return trials


def _forwarded_environment(
    mode: str,
    raw_path: Path,
    e2e_path: Path,
) -> dict[str, str]:
    keys = {
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NCCL_CUMEM_ENABLE",
        "SGLANG_NCCL_SO_PATH",
    }
    if mode == "external":
        keys.update(
            {
                "MX_SERVER_ADDRESS",
                "SGLANG_PLUGINS",
                "MX_MILES_RUN_ID",
                "MX_MILES_VERIFY_TENSOR_EQUALITY",
                "MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S",
                "MX_NCCL_REFIT_GROUP_TIMEOUT_S",
                "MX_NCCL_REFIT_NUM_STREAMS",
                "MX_NCCL_REFIT_POLL_INTERVAL_S",
                "MX_NCCL_REFIT_TRANSFER_TIMEOUT_S",
            }
        )
    environment = {key: os.environ[key] for key in keys if key in os.environ}
    environment.update(
        {
            "MILES_BENCH_ENABLE_RUNTIME_RECEIPTS": "1",
            "MILES_BENCH_E2E_RECEIPTS": str(e2e_path),
            "MILES_BENCH_RAW_RECEIPTS": str(raw_path),
            "MILES_BENCH_REQUIRE_WEIGHT_CHECKER": _required_environment(
                "MILES_BENCH_REQUIRE_WEIGHT_CHECKER"
            ),
            "MILES_WEIGHT_TRANSFER_MODE": mode,
            "PYTHONPATH": (
                "/opt/miles-benchmark"
                + (
                    f":{os.environ['PYTHONPATH']}"
                    if os.environ.get("PYTHONPATH")
                    else ""
                )
            ),
        }
    )
    return environment


def main() -> None:
    from miles.utils.external_utils import command_utils

    mode = _required_environment("MILES_WEIGHT_TRANSFER_MODE")
    rollouts = int(_required_environment("MILES_BENCH_ROLLOUTS"))
    model_path = Path(_required_environment("MILES_MODEL_PATH"))
    dataset_path = Path(_required_environment("MILES_DATASET_PATH"))
    validate_workload_files(
        checkpoint_manifest_path=Path(
            _required_environment("MILES_CHECKPOINT_MANIFEST_PATH")
        ),
        checkpoint_revision=_required_environment("MILES_CHECKPOINT_REVISION"),
        checkpoint_sha256=_required_environment("MILES_CHECKPOINT_SHA256"),
        model_path=model_path,
        model_id=_required_environment("MODEL_ID"),
        dataset_path=dataset_path,
        dataset_revision=_required_environment("MILES_DATASET_REVISION"),
        dataset_sha256=_required_environment("MILES_DATASET_SHA256"),
    )
    if mode == "external":
        validate_external_runtime_contract()
        validate_external_protocol_factory()
        subprocess.run(
            ["python3", "/opt/modelexpress-miles-m2n/verify_runtime.py"],
            check=True,
        )
    raw_path = Path(_required_environment("MILES_BENCH_RAW_RECEIPTS"))
    e2e_path = Path(_required_environment("MILES_BENCH_E2E_RECEIPTS"))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.unlink(missing_ok=True)
    e2e_path.unlink(missing_ok=True)
    train_args = build_train_args(
        model_path=model_path,
        dataset_path=dataset_path,
        transfer_mode=mode,
        num_rollouts=rollouts,
        actor_gpus=int(_required_environment("ACTOR_GPUS")),
        pipeline_parallel_size=int(_required_environment("PIPELINE_PARALLEL_SIZE")),
        rollout_gpus=int(_required_environment("ROLLOUT_GPUS")),
        rollout_gpus_per_engine=int(_required_environment("ROLLOUT_GPUS_PER_ENGINE")),
    )
    command_utils.execute_train(
        train_args=train_args,
        num_gpus_per_node=int(_required_environment("TOTAL_GPUS")),
        megatron_model_type=_required_environment("MILES_MODEL_TYPE"),
        train_script="train_async.py",
        extra_env_vars=_forwarded_environment(mode, raw_path, e2e_path),
    )
    finalize_receipts(
        raw_path,
        e2e_path=e2e_path,
        mode=mode,
        expected_updates=rollouts + 1,
    )


if __name__ == "__main__":
    main()
