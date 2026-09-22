# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-loop microbench for the native MILES NCCL broadcast sender.

The source rank calls the real MILES
``broadcast.update_weights_from_distributed`` function. Receiver ranks mirror
the NCCL receive loop, while validating in-process clients replace SGLang HTTP
and model loading. Results therefore measure the native sender transport loop,
not the complete native MILES implementation or an application weight update.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA_VERSION = 1
BF16_BYTES = 2
DEFAULT_BUCKET_BYTES = 512 * 1024**2
EXPECTED_TENSOR_COUNT = 434
EXPECTED_LOGICAL_BYTES = 6_171_877_376
EXPECTED_MANIFEST_SHA256 = (
    "dc31e71f48bcd8d91bf95e054ab7458c97d89f29afc598d5c68965dd6d94d446"
)
MILES_BROADCAST_CALLABLE = (
    "miles.backends.training_utils.weight_update.protocols.broadcast."
    "update_weights_from_distributed"
)


@dataclass(frozen=True)
class TensorSpec:
    """One exported Hugging Face BF16 tensor and its indivisible source unit."""

    name: str
    shape: tuple[int, ...]
    layer: int | None
    source_unit: str

    @property
    def bytes(self) -> int:
        elements = 1
        for extent in self.shape:
            elements *= extent
        return elements * BF16_BYTES


@dataclass(frozen=True)
class Topology:
    """Production-shaped gathered source topology for four TP1 receivers."""

    source_rank: int = 0
    receiver_ranks: tuple[int, ...] = (1, 2, 3, 4)

    @property
    def group_ranks(self) -> tuple[int, ...]:
        return (self.source_rank, *self.receiver_ranks)

    @property
    def world_size(self) -> int:
        return len(self.group_ranks)

    def validate(self, *, rank: int, world_size: int) -> None:
        if world_size != self.world_size:
            raise ValueError(
                f"native broadcast transport microbench requires world size "
                f"{self.world_size}, got {world_size}"
            )
        if rank not in self.group_ranks:
            raise ValueError(f"rank {rank} is outside group ranks {self.group_ranks}")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Inputs that affect the measured transport loop."""

    output: Path
    warmup: int
    repetitions: int
    timeout_s: float
    bucket_bytes: int
    group_name: str
    selector: str

    def validate(self) -> None:
        validate_output_path(self.output)
        if self.warmup < 0:
            raise ValueError("warmup must not be negative")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.bucket_bytes <= 0:
            raise ValueError("bucket_bytes must be positive")
        if not self.group_name:
            raise ValueError("group_name must not be empty")
        if not self.selector:
            raise ValueError("selector must not be empty")


@dataclass(frozen=True)
class ExpectedMetadataCall:
    """Metadata expected from one native sender invocation."""

    names: tuple[str, ...]
    dtypes: tuple[Any, ...]
    shapes: tuple[tuple[int, ...], ...]


class ValidatingMetadataClient:
    """Async SGLang-client substitute that validates every metadata request."""

    def __init__(
        self,
        *,
        client_id: str,
        expected_names: Sequence[str] | None = None,
        expected_dtypes: Sequence[Any] | None = None,
        expected_shapes: Sequence[Sequence[int]] | None = None,
        expected_calls: Sequence[ExpectedMetadataCall] | None = None,
        expected_selector: str = "all",
        expected_group_name: str = "miles-pp_0",
    ) -> None:
        if expected_calls is None:
            if (
                expected_names is None
                or expected_dtypes is None
                or expected_shapes is None
            ):
                raise ValueError("provide expected_calls or names, dtypes, and shapes")
            expected_calls = (
                ExpectedMetadataCall(
                    names=tuple(expected_names),
                    dtypes=tuple(expected_dtypes),
                    shapes=tuple(tuple(shape) for shape in expected_shapes),
                ),
            )
        self.client_id = client_id
        self._expected_calls = tuple(expected_calls)
        self._expected_selector = expected_selector
        self._expected_group_name = expected_group_name
        self.calls = 0

    async def update_weights_from_distributed(
        self,
        *,
        names: Sequence[str],
        dtypes: Sequence[Any],
        shapes: Sequence[Sequence[int]],
        selector: str,
        group_name: str,
    ) -> None:
        if self.calls >= len(self._expected_calls):
            raise AssertionError(
                f"{self.client_id}: unexpected metadata call {self.calls + 1}"
            )
        expected = self._expected_calls[self.calls]
        if tuple(names) != expected.names:
            raise AssertionError(f"{self.client_id}: names mismatch")
        if tuple(dtypes) != expected.dtypes:
            raise AssertionError(f"{self.client_id}: dtypes mismatch")
        if tuple(tuple(shape) for shape in shapes) != expected.shapes:
            raise AssertionError(f"{self.client_id}: shapes mismatch")
        if selector != self._expected_selector:
            raise AssertionError(f"{self.client_id}: selector mismatch")
        if group_name != self._expected_group_name:
            raise AssertionError(f"{self.client_id}: group_name mismatch")
        self.calls += 1


def qwen_25_3b_manifest() -> tuple[TensorSpec, ...]:
    """Return the gathered Bridge export order for Qwen2.5-3B BF16."""
    manifest = [
        TensorSpec(
            "model.norm.weight",
            (2_048,),
            None,
            "decoder.final_layernorm.weight",
        ),
        TensorSpec(
            "model.embed_tokens.weight",
            (151_936, 2_048),
            None,
            "embedding.word_embeddings.weight",
        ),
    ]
    for layer in range(36):
        prefix = f"model.layers.{layer}"
        source_prefix = f"decoder.layers.{layer}"
        manifest.extend(
            (
                TensorSpec(
                    f"{prefix}.post_attention_layernorm.weight",
                    (2_048,),
                    layer,
                    f"{source_prefix}.mlp.linear_fc1.layer_norm_weight",
                ),
                TensorSpec(
                    f"{prefix}.mlp.gate_proj.weight",
                    (11_008, 2_048),
                    layer,
                    f"{source_prefix}.mlp.linear_fc1.weight",
                ),
                TensorSpec(
                    f"{prefix}.mlp.up_proj.weight",
                    (11_008, 2_048),
                    layer,
                    f"{source_prefix}.mlp.linear_fc1.weight",
                ),
                TensorSpec(
                    f"{prefix}.mlp.down_proj.weight",
                    (2_048, 11_008),
                    layer,
                    f"{source_prefix}.mlp.linear_fc2.weight",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.o_proj.weight",
                    (2_048, 2_048),
                    layer,
                    f"{source_prefix}.self_attention.linear_proj.weight",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.q_proj.bias",
                    (2_048,),
                    layer,
                    f"{source_prefix}.self_attention.linear_qkv.bias",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.k_proj.bias",
                    (256,),
                    layer,
                    f"{source_prefix}.self_attention.linear_qkv.bias",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.v_proj.bias",
                    (256,),
                    layer,
                    f"{source_prefix}.self_attention.linear_qkv.bias",
                ),
                TensorSpec(
                    f"{prefix}.input_layernorm.weight",
                    (2_048,),
                    layer,
                    f"{source_prefix}.self_attention.linear_qkv.layer_norm_weight",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.q_proj.weight",
                    (2_048, 2_048),
                    layer,
                    f"{source_prefix}.self_attention.linear_qkv.weight",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.k_proj.weight",
                    (256, 2_048),
                    layer,
                    f"{source_prefix}.self_attention.linear_qkv.weight",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.v_proj.weight",
                    (256, 2_048),
                    layer,
                    f"{source_prefix}.self_attention.linear_qkv.weight",
                ),
            )
        )
    return tuple(manifest)


def pack_manifest(
    manifest: Sequence[TensorSpec], max_bytes: int
) -> tuple[tuple[TensorSpec, ...], ...]:
    """Pack Bridge source units with MILES' current >= boundary behavior."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    units: list[list[TensorSpec]] = []
    seen_units: set[str] = set()
    for spec in manifest:
        if not units or units[-1][0].source_unit != spec.source_unit:
            if spec.source_unit in seen_units:
                raise ValueError(f"non-contiguous source unit {spec.source_unit}")
            seen_units.add(spec.source_unit)
            units.append([])
        units[-1].append(spec)

    buckets: list[tuple[TensorSpec, ...]] = []
    current: list[TensorSpec] = []
    current_bytes = 0
    for unit in units:
        unit_bytes = sum(spec.bytes for spec in unit)
        if current and current_bytes + unit_bytes >= max_bytes:
            buckets.append(tuple(current))
            current = []
            current_bytes = 0
        current.extend(unit)
        current_bytes += unit_bytes
    if current:
        buckets.append(tuple(current))
    return tuple(buckets)


def manifest_fingerprint(manifest: Sequence[TensorSpec]) -> str:
    payload = [asdict(spec) for spec in manifest]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest(manifest: Sequence[TensorSpec]) -> None:
    """Fail closed unless the benchmark payload is the reviewed exact manifest."""
    tensor_count = len(manifest)
    logical_bytes = sum(spec.bytes for spec in manifest)
    fingerprint = manifest_fingerprint(manifest)
    if tensor_count != EXPECTED_TENSOR_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TENSOR_COUNT} tensors, got {tensor_count}"
        )
    if logical_bytes != EXPECTED_LOGICAL_BYTES:
        raise ValueError(
            f"expected {EXPECTED_LOGICAL_BYTES} bytes, got {logical_bytes}"
        )
    if fingerprint != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"expected manifest {EXPECTED_MANIFEST_SHA256}, got {fingerprint}"
        )


def validate_output_path(path: Path) -> Path:
    """Require an absolute output outside common ephemeral filesystems."""
    if not path.is_absolute():
        raise ValueError(f"output must be an absolute persistent path, got {path}")
    resolved = path.resolve(strict=False)
    ephemeral_roots = (Path("/tmp"), Path("/dev/shm"), Path("/run"))
    for root in ephemeral_roots:
        if resolved == root or root in resolved.parents:
            raise ValueError(f"output must not use ephemeral storage under {root}")
    return resolved


def invoke_miles_broadcast(
    *,
    broadcast_fn: Callable[..., list[Any]],
    group: Any,
    clients: Sequence[Any],
    tensors: Sequence[tuple[str, Any]],
    group_name: str,
    selector: str,
) -> list[Any]:
    """Call the production MILES function without reproducing its sender loop."""
    return broadcast_fn(
        group_name,
        group,
        clients,
        tensors,
        selector=selector,
    )


def _load_miles_sender() -> tuple[Callable[..., list[Any]], Callable[[Any], Any]]:
    module = importlib.import_module(
        "miles.backends.training_utils.weight_update.protocols.broadcast"
    )
    async_utils = importlib.import_module("miles.utils.async_utils")
    sender = module.update_weights_from_distributed
    if (
        sender.__module__
        != "miles.backends.training_utils.weight_update.protocols.broadcast"
        or sender.__name__ != "update_weights_from_distributed"
    ):
        raise RuntimeError(f"unexpected MILES sender callable: {sender!r}")
    return sender, async_utils.wait_futures


def _require_gpu_environment() -> None:
    if os.environ.get("NCCL_CUMEM_ENABLE") != "1":
        raise RuntimeError("set NCCL_CUMEM_ENABLE=1 for the pinned runtime")
    if os.environ.get("NCCL_COMM_ID"):
        raise RuntimeError("NCCL_COMM_ID must not be forced")


def _allocate_tensors(
    manifest: Sequence[TensorSpec], device: Any, torch: Any
) -> list[tuple[TensorSpec, Any]]:
    return [
        (
            spec,
            torch.empty(spec.shape, dtype=torch.bfloat16, device=device),
        )
        for spec in manifest
    ]


def _byte_pattern(tensor_index: int, round_index: int) -> int:
    return ((tensor_index * 17 + round_index * 29) % 251) + 1


def _fill_source_tensors(
    tensors: Sequence[tuple[TensorSpec, Any]], round_index: int, torch: Any
) -> None:
    with torch.no_grad():
        for tensor_index, (_spec, tensor) in enumerate(tensors):
            tensor.view(torch.uint8).fill_(_byte_pattern(tensor_index, round_index))


def _verify_exact_bytes(
    tensors: Sequence[tuple[TensorSpec, Any]], round_index: int, torch: Any
) -> int:
    mismatched = torch.zeros((), dtype=torch.int64, device=tensors[0][1].device)
    for tensor_index, (_spec, tensor) in enumerate(tensors):
        expected = _byte_pattern(tensor_index, round_index)
        mismatched.add_(torch.count_nonzero(tensor.view(torch.uint8) != expected))
    return int(mismatched.item())


def _named_buckets(
    tensors: Sequence[tuple[TensorSpec, Any]],
    bucket_specs: Sequence[Sequence[TensorSpec]],
) -> list[list[tuple[str, Any]]]:
    by_name = {spec.name: tensor for spec, tensor in tensors}
    return [
        [(spec.name, by_name[spec.name]) for spec in specs] for specs in bucket_specs
    ]


def _expected_metadata_calls(
    named_buckets: Sequence[Sequence[tuple[str, Any]]],
) -> tuple[ExpectedMetadataCall, ...]:
    return tuple(
        ExpectedMetadataCall(
            names=tuple(name for name, _tensor in bucket),
            dtypes=tuple(tensor.dtype for _name, tensor in bucket),
            shapes=tuple(tuple(tensor.shape) for _name, tensor in bucket),
        )
        for bucket in named_buckets
    )


def _receive_buckets(group: Any, named_buckets: Sequence[Any], dist: Any) -> None:
    for bucket in named_buckets:
        handles = [
            dist.broadcast(tensor, 0, group=group, async_op=True)
            for _name, tensor in bucket
        ]
        for handle in handles:
            handle.wait()


def _collect_rank_scalars(
    value: float | int, *, dtype: Any, group: Any, torch: Any, dist: Any
) -> list[Any]:
    device = torch.device("cuda", torch.cuda.current_device())
    local = torch.tensor(value, dtype=dtype, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    return [item.item() for item in gathered]


def _git_revision(path: Path) -> str | None:
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            completed = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                return completed.stdout.strip()
            return None
    return None


def _callable_provenance(callable_obj: Callable[..., Any]) -> dict[str, Any]:
    source = inspect.getsourcefile(callable_obj)
    if source is None:
        return {
            "callable": MILES_BROADCAST_CALLABLE,
            "source_file": None,
            "source_sha256": None,
            "git_revision": None,
        }
    source_path = Path(source).resolve()
    return {
        "callable": MILES_BROADCAST_CALLABLE,
        "source_file": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "git_revision": _git_revision(source_path),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_gpu(config: BenchmarkConfig) -> dict[str, Any] | None:
    """Run under ``torchrun --nproc-per-node=5`` on one GPU node."""
    config.validate()
    _require_gpu_environment()

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    topology = Topology()
    topology.validate(rank=rank, world_size=world_size)
    if torch.cuda.device_count() <= local_rank:
        raise RuntimeError(
            f"local rank {local_rank} has no CUDA device; visible count is "
            f"{torch.cuda.device_count()}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    timeout = datetime.timedelta(seconds=config.timeout_s)
    dist.init_process_group(backend="nccl", timeout=timeout)
    group = None
    try:
        group = dist.new_group(
            ranks=list(topology.group_ranks),
            backend="nccl",
            timeout=timeout,
        )
        manifest = qwen_25_3b_manifest()
        validate_manifest(manifest)
        bucket_specs = pack_manifest(manifest, config.bucket_bytes)
        tensors = _allocate_tensors(manifest, device, torch)
        named_buckets = _named_buckets(tensors, bucket_specs)
        total_rounds = config.warmup + config.repetitions

        broadcast_fn = None
        wait_futures = None
        clients: list[ValidatingMetadataClient] = []
        callable_provenance = None
        if rank == topology.source_rank:
            broadcast_fn, wait_futures = _load_miles_sender()
            expected_once = _expected_metadata_calls(named_buckets)
            expected_all = expected_once * total_rounds
            clients = [
                ValidatingMetadataClient(
                    client_id=f"receiver-{receiver_rank}",
                    expected_calls=expected_all,
                    expected_selector=config.selector,
                    expected_group_name=config.group_name,
                )
                for receiver_rank in topology.receiver_ranks
            ]
            callable_provenance = _callable_provenance(broadcast_fn)

        records: list[dict[str, Any]] = []
        for round_index in range(total_rounds):
            if rank == topology.source_rank:
                _fill_source_tensors(tensors, round_index, torch)
            dist.barrier(group=group)
            torch.cuda.synchronize(device)
            started = time.perf_counter()

            if rank == topology.source_rank:
                if broadcast_fn is None or wait_futures is None:
                    raise RuntimeError("source rank did not load the MILES sender")
                for bucket in named_buckets:
                    futures = invoke_miles_broadcast(
                        broadcast_fn=broadcast_fn,
                        group=group,
                        clients=clients,
                        tensors=bucket,
                        group_name=config.group_name,
                        selector=config.selector,
                    )
                    wait_futures(futures)
            else:
                _receive_buckets(group, named_buckets, dist)

            torch.cuda.synchronize(device)
            elapsed_s = time.perf_counter() - started
            rank_elapsed = _collect_rank_scalars(
                elapsed_s,
                dtype=torch.float64,
                group=group,
                torch=torch,
                dist=dist,
            )
            mismatched_bytes = _verify_exact_bytes(tensors, round_index, torch)
            mismatches_by_rank = _collect_rank_scalars(
                mismatched_bytes,
                dtype=torch.int64,
                group=group,
                torch=torch,
                dist=dist,
            )
            verified_exact = all(value == 0 for value in mismatches_by_rank)

            if round_index >= config.warmup:
                sample_index = round_index - config.warmup
                max_wall_s = max(rank_elapsed)
                records.append(
                    {
                        "sample_index": sample_index,
                        "rank_elapsed_s": [
                            {
                                "rank": participant_rank,
                                "role": (
                                    "gathered-source"
                                    if participant_rank == topology.source_rank
                                    else "synthetic-receiver"
                                ),
                                "elapsed_s": rank_elapsed[participant_rank],
                            }
                            for participant_rank in topology.group_ranks
                        ],
                        "max_wall_s": max_wall_s,
                        "source_payload_gib_per_s": (
                            sum(spec.bytes for spec in manifest) / max_wall_s / 1024**3
                        ),
                        "aggregate_delivered_gib_per_s": (
                            sum(spec.bytes for spec in manifest)
                            * len(topology.receiver_ranks)
                            / max_wall_s
                            / 1024**3
                        ),
                        "mismatched_bytes_by_rank": mismatches_by_rank,
                        "verified_exact": verified_exact,
                    }
                )

        if rank != topology.source_rank:
            return None

        if callable_provenance is None:
            raise RuntimeError("source rank did not record callable provenance")
        max_wall_samples = [record["max_wall_s"] for record in records]
        expected_client_calls = total_rounds * len(bucket_specs)
        client_call_counts = {client.client_id: client.calls for client in clients}
        report = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": {
                "name": "native-miles-broadcast-transport-loop",
                "classification": "transport-loop microbenchmark",
                "production_fidelity": False,
                "measures": (
                    "actual MILES sender metadata scheduling and NCCL broadcast "
                    "loop with synthetic receiver loops"
                ),
                "excludes": [
                    "real HTTP requests",
                    "SGLang server allocation and receive implementation",
                    "model.load_weights",
                    "MILES pause, begin, end, version, and resume session calls",
                    "Megatron PP export, conversion, and gather",
                ],
                "comparison_warning": (
                    "Do not report this as the full native MILES implementation "
                    "or compare it directly with an application end-to-end time."
                ),
            },
            "config": {
                **asdict(config),
                "output": str(config.output),
            },
            "topology": {
                **asdict(topology),
                "group_ranks": topology.group_ranks,
                "world_size": topology.world_size,
                "production_shape": (
                    "one gather_pp source and four TP1 rollout participants"
                ),
                "receiver_implementation": (
                    "matching torch.distributed.broadcast loop, not SGLang"
                ),
            },
            "manifest": {
                "name": "Qwen2.5-3B gathered Bridge HF BF16 export",
                "tensor_count": len(manifest),
                "logical_bytes": sum(spec.bytes for spec in manifest),
                "fingerprint_sha256": manifest_fingerprint(manifest),
                "matches_production_tensor_shapes": True,
                "matches_production_export_order": True,
                "source": (
                    "Megatron-Bridge sorted gathered parameter tasks and "
                    "Qwen2 source mappings"
                ),
            },
            "packing": {
                "max_bytes": config.bucket_bytes,
                "bucket_count": len(bucket_specs),
                "bucket_tensor_counts": [len(bucket) for bucket in bucket_specs],
                "bucket_bytes": [
                    sum(spec.bytes for spec in bucket) for bucket in bucket_specs
                ],
                "classification": (
                    "production Bridge source-unit order with indivisible "
                    "conversion units and MILES >= size boundary"
                ),
            },
            "execution": {
                "warmup_rounds": config.warmup,
                "measured_rounds": config.repetitions,
                "actual_miles_function_calls_per_round": len(bucket_specs),
                "validating_metadata_clients": len(clients),
                "expected_calls_per_client": expected_client_calls,
                "calls_per_client": client_call_counts,
                "all_client_call_counts_exact": all(
                    count == expected_client_calls
                    for count in client_call_counts.values()
                ),
            },
            "samples": records,
            "summary": {
                "median_max_wall_s": statistics.median(max_wall_samples),
                "min_max_wall_s": min(max_wall_samples),
                "max_max_wall_s": max(max_wall_samples),
                "median_source_payload_gib_per_s": statistics.median(
                    record["source_payload_gib_per_s"] for record in records
                ),
                "median_aggregate_delivered_gib_per_s": statistics.median(
                    record["aggregate_delivered_gib_per_s"] for record in records
                ),
                "all_samples_verified_exact": all(
                    record["verified_exact"] for record in records
                ),
            },
            "provenance": {
                "miles_sender": callable_provenance,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "gpu": torch.cuda.get_device_name(device),
                "hostname": platform.node(),
                "nccl_environment": {
                    key: value
                    for key, value in sorted(os.environ.items())
                    if key.startswith("NCCL_")
                },
            },
        }
        return report
    finally:
        if group is not None:
            dist.destroy_process_group(group)
        if dist.is_initialized():
            dist.destroy_process_group()


def _parse_args(argv: Sequence[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--bucket-bytes", type=int, default=DEFAULT_BUCKET_BYTES)
    parser.add_argument("--group-name", default="miles-pp_0")
    parser.add_argument("--selector", default="all")
    args = parser.parse_args(argv)
    return BenchmarkConfig(
        output=args.output,
        warmup=args.warmup,
        repetitions=args.repetitions,
        timeout_s=args.timeout_s,
        bucket_bytes=args.bucket_bytes,
        group_name=args.group_name,
        selector=args.selector,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_args(argv)
    report = run_gpu(config)
    if report is None:
        return 0
    _atomic_write_json(config.output, report)
    if not report["summary"]["all_samples_verified_exact"]:
        raise RuntimeError("native broadcast transport verification failed")
    if not report["execution"]["all_client_call_counts_exact"]:
        raise RuntimeError("metadata client call-count verification failed")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
