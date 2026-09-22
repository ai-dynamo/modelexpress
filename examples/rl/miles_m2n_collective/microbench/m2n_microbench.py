# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone behavioral and native NCCL M2N microbench.

The GPU mode deliberately uses the production ``_reshard`` binding and
``LaneCommunicator`` drain path. It is a transport microbench, not a MILES
training or ModelExpress control-plane benchmark.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA_VERSION = 2
BF16_BYTES = 2
EMBEDDING_BYTES = 622_329_856
VRAM_HEADROOM_FRACTION = 1.25
SUPPORTED_PARTITIONS = (1, 2, 4)
SUPPORTED_FANOUT = (1, 2, 4, 8)
SUPPORTED_STREAMS = (1, 2, 4)
SUPPORTED_SCHEDULES = ("miles-current", "synthetic")
SUPPORTED_GROUPING = ("singleton", "layer", "bucket")
SUPPORTED_DRAIN = ("per-tensor", "per-group", "end")
SUPPORTED_GROUP_KEYS = ("unique", "shared")


@dataclass(frozen=True)
class TensorSpec:
    """A canonical BF16 tensor in the MILES external-protocol namespace."""

    name: str
    shape: tuple[int, ...]
    layer: int | None

    @property
    def bytes(self) -> int:
        count = 1
        for extent in self.shape:
            count *= extent
        return count * BF16_BYTES


@dataclass(frozen=True)
class Case:
    """Normalized microbench inputs that change data-plane scheduling."""

    profile: str
    source_partitions: int
    destination_fanout: int
    streams: int
    schedule: str
    grouping: str
    bucket_bytes: int
    drain: str
    group_keys: str
    warmup: int
    repetitions: int
    timeout_s: float

    def validate(self) -> None:
        if self.source_partitions not in SUPPORTED_PARTITIONS:
            raise ValueError(
                f"source_partitions must be one of {SUPPORTED_PARTITIONS}, got "
                f"{self.source_partitions}"
            )
        if self.destination_fanout not in SUPPORTED_FANOUT:
            raise ValueError(
                f"destination_fanout must be one of {SUPPORTED_FANOUT}, got "
                f"{self.destination_fanout}"
            )
        if self.streams not in SUPPORTED_STREAMS:
            raise ValueError(
                f"streams must be one of {SUPPORTED_STREAMS}, got {self.streams}"
            )
        if self.schedule not in SUPPORTED_SCHEDULES:
            raise ValueError(
                f"schedule must be one of {SUPPORTED_SCHEDULES}, got {self.schedule!r}"
            )
        if self.grouping not in SUPPORTED_GROUPING:
            raise ValueError(
                f"grouping must be one of {SUPPORTED_GROUPING}, got {self.grouping!r}"
            )
        if self.drain not in SUPPORTED_DRAIN:
            raise ValueError(
                f"drain must be one of {SUPPORTED_DRAIN}, got {self.drain!r}"
            )
        if self.group_keys not in SUPPORTED_GROUP_KEYS:
            raise ValueError(
                f"group_keys must be one of {SUPPORTED_GROUP_KEYS}, got "
                f"{self.group_keys!r}"
            )
        if self.bucket_bytes <= 0:
            raise ValueError("bucket_bytes must be positive")
        if self.warmup < 0:
            raise ValueError("warmup must not be negative")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    @property
    def world_size(self) -> int:
        return self.source_partitions + self.destination_fanout


@dataclass(frozen=True)
class PlanEntry:
    """One canonical tensor routed through a source-partition lane."""

    tensor: TensorSpec
    partition_id: int
    group_id: int
    group_key: str


def _smoke_manifest() -> tuple[TensorSpec, ...]:
    return (
        TensorSpec("model.embed_tokens.weight", (2048, 1024), None),
        TensorSpec("model.layers.0.input_layernorm.weight", (1024,), 0),
        TensorSpec("model.layers.0.self_attn.qkv_proj.weight", (1280, 1024), 0),
        TensorSpec("model.layers.0.self_attn.o_proj.weight", (1024, 1024), 0),
        TensorSpec("model.layers.0.mlp.gate_proj.weight", (4096, 1024), 0),
        TensorSpec("model.layers.1.self_attn.qkv_proj.weight", (1280, 1024), 1),
        TensorSpec("model.layers.1.mlp.down_proj.weight", (1024, 4096), 1),
        TensorSpec("lm_head.weight", (2048, 1024), None),
    )


def _qwen_25_3b_manifest() -> tuple[TensorSpec, ...]:
    """A 290-tensor BF16 layout with the real 622,329,856-byte embedding."""
    manifest = [
        TensorSpec("model.embed_tokens.weight", (151_936, 2_048), None),
        TensorSpec("model.norm.weight", (2_048,), None),
    ]
    for layer in range(36):
        prefix = f"model.layers.{layer}"
        manifest.extend(
            (
                TensorSpec(f"{prefix}.input_layernorm.weight", (2_048,), layer),
                TensorSpec(
                    f"{prefix}.self_attn.qkv_proj.weight",
                    (2_560, 2_048),
                    layer,
                ),
                TensorSpec(
                    f"{prefix}.self_attn.o_proj.weight",
                    (2_048, 2_048),
                    layer,
                ),
                TensorSpec(
                    f"{prefix}.post_attention_layernorm.weight", (2_048,), layer
                ),
                TensorSpec(f"{prefix}.mlp.gate_proj.weight", (11_008, 2_048), layer),
                TensorSpec(f"{prefix}.mlp.up_proj.weight", (11_008, 2_048), layer),
                TensorSpec(f"{prefix}.mlp.down_proj.weight", (2_048, 11_008), layer),
                TensorSpec(f"{prefix}.self_attn.k_norm.weight", (128,), layer),
            )
        )
    return tuple(manifest)


def manifest_for(profile: str) -> tuple[TensorSpec, ...]:
    if profile == "smoke":
        return _smoke_manifest()
    if profile == "qwen-2.5-3b":
        return _qwen_25_3b_manifest()
    raise ValueError(f"unknown profile {profile!r}")


def manifest_fingerprint(manifest: Iterable[TensorSpec]) -> str:
    canonical = [
        {"name": item.name, "shape": item.shape, "layer": item.layer}
        for item in manifest
    ]
    return _fingerprint(canonical)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def case_id(case: Case, manifest: Iterable[TensorSpec]) -> str:
    payload = {"case": asdict(case), "manifest": manifest_fingerprint(manifest)}
    return f"m2n-{_fingerprint(payload)[:16]}"


def group_manifest(
    case: Case, manifest: tuple[TensorSpec, ...]
) -> list[list[TensorSpec]]:
    """Build the effective deterministic grouping for the requested schedule."""
    if case.schedule == "miles-current":
        return [[item] for item in manifest]
    return _synthetic_group_manifest(case, manifest)


def _synthetic_group_manifest(
    case: Case, manifest: tuple[TensorSpec, ...]
) -> list[list[TensorSpec]]:
    """Build deterministic singleton, layer, or bounded-byte synthetic groups."""
    if case.grouping == "singleton":
        return [[item] for item in manifest]
    if case.grouping == "layer":
        grouped: OrderedDict[str, list[TensorSpec]] = OrderedDict()
        for item in manifest:
            key = f"layer-{item.layer}" if item.layer is not None else item.name
            grouped.setdefault(key, []).append(item)
        return list(grouped.values())

    groups: list[list[TensorSpec]] = []
    current: list[TensorSpec] = []
    current_bytes = 0
    for item in manifest:
        if current and current_bytes + item.bytes > case.bucket_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item.bytes
    if current:
        groups.append(current)
    return groups


def schedule_metadata(case: Case) -> dict[str, str]:
    if case.schedule == "miles-current":
        return {
            "name": "miles-current",
            "classification": "shipping MILES schedule",
            "grouping": "singleton parameter groups",
            "sender_drain": "after all groups, in finish_weight_update",
            "receiver_drain": "after every singleton group, before loader use",
        }
    return {
        "name": "synthetic",
        "classification": "synthetic scheduling experiment",
        "grouping": case.grouping,
        "sender_drain": case.drain,
        "receiver_drain": case.drain,
    }


def build_entries(case: Case, manifest: tuple[TensorSpec, ...]) -> list[PlanEntry]:
    entries: list[PlanEntry] = []
    for group_id, group in enumerate(group_manifest(case, manifest)):
        for item in group:
            key = item.name if case.group_keys == "unique" else f"group-{group_id}"
            entries.append(
                PlanEntry(
                    tensor=item,
                    partition_id=len(entries) % case.source_partitions,
                    group_id=group_id,
                    group_key=key,
                )
            )
    return entries


def native_fusion_supported() -> bool:
    """The current production binding exposes no native group-key argument."""
    try:
        from modelexpress_rl.collective import backend
    except ImportError:
        return False
    return "group_key" in backend._reshard.__code__.co_varnames


def _token(entry: PlanEntry, round_index: int) -> str:
    return _fingerprint(
        {
            "name": entry.tensor.name,
            "bytes": entry.tensor.bytes,
            "partition": entry.partition_id,
            "round": round_index,
        }
    )


def expected_global_drains(entries: list[PlanEntry], drain: str) -> int:
    """Logical lane synchronizations for one full all-destination schedule."""
    if drain == "per-tensor":
        return len(entries)
    by_group: OrderedDict[int, set[int]] = OrderedDict()
    for entry in entries:
        by_group.setdefault(entry.group_id, set()).add(entry.partition_id)
    if drain == "per-group":
        return sum(len(partitions) for partitions in by_group.values())
    return len({entry.partition_id for entry in entries})


def expected_role_drains(entries: list[PlanEntry], case: Case) -> dict[str, int]:
    """Expected per-participant drain count for the selected schedule."""
    if case.schedule == "miles-current":
        return {
            "sender_per_rank": int(bool(entries)),
            "receiver_per_rank": len(entries),
        }
    entries_by_partition = [
        sum(entry.partition_id == partition for entry in entries)
        for partition in range(case.source_partitions)
    ]
    groups_by_partition = [
        len({entry.group_id for entry in entries if entry.partition_id == partition})
        for partition in range(case.source_partitions)
    ]
    return {
        "sender_per_rank": (
            max(entries_by_partition, default=0)
            if case.drain == "per-tensor"
            else max(groups_by_partition, default=0)
            if case.drain == "per-group"
            else int(bool(entries))
        ),
        "receiver_per_rank": expected_global_drains(entries, case.drain),
    }


def run_mock(case: Case) -> dict[str, Any]:
    """Run the deterministic scheduler check without CUDA or NCCL."""
    case.validate()
    manifest = manifest_for(case.profile)
    entries = build_entries(case, manifest)
    rounds: list[dict[str, Any]] = []
    for round_index in range(case.warmup + case.repetitions):
        destinations: list[dict[str, str]] = [
            {} for _ in range(case.destination_fanout)
        ]
        started_ns = time.perf_counter_ns()
        for entry in entries:
            token = _token(entry, round_index)
            for destination in destinations:
                destination[entry.tensor.name] = token
        elapsed_ns = time.perf_counter_ns() - started_ns
        for destination in destinations:
            for entry in entries:
                if destination[entry.tensor.name] != _token(entry, round_index):
                    raise AssertionError(
                        f"mock equality failed for {entry.tensor.name}"
                    )
        role_drains = expected_role_drains(entries, case)
        if round_index >= case.warmup:
            rounds.append(
                _round_record(
                    round_index=round_index - case.warmup,
                    elapsed_ns=elapsed_ns,
                    logical_bytes=sum(item.bytes for item in manifest),
                    fanout=case.destination_fanout,
                    issue_count=len(entries),
                    drain_count=role_drains["receiver_per_rank"],
                    role_drain_counts=role_drains,
                    verified=True,
                    rank_stats=None,
                )
            )
    return _report(
        mode="mock",
        case=case,
        manifest=manifest,
        entries=entries,
        rounds=rounds,
        verification="deterministic logical token equality",
    )


def _round_record(
    *,
    round_index: int,
    elapsed_ns: int,
    logical_bytes: int,
    fanout: int,
    issue_count: int,
    drain_count: int,
    role_drain_counts: dict[str, int],
    verified: bool,
    rank_stats: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if elapsed_ns <= 0:
        raise ValueError("elapsed_ns must be positive")
    seconds = elapsed_ns / 1_000_000_000
    delivered_bytes = logical_bytes * fanout
    return {
        "round": round_index,
        "elapsed_ns": elapsed_ns,
        "logical_bytes": logical_bytes,
        "delivered_bytes": delivered_bytes,
        "logical_throughput_gib_s": logical_bytes / seconds / (1024**3),
        "delivered_effective_bandwidth_gib_s": delivered_bytes / seconds / (1024**3),
        "issue_count": issue_count,
        "drain_count": drain_count,
        "role_drain_counts": role_drain_counts,
        "verified_exact": verified,
        "rank_stats": rank_stats,
    }


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _module_provenance(module_name: str, distribution: str | None) -> dict[str, Any]:
    """Describe the module actually imported by the benchmark process."""
    try:
        module = importlib.import_module(module_name)
        module_file = Path(inspect.getfile(module)).resolve()
    except (ImportError, OSError, TypeError) as error:
        return {
            "module": module_name,
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }

    checkout: Path | None = None
    for candidate in (module_file.parent, *module_file.parents):
        if (candidate / ".git").exists():
            checkout = candidate
            break
    package_version: str | None = None
    if distribution is not None:
        try:
            package_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "module": module_name,
        "available": True,
        "module_file": str(module_file),
        "distribution": distribution,
        "package_version": package_version,
        "checkout": str(checkout) if checkout is not None else None,
        "revision": _git_revision(checkout) if checkout is not None else None,
    }


def _environment_metadata() -> dict[str, str]:
    keep = (
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "MX_NCCL_REFIT_NUM_STREAMS",
        "MX_NCCL_REFIT_TRANSFER_TIMEOUT_S",
        "NCCL_CUMEM_ENABLE",
        "NCCL_DEBUG",
        "NCCL_NVLS_ENABLE",
    )
    return {key: os.environ[key] for key in keep if key in os.environ}


def _vram_envelope(case: Case, entries: list[PlanEntry]) -> dict[str, Any]:
    """Minimum tensor-buffer capacity required before allocator overhead."""
    source_bytes = [0] * case.source_partitions
    for entry in entries:
        source_bytes[entry.partition_id] += entry.tensor.bytes
    destination_bytes = sum(entry.tensor.bytes for entry in entries)
    per_rank_required = {
        **{
            f"source-{partition}": source_bytes[partition]
            for partition in range(case.source_partitions)
        },
        **{
            f"destination-{destination}": destination_bytes
            for destination in range(case.destination_fanout)
        },
    }
    largest_required = max(per_rank_required.values(), default=0)
    return {
        "source_buffer_bytes_by_partition": source_bytes,
        "destination_buffer_bytes_per_rank": destination_bytes,
        "verification_temporary_bytes": 0,
        "largest_required_buffer_bytes": largest_required,
        "minimum_free_bytes_with_headroom": int(
            largest_required * VRAM_HEADROOM_FRACTION
        ),
        "headroom_fraction": VRAM_HEADROOM_FRACTION,
    }


def _report(
    *,
    mode: str,
    case: Case,
    manifest: tuple[TensorSpec, ...],
    entries: list[PlanEntry],
    rounds: list[dict[str, Any]],
    verification: str,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_bytes = sum(item.bytes for item in manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "case_id": case_id(case, manifest),
        "case": asdict(case),
        "manifest": {
            "profile": case.profile,
            "source": (
                "synthetic representative layout"
                if case.profile == "qwen-2.5-3b"
                else "synthetic smoke layout"
            ),
            "matches_production_plan": False,
            "fingerprint": manifest_fingerprint(manifest),
            "tensor_count": len(manifest),
            "logical_bf16_bytes": total_bytes,
            "largest_tensors": [
                {"name": item.name, "shape": item.shape, "bytes": item.bytes}
                for item in sorted(manifest, key=lambda item: item.bytes, reverse=True)[
                    :10
                ]
            ],
        },
        "plan": {
            "fingerprint": _fingerprint(
                [
                    {
                        "name": entry.tensor.name,
                        "partition_id": entry.partition_id,
                        "group_id": entry.group_id,
                        "group_key": entry.group_key,
                    }
                    for entry in entries
                ]
            ),
            "group_count": len({entry.group_id for entry in entries}),
            "native_fusion_supported": native_fusion_supported(),
            "group_key_behavior": (
                "native group-key forwarding"
                if native_fusion_supported()
                else "recorded only; current production _reshard does not forward it"
            ),
        },
        "schedule": schedule_metadata(case),
        "vram_envelope": _vram_envelope(case, entries),
        "verification": verification,
        "immutable_metadata": {
            "imports": {
                "modelexpress_collective": _module_provenance(
                    "modelexpress_rl.collective.backend", "modelexpress"
                ),
                "miles": _module_provenance("miles", None),
                "megatron_bridge": _module_provenance("megatron.bridge", None),
            },
            "python": sys.version,
            "platform": platform.platform(),
            "environment": _environment_metadata(),
            "runtime": runtime or {},
        },
        "rounds": rounds,
    }


def _native_entry(entry: PlanEntry, case: Case):
    from modelexpress_rl.collective import MeshSpec, ParamPlan, Placement

    return ParamPlan(
        name=entry.tensor.name,
        global_shape=entry.tensor.shape,
        dtype="bfloat16",
        partition_id=entry.partition_id,
        src_mesh=MeshSpec((1,), rank_offset=0),
        src_placements=(Placement.replicate(),),
        dst_mesh=MeshSpec(
            (case.destination_fanout,),
            rank_offset=1,
        ),
        dst_placements=(Placement.replicate(),),
        group_key=entry.group_key,
    )


def _require_gpu_environment(
    reject_forced_communicator_id: Callable[[], None] | None = None,
) -> None:
    """Refuse configuration known to make M2N bootstrap invalid or hang."""
    if os.environ.get("NCCL_CUMEM_ENABLE") != "1":
        raise RuntimeError(
            "MILES M2N requires NCCL_CUMEM_ENABLE=1 for symmetric-memory "
            "device communicators"
        )
    if reject_forced_communicator_id is not None:
        reject_forced_communicator_id()
    elif os.environ.get("NCCL_COMM_ID"):
        raise RuntimeError(
            "NCCL_COMM_ID is incompatible with MX-brokered communicator bootstrap; "
            "unset NCCL_COMM_ID on every worker before initializing NCCL"
        )


def _native_requirements(case: Case, rank: int, local_rank: int) -> tuple[Any, Any]:
    import torch

    from modelexpress_rl.collective.backend import MIN_NCCL, loaded_nccl_version
    from modelexpress_rl.collective.backend import require_nccl_m2n
    from modelexpress_rl.collective.comm import _reject_forced_communicator_id

    _require_gpu_environment(_reject_forced_communicator_id)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU mode requires an available CUDA runtime")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"rank {rank} maps to local CUDA device {local_rank}, but this host has "
            f"{torch.cuda.device_count()} visible device(s)"
        )
    require_nccl_m2n()
    version = loaded_nccl_version()
    if version is not None and version < MIN_NCCL:
        raise RuntimeError(
            f"loaded NCCL {version} is older than the required {MIN_NCCL}"
        )
    torch.cuda.set_device(local_rank)
    return torch, torch.device(f"cuda:{local_rank}")


def _format_nccl_version(version: tuple[int, int, int] | None) -> str | None:
    return ".".join(str(part) for part in version) if version is not None else None


def _nvidia_smi_query(local_rank: int) -> dict[str, str | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                str(local_rank),
                "--query-gpu=uuid,pci.bus_id,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"uuid": None, "pci_bus_id": None, "driver_version": None}
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "uuid": rows[0].split(", ")[0] if rows else None,
        "pci_bus_id": rows[0].split(", ")[1] if rows and ", " in rows[0] else None,
        "driver_version": (
            rows[0].split(", ")[2] if rows and rows[0].count(", ") >= 2 else None
        ),
    }


def _nvidia_topology() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _rank_runtime_metadata(
    *,
    torch: Any,
    device: Any,
    rank: int,
    local_rank: int,
    envelope: dict[str, Any],
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    nvidia_smi = _nvidia_smi_query(local_rank)
    required_buffer_bytes = (
        envelope["source_buffer_bytes_by_partition"][rank]
        if rank < len(envelope["source_buffer_bytes_by_partition"])
        else envelope["destination_buffer_bytes_per_rank"]
    )
    return {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "device": str(device),
        "gpu_name": properties.name,
        "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_free_memory_bytes_preflight": free_bytes,
        "gpu_uuid": nvidia_smi["uuid"],
        "gpu_pci_bus_id": nvidia_smi["pci_bus_id"],
        "cuda_driver_version": nvidia_smi["driver_version"],
        "required_buffer_bytes": required_buffer_bytes,
        "required_free_bytes_with_headroom": int(
            required_buffer_bytes * envelope["headroom_fraction"]
        ),
    }


def _preflight_vram(rank_metadata: dict[str, Any]) -> None:
    if (
        rank_metadata["gpu_free_memory_bytes_preflight"]
        < rank_metadata["required_free_bytes_with_headroom"]
    ):
        raise RuntimeError(
            "insufficient free CUDA memory for microbench buffers: "
            f"free={rank_metadata['gpu_free_memory_bytes_preflight']} required="
            f"{rank_metadata['required_free_bytes_with_headroom']} "
            f"(including {VRAM_HEADROOM_FRACTION:.2f}x headroom)"
        )


def _broadcast_unique_id(dist: Any, rank: int, partition: int) -> bytes:
    packet: list[bytes | None] = [None]
    if rank == partition:
        from nccl.core import utils

        unique_id = utils.get_unique_id()
        raw = getattr(unique_id, "as_bytes", None)
        packet[0] = bytes(raw() if callable(raw) else raw)
    dist.broadcast_object_list(packet, src=partition)
    if packet[0] is None:
        raise RuntimeError(f"source partition {partition} did not publish an NCCL id")
    return packet[0]


def _create_native_lanes(
    *,
    case: Case,
    dist: Any,
    rank: int,
    device: Any,
    streams: list[Any],
    group_id: str,
) -> tuple[Any, dict[int, Any]]:
    from modelexpress_rl.collective.comm import CommunicatorCache, LaneKey

    cache = CommunicatorCache()
    lanes: dict[int, Any] = {}
    for partition in range(case.source_partitions):
        unique_id = _broadcast_unique_id(dist, rank, partition)
        participates = rank == partition or rank >= case.source_partitions
        if participates:
            lane_rank = 0 if rank == partition else 1 + rank - case.source_partitions
            key = LaneKey(group_id=group_id, epoch=1, lane_id=partition)
            lanes[partition] = cache.create(
                key,
                rank=lane_rank,
                world_size=1 + case.destination_fanout,
                unique_id=unique_id,
                device=device,
                stream=streams[partition % len(streams)],
                timeout_s=case.timeout_s,
            )
        dist.barrier()
    return cache, lanes


def _round_byte(entry_index: int, round_index: int) -> int:
    return (entry_index * 17 + round_index * 31) % 251


def _prepare_native_buffers(
    *,
    torch: Any,
    device: Any,
    case: Case,
    entries: list[PlanEntry],
    rank: int,
) -> dict[str, Any]:
    buffers: dict[str, Any] = {}
    for entry in entries:
        owns_source = rank == entry.partition_id
        is_destination = rank >= case.source_partitions
        if owns_source or is_destination:
            buffers[entry.tensor.name] = torch.empty(
                entry.tensor.shape,
                dtype=torch.bfloat16,
                device=device,
            )
    return buffers


def _fill_round_buffers(
    *,
    torch: Any,
    buffers: dict[str, Any],
    entries: list[PlanEntry],
    rank: int,
    case: Case,
    round_index: int,
) -> None:
    for entry_index, entry in enumerate(entries):
        if rank == entry.partition_id:
            buffers[entry.tensor.name].view(torch.uint8).fill_(
                _round_byte(entry_index, round_index)
            )
        elif rank >= case.source_partitions:
            buffers[entry.tensor.name].view(torch.uint8).fill_(0xFF)


def _verify_native_buffers(
    *,
    torch: Any,
    buffers: dict[str, Any],
    entries: list[PlanEntry],
    rank: int,
    case: Case,
    round_index: int,
) -> bool:
    if rank < case.source_partitions:
        return True
    for entry_index, entry in enumerate(entries):
        raw = buffers[entry.tensor.name].view(torch.uint8)
        minimum, maximum = torch.aminmax(raw)
        expected = _round_byte(entry_index, round_index)
        if int(minimum.item()) != expected or int(maximum.item()) != expected:
            return False
    return True


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("the full microbench round exceeded --timeout-s")
    return remaining


def _native_round(
    *,
    torch: Any,
    dist: Any,
    rank: int,
    case: Case,
    entries: list[PlanEntry],
    native_entries: list[Any],
    lanes: dict[int, Any],
    buffers: dict[str, Any],
    round_index: int,
) -> tuple[int, int, int, dict[str, int], bool, list[dict[str, Any]]]:
    """Issue one full schedule and return max-rank timing plus proof details."""
    _fill_round_buffers(
        torch=torch,
        buffers=buffers,
        entries=entries,
        rank=rank,
        case=case,
        round_index=round_index,
    )
    torch.cuda.synchronize()
    dist.barrier()
    started_ns = time.perf_counter_ns()
    deadline = time.monotonic() + case.timeout_s
    issued_lanes: OrderedDict[int, Any] = OrderedDict()
    active_group_id: int | None = None
    active_group_lanes: OrderedDict[int, Any] = OrderedDict()
    issue_count = 0
    drain_count = 0
    from modelexpress_rl.collective import backend

    for entry, native_entry in zip(entries, native_entries, strict=True):
        if (
            case.schedule == "synthetic"
            and case.drain == "per-group"
            and active_group_id is not None
            and entry.group_id != active_group_id
        ):
            for lane in active_group_lanes.values():
                lane.synchronize(timeout_s=_remaining_timeout(deadline))
                drain_count += 1
            active_group_lanes.clear()
        owns_source = rank == entry.partition_id
        is_destination = rank >= case.source_partitions
        if not owns_source and not is_destination:
            continue
        lane = lanes[entry.partition_id]
        backend._reshard(
            comm=lane,
            entry=native_entry,
            src=buffers[entry.tensor.name] if owns_source else None,
            dst=buffers[entry.tensor.name] if is_destination else None,
        )
        issue_count += 1
        issued_lanes.setdefault(entry.partition_id, lane)
        active_group_id = entry.group_id
        active_group_lanes[entry.partition_id] = lane
        if case.schedule == "miles-current" and is_destination:
            lane.synchronize(timeout_s=_remaining_timeout(deadline))
            drain_count += 1
        elif case.schedule == "synthetic" and case.drain == "per-tensor":
            lane.synchronize(timeout_s=_remaining_timeout(deadline))
            drain_count += 1
    if case.schedule == "synthetic" and case.drain == "per-group":
        for lane in active_group_lanes.values():
            lane.synchronize(timeout_s=_remaining_timeout(deadline))
            drain_count += 1
    if case.schedule == "miles-current" and rank < case.source_partitions:
        for lane in issued_lanes.values():
            lane.synchronize(timeout_s=_remaining_timeout(deadline))
            drain_count += 1
    elif case.schedule == "synthetic" and case.drain == "end":
        for lane in issued_lanes.values():
            lane.synchronize(timeout_s=_remaining_timeout(deadline))
            drain_count += 1
    elapsed_ns = time.perf_counter_ns() - started_ns
    verified = _verify_native_buffers(
        torch=torch,
        buffers=buffers,
        entries=entries,
        rank=rank,
        case=case,
        round_index=round_index,
    )
    proof = torch.tensor(int(verified), device="cpu", dtype=torch.int32)
    dist.all_reduce(proof)
    all_verified = int(proof.item()) == case.world_size
    local = {
        "rank": rank,
        "elapsed_ns": elapsed_ns,
        "issue_count": issue_count,
        "drain_count": drain_count,
        "verified_exact": verified,
    }
    gathered: list[dict[str, Any] | None] = [None] * case.world_size
    dist.all_gather_object(gathered, local)
    rank_stats = [item for item in gathered if item is not None]
    sender_drains = [
        item["drain_count"]
        for item in rank_stats
        if item["rank"] < case.source_partitions
    ]
    receiver_drains = [
        item["drain_count"]
        for item in rank_stats
        if item["rank"] >= case.source_partitions
    ]
    role_drain_counts = {
        "sender_per_rank": max(sender_drains, default=0),
        "receiver_per_rank": max(receiver_drains, default=0),
    }
    return (
        max(item["elapsed_ns"] for item in rank_stats),
        max(item["issue_count"] for item in rank_stats),
        max(item["drain_count"] for item in rank_stats),
        role_drain_counts,
        all_verified,
        rank_stats,
    )


def _run_identity(dist: Any, rank: int, supplied_run_id: str | None) -> dict[str, str]:
    identity: list[dict[str, str] | None] = [None]
    if rank == 0:
        identity[0] = {
            "run_id": supplied_run_id or uuid.uuid4().hex,
            "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    dist.broadcast_object_list(identity, src=0)
    if identity[0] is None:
        raise RuntimeError("rank 0 did not publish the microbench run identity")
    return identity[0]


def _abort_group(cache: Any | None, group_id: str | None) -> int:
    """Abort all peer lanes before Gloo teardown after any failed native run."""
    if cache is None or group_id is None:
        return 0
    return cache.abort_group(group_id)


def run_gpu(
    case: Case,
    *,
    allow_large_profile: bool,
    output: Path | None,
    run_id: str | None,
) -> int:
    """Run one native cohort. Must be called through torchrun."""
    case.validate()
    manifest = manifest_for(case.profile)
    if case.profile == "qwen-2.5-3b" and not allow_large_profile:
        raise ValueError(
            "qwen-2.5-3b allocates multi-gigabyte BF16 buffers; pass "
            "--allow-large-profile to run it"
        )
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != case.world_size:
        raise ValueError(
            f"GPU case needs WORLD_SIZE={case.world_size}, got {world_size}; launch "
            "one source participant per partition plus one participant per destination"
        )
    entries = build_entries(case, manifest)
    envelope = _vram_envelope(case, entries)
    from modelexpress_rl.collective.comm import _reject_forced_communicator_id

    _require_gpu_environment(_reject_forced_communicator_id)
    import torch.distributed as dist

    cache: Any | None = None
    group_id: str | None = None
    initialized = False
    dist.init_process_group(
        backend="gloo",
        timeout=datetime.timedelta(seconds=case.timeout_s + 30),
    )
    initialized = True
    try:
        torch, device = _native_requirements(case, rank, local_rank)
        rank_runtime = _rank_runtime_metadata(
            torch=torch,
            device=device,
            rank=rank,
            local_rank=local_rank,
            envelope=envelope,
        )
        _preflight_vram(rank_runtime)
        torch.cuda.reset_peak_memory_stats(device)
        streams = [torch.cuda.Stream(device=device) for _ in range(case.streams)]
        native_entries = [_native_entry(entry, case) for entry in entries]
        run_identity = _run_identity(dist, rank, run_id)
        group_id = f"{case_id(case, manifest)}-{run_identity['run_id']}"
        cache, lanes = _create_native_lanes(
            case=case,
            dist=dist,
            rank=rank,
            device=device,
            streams=streams,
            group_id=group_id,
        )
        buffers = _prepare_native_buffers(
            torch=torch,
            device=device,
            case=case,
            entries=entries,
            rank=rank,
        )
        for warmup_round in range(case.warmup):
            (
                _elapsed,
                _issued,
                _drains,
                _role_drains,
                verified,
                _stats,
            ) = _native_round(
                torch=torch,
                dist=dist,
                rank=rank,
                case=case,
                entries=entries,
                native_entries=native_entries,
                lanes=lanes,
                buffers=buffers,
                round_index=warmup_round,
            )
            if not verified:
                raise RuntimeError(f"warmup round {warmup_round} failed exact equality")
        rounds: list[dict[str, Any]] = []
        logical_bytes = sum(item.bytes for item in manifest)
        for measured_round in range(case.repetitions):
            round_index = case.warmup + measured_round
            (
                elapsed_ns,
                issue_count,
                drain_count,
                role_drain_counts,
                verified,
                rank_stats,
            ) = _native_round(
                torch=torch,
                dist=dist,
                rank=rank,
                case=case,
                entries=entries,
                native_entries=native_entries,
                lanes=lanes,
                buffers=buffers,
                round_index=round_index,
            )
            if not verified:
                raise RuntimeError(
                    f"measured round {measured_round} failed exact destination equality"
                )
            if rank == 0:
                rounds.append(
                    _round_record(
                        round_index=measured_round,
                        elapsed_ns=elapsed_ns,
                        logical_bytes=logical_bytes,
                        fanout=case.destination_fanout,
                        issue_count=issue_count,
                        drain_count=drain_count,
                        role_drain_counts=role_drain_counts,
                        verified=True,
                        rank_stats=rank_stats,
                    )
                )
        rank_runtime["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        rank_runtime["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        all_rank_runtime: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(all_rank_runtime, rank_runtime)
        if rank == 0:
            from modelexpress_rl.collective.backend import loaded_nccl_version

            runtime = {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "loaded_nccl_version": _format_nccl_version(loaded_nccl_version()),
                "world_size": world_size,
                "host": socket.gethostname(),
                "nvidia_smi_topology": _nvidia_topology(),
                "run_identity": run_identity,
                "rank_mapping": [item for item in all_rank_runtime if item is not None],
            }
            report = _report(
                mode="gpu",
                case=case,
                manifest=manifest,
                entries=entries,
                rounds=rounds,
                verification=(
                    "uint8 amin/max equality against a deterministic uniform byte "
                    "value for every destination tensor after every round"
                ),
                runtime=runtime,
            )
            _write_json(output, report)
        return 0
    except BaseException:
        _abort_group(cache, group_id)
        raise
    finally:
        if initialized:
            dist.destroy_process_group()


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(rendered)
    temporary.replace(path)


def _parse_case(args: argparse.Namespace) -> Case:
    return Case(
        profile=args.profile,
        source_partitions=args.source_partitions,
        destination_fanout=args.destination_fanout,
        streams=args.streams,
        schedule=args.schedule,
        grouping=args.grouping,
        bucket_bytes=args.bucket_bytes,
        drain=args.drain,
        group_keys=args.group_keys,
        warmup=args.warmup,
        repetitions=args.repetitions,
        timeout_s=args.timeout_s,
    )


def _parse_int_set(raw: str, choices: tuple[int, ...], label: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item]
    if not values or any(item not in choices for item in values):
        raise ValueError(f"{label} must be a non-empty comma list from {choices}")
    return values


def _matrix_cases(args: argparse.Namespace) -> Iterable[Case]:
    dimensions = (
        _parse_int_set(args.matrix_partitions, SUPPORTED_PARTITIONS, "partitions"),
        _parse_int_set(args.matrix_fanout, SUPPORTED_FANOUT, "fanout"),
        _parse_int_set(args.matrix_streams, SUPPORTED_STREAMS, "streams"),
        args.matrix_schedule.split(","),
        args.matrix_grouping.split(","),
        args.matrix_drain.split(","),
        args.matrix_group_keys.split(","),
    )
    for (
        partition,
        fanout,
        streams,
        schedule,
        grouping,
        drain,
        group_keys,
    ) in product(*dimensions):
        case = Case(
            profile=args.profile,
            source_partitions=partition,
            destination_fanout=fanout,
            streams=streams,
            schedule=schedule,
            grouping=grouping,
            bucket_bytes=args.bucket_bytes,
            drain=drain,
            group_keys=group_keys,
            warmup=args.warmup,
            repetitions=args.repetitions,
            timeout_s=args.timeout_s,
        )
        case.validate()
        yield case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "gpu"), required=True)
    parser.add_argument("--profile", choices=("smoke", "qwen-2.5-3b"), default="smoke")
    parser.add_argument("--source-partitions", type=int, default=1)
    parser.add_argument("--destination-fanout", type=int, default=1)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument(
        "--schedule",
        choices=SUPPORTED_SCHEDULES,
        default="miles-current",
        help=(
            "miles-current reproduces MILES singleton groups with role-specific "
            "drains; synthetic enables grouping/drain experiments"
        ),
    )
    parser.add_argument("--grouping", choices=SUPPORTED_GROUPING, default="singleton")
    parser.add_argument("--bucket-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--drain", choices=SUPPORTED_DRAIN, default="end")
    parser.add_argument("--group-keys", choices=SUPPORTED_GROUP_KEYS, default="unique")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--allow-large-profile", action="store_true")
    parser.add_argument(
        "--run-id",
        help="Optional immutable GPU run identity; a UUID is generated when omitted.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run a CPU mock matrix. GPU matrix execution belongs in run_microbench.sh.",
    )
    parser.add_argument("--matrix-partitions", default="1,2,4")
    parser.add_argument("--matrix-fanout", default="1,2,4,8")
    parser.add_argument("--matrix-streams", default="1,2,4")
    parser.add_argument("--matrix-schedule", default="miles-current,synthetic")
    parser.add_argument("--matrix-grouping", default="singleton,layer,bucket")
    parser.add_argument("--matrix-drain", default="per-tensor,per-group,end")
    parser.add_argument("--matrix-group-keys", default="unique,shared")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "gpu" and args.matrix:
        raise ValueError("GPU matrix execution must use run_microbench.sh")
    if args.mode == "mock":
        if args.matrix:
            reports = [run_mock(case) for case in _matrix_cases(args)]
            _write_json(
                args.json_out, {"schema_version": SCHEMA_VERSION, "cases": reports}
            )
        else:
            _write_json(args.json_out, run_mock(_parse_case(args)))
        return 0
    return run_gpu(
        _parse_case(args),
        allow_large_profile=args.allow_large_profile,
        output=args.json_out,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - CLI must fail closed with context.
        print(
            f"m2n microbench failed: {type(error).__name__}: {error}", file=sys.stderr
        )
        raise SystemExit(2)
