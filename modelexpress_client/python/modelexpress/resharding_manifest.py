# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest extraction helpers for cross-parallelism refit planning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Mapping
import urllib.request

from .resharding import (
    QuantizationMetadataError,
    QuantizationScope,
    SliceOwnership,
    SliceRequest,
    classify_tensor_family,
    plan_segments,
)

DEFAULT_REMOTE_HEADER_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class TensorManifestEntry:
    """Classified tensor metadata for resharding/refit planning."""

    model_name: str
    model_version: str
    tensor_name: str
    global_shape: tuple[int, ...]
    dtype: str
    tensor_family: str
    quantization_scope: QuantizationScope | str = QuantizationScope.ABSENT
    layout_tags: dict[str, str | int | bool] = field(default_factory=dict)
    requires_special_handling: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "global_shape",
            tuple(int(dim) for dim in self.global_shape),
        )
        object.__setattr__(
            self,
            "quantization_scope",
            QuantizationScope(self.quantization_scope),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "tensor_name": self.tensor_name,
            "global_shape": list(self.global_shape),
            "dtype": self.dtype,
            "tensor_family": self.tensor_family,
            "quantization_scope": self.quantization_scope.value,
            "layout_tags": dict(self.layout_tags),
            "requires_special_handling": self.requires_special_handling,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorManifestEntry":
        return cls(
            model_name=str(data["model_name"]),
            model_version=str(data["model_version"]),
            tensor_name=str(data["tensor_name"]),
            global_shape=tuple(int(dim) for dim in data["global_shape"]),
            dtype=str(data["dtype"]),
            tensor_family=str(data["tensor_family"]),
            quantization_scope=data.get(
                "quantization_scope",
                QuantizationScope.ABSENT,
            ),
            layout_tags=dict(data.get("layout_tags", {})),
            requires_special_handling=bool(
                data.get("requires_special_handling", False)
            ),
            reason=str(data.get("reason", "")),
        )


def extract_qwen_moe_manifest(
    tensors: Mapping[str, Any] | Iterable[tuple[str, Any]],
    *,
    model_name: str,
    model_version: str,
    default_dtype: str = "unknown",
) -> list[TensorManifestEntry]:
    """Classify Qwen-style MoE tensors for resharding planning.

    ``tensors`` may be a ``state_dict``-like mapping, ``named_parameters`` style
    iterable, or a mapping to small metadata dicts containing ``shape`` and
    ``dtype``. The extractor intentionally avoids touching tensor data.
    """

    items = tensors.items() if isinstance(tensors, Mapping) else tensors
    manifest = [
        classify_qwen_moe_tensor(
            name,
            value,
            model_name=model_name,
            model_version=model_version,
            default_dtype=default_dtype,
        )
        for name, value in items
    ]
    return sorted(manifest, key=lambda entry: entry.tensor_name)


def extract_qwen_moe_manifest_from_safetensors_metadata(
    metadata_by_tensor: Mapping[str, Mapping[str, Any]],
    *,
    model_name: str,
    model_version: str,
    default_dtype: str = "unknown",
) -> list[TensorManifestEntry]:
    """Classify tensors from safetensors header metadata.

    Safetensors headers include tensor ``shape`` and ``dtype`` without requiring
    payload reads, which makes this suitable for very large Qwen MoE checkpoints.
    """

    tensors = safetensors_header_to_tensor_metadata(
        metadata_by_tensor,
        default_dtype=default_dtype,
    )
    return extract_qwen_moe_manifest(
        tensors,
        model_name=model_name,
        model_version=model_version,
        default_dtype=default_dtype,
    )


def safetensors_header_to_tensor_metadata(
    header: Mapping[str, Mapping[str, Any]],
    *,
    default_dtype: str = "unknown",
    source_file: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Convert safetensors header entries to classifier input metadata."""

    tensors: dict[str, dict[str, Any]] = {}
    for name, tensor_meta in header.items():
        if name.startswith("__"):
            continue
        if not isinstance(tensor_meta, Mapping):
            raise ValueError(f"safetensors metadata for {name} must be a mapping")
        if "shape" not in tensor_meta:
            raise ValueError(f"safetensors metadata for {name} is missing shape")

        entry = {
            "shape": tuple(int(dim) for dim in tensor_meta["shape"]),
            "dtype": str(tensor_meta.get("dtype", default_dtype)),
        }
        effective_source_file = source_file
        if effective_source_file is None and tensor_meta.get("source_file") is not None:
            effective_source_file = str(tensor_meta["source_file"])
        if effective_source_file is not None:
            entry["source_file"] = effective_source_file
        tensors[name] = entry
    return tensors


def read_safetensors_header(path: Path | str) -> dict[str, Any]:
    """Read a local safetensors header without loading tensor payloads."""

    path = Path(path)
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        (header_size,) = struct.unpack("<Q", header_size_bytes)
        header = handle.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"{path} ended before safetensors header was complete")
    return json.loads(header.decode("utf-8"))


def extract_qwen_moe_manifest_from_safetensors_files(
    paths: Iterable[Path],
    *,
    model_name: str,
    model_version: str,
    default_dtype: str = "unknown",
) -> list[TensorManifestEntry]:
    """Classify Qwen-style tensors from one or more local safetensors files."""

    merged: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        header = read_safetensors_header(path)
        merged.update(
            safetensors_header_to_tensor_metadata(
                header,
                source_file=str(path),
                default_dtype=default_dtype,
            )
        )
    return extract_qwen_moe_manifest_from_safetensors_metadata(
        merged,
        model_name=model_name,
        model_version=model_version,
        default_dtype=default_dtype,
    )


def read_safetensors_header_from_url(
    url: str,
    *,
    token: str | None = None,
    timeout_seconds: float = DEFAULT_REMOTE_HEADER_TIMEOUT_SECONDS,
    max_header_bytes: int = DEFAULT_MAX_SAFETENSORS_HEADER_BYTES,
) -> dict[str, Any]:
    """Read a remote safetensors header using HTTP range requests only."""

    header_size_bytes = _read_url_range(
        url,
        start=0,
        end=7,
        token=token,
        timeout_seconds=timeout_seconds,
    )
    (header_size,) = struct.unpack("<Q", header_size_bytes)
    if header_size > max_header_bytes:
        raise ValueError(
            f"safetensors header is {header_size} bytes, above {max_header_bytes}"
        )

    header = _read_url_range(
        url,
        start=8,
        end=8 + header_size - 1,
        token=token,
        timeout_seconds=timeout_seconds,
    )
    return json.loads(header.decode("utf-8"))


def read_hf_safetensors_headers(
    repo_id: str,
    *,
    filenames: Iterable[str] | None = None,
    revision: str | None = None,
    token: str | None = None,
    timeout_seconds: float = DEFAULT_REMOTE_HEADER_TIMEOUT_SECONDS,
) -> dict[str, dict[str, Any]]:
    """Read safetensors headers from a Hugging Face repo without payloads."""

    from huggingface_hub import HfApi, hf_hub_url

    if filenames is None:
        api = HfApi(token=token)
        filenames = [
            filename
            for filename in api.list_repo_files(repo_id=repo_id, revision=revision)
            if filename.endswith(".safetensors")
        ]

    merged: dict[str, dict[str, Any]] = {}
    for filename in filenames:
        url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
        header = read_safetensors_header_from_url(
            url,
            token=token,
            timeout_seconds=timeout_seconds,
        )
        merged.update(
            safetensors_header_to_tensor_metadata(header, source_file=filename)
        )
    return merged


def extract_qwen_moe_manifest_from_hf_repo(
    repo_id: str,
    *,
    model_name: str,
    model_version: str,
    filenames: Iterable[str] | None = None,
    revision: str | None = None,
    token: str | None = None,
    default_dtype: str = "unknown",
    timeout_seconds: float = DEFAULT_REMOTE_HEADER_TIMEOUT_SECONDS,
) -> list[TensorManifestEntry]:
    """Classify Qwen-style MoE tensors from Hugging Face safetensors headers."""

    metadata = read_hf_safetensors_headers(
        repo_id,
        filenames=filenames,
        revision=revision,
        token=token,
        timeout_seconds=timeout_seconds,
    )
    return extract_qwen_moe_manifest_from_safetensors_metadata(
        metadata,
        model_name=model_name,
        model_version=model_version,
        default_dtype=default_dtype,
    )


def classify_qwen_moe_tensor(
    tensor_name: str,
    value: Any = None,
    *,
    model_name: str,
    model_version: str,
    default_dtype: str = "unknown",
    explicit_quantization_scope: QuantizationScope | str | None = None,
) -> TensorManifestEntry:
    """Classify one Qwen-style tensor by name, shape, and dtype metadata."""

    shape, dtype = _shape_and_dtype(value, default_dtype=default_dtype)
    layout_tags = _qwen_layout_tags(tensor_name, shape)
    if isinstance(value, Mapping) and value.get("source_file") is not None:
        layout_tags["source_file"] = str(value["source_file"])
    quantization_scope = _qwen_quantization_scope(
        tensor_name,
        explicit=explicit_quantization_scope,
    )
    tensor_family = classify_tensor_family(
        tensor_name,
        layout_tags=layout_tags,
        quantization_scope=quantization_scope,
    )
    requires_special_handling, reason = _special_handling(
        tensor_name,
        tensor_family,
        quantization_scope,
    )

    return TensorManifestEntry(
        model_name=model_name,
        model_version=model_version,
        tensor_name=tensor_name,
        global_shape=shape,
        dtype=dtype,
        tensor_family=tensor_family,
        quantization_scope=quantization_scope,
        layout_tags=layout_tags,
        requires_special_handling=requires_special_handling,
        reason=reason,
    )


def write_manifest_artifact(
    manifest: Iterable[TensorManifestEntry],
    artifact_path: Path,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [entry.to_dict() for entry in manifest]
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def manifest_coverage_summary(
    manifest: Iterable[TensorManifestEntry],
) -> dict[str, Any]:
    entries = list(manifest)
    return {
        "tensor_count": len(entries),
        "tensor_family_counts": _count_by(entries, "tensor_family"),
        "quantization_scope_counts": {
            scope: count
            for scope, count in _count_by(entries, "quantization_scope").items()
        },
        "requires_special_handling_count": sum(
            1 for entry in entries if entry.requires_special_handling
        ),
        "source_file_counts": _count_source_files(entries),
    }


def write_manifest_coverage_artifact(
    manifest: Iterable[TensorManifestEntry],
    artifact_path: Path,
    *,
    source: Mapping[str, Any] | None = None,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    entries = list(manifest)
    payload = {
        "source": dict(source or {}),
        "summary": manifest_coverage_summary(entries),
        "entries": [entry.to_dict() for entry in entries],
    }
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_global_required_zero_copy_fallback(
    manifest: Iterable[TensorManifestEntry],
    *,
    runtime_framework: str = "vllm",
    tensor_name: str | None = None,
) -> dict[str, Any]:
    """Verify real quantization metadata is rejected for zero-copy planning."""

    selected = _select_global_required_entry(manifest, tensor_name=tensor_name)
    tensor_range = tuple((0, int(dim)) for dim in selected.global_shape)
    ownership = SliceOwnership(
        model_name=selected.model_name,
        model_version=selected.model_version,
        tensor_name=selected.tensor_name,
        global_shape=selected.global_shape,
        dtype=selected.dtype,
        source_range=tensor_range,
        worker_id="qwen-fp8-source-worker",
        source_id="qwen-fp8-source",
        source_lease="qwen-fp8-lease",
        nixl_descriptor_id="qwen-fp8-desc",
        layout_tags=dict(selected.layout_tags),
        quantization_scope=selected.quantization_scope,
    )
    request = SliceRequest(
        tensor_name=selected.tensor_name,
        requested_range=tensor_range,
        target_shape=selected.global_shape,
        dtype=selected.dtype,
        target_id=f"{runtime_framework}:{selected.tensor_name}",
        model_name=selected.model_name,
        model_version=selected.model_version,
        runtime_framework=runtime_framework,
        layout_tags=dict(selected.layout_tags),
        quantization_scope=QuantizationScope.ABSENT,
    )

    try:
        plans = plan_segments([ownership], [request])
    except QuantizationMetadataError as exc:
        return {
            "passed": True,
            "fallback_required": True,
            "zero_copy_plan_created": False,
            "runtime_framework": runtime_framework,
            "selected_tensor": selected.to_dict(),
            "planner_error": str(exc),
        }

    return {
        "passed": False,
        "fallback_required": False,
        "zero_copy_plan_created": True,
        "runtime_framework": runtime_framework,
        "selected_tensor": selected.to_dict(),
        "segment_count": len(plans),
    }


def _shape_and_dtype(value: Any, *, default_dtype: str) -> tuple[tuple[int, ...], str]:
    if isinstance(value, Mapping):
        raw_shape = value.get("global_shape", value.get("shape", ()))
        dtype = str(value.get("dtype", default_dtype))
        return tuple(int(dim) for dim in raw_shape), _normalize_dtype(dtype)

    if isinstance(value, (tuple, list)) and all(isinstance(dim, int) for dim in value):
        return tuple(int(dim) for dim in value), _normalize_dtype(default_dtype)

    shape = getattr(value, "shape", ())
    dtype = getattr(value, "dtype", default_dtype)
    return tuple(int(dim) for dim in shape), _normalize_dtype(str(dtype))


def _normalize_dtype(dtype: str) -> str:
    normalized = dtype.replace("torch.", "").lower()
    if normalized == "f32":
        return "float32"
    if normalized == "f16":
        return "float16"
    if normalized == "bf16":
        return "bfloat16"
    return normalized


def _count_by(
    entries: Iterable[TensorManifestEntry],
    field_name: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = getattr(entry, field_name)
        if isinstance(value, QuantizationScope):
            key = value.value
        else:
            key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _select_global_required_entry(
    manifest: Iterable[TensorManifestEntry],
    *,
    tensor_name: str | None,
) -> TensorManifestEntry:
    entries = list(manifest)
    for entry in entries:
        if tensor_name is not None and entry.tensor_name != tensor_name:
            continue
        if entry.quantization_scope == QuantizationScope.GLOBAL_REQUIRED:
            return entry

    if tensor_name is not None:
        raise ValueError(f"{tensor_name!r} is not global-required quantization metadata")
    raise ValueError("manifest contains no global-required quantization metadata")


def _count_source_files(entries: Iterable[TensorManifestEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        source_file = entry.layout_tags.get("source_file")
        if source_file is None:
            continue
        key = str(source_file)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _qwen_layout_tags(
    tensor_name: str,
    shape: tuple[int, ...],
) -> dict[str, str | int | bool]:
    lowered = tensor_name.lower()
    tags: dict[str, str | int | bool] = {
        "model_family": "qwen-moe",
        "storage_layout": "row-major",
    }
    if ".mlp.experts." in lowered or ".experts." in lowered:
        expert_match = re.search(r"\.experts\.(\d+)\.", lowered)
        if expert_match:
            tags["expert_storage"] = "per-expert-tensor"
            tags["moe_expert_index"] = int(expert_match.group(1))
        else:
            tags["expert_storage"] = "stacked-expert-axis"
            tags["moe_expert_axis"] = 0
            if shape:
                tags["num_experts"] = shape[0]
        tags["tensor_role"] = _qwen_tensor_role(lowered)
    elif ".mlp.shared_expert." in lowered:
        tags["tensor_role"] = "shared-expert"
        tags["shared_expert"] = True
    elif "rotary_emb.inv_freq" in lowered:
        tags["tensor_role"] = "rotary-inv-freq"
    elif ".norm." in lowered or lowered.endswith(".norm.weight"):
        tags["tensor_role"] = "normalization"
    elif "embed_tokens" in lowered:
        tags["tensor_role"] = "embedding"
    elif lowered.endswith("lm_head.weight"):
        tags["tensor_role"] = "lm-head"
    else:
        tags["tensor_role"] = "dense-or-unknown"
    return tags


def _qwen_tensor_role(lowered_name: str) -> str:
    if "weight_scale" in lowered_name or "scale_inv" in lowered_name:
        return "expert-quant-metadata"
    if ".w1." in lowered_name or "gate_proj" in lowered_name:
        return "expert-gate"
    if ".w2." in lowered_name or "down_proj" in lowered_name:
        return "expert-down"
    if ".w3." in lowered_name or "up_proj" in lowered_name:
        return "expert-up"
    return "expert-unknown"


def _qwen_quantization_scope(
    tensor_name: str,
    *,
    explicit: QuantizationScope | str | None,
) -> QuantizationScope:
    if explicit is not None:
        return QuantizationScope(explicit)

    lowered = tensor_name.lower()
    if "rotary_emb.inv_freq" in lowered:
        return QuantizationScope.GENERATED_ON_TARGET
    if any(
        token in lowered
        for token in (
            "scale_inv",
            "weight_scale",
            "activation_scale",
            ".scales",
            ".qzeros",
            ".g_idx",
        )
    ):
        return QuantizationScope.GLOBAL_REQUIRED
    return QuantizationScope.ABSENT


def _special_handling(
    tensor_name: str,
    tensor_family: str,
    quantization_scope: QuantizationScope,
) -> tuple[bool, str]:
    if quantization_scope == QuantizationScope.GLOBAL_REQUIRED:
        return True, "global quantization metadata required before zero-copy refit"
    if quantization_scope == QuantizationScope.GENERATED_ON_TARGET:
        return True, "target runtime can regenerate this tensor"
    if tensor_family == "moe-expert-axis-shard":
        return True, "MoE expert-axis ownership must be preserved"
    if "rotary_emb.inv_freq" in tensor_name.lower():
        return True, "layout-sensitive generated tensor"
    return False, ""


def _read_url_range(
    url: str,
    *,
    start: int,
    end: int,
    token: str | None,
    timeout_seconds: float,
) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise ValueError(
                f"{url} did not honor HTTP range request {start}-{end}: {status}"
            )
        expected = end - start + 1
        payload = response.read(expected)
        if len(payload) != expected:
            raise ValueError(
                f"{url} returned {len(payload)} bytes for range {start}-{end}"
            )
        return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify Qwen-style MoE tensors from safetensors metadata."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--default-dtype", default="unknown")
    parser.add_argument(
        "--safetensors",
        action="append",
        default=[],
        type=Path,
        help="Local safetensors file to inspect; may be repeated.",
    )
    parser.add_argument("--hf-repo", help="Hugging Face repo id to inspect.")
    parser.add_argument("--hf-revision")
    parser.add_argument(
        "--hf-file",
        action="append",
        default=[],
        help="Specific safetensors file in the Hugging Face repo; may repeat.",
    )
    parser.add_argument(
        "--hf-token",
        help="Optional Hugging Face token for private repos.",
    )
    parser.add_argument(
        "--entries-only",
        action="store_true",
        help="Write only manifest entries instead of a coverage summary object.",
    )
    args = parser.parse_args(argv)

    if bool(args.safetensors) == bool(args.hf_repo):
        raise SystemExit("provide exactly one source: --safetensors or --hf-repo")

    if args.hf_repo:
        manifest = extract_qwen_moe_manifest_from_hf_repo(
            args.hf_repo,
            model_name=args.model_name,
            model_version=args.model_version,
            filenames=args.hf_file or None,
            revision=args.hf_revision,
            token=args.hf_token,
            default_dtype=args.default_dtype,
        )
        source = {
            "kind": "huggingface",
            "repo_id": args.hf_repo,
            "revision": args.hf_revision,
            "files": args.hf_file or "all safetensors files",
        }
    else:
        manifest = extract_qwen_moe_manifest_from_safetensors_files(
            args.safetensors,
            model_name=args.model_name,
            model_version=args.model_version,
            default_dtype=args.default_dtype,
        )
        source = {
            "kind": "local-safetensors",
            "files": [str(path) for path in args.safetensors],
        }

    if args.entries_only:
        write_manifest_artifact(manifest, args.artifact)
    else:
        write_manifest_coverage_artifact(manifest, args.artifact, source=source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
