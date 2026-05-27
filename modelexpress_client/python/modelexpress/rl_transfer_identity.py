# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL transfer source identity helpers."""

from __future__ import annotations

from modelexpress import p2p_pb2
from modelexpress.rl_metadata import RlSourceCandidate, candidates_from_response

_BACKEND_FRAMEWORKS = {
    "vllm": p2p_pb2.BACKEND_FRAMEWORK_VLLM,
    "sglang": p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
    "trtllm": p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
    "trt_llm": p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
    "trt-llm": p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
}


def backend_framework_value(value: str | int) -> int:
    """Return the SourceIdentity enum value for a framework name."""
    if isinstance(value, int):
        return value
    normalized = value.strip().lower()
    if normalized not in _BACKEND_FRAMEWORKS:
        raise ValueError(
            f"unsupported backend_framework {value!r}; expected one of "
            f"{sorted(_BACKEND_FRAMEWORKS)}"
        )
    return _BACKEND_FRAMEWORKS[normalized]


def build_rl_base_identity(
    *,
    model_name: str,
    mx_version: str,
    backend_framework: str | int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    dtype: str,
    quantization: str,
    revision: str,
) -> "p2p_pb2.SourceIdentity":
    """Build the stable, non-versioned SourceIdentity shared by RL sources."""
    if not model_name:
        raise ValueError("ModelExpress RL transfer requires model_name")
    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model_name,
        backend_framework=backend_framework_value(backend_framework),
        tensor_parallel_size=int(tensor_parallel_size),
        pipeline_parallel_size=int(pipeline_parallel_size),
        expert_parallel_size=int(expert_parallel_size),
        dtype=dtype,
        quantization=quantization,
        revision=revision,
    )


def identity_matches_base(
    identity: "p2p_pb2.SourceIdentity",
    base_identity: "p2p_pb2.SourceIdentity",
) -> bool:
    """Return true when an RL source identity matches the non-versioned base."""
    return (
        identity.mx_version == base_identity.mx_version
        and identity.mx_source_type == base_identity.mx_source_type
        and identity.model_name == base_identity.model_name
        and identity.backend_framework == base_identity.backend_framework
        and identity.tensor_parallel_size == base_identity.tensor_parallel_size
        and identity.pipeline_parallel_size == base_identity.pipeline_parallel_size
        and identity.expert_parallel_size == base_identity.expert_parallel_size
        and identity.dtype == base_identity.dtype
        and identity.quantization == base_identity.quantization
        and identity.revision == base_identity.revision
    )


def candidates_for_base_identity(
    response: "p2p_pb2.ListSourcesResponse",
    base_identity: "p2p_pb2.SourceIdentity",
) -> list[RlSourceCandidate]:
    """Parse RL candidates from a broad ListSources response."""
    filtered_response = p2p_pb2.ListSourcesResponse()
    for ref in response.instances:
        if not ref.HasField("identity"):
            continue
        if not identity_matches_base(ref.identity, base_identity):
            continue
        filtered_response.instances.append(ref)
    return candidates_from_response(filtered_response)
