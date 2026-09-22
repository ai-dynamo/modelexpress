# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SourceIdentity construction from vLLM config objects.

Lives under the vLLM engine rather than in metadata/ because every field here
is read off vLLM's own config objects. The expert-parallel derivation in
particular depends on vLLM internals, and keeping it in an engine-agnostic
module is how a read of a nonexistent ParallelConfig attribute went unnoticed.
"""

from __future__ import annotations

from ... import envs
from ... import p2p_pb2


def _derive_expert_parallel_size(parallel) -> int:
    """Derive the expert-parallel world size from a vLLM ParallelConfig.

    vLLM has no ``expert_parallel_size`` attribute. ``ParallelConfig`` carries
    only ``enable_expert_parallel``, and the effective expert-parallel world
    size is the tensor-parallel size flattened across data parallelism and
    prefill-context parallelism, per
    ``vllm.model_executor.layers.fused_moe.config``. The expert-parallel group
    spans data-parallel replicas, so two deployments differing only in
    data-parallel degree hold different expert subsets per rank and must not
    share an ``mx_source_id``.

    ``prefill_context_parallel_size`` is absent on vLLM releases predating
    prefill-context parallelism, where a default of 1 is correct.
    ``decode_context_parallel_size`` is deliberately excluded: it does not
    participate in the expert-parallel group.
    """
    if not getattr(parallel, "enable_expert_parallel", False):
        return 1

    tp_size = getattr(parallel, "tensor_parallel_size", 1) or 1
    dp_size = getattr(parallel, "data_parallel_size", 1) or 1
    pcp_size = getattr(parallel, "prefill_context_parallel_size", 1) or 1
    return max(1, tp_size * dp_size * pcp_size)


def build_source_identity(
    vllm_config, model_config,
) -> p2p_pb2.SourceIdentity:
    """Build a SourceIdentity from vLLM config objects."""
    from importlib.metadata import version as pkg_version

    try:
        mx_version = pkg_version("modelexpress")
    except Exception:
        mx_version = "0.0.0"

    parallel = vllm_config.parallel_config
    tp_size = getattr(parallel, "tensor_parallel_size", 1)
    pp_size = getattr(parallel, "pipeline_parallel_size", 1)
    ep_size = _derive_expert_parallel_size(parallel)

    # torch.dtype.__str__ returns e.g. "torch.bfloat16"; strip the prefix
    dtype = str(model_config.dtype).replace("torch.", "")
    quantization = model_config.quantization or ""

    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model_config.model,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        expert_parallel_size=ep_size,
        dtype=dtype,
        quantization=quantization,
        revision=_resolve_model_revision(model_config),
    )


def _resolve_model_revision(model_config) -> str:
    """Resolve the model revision for content-addressed identity.

    Priority:
    1. MX_MODEL_REVISION env var (explicit deployer override, useful
       for local checkpoints or non-HF sources).
    2. model_config.revision (from vLLM's ModelConfig; typically the
       HuggingFace commit SHA or branch/tag that was loaded).
    3. Empty string (unknown revision; handshake relies on the other
       identity fields only, and decentralized deployments lose the
       bit-identical guarantee).
    """
    override = envs.MX_MODEL_REVISION
    if override:
        return override
    revision = getattr(model_config, "revision", None)
    return revision or ""
