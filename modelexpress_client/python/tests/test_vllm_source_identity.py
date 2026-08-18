# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_source_identity on the vLLM engine path.

The expert-parallel coverage here exists because vLLM's ParallelConfig has no
expert_parallel_size attribute. Reading one with a getattr default silently
published a constant, so every assertion below pins a value that a missing
attribute cannot produce.
"""

from types import SimpleNamespace

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.engines.vllm.source_identity import (
    _derive_expert_parallel_size,
    build_source_identity,
)


def _parallel_config(**overrides):
    """A ParallelConfig stand-in carrying only the attributes vLLM really has.

    Deliberately does NOT define expert_parallel_size: the defect under test
    was reading an attribute that does not exist on the real object.
    """
    fields = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_expert_parallel": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _model_config(**overrides):
    fields = {
        "model": "deepseek-ai/DeepSeek-V2-Lite",
        "dtype": torch.bfloat16,
        "quantization": None,
        "revision": "abc123",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _vllm_config(parallel):
    return SimpleNamespace(parallel_config=parallel)


def test_expert_parallel_disabled_is_one_not_zero():
    assert _derive_expert_parallel_size(_parallel_config()) == 1
    assert (
        _derive_expert_parallel_size(
            _parallel_config(tensor_parallel_size=2, data_parallel_size=4)
        )
        == 1
    )


def test_expert_parallel_world_size_spans_data_parallelism():
    """The MX-440 breaking pair: same flags, different DP, must not collide."""
    dp1 = _derive_expert_parallel_size(
        _parallel_config(tensor_parallel_size=2, enable_expert_parallel=True)
    )
    dp2 = _derive_expert_parallel_size(
        _parallel_config(
            tensor_parallel_size=2, data_parallel_size=2, enable_expert_parallel=True
        )
    )

    assert dp1 == 2
    assert dp2 == 4
    assert dp1 != dp2


def test_expert_parallel_world_size_includes_prefill_context_parallelism():
    assert (
        _derive_expert_parallel_size(
            _parallel_config(
                tensor_parallel_size=2,
                data_parallel_size=2,
                prefill_context_parallel_size=3,
                enable_expert_parallel=True,
            )
        )
        == 12
    )


def test_prefill_context_parallel_size_absent_defaults_to_one():
    """Older vLLM has no prefill-context parallelism; absence means 1, not 0."""
    parallel = _parallel_config(tensor_parallel_size=2, enable_expert_parallel=True)
    del parallel.prefill_context_parallel_size

    assert _derive_expert_parallel_size(parallel) == 2


def test_decode_context_parallel_size_is_not_in_the_product():
    parallel = _parallel_config(tensor_parallel_size=2, enable_expert_parallel=True)
    parallel.decode_context_parallel_size = 8

    assert _derive_expert_parallel_size(parallel) == 2


@pytest.mark.parametrize(
    "overrides, expected_ep",
    [
        ({}, 1),
        ({"enable_expert_parallel": True, "tensor_parallel_size": 2}, 2),
        (
            {
                "enable_expert_parallel": True,
                "tensor_parallel_size": 2,
                "data_parallel_size": 2,
            },
            4,
        ),
    ],
)
def test_build_source_identity_publishes_derived_expert_parallel_size(
    overrides, expected_ep
):
    parallel = _parallel_config(**overrides)
    identity = build_source_identity(_vllm_config(parallel), _model_config())

    assert identity.expert_parallel_size == expected_ep


def test_build_source_identity_carries_the_remaining_vllm_fields():
    parallel = _parallel_config(
        tensor_parallel_size=4, pipeline_parallel_size=2, enable_expert_parallel=True
    )
    identity = build_source_identity(
        _vllm_config(parallel), _model_config(quantization="fp8")
    )

    assert identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_WEIGHTS
    assert identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_VLLM
    assert identity.model_name == "deepseek-ai/DeepSeek-V2-Lite"
    assert identity.tensor_parallel_size == 4
    assert identity.pipeline_parallel_size == 2
    # Pipeline parallelism is NOT part of the expert-parallel product: the EP
    # group row spans data, prefill-context and tensor parallelism only.
    assert identity.expert_parallel_size == 4
    assert identity.dtype == "bfloat16"
    assert identity.quantization == "fp8"
    assert identity.revision == "abc123"


def test_expert_parallel_config_never_reads_a_nonexistent_attribute():
    """Guard against a regression to getattr(parallel, 'expert_parallel_size').

    A ParallelConfig that happens to carry the attribute must still be ignored,
    because the real vLLM object does not have it and the derived value is the
    only correct source.
    """
    parallel = _parallel_config(tensor_parallel_size=2, enable_expert_parallel=True)
    parallel.expert_parallel_size = 99

    assert _derive_expert_parallel_size(parallel) == 2
