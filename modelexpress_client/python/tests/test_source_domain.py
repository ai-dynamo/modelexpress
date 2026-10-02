# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MX_SOURCE_DOMAIN partitions weights and artifact discovery by locality."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.metadata.source_domain import (
    SOURCE_DOMAIN_KEY,
    apply_source_domain,
    source_domain,
)
from modelexpress.metadata.source_id import compute_mx_source_id


def _identity(**extra) -> p2p_pb2.SourceIdentity:
    return p2p_pb2.SourceIdentity(
        mx_version="0.6.0",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name="deepseek-ai/DeepSeek-V3",
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        tensor_parallel_size=8,
        dtype="bfloat16",
        extra_parameters=extra,
    )


class TestApplySourceDomain:
    def test_unset_leaves_identity_and_id_unchanged(self, monkeypatch):
        monkeypatch.delenv("MX_SOURCE_DOMAIN", raising=False)
        before = compute_mx_source_id(_identity())
        identity = apply_source_domain(_identity())
        assert SOURCE_DOMAIN_KEY not in identity.extra_parameters
        assert compute_mx_source_id(identity) == before
        assert source_domain() == ""

    def test_blank_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv("MX_SOURCE_DOMAIN", "   ")
        identity = apply_source_domain(_identity())
        assert SOURCE_DOMAIN_KEY not in identity.extra_parameters

    def test_set_folds_domain_into_extra_parameters(self, monkeypatch):
        monkeypatch.setenv("MX_SOURCE_DOMAIN", " mid-training ")
        identity = _identity(role="x")
        returned = apply_source_domain(identity)
        assert returned is identity
        assert identity.extra_parameters[SOURCE_DOMAIN_KEY] == "mid-training"
        assert identity.extra_parameters["role"] == "x"

    def test_different_domains_never_share_a_source_id(self, monkeypatch):
        monkeypatch.delenv("MX_SOURCE_DOMAIN", raising=False)
        unpartitioned = compute_mx_source_id(apply_source_domain(_identity()))
        monkeypatch.setenv("MX_SOURCE_DOMAIN", "ns-a")
        ns_a = compute_mx_source_id(apply_source_domain(_identity()))
        monkeypatch.setenv("MX_SOURCE_DOMAIN", "ns-b")
        ns_b = compute_mx_source_id(apply_source_domain(_identity()))
        monkeypatch.setenv("MX_SOURCE_DOMAIN", "ns-a")
        ns_a_again = compute_mx_source_id(apply_source_domain(_identity()))
        assert len({unpartitioned, ns_a, ns_b}) == 3
        assert ns_a == ns_a_again

    def test_announces_once_per_domain(self, monkeypatch, caplog):
        from modelexpress.metadata import source_domain as mod

        monkeypatch.setattr(mod, "_announced", set())
        monkeypatch.setenv("MX_SOURCE_DOMAIN", "ns-log")
        with caplog.at_level("INFO", logger="modelexpress.metadata.source_domain"):
            apply_source_domain(_identity())
            apply_source_domain(_identity())
        assert sum("MX_SOURCE_DOMAIN" in r.message for r in caplog.records) == 1


@pytest.fixture
def domain(monkeypatch):
    monkeypatch.setenv("MX_SOURCE_DOMAIN", "mid-training")
    return "mid-training"


def _sglang_model_config():
    return SimpleNamespace(
        model_path="zai-org/GLM-5.3",
        dtype=torch.bfloat16,
        quantization="fp8",
        revision="",
    )


def test_sglang_weights_identity_carries_domain(domain):
    from modelexpress.engines.sglang.adapter import build_sglang_source_identity

    identity = build_sglang_source_identity(_sglang_model_config())
    assert identity.extra_parameters[SOURCE_DOMAIN_KEY] == domain


def test_sglang_weights_identity_has_no_extras_when_unset(monkeypatch):
    monkeypatch.delenv("MX_SOURCE_DOMAIN", raising=False)
    from modelexpress.engines.sglang.adapter import build_sglang_source_identity

    identity = build_sglang_source_identity(_sglang_model_config())
    assert dict(identity.extra_parameters) == {}


def test_vllm_weights_identity_carries_domain(domain):
    from modelexpress.engines.vllm.source_identity import build_source_identity

    parallel = SimpleNamespace(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_expert_parallel=False,
    )
    identity = build_source_identity(
        SimpleNamespace(parallel_config=parallel),
        SimpleNamespace(model="m", dtype=torch.bfloat16, quantization=None, revision="r"),
    )
    assert identity.extra_parameters[SOURCE_DOMAIN_KEY] == domain


def test_trtllm_weights_identity_carries_domain(domain):
    from modelexpress.engines.trtllm.adapter import build_mx_identity

    source_identity = SimpleNamespace(
        to_dict=lambda: {"model_name": "m", "tp_size": 1},
        tp_size=1,
        pp_size=1,
        ep_size=1,
        dtype="bfloat16",
    )
    identity = build_mx_identity(source_identity, transform_protocol_version=1)
    assert identity.extra_parameters[SOURCE_DOMAIN_KEY] == domain
    assert identity.extra_parameters["trtllm_weight_layout"] == "post_transform"


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_artifact_identities_carry_domain(domain, engine, monkeypatch):
    artifacts = importlib.import_module(f"modelexpress.engines.{engine}.artifacts")
    if engine == "sglang":
        monkeypatch.setattr(
            artifacts._common_artifacts, "gpu_arch", lambda device_id: "sm100"
        )
        monkeypatch.setattr(
            artifacts._common_artifacts, "tvm_ffi_version", lambda: "0.1.6"
        )
    else:
        monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: "sm100")
    ctx = SimpleNamespace(
        device_id=0,
        identity=p2p_pb2.SourceIdentity(
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name="test/model",
            tensor_parallel_size=8,
            dtype="bfloat16",
            quantization="fp8",
            revision="abc",
        ),
    )
    artifact_types = [
        p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
        p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
        p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE,
        p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE,
    ]
    if engine == "sglang":
        artifact_types.append(p2p_pb2.MX_SOURCE_TYPE_TVM_FFI_CACHE)
    for source_type in artifact_types:
        identity = artifacts._artifact_identity(ctx, source_type)
        assert identity.extra_parameters[SOURCE_DOMAIN_KEY] == domain, source_type
