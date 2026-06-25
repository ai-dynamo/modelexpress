<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compatibility

This page summarizes the runtime combinations currently exercised by CI or
maintained as public examples. For detailed scenario coverage and open gaps,
see [`ci/TEST_PLAN.md`](../ci/TEST_PLAN.md).

Use the documentation from the release tag you deploy. The values below track
the current `main` branch.

## Support Matrix

| Capability | Engine / mode | Status | Validation |
|------------|---------------|--------|------------|
| Server + CLI model-cache management | Standalone | Supported | Rust integration tests and CLI docs |
| P2P weight transfer | vLLM | Supported | In CI |
| Multi-node tensor parallel P2P | vLLM | Supported | In CI for TP=2 |
| Dynamo integration | Dynamo + vLLM | Supported | In CI for aggregated and disaggregated vLLM paths |
| P2P weight transfer | SGLang + NIXL | Supported | In CI with a known-good SGLang release image |
| P2P weight transfer | SGLang + Mooncake TransferEngine | Supported | In CI |
| P2P weight transfer | TensorRT-LLM | Beta | In CI, requires TRT-LLM/Dynamo-specific image and patches |
| ModelStreamer storage loading | vLLM | Supported | In CI for S3; examples cover S3, Azure Blob, and local/PVC |
| ModelStreamer storage loading | SGLang | Experimental | Adapter and launch coverage still gated |
| Metadata backend | Redis or Kubernetes CRD | Supported | Used by P2P and model-cache examples |
| Metadata backend | `k8s-service` | Specialized | Stable-weight inference only; no central MX server |
| Metadata backend | In-memory | Dev/test only | Feature-gated; not for production deployments |
| GPUDirect Storage | vLLM/TRT-LLM/SGLang | Experimental | Hardware-dependent; CI coverage pending |
| RL/live refit workflows | Framework integrations | Emerging | Use Redis or Kubernetes CRD; APIs and examples are still evolving |

## ModelExpress Artifacts

| Artifact | Current pin | Source |
|----------|-------------|--------|
| Rust workspace crates | `0.5.0` | [`Cargo.toml`](../Cargo.toml) |
| Python client package | `0.5.0` | [`modelexpress_client/python/pyproject.toml`](../modelexpress_client/python/pyproject.toml) |
| Helm chart | `0.5.0` | [`helm/Chart.yaml`](../helm/Chart.yaml) |
| Helm app version | `0.5.0` | [`helm/Chart.yaml`](../helm/Chart.yaml) |
| Server image used by examples | `nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.5.0` | [`helm/values.yaml`](../helm/values.yaml) and Kubernetes examples |

## Engine Integrations

| Integration | Tested or documented runtime | Status | Notes |
|-------------|------------------------------|--------|-------|
| vLLM P2P | `vllm/vllm-openai:v0.17.1` | In CI | Uses `--load-format modelexpress`; CI covers single-replica P2P plus TP=2 single-node and multi-node vLLM paths. |
| Dynamo + vLLM | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1`; Dynamo operator image tag `1.0.2` | In CI | CI covers aggregated and disaggregated DynamoGraphDeployment paths with vLLM. |
| SGLang P2P with NIXL | `lmsysorg/sglang:v0.5.13.post1` | In CI | This image includes the upstream ModelExpress delegation hook from `sgl-project/sglang#24723`. Install ModelExpress with `--no-deps` inside SGLang images. |
| SGLang P2P with Mooncake TransferEngine | `lmsysorg/sglang:v0.5.13.post1` plus `mooncake-transfer-engine-cuda13` | In CI | Uses the same SGLang `remote_instance` path with `modelexpress-config` selecting `{"transport": "transfer_engine"}`. |
| TensorRT-LLM P2P | `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.1` | Beta, in CI | Runtime includes TRT-LLM `1.3.0rc13`, NIXL `0.10.1`, and `nixl-cu12` `0.10.1`; ModelExpress applies the PRESHARDED patch flow during image build. |
| ModelStreamer with vLLM | `vllm/vllm-openai:v0.17.1` | In CI for S3 | Examples also cover Azure Blob Storage and local/PVC sources. |
| ModelStreamer with SGLang | `lmsysorg/sglang:v0.5.13.post1` | Experimental | The CI matrix is gated until the SGLang adapter and framework-aware launch path are completed. |
| vLLM on AWS EFA | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0-efa-amd64` | Example only | Set `MX_NIXL_BACKEND=LIBFABRIC`; this path is documented but not part of the default PR CI matrix. |

## Platform Pins

| Area | Pin or requirement | Notes |
|------|--------------------|-------|
| Python | `>=3.10` | Python package classifiers cover Python 3.10, 3.11, and 3.12. |
| Rust | `1.90.0` | See [`rust-toolchain.toml`](../rust-toolchain.toml). |
| Kubernetes | Helm docs require Kubernetes 1.19+; CI vCluster uses `v1.32.13` | Kubernetes CRD code is built with the `k8s-openapi` `v1_31` feature. |
| vCluster CLI | `v0.33.0` | Used by the Kubernetes CI bootstrap action. |
| NIXL | Normal Python installs depend on `nixl[cu12]` on Linux | Engine images often own the CUDA/NIXL stack. For SGLang, install ModelExpress with `--no-deps`; for TRT-LLM, use the Dynamo TRT-LLM runtime pin above. |
| CUDA | Runtime-owned | Dynamo source-build examples use CUDA `12.9`; SGLang source-build examples use CUDA `13.0.1`; TRT-LLM CI uses the CUDA 12 NIXL stack from the runtime image. |

## Update Rule

When bumping a runtime image, ModelExpress release version, or CI platform pin,
update this file in the same change as the Dockerfile, workflow, Helm chart,
or example manifest that introduced the bump.
