<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compatibility

This page separates what is supported by the current code, what has a checked-in example, and what the current CI workflow actually runs. Use the documentation from the release tag that you deploy; this page tracks the `main` branch at ModelExpress `0.5.1`.

## Capability matrix

| Capability | Runtime or mode | Status | Evidence |
|---|---|---|---|
| Server and CLI model-cache management | Standalone | Supported | Rust tests, CLI reference, Docker Compose/Helm examples |
| P2P weight transfer | vLLM | Supported | Active P2P, TP, DP, EP, fleet, rolling-update, and stale-metadata CI jobs |
| P2P weight transfer | SGLang with NIXL | Supported with pinned image | Active CI matrix and [`docs/SGLANG.md`](SGLANG.md) |
| P2P weight transfer | SGLang with Mooncake TransferEngine | Supported with pinned image | Active CI matrix; weight-only path |
| P2P weight transfer | TensorRT-LLM | Beta/example | [`examples/p2p_transfer_k8s/client/trtllm`](../examples/p2p_transfer_k8s/client/trtllm/); not in active CI |
| ModelStreamer storage loading | vLLM | Supported for CI S3 path | Active direct and fallback S3 jobs; examples also cover Azure and local/PVC |
| ModelStreamer storage loading | SGLang | Supported for CI S3 path | Active direct and fallback S3 jobs; examples also cover Azure and local/PVC |
| Metadata backend | Redis or Kubernetes CRD | Supported | Active P2P and server/registry tests |
| Metadata backend | `k8s-service` | Specialized | Stable-weight inference; no central server; see [`K8S_SERVICE_BACKEND.md`](K8S_SERVICE_BACKEND.md) |
| Metadata backend | In-memory | Development/test only | Feature-gated server backend |
| GPUDirect Storage | Runtime-dependent | Experimental | Code path and documentation exist; no active hardware CI job |
| llm-d integration | llm-d Optimized Baseline + vLLM | Upstream guide | [llm-d ModelExpress P2P guide](https://github.com/llm-d/llm-d/tree/main/guides/modelexpress-p2p); no MX-side llm-d CI job |

## Runtime and image pins

| Integration | Current pin in this repository | Validation status |
|---|---|---|
| ModelExpress Rust/Python/Helm | `0.5.1` | `Cargo.toml`, Python `pyproject.toml`, and `helm/Chart.yaml` |
| vLLM public P2P example | `vllm/vllm-openai:v0.23.0` | Checked-in Dockerfile and example manifests |
| vLLM K8s CI image | `vllm/vllm-openai:v0.17.1` | Active CI; uses the compatibility plugin path |
| SGLang P2P and ModelStreamer examples/CI | `lmsysorg/sglang:v0.5.13.post1` | Active CI and checked-in Dockerfiles |
| Dynamo vLLM CI image | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1` | Active aggregated/disaggregated CI |
| TensorRT-LLM example | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22` | Beta/example only; Dockerfile warns that the full upstream MX PR chain must land before release support |
| llm-d upstream guide | `v0.5.0` in guide defaults | Upstream guide snapshot; update all server/client/CRD pins together before using MX `0.5.1` |

## Platform requirements

| Area | Requirement |
|---|---|
| Python | `>=3.10` for the Python client |
| Rust | Pinned by [`rust-toolchain.toml`](../rust-toolchain.toml) |
| Kubernetes | CRD and RBAC support for the Kubernetes backend; GPU/NIXL examples also need the relevant device plugins and fabric resources |
| P2P fabric | NIXL-compatible InfiniBand, RoCE, EFA, NVLink, or another supported transport for the selected runtime image |
| ModelStreamer | Worker credentials and network access to the selected storage provider |
| GDS | GDS-capable hardware and runtime support; availability is detected at runtime |

## CI coverage map

The active workflow is [`.github/workflows/modelexpress-ci-tests.yml`](../.github/workflows/modelexpress-ci-tests.yml). It currently contains these user-visible validation areas:

- vLLM, SGLang, and SGLang Mooncake P2P image builds and P2P tests.
- vLLM TP, DP, EP, MLA, fleet-scale, rolling-update, and stale-metadata tests.
- Dynamo vLLM aggregated and disaggregated serving.
- vLLM and SGLang direct S3 ModelStreamer loading and S3 failure fallback.
- Rust workspace, Python, protobuf, Helm CRD-sync, and server/client integration checks in the main CI workflow.

The matrix does not currently validate llm-d composition, GCS, Azure Blob, GDS hardware, or TensorRT-LLM. A checked-in example for one of those areas is useful evidence, but it should not be described as CI support.

## Update rule

When a runtime image, ModelExpress version, Helm chart, example manifest, or CI matrix entry changes, update this page in the same change. If an integration is only tested upstream, name the upstream repository and keep its version relationship explicit.
