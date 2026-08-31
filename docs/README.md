<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress documentation

ModelExpress accelerates model startup by choosing an available path from model storage, a server cache, or a serving peer's GPU memory. Start with the question that matches your deployment instead of reading the implementation reference first.

## Choose a path

| I need to... | Start here |
|---|---|
| Add replicas without a shared model filesystem | [P2P without shared storage](guides/choose-a-path.md#i-need-replicas-without-shared-model-storage) |
| Load weights from S3, GCS, Azure Blob, or a local path | [ModelStreamer storage loading](guides/choose-a-path.md#can-i-load-from-s3-gcs-azure-blob-or-a-local-path) |
| Understand which loader will run | [Loader selection and configuration](guides/choose-a-path.md#can-i-configure-the-loader) |
| Run with vLLM | [vLLM integration](integrations/runtimes/vllm.md) |
| Run with SGLang | [SGLang integration](integrations/runtimes/sglang.md) |
| Run with TensorRT-LLM | [TensorRT-LLM integration](integrations/runtimes/tensorrt-llm.md) |
| Run ModelExpress through Dynamo | [Dynamo integration](integrations/orchestrators/dynamo.md) |
| Run ModelExpress through llm-d | [llm-d integration](integrations/orchestrators/llm-d.md) |
| Operate the standalone server or CLI | [Deployment guide](DEPLOYMENT.md) and [CLI reference](CLI.md) |

## Documentation map

### Guides

- [Choose a ModelExpress path](guides/choose-a-path.md) maps common deployment questions to the smallest working example.
- [Deployment](DEPLOYMENT.md) covers server prerequisites, metadata topologies, Docker, Helm, and Kubernetes rollout choices.
- [Configuration](CONFIGURATION.md) is the canonical reference for ModelExpress-owned environment variables, defaults, and eligibility gates.
- [Troubleshooting](TROUBLESHOOTING.md) starts from observable symptoms and points to the relevant logs, probes, and configuration checks.

### Integrations

- [Runtime integrations](integrations/runtimes/) covers vLLM, SGLang, and TensorRT-LLM independently.
- [Orchestrator integrations](integrations/orchestrators/) covers Dynamo and llm-d. These pages describe the contract with the orchestrator and link to its runnable manifests.
- [Loading paths](integrations/loading-paths/) covers P2P and ModelStreamer details that apply across runtimes.

### Reference and internals

- [Compatibility](COMPATIBILITY.md) records the current release, examples, runtime pins, and what CI actually exercises.
- [Internals index](internals/README.md) groups the implementation and protocol references.
- [Architecture](ARCHITECTURE.md) is implementation reference material: components, protocols, metadata, and transfer internals.
- [Metadata](metadata.md) documents the coordination protocol and backend storage shapes.
- [Metrics](METRICS.md) documents server and client Prometheus exposition.
- [GCS provider](GCS_PROVIDER.md) documents the standalone model-cache provider implementation.
- [Kubernetes Service backend](K8S_SERVICE_BACKEND.md) explains the decentralized source-discovery design and its limitations.
- [Benchmarks](BENCHMARKS.md) records measured loading-path and artifact-transfer results.

## Current release

The checked-out `main` branch and the Helm chart are at `0.5.1`. Use the documentation from the release tag that you deploy; examples and compatibility notes on this page track `main`.

## How the docs are kept honest

Relative links are checked against the repository, Kubernetes manifests are parsed as YAML, the Helm chart is linted, and the CRD bundle is compared with the chart copy in CI. Maintainers cross-check scenario commands and environment variables against the Python loader chain, Helm values, examples, and CI matrices. The [compatibility page](COMPATIBILITY.md) distinguishes runnable examples from paths that are only unit-tested or hardware-gated.
