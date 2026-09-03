<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress documentation

Start with the deployment scenario that matches where your model lives and how workers reach it.

## Start here

| Goal | Guide |
|---|---|
| Scale replicas without shared model storage | [P2P without shared storage](guides/choose-a-path.md#p2p-without-shared-storage) |
| Load from S3, GCS, Azure Blob, or a local path | [Storage loading](guides/choose-a-path.md#load-from-object-storage-or-a-local-path) |
| Understand or tune loader selection | [Loader behavior](guides/choose-a-path.md#configure-loader-behavior) |
| Integrate an inference runtime | [vLLM](integrations/runtimes/vllm.md), [SGLang](integrations/runtimes/sglang.md), or [TensorRT-LLM](integrations/runtimes/tensorrt-llm.md) |
| Deploy through an orchestrator | [Dynamo](integrations/orchestrators/dynamo.md) or [llm-d](integrations/orchestrators/llm-d.md) |
| Run the standalone server or CLI | [Deployment](DEPLOYMENT.md) and [CLI](CLI.md) |

## Guides and reference

- [Choose a ModelExpress path](guides/choose-a-path.md) maps common deployment scenarios to checked-in examples.
- [Configuration](CONFIGURATION.md) covers defaults, loader eligibility, metadata, transport, artifacts, and metrics.
- [Troubleshooting](TROUBLESHOOTING.md) starts from observable symptoms.
- [Integrations](integrations/README.md) separates runtime loaders from orchestrators.
- [Deployment](DEPLOYMENT.md) covers server, Docker, Helm, Kubernetes, and rollout details.
- [Architecture](ARCHITECTURE.md) and [metadata](metadata.md) contain implementation and protocol details.
- [Metrics](METRICS.md), [Kubernetes Service backend](K8S_SERVICE_BACKEND.md), and [benchmarks](BENCHMARKS.md) cover their respective subsystems.

## Support boundaries

The `main` branch and Helm chart currently report ModelExpress `0.5.1`. Use docs from the release tag you deploy, and treat pinned runtime images in examples as qualification snapshots. The active CI workflows are the source of truth for automated coverage; hardware-dependent P2P, GDS, and runtime combinations still need validation on the target GPU, fabric, driver, and storage stack.
