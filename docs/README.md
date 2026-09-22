<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress documentation

Start with the job you want ModelExpress to do:

| Goal | Start here |
|---|---|
| **Inference: start and scale replicas** | [vLLM Kubernetes quick start](integrations/runtimes/vllm.md), [SGLang](integrations/runtimes/sglang.md), or [TensorRT-LLM](integrations/runtimes/tensorrt-llm.md) |
| **RL: update rollout weights** | [RL guide](guides/rl.md), with direct-transfer and object-storage examples |
| Load weights from object storage or avoid shared storage | [Choose a loading path](guides/choose-a-path.md) |
| Use Dynamo or llm-d | [Dynamo](integrations/orchestrators/dynamo.md) or [llm-d](integrations/orchestrators/llm-d.md) |
| Run the server or CLI without a GPU | [Deployment](DEPLOYMENT.md) and [CLI](CLI.md) |

## Reference

- [Compatibility](COMPATIBILITY.md): which runtime images and paths the examples and CI cover.
- [Configuration](CONFIGURATION.md): settings, defaults, and loader selection.
- [Deployment](DEPLOYMENT.md) and [Helm](../helm/README.md): server setup and operational requirements.
- [Troubleshooting](TROUBLESHOOTING.md): diagnose the selected loader and failed dependencies.
- [Metrics](METRICS.md): observe the server and inference workers.
- [Architecture](ARCHITECTURE.md), [metadata](metadata.md), and [Kubernetes Service discovery](K8S_SERVICE_BACKEND.md): implementation details.
- [Benchmarks](BENCHMARKS.md): measured loading and startup performance, with conditions.

Use documentation from the release tag you deploy. A runtime Dockerfile or CI test covers a particular combination; it is not a guarantee for every GPU, fabric, driver, or model.
