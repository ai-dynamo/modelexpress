<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Integrations

Runtime integrations implement model loading. Orchestrator integrations place and scale those runtime workers.

| Layer | Integrations |
|---|---|
| Runtime | [vLLM](runtimes/vllm.md), [SGLang](runtimes/sglang.md), [TensorRT-LLM](runtimes/tensorrt-llm.md) |
| Orchestrator | [NVIDIA Dynamo](orchestrators/dynamo.md), [llm-d](orchestrators/llm-d.md) |

Choose the loading path first in [Choose a ModelExpress path](../guides/choose-a-path.md), then use the runtime or orchestrator page for integration-specific commands and limits.
