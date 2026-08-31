<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Integrations

ModelExpress integrates at two layers: runtime loaders receive or stream weight tensors, while orchestrators place and scale those runtime workers. Choose the page for the layer you are changing.

## Runtime loaders

- [vLLM](runtimes/vllm.md)
- [SGLang](runtimes/sglang.md)
- [TensorRT-LLM](runtimes/tensorrt-llm.md)

## Orchestrators

- [NVIDIA Dynamo](orchestrators/dynamo.md)
- [llm-d](orchestrators/llm-d.md)

## Loading paths

- [P2P weight transfer](loading-paths/p2p.md)
- [ModelStreamer storage loading](loading-paths/model-streamer.md)
