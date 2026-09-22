<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Refit implementation

This directory contains the shared geometry, transfer planning, NIXL transport, and timing primitives. Framework-facing clients live in [`modelexpress_rl`](../../modelexpress_rl/).

Start with [RL weight updates](../../../../docs/guides/rl.md). The integration and design documentation now lives in the [RL integration reference](../../../../docs/RL_REFIT.md).

| Topic | Documentation |
|---|---|
| <a id="overview"></a><a id="the-design"></a>Receiver design | [Receiver internals](../../../../docs/RL_REFIT.md#receiver-internals) |
| <a id="integration-contract"></a>Adapter hooks | [Integration contract](../../../../docs/RL_REFIT.md#integration-contract) |
| <a id="implementation-status"></a>Implementation status and limits | [Implementation status](../../../../docs/RL_REFIT.md#implementation-status) |
| <a id="timing-and-configuration"></a>Configuration | [Client settings](../../../../docs/RL_REFIT.md#client-settings) and [timing](../../../../docs/RL_REFIT.md#timing-and-configuration) |
| <a id="validation"></a>Validation | [Tests and acceptance criteria](../../../../docs/RL_REFIT.md#validation) |
| <a id="tradeoffs-and-failure-modes"></a>Tradeoffs and failure modes | [Reference](../../../../docs/RL_REFIT.md#tradeoffs-and-failure-modes) |

## Package map

| Path | Role |
|---|---|
| [`timing.py`](timing.py) | Normalized timing stages and context propagation |
| [`reshard/geometry.py`](reshard/geometry.py) | Record the engine loader's source views and destination writes |
| [`reshard/slice_plan.py`](reshard/slice_plan.py) | Resolve views, intersect shard boxes, emit contiguous runs |
| [`reshard/transfer_plan.py`](reshard/transfer_plan.py) | Build and execute receiver-local plans |
| [`reshard/rendezvous.py`](reshard/rendezvous.py) | Publish/discover shard ownership and NIXL endpoints |
| [`reshard/receiver.py`](reshard/receiver.py) | Shared receiver lifecycle, buffers, staging, and install hooks |
| [`reshard/transport/`](reshard/transport/) | Reference and NIXL transport adapters |
| [`modelexpress_rl/inference/nixl_staged_transfer.py`](../../modelexpress_rl/inference/nixl_staged_transfer.py) | RL exact-manifest planning, staged NIXL transfer, and verification |
| [`modelexpress_rl/inference/engines/vllm/installer.py`](../../modelexpress_rl/inference/engines/vllm/installer.py) | vLLM geometry capture and graph-safe layerwise install |
| [`engines/vllm/refit/installer.py`](../engines/vllm/refit/installer.py) | Optional vLLM mapped direct installer |
