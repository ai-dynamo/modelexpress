<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Dynamo

Dynamo supplies the orchestration and serving graph; ModelExpress supplies the model weight loading and P2P source/target behavior inside the runtime worker. The integration is not a separate loader: the Dynamo vLLM worker uses the same `modelexpress` client path as standalone vLLM.

## Choose a Dynamo topology

| Dynamo shape | Example | What it validates |
|---|---|---|
| Aggregated serving | [`vllm-multi-node-aggregated.yaml`](../../../examples/dynamo_p2p_transfer_k8s/vllm/vllm-multi-node-aggregated.yaml) | DynamoGraphDeployment with one vLLM worker service and P2P scale-out |
| Disaggregated serving | [`vllm-single-node-disaggregated.yaml`](../../../examples/dynamo_p2p_transfer_k8s/vllm/vllm-single-node-disaggregated.yaml) | Separate prefill/decode services, KV transfer, and decode-worker P2P scale-out |

The active ModelExpress CI workflow runs both aggregated and disaggregated vLLM Dynamo tests. It does not currently establish Dynamo integrations for SGLang or TensorRT-LLM.

## Integration requirements

1. Install the Dynamo operator and apply the ModelExpress CRDs when the deployment uses the Kubernetes metadata backend.
2. Build a worker image that combines the Dynamo vLLM runtime with the ModelExpress Python client. The checked-in CI image uses `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1`; the example Dockerfile layers the client onto a supplied Dynamo runtime image.
3. Configure the ModelExpress server address in the worker environment. Dynamo manifests historically use `MODEL_EXPRESS_URL`; `MX_SERVER_ADDRESS` is the preferred name for new deployments, but set both to the same value when an integration path reads the legacy variable.
4. Give the first worker a storage/bootstrap path and scale only after it is ready so later workers have a source to discover.

## Verify

Use the DGD status and frontend health for serving readiness, then inspect worker logs for `Eligible loaders`, `Trying strategy: rdma`, and `Transfer complete`. The [Dynamo example](../../../examples/dynamo_p2p_transfer_k8s/README.md) contains the exact `kubectl` and image commands.
