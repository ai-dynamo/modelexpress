<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Dynamo

Dynamo owns the serving graph and worker lifecycle; ModelExpress runs inside the runtime worker and supplies model loading, source discovery, and P2P transfer.

| Topology | Example |
|---|---|
| Aggregated vLLM | [`vllm-multi-node-aggregated.yaml`](../../../examples/dynamo_p2p_transfer_k8s/vllm/vllm-multi-node-aggregated.yaml) |
| Disaggregated vLLM | [`vllm-single-node-disaggregated.yaml`](../../../examples/dynamo_p2p_transfer_k8s/vllm/vllm-single-node-disaggregated.yaml) |

Install the Dynamo operator and ModelExpress CRDs, build a Dynamo vLLM image containing the ModelExpress client, configure the ModelExpress server address, and give the first worker a bootstrap source. The checked-in examples currently use `MODEL_EXPRESS_URL`; new integrations should prefer `MX_SERVER_ADDRESS`, or set both to the same value when the integration still reads the legacy name.

Wait for the first worker to become ready before scaling. Confirm the `DynamoGraphDeployment` status, then inspect worker logs for `Eligible loaders`, `Trying strategy: rdma`, and the transfer-completion message. See the [Dynamo example](../../../examples/dynamo_p2p_transfer_k8s/README.md) for build and deployment commands.

ModelExpress CI exercises the aggregated and disaggregated vLLM topologies. It does not currently qualify Dynamo with SGLang or TensorRT-LLM.
