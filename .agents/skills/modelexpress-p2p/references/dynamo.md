# ModelExpress P2P Under NVIDIA Dynamo

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Dynamo owns the serving graph and worker lifecycle; ModelExpress (MX) runs
inside each worker and supplies loading, source discovery, and P2P transfer.
This reference lists what MX needs from a Dynamo worker. For where each
setting goes in a `DynamoGraphDeployment` (DGD) and how to scale a component,
follow Dynamo's documentation for the manifest's `apiVersion`.

This reference covers vLLM and SGLang workers. For TensorRT-LLM, see the
standalone section of `references/runtimes.md`.

## Before You Start

- MX client: Dynamo's vLLM and SGLang runtime images are built with it by
  default. With another image, install the ModelExpress Python client in it.
- MX server (`redis` or `kubernetes` backend): the Dynamo platform chart does
  not install it. Install and health-check it as in
  `references/server-and-backends.md`. The `k8s-service` backend needs no
  server.

## Worker Settings

Add to the `main` container of each worker component:

- Args. vLLM (`python3 -m dynamo.vllm`): `--load-format modelexpress`.
  SGLang (`python3 -m dynamo.sglang`): `--load-format remote_instance
  --remote-instance-weight-loader-backend modelexpress
  --modelexpress-config '{"transport":"nixl"}'`.
- Env: with `redis` or `kubernetes`, `MX_SERVER_ADDRESS` and
  `MODEL_EXPRESS_URL`, both set to the server address, for example
  `<server-service>.<server-namespace>.svc.cluster.local:8001`. With
  `k8s-service`, `MX_METADATA_BACKEND=k8s-service` instead.
- Resources: the cluster's RDMA device resource (for example `rdma/ib`) next
  to the GPUs.
- Capability: `IPC_LOCK`.

Leave `VLLM_PLUGINS` unset: Dynamo's docs require `dynamo_nixl_telemetry` in
it whenever it is set.

Optional (see `docs/DEPLOYMENT.md`, P2P Environment Variables):
`MX_RDMA_NIC_PIN=auto` for per-rank NIC pinning.

The MX client reads `MODEL_EXPRESS_URL` before `MX_SERVER_ADDRESS`. The Dynamo
operator injects `MODEL_EXPRESS_URL` into every pod when the platform chart
sets `dynamo-operator.modelExpressURL`, but only into containers that do not
already set it. Setting both in the worker keeps an operator-wide default from
overriding the worker's own address.

With these load formats Dynamo skips its own weight prefetch for Hugging Face
model names and lets MX acquire the weights. Every worker still needs the
model's configuration and tokenizer files: mount a shared model cache or
download the non-weight files before the worker starts.

## Roll Out

Deploy the DGD with one replica of the worker component. It loads from storage
and publishes itself as a source. Once the DGD is ready, scale the worker
component.

In a disaggregated graph, prefill and decode workers can pull from each other
only when their full source identities match: the same ModelExpress client
version (for vLLM, also the same vLLM version) and the same model arguments,
including the model string, revision, dtype or quantization, and tensor,
pipeline, and expert parallel sizes.

## Confirm

On a worker started after the first, the `main` container's log should
contain `RDMA transfer complete`. For multi-node workers, check the `main`
container of each of the worker's pods.

## Relation To `deploy-dynamo-recipe`

Dynamo's `deploy-dynamo-recipe` skill deploys an assigned DGD as handed off; it
does not change engine arguments or install cluster services. To use P2P there,
add the settings above to the DGD before it is handed off. If the workers use
the `redis` or `kubernetes` backend, the MX server must also be running before
the DGD is deployed; the `k8s-service` backend needs no server.
