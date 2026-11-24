---
name: modelexpress-p2p
description: Set up ModelExpress P2P weight loading so new inference replicas pull weights from a running replica over RDMA instead of reading storage. Use to choose a metadata backend, install and health-check the ModelExpress server, enable P2P on vLLM, SGLang, or TensorRT-LLM workers, including vLLM and SGLang under NVIDIA Dynamo, and confirm the weights arrived over P2P; not for RL weight refit.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - modelexpress
    - p2p
    - nixl
    - rdma
    - kubernetes
    - vllm
    - sglang
    - dynamo
---

# ModelExpress P2P

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Purpose

Make inference replicas start faster by reading weights over NIXL from a
replica that already has them in memory, instead of from storage. Weights on
the accelerator (CUDA or Intel XPU) move accelerator to accelerator; with vLLM
and SGLang, MX clients that support it also move parameters the engine keeps in
host memory. The first replica loads from storage and publishes itself as a
source; later replicas with the same model and layout discover it through
ModelExpress (MX) and read its weights directly. Weight bytes never pass
through the MX server.

## Prerequisites

- RDMA-capable NICs exposed to worker pods (for example an `rdma/ib` device
  plugin resource, or AWS EFA) and NIXL in the worker image. MX checks that
  NIXL is importable but not that the fabric is RDMA, so a pod without RDMA
  resources can still attempt P2P. Depending on the UCX configuration, that
  transfer may run over TCP, which works but is much slower and gives no
  warning. Check transfer throughput in the worker logs to confirm RDMA is in
  use.
- The MX Python client in the worker image. NVIDIA Dynamo's vLLM and SGLang
  runtime images are built with it by default.
- Every target can read the model's configuration and tokenizer at its model
  path, even when all weights arrive over RDMA.
- Replicas that should share weights run the same ModelExpress client version
  (for vLLM, also the same vLLM version) with the same model arguments: the
  same model string, revision, dtype or quantization, and parallel layout.
  Sources pair with targets rank to rank.

## Instructions

### 1. Choose A Metadata Backend

| Workload | Backend | MX server needed |
|---|---|---|
| Weights fixed for each pod's lifetime, every pod behind a Service serves the same checkpoint | `k8s-service` | No |
| Mixed revisions, heterogeneous hardware, or anything beyond the case above | `redis` or `kubernetes` | Yes |

`redis` needs a Redis instance you provide. `kubernetes` stores source
metadata in `ModelMetadata` custom resources and needs the CRDs and RBAC.
Details and trade-offs: `references/server-and-backends.md`.

### 2. Deploy And Health-Check The Server

Skip for `k8s-service`. Install the Helm chart from a ModelExpress checkout,
set `MX_METADATA_BACKEND` (plus `REDIS_URL` for Redis), and wait for the
rollout. The readiness probe targets the gRPC port (default `8001`). Full
commands and checks: `references/server-and-backends.md`.

### 3. Configure The Workers

With `redis` or `kubernetes`, point every worker at the server with
`MX_SERVER_ADDRESS=<service>:<port>` (`8001` with the Helm chart). With
`k8s-service`, set `MX_METADATA_BACKEND=k8s-service` on the workers instead
(see `references/server-and-backends.md`). Then select the MX loader for each
runtime:

| Runtime | Worker setting |
|---|---|
| vLLM | `--load-format modelexpress`; on standalone vLLM older than 0.23.0, the ModelExpress plugin must load (leave `VLLM_PLUGINS` unset or include `modelexpress`) |
| SGLang | `--load-format remote_instance --remote-instance-weight-loader-backend modelexpress --modelexpress-config '{"transport":"nixl"}'` |
| TensorRT-LLM | `checkpoint_format: MX` with `mx_config.server_url` |

Under NVIDIA Dynamo, set these in the `DynamoGraphDeployment` worker spec; see
`references/dynamo.md`. Standalone manifests and per-runtime details:
`references/runtimes.md`.

`MODEL_EXPRESS_URL` is a deprecated alias that still takes precedence over
`MX_SERVER_ADDRESS`. If anything injects it (the Dynamo operator can), set it
to the same address.

### 4. Roll Out One Replica, Then Scale

Start one replica and wait until it is ready: it loads from storage and
publishes itself as a source. Then scale. Replicas started before any source
is ready also load from storage.

### 5. Confirm The Weights Came Over P2P

On a replica started after the first, the worker log should contain
`RDMA transfer complete`. If it is missing, see Troubleshooting below. SGLang's
`transfer_engine` transport has no equivalent line: its completion message is
logged even when it falls back to storage.

## Output Contract

Return:

- the chosen backend and why
- the server install commands and the health-check result, or why no server is needed
- the exact worker settings per runtime, and where they go (manifest, DGD, or engine config)
- the P2P evidence: the log lines from a scaled replica, or the failing signal
  and next action

## Limitations

- P2P is built for RDMA between source and target. MX does not verify the
  fabric before trying it.
- This skill covers TensorRT-LLM standalone only.
- RL weight refit is out of scope; see `docs/rl/README.md`.

## Troubleshooting

| Symptom | Likely cause | Next step |
|---|---|---|
| `Eligible loaders` lacks `rdma` | No server address, or NIXL not importable | Check each cause; log messages such as `No MX server configured, skipping RDMA` point to some of them |
| `Strategy rdma failed, trying next: <reason>` | Either no ready source matches this model on this server (`No RDMA source available`), or a source was found but could not be used or its transfer failed | For no source: wait for the first replica, compare the ModelExpress client version (and vLLM version) and the model arguments (model string, revision, dtype, quantization, parallel layout), and check the server address. Otherwise: read the warnings logged earlier for each source tried |
| `NIXL registration failed` | The target could not register its buffers for RDMA | Read the registration error logged just before; check the RDMA device resources and `IPC_LOCK` |
| `[TIMING] RDMA transfer complete:` with low Gbps | Traffic on TCP or the wrong NIC | Set `MX_RDMA_NIC_PIN=auto` or `UCX_NET_DEVICES`; on AWS EFA, follow `references/runtimes.md` for the libfabric setup |
| Worker reaches a different server than `MX_SERVER_ADDRESS` | `MODEL_EXPRESS_URL` set elsewhere and taking precedence | Set both to the same address |

## References

- `references/server-and-backends.md`: backend choice, Helm install, health checks, pointing workers at the server, stale sources.
- `references/runtimes.md`: standalone vLLM, SGLang, and TensorRT-LLM worker configuration.
- `references/dynamo.md`: enabling P2P on Dynamo workers.
- Source docs: `docs/guides/choose-a-path.md`, `docs/DEPLOYMENT.md`, `docs/integrations/`, `docs/TROUBLESHOOTING.md`.
