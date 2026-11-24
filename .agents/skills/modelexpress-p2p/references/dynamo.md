# ModelExpress P2P Under NVIDIA Dynamo

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Dynamo owns the serving graph and worker lifecycle; ModelExpress (MX) runs
inside each worker and supplies loading, source discovery, and P2P transfer.
Authoritative sources: `docs/integrations/orchestrators/dynamo.md` and
`examples/dynamo_p2p_transfer_k8s/README.md`.

This reference covers vLLM and SGLang workers. For TensorRT-LLM, see the
standalone section of `references/runtimes.md`; Dynamo's TensorRT-LLM runtime
image does not include the MX client.

## What Dynamo Provides And What You Add

| Piece | Provided by Dynamo | You add |
|---|---|---|
| MX client in the worker image | Dynamo's vLLM and SGLang runtime images, when built with ModelExpress enabled | Nothing, if the image has it; otherwise layer it with `examples/dynamo_p2p_transfer_k8s/Dockerfile` |
| MX server | No; the Dynamo platform chart does not install it | Install it (`references/server-and-backends.md`) |
| Server address in pods | Optional: the operator injects `MODEL_EXPRESS_URL` when the platform chart sets `dynamo-operator.modelExpressURL` | The address in the DGD worker env |
| Worker load format | No | The runtime flag in the DGD worker args |

## MX Server Bring-Up

Install and health-check the server as in `references/server-and-backends.md`.
Then point the workers at it by setting both variables to the same value in
every worker's `envs`:

```yaml
envs:
  - name: MX_SERVER_ADDRESS
    value: "<server-service>.<server-namespace>.svc.cluster.local:8001"
  - name: MODEL_EXPRESS_URL
    value: "<server-service>.<server-namespace>.svc.cluster.local:8001"
```

The MX client reads `MODEL_EXPRESS_URL` before `MX_SERVER_ADDRESS`. The
operator's injected `MODEL_EXPRESS_URL` applies only to containers that do not
already set it, so setting it in the DGD keeps an operator-wide default from
silently overriding the DGD's own address.

## Worker Spec

Add to each worker service in the `DynamoGraphDeployment`:

vLLM (`python3 -m dynamo.vllm`):

```yaml
args:
  - --load-format
  - modelexpress
```

SGLang (`python3 -m dynamo.sglang`):

```yaml
args:
  - --load-format
  - remote_instance
  - --remote-instance-weight-loader-backend
  - modelexpress
  - --modelexpress-config
  - '{"transport":"nixl"}'
```

RDMA access, as in the checked-in examples (the resource name and count depend
on the cluster's device plugin):

```yaml
resources:
  limits:
    gpu: "4"
    custom:
      rdma/ib: "4"
extraPodSpec:
  mainContainer:
    securityContext:
      capabilities:
        add:
          - IPC_LOCK
```

With these load formats Dynamo skips its own weight prefetch for Hugging Face
model names and lets MX acquire the weights. Every worker still needs the
model's configuration and tokenizer files. Mount a shared model cache (as the
examples do with `HF_HOME` on a PVC) or download the non-weight files before
the worker starts.

Optional settings used by the examples (see `docs/DEPLOYMENT.md`, P2P
Environment Variables): `MX_RDMA_NIC_PIN=auto` for per-rank NIC pinning, and
`MX_ARTIFACT_TRANSFER=1` with `MX_ARTIFACT_READY_URL` to share compiled caches
as well as weights.

## Roll Out

1. Apply the DGD with one worker replica. It loads from storage and publishes
   itself as a source.
2. Wait for the DGD to be ready.
3. Scale the worker service:

   ```bash
   kubectl patch dgd <dgd-name> -n <namespace> --type merge \
     -p '{"spec":{"services":{"<WorkerService>":{"replicas":2}}}}'
   ```

In a disaggregated graph, prefill and decode workers of the same model publish
under the same source identity, so a new worker of either service can pull from
a running worker of either service.

## Confirm

Worker pods carry `nvidia.com/dynamo-graph-deployment-name=<dgd-name>` and
`nvidia.com/dynamo-component-type=worker`. On the replica started after the
first, the `main` container's log should contain `RDMA transfer complete`:

```bash
kubectl logs -n <namespace> <new-worker-pod> -c main | grep "RDMA transfer complete"
```

For multi-node workers, read the leader pod.

## Relation To `deploy-dynamo-recipe`

Dynamo's `deploy-dynamo-recipe` skill deploys an assigned DGD as handed off and
does not change its engine arguments or install cluster services. To use P2P
there, the DGD it receives must already carry the settings above, and the MX
server must already be running.
