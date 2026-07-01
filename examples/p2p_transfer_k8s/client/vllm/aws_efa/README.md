# ModelExpress on AWS EFA (libfabric NIXL backend)

This example deploys a vLLM + ModelExpress pod on AWS EKS / dgxc-k8s clusters
that have EFA-capable nodes, using the libfabric NIXL backend for GPU-to-GPU
RDMA over the EFA fabric. Scale the Deployment to N>=2 to demonstrate P2P
weight transfer between replicas.

The manifest uses `deepseek-ai/DeepSeek-V4-Pro` and requires an EFA-enabled
Dynamo runtime built with vLLM 0.23.0.

The default NIXL backend (`UCX`) is tuned for InfiniBand and RoCE clusters. On
AWS EFA, UCX can silently fall back to TCP depending on the libibverbs / EFA
installer combination on the host - `MX_NIXL_BACKEND=LIBFABRIC` selects the
libfabric plugin instead, which is the supported AWS path.

## Prerequisites

- ModelExpress server deployed (see [`../../server/`](../../server/)).
- AWS EKS or dgxc-k8s cluster with EFA-capable nodes (e.g. `p5.48xlarge` for
  H100/EFAv2, `p6e-gb200.36xlarge` for GB200/EFAv3).
- `aws-efa-k8s-device-plugin` DaemonSet running on the cluster. Exposes the
  `vpc.amazonaws.com/efa` extended resource on EFA nodes.
- EFA driver `>= 3.0.0g` installed on the worker nodes. The 2.17.3g driver
  that ships on default P4d / P5 AMIs is not sufficient.
- `nvidia-peermem` kernel module loaded on the worker nodes (required for
  GPUDirect RDMA). Verify with `lsmod | grep nvidia_peermem`.
- HuggingFace token secret in the target namespace:
  ```bash
  kubectl create secret generic hf-token-secret \
    --from-literal=HF_TOKEN=<your-token>
  ```

## Build

The example image extends the dynamo vLLM runtime's `-efa-amd64` variant, which
already includes the AWS EFA installer, libfabric provider, and aws-ofi-nccl.
We layer the ModelExpress Python client on top.

From the repository root:

```bash
docker build \
  -f examples/p2p_transfer_k8s/client/vllm/aws_efa/Dockerfile \
  --build-arg DYNAMO_VLLM_EFA_RUNTIME_IMAGE=<efa-runtime-with-vllm-0.23.0> \
  -t <YOUR_REGISTRY>/modelexpress-aws-efa:latest \
  .

docker push <YOUR_REGISTRY>/modelexpress-aws-efa:latest
```

Edit the `image:` field in `vllm-aws-efa.yaml` to point at your pushed tag.
The manifest defaults to `deepseek-ai/DeepSeek-V4-Pro` with
`--tensor-parallel-size 8` and eight GPUs. If you switch models, update the
tensor-parallel size and GPU request/limit to match that model.

## Deploy

```bash
kubectl apply -f examples/p2p_transfer_k8s/client/vllm/aws_efa/vllm-aws-efa.yaml
```

The first replica loads the model from HuggingFace and registers itself as a
P2P source with the ModelExpress server. Scale up to see a second replica pull
weights from the first over EFA via NIXL:

```bash
kubectl scale deploy/mx-vllm-aws-efa --replicas=2
```

## Verify

The libfabric backend is selected at MX startup, before any transfer. Confirm
on the pods:

```bash
kubectl logs deploy/mx-vllm-aws-efa | grep 'NIXL agent.*created'
```

Lines should contain `(backend=LIBFABRIC)`. If they say `(backend=UCX)`,
`MX_NIXL_BACKEND` is not propagating.

To confirm that the transfer actually used the EFA fabric (and not a silent
TCP fallback), look for the libfabric provider's EFA memory-registration
log lines:

```bash
kubectl logs deploy/mx-vllm-aws-efa | grep 'efa:mr:efa_mr_reg_impl' | head -3
```

These lines come from libfabric's EFA provider registering CUDA memory
against an `ibv pd` (a real EFA protection domain). Their presence is
direct evidence that the EFA fabric is being used; their absence means
libfabric chose a different provider or the registration path failed.

Note: the standard Mellanox approach of checking
`/sys/class/infiniband/*/ports/1/hw_counters/rdma_write_bytes` does not
work on EFA. The AWS EFA driver does not expose hw_counters in sysfs at
all; the libfabric provider log is the verification path.

The other strong signal is the transfer-complete log line on a target
replica (replica 2+):

```bash
kubectl logs deploy/mx-vllm-aws-efa | grep 'Transfer complete'
```

It reports tensor count, total bytes, and elapsed time, e.g.
`Transfer complete: 674 tensors, 16.27 GB in 1.76s (73.8 Gbps)`. The
throughput floor implied by that number is well above any plausible TCP
fallback for a single cross-pod transfer, so it doubles as a sanity check.

## EFA generation notes

| Instance | GPU | EFA generation | NICs per node | Per-NIC rate |
| --- | --- | --- | --- | --- |
| `p5.48xlarge` | 8x H100 | EFAv2 | 32 | 100 Gb/s |
| `p6e-gb200.36xlarge` | 4x GB200 | EFAv3 | 4 | 400 Gb/s |

Adjust the `vpc.amazonaws.com/efa` resource count in the manifest to match
the instance type. On `p6e-gb200`, requesting more than 4 EFA NICs per pod
prevents scheduling. On `p5.48xlarge`, requesting all 32 NICs claims the
whole node; pick a smaller number for shared workloads. Note that the AWS
EFA k8s device plugin does not isolate NICs per-pod the way nvidia-device-plugin
isolates GPUs: the resource request gates scheduling, but the running
container sees every EFA NIC on its host node.

## Architecture notes

The base passed through `DYNAMO_VLLM_EFA_RUNTIME_IMAGE` determines the target
architecture. Use an amd64 EFA image for AWS EFAv2 H100 instances (`p5.*`) or
an arm64 EFA image for AWS EFAv3 GB200 instances (`p6e-*`). The wrapper
Dockerfile rejects images that do not contain vLLM 0.23.0.
