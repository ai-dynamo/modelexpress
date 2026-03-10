# Kimi K2.5 P2P Weight Transfer on GCP GB200

Fast GPU-to-GPU weight loading for Kimi K2.5 (589 GB, EP=4) via ModelExpress NIXL RDMA.
A target instance loads weights from a running source in ~30s instead of ~75 min from disk.

## Quick Start

### Prerequisites

```bash
# 1. Namespace with Dynamo platform
kubectl -n kavin get pods  # should show etcd, nats running

# 2. ModelExpress server + Redis
kubectl -n kavin apply -f mx-infra.yaml

# 3. ComputeDomain (for GPU allocation via DRA)
cat <<EOF | kubectl -n kavin apply -f -
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: kavin-compute-domain
spec:
  numNodes: 0
  channel:
    resourceClaimTemplate:
      name: kavin-compute-domain-channel
EOF

# 4. Secrets
kubectl -n kavin get secret hf-token-secret        # HuggingFace token
kubectl -n kavin get secret nvcr-imagepullsecret    # NVCR pull secret
```

### Deploy Source (loads from disk, publishes weights)

```bash
# Flush any stale metadata
kubectl -n kavin exec deploy/redis -- redis-cli FLUSHALL

# Deploy source — takes ~75 min to load Kimi K2.5
kubectl -n kavin apply -f kimi-source-deploy.yaml

# Monitor progress
kubectl -n kavin logs -f deploy/kimi-source-deploy
# Look for: "ModelExpress source: all 4 workers published"
```

### Deploy Target (receives weights via RDMA)

```bash
# Only after source publishes successfully
kubectl -n kavin apply -f kimi-target-deploy.yaml

# Check per-rank transfer logs
kubectl -n kavin exec deploy/kimi-target-deploy -- cat /tmp/mx_logs/rank0.log
# Expected: "transferred 1815 params (162.09 GB) in Xs (Y Gbps)"
```

## Image

```
nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx-v1.5.0
```

Built from `Dockerfile.ph3-gcp-gb200` on top of karenc's `dynamo-trtllm-v1.0.0-a9b6f95`.
Includes ModelExpress client, TRT-LLM PRESHARDED patches, and worker-side publish hook.

## Deployment Options

| File | Type | Use When |
|------|------|----------|
| `kimi-source-deploy.yaml` | Plain Deployment | Operator unavailable |
| `kimi-target-deploy.yaml` | Plain Deployment | Operator unavailable |
| `kimi-source-dgd.yaml` | DynamoGraphDeployment | Operator running |
| `kimi-target-agg-mx.yaml` | DynamoGraphDeployment | Operator running |
| `qwen-source-deploy.yaml` | Plain Deployment | Fast testing (TP=2, loads in 30s) |
| `qwen-target-deploy.yaml` | Plain Deployment | Fast testing |

## Required Pod Config for GB200

```yaml
securityContext:
  privileged: true                    # for /dev/infiniband access

env:
  UCX_TLS: "rc_v,rc_x,rc,dc_x,dc,cuda_copy,tcp"   # NO cuda_ipc
  OMPI_MCA_pml: "ob1"                               # avoid UCX UD timeout
  OMPI_MCA_btl: "tcp,self,vader"                     # MPI over TCP+shmem

volumes:
  /dev/shm:  emptyDir (100Gi, Memory)               # NCCL shared memory
  /dev/infiniband: hostPath                          # RoCE devices

resourceClaims:
  compute-domain-channel                             # GPU allocation via DRA

affinity:
  topologyKey: nvidia.com/gpu.clique                 # same NVLink domain
```

## Key Findings

- **No `cuda_ipc` in UCX_TLS** — `cuIpcOpenMemHandle` fails on GB200, causing NIXL to reject remote metadata. Remove `cuda_ipc` to use host-staged RoCE RDMA instead.
- **`/dev/shm` required** — NCCL needs >64MB for shared memory segments. K8s default is 64MB. Must mount explicit emptyDir.
- **`OMPI_MCA_pml=ob1`** — UCX UD endpoint times out during MPI bootstrap with TP=4 on degraded nodes. Force MPI to use TCP+vader instead. Does not affect NCCL or NIXL performance.
- **ComputeDomain** — required for IMEX channels on GB200. Without it, NIXL `loadRemoteMD` fails.

## Validated Results

| Model | TP | Transfer | Speed | Transport |
|-------|-----|----------|-------|-----------|
| Qwen 0.5B | 2 | 0.63 GB/rank | 25-33 Gbps | rc_mlx5 (RoCE) |
| Kimi K2.5 | 4 | 162 GB/rank | TBD | TBD |

## Operator Webhook Fix

If DGD operator webhook is down (`kube-rbac-proxy` image pull error):

```bash
kubectl -n dynamo-system patch deploy \
  dynamo-platform-dynamo-operator-controller-manager \
  --type='json' \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/image","value":"registry.k8s.io/kubebuilder/kube-rbac-proxy:v0.15.0"}]'
```

## Branches

- **modelexpress:** `kavink/trtllm` on `github.com:ai-dynamo/modelexpress`
- **dynamo:** `kavink/trtllm-p2p` on `github.com:ai-dynamo/dynamo`
