# TRT-LLM Dynamo Phase 2.5: Kimi K2.5 P2P on GCP GB200

**Status**: P2P VALIDATED via Dynamo engine — RoCE 25-33 Gbps on Qwen TP=2
**Date**: March 9, 2026 (updated)
**Branch (modelexpress)**: `kavink/trtllm`
**Branch (dynamo)**: `kavink/trtllm-p2p` (rebased on main with Kimi K2.5 recipe)
**Cluster**: `dynamo-gcp-dev-01` (GCP GB200 NVL36, ARM64)
**Namespace**: `kavin` (separate from karenc-dynamo)

---

## 1. Summary

Kimi K2.5 (589 GB, EP=4) P2P weight transfer validated end-to-end on GCP GB200
NVL36 cluster. All 4 ranks transfer 162 GB each with 100% param match and
verified checksums. Currently over TCP (~2-4 Gbps); pending RDMA enablement
for NVLink/RoCE speeds.

**Model**: [`baseten-admin/Kimi-2.5-text-nvfp4-v3`](https://huggingface.co/baseten-admin/Kimi-2.5-text-nvfp4-v3)
- Architecture: DeepSeek V3 family (same as DSV3.2 — `deepseek_v3`)
- 589 GB, 118 safetensors, NVFP4 quantized via ModelOpt
- MoE with 256 routed experts (same as DSV3.2)

**Validation Results (Kimi K2.5 NVFP4, EP=4, cross-node, GCP GB200):**

| Rank | Params | Data | Time | Bandwidth | Status |
|------|--------|------|------|-----------|--------|
| 0 | 1755/1755 | 162.09 GB | 635s | ~2.0 Gbps | PASS |
| 1 | 1755/1755 | 162.09 GB | 501s | ~2.6 Gbps | PASS |
| 2 | 1755/1755 | 162.09 GB | 365s | ~3.6 Gbps | PASS |
| 3 | 1755/1755 | 162.09 GB | 662s | ~2.0 Gbps | PASS |
| **Total** | **7020** | **648.36 GB** | **~11 min** | TCP fallback | **ALL PASS** |

Transfer is over TCP because RDMA transports are not available to UCX (see
Section 10 for details). With RDMA (RoCE or MNNVL), expect ~3-15 seconds
based on H200 results.

---

## 2. What We Built

### 2.1 ARM64 Docker image with ModelExpress

**Image**: `nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx-v1.0.0`
**SHA**: `386c3e6fc8aa5cf775f7108544ebb5c44b47617af0bbdff100ad8664312a8b57`

Built by layering on karenc's working Dynamo v1.0.0 image. We tried the full
Dynamo build system (`render.py --platform arm64`) first, but cross-compiling
a 931-line Dockerfile via QEMU was too slow (canceled during ffmpeg compilation
after 20+ minutes). The layered approach completed in ~10 minutes.

**Base**: `nvcr.io/nvidian/dynamo-dev/karenc:dynamo-trtllm-v1.0.0-a9b6f95`
- TRT-LLM **1.3.0rc5** (newer than our previous patches)
- NIXL pre-installed and functional
- ARM64 (aarch64) for GB200 GPU nodes
- Dynamo worker (`python3 -m dynamo.trtllm`)

**Layers added** (`Dockerfile.ph3-gcp-gb200`):
1. `USER root` — base image runs as non-root
2. ModelExpress client v0.2.2 (`pip install` from source)
3. gRPC proto generation (`p2p.proto` → `p2p_pb2.py`, `p2p_pb2_grpc.py`)
4. Import fix (`sed` to convert absolute to relative imports in generated gRPC)
5. PRESHARDED patches via `apply_patches.py`
6. Verification step (`grep` for PRESHARDED — can't import TRT-LLM without GPU)

**Build command**:
```bash
cd /home/kavink/work/github/modelexpress
docker buildx build --platform linux/arm64 --no-cache \
    -f examples/p2p_transfer_trtllm/Dockerfile.ph3-gcp-gb200 \
    -t nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx-v1.0.0 \
    --push .
```

### 2.2 TRT-LLM 1.3.0rc5 PRESHARDED patches

File structure changed significantly from 1.3.0rc3 to 1.3.0rc5:

| File | v1.3.0rc3 path | v1.3.0rc5 path |
|------|----------------|----------------|
| `llm_args.py` | `llmapi/llm_args.py` | `llmapi/llm_args.py` (same) |
| `model_loader.py` | `_torch/models/model_loader.py` | `_torch/pyexecutor/model_loader.py` |
| `linear.py` | `_torch/models/linear.py` | `_torch/modules/linear.py` |

Instead of full file replacements, we created `apply_patches.py` — a Python
script that performs surgical text replacements on the installed TRT-LLM files:

- `llm_args.py`: Adds `PRESHARDED = 3` after `VISION_ONLY = 2`
- `model_loader.py`: Inserts PRESHARDED branch before VISION_ONLY branch
- `linear.py`: Adds `_tp_size` variable to 3 helper functions, replaces
  `module.tp_size` with `_tp_size` in `load_weight_shard()` calls

**File**: `trtllm_patches/v1.3.0rc5/apply_patches.py`

### 2.3 Build issues resolved

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `p2p_pb2.py: Permission denied` | Base image runs as non-root, COPY'd files read-only | Added `USER root` |
| `chmod: Operation not permitted` | Same non-root issue, `chmod` also fails | `USER root` (not chmod) |
| `import p2p_pb2` (absolute import) | `grpc_tools.protoc` generates absolute imports | `sed -i 's/^import p2p_pb2/from . import p2p_pb2/'` |
| `libcuda.so.1: cannot open` | No GPU in Docker buildx QEMU sandbox | Changed verify to `grep` instead of Python import |
| Full Dynamo build too slow | 931-line Dockerfile cross-compiling via QEMU | Switched to layered approach on karenc's image |

### 2.4 Dynamo repo rebased

Rebased `kavink/trtllm-p2p` onto latest `main` (clean, no conflicts):
- Now includes Kimi K2.5 recipe (commit `62ec9f5b0`)
- Our commit: `4d48d97cc feat: add ModelExpress P2P weight transfer support`
- Dynamo `main` has `enable_modelexpress_p2p` build arg (TRT-LLM 1.3.0rc3)

---

## 3. Current Deployment (namespace `kavin`)

### 3.1 Infrastructure

| Pod | Status | Node Pool | Purpose |
|-----|--------|-----------|---------|
| `modelexpress-server` | Running | customer-cpu (amd64) | gRPC metadata coordinator |
| `redis` | Running | customer-cpu (amd64) | Persistent metadata store |
| `kimi-source` | Running (loading) | customer-gpu (arm64) | Kimi 2.5 source, EP=4, single-node |

### 3.2 Resources

| Resource | Details |
|----------|---------|
| Namespace | `kavin` (on `dynamo-gcp-dev-01`) |
| Secrets | `nvcr-imagepullsecret`, `hf-token-secret` |
| PVC | `model-cache` (1500Gi, `filestore-rwx`) |

### 3.3 Source pod status — PUBLISHED (stable)

The source pod runs 4 MPI ranks via `mpirun -np 4` on 4 GB200 GPUs (EP=4).
All 4 ranks load model weights and publish NIXL metadata. Source sleeps
indefinitely after publishing, holding weights in GPU memory for RDMA reads.

**Loading timeline** (per deployment cycle):
1. Config + MetaInit + materialize: ~50s
2. Safetensors parallel loading (118 files from PVC): ~22 min
3. `model.load_weights()` (1468 params): ~50 min
4. NIXL registration (1755 tensors/rank): ~1s
5. Gather + publish (all 4 workers in single gRPC call): ~1s

**Total source startup**: ~75 minutes

### 3.4 Target pod status — P2P VALIDATED

Standalone target pod creates the same model structure (MetaInit, no disk
weights), then receives all weights via NIXL from the source. Validates
by comparing pre/post checksums.

**Target timeline**:
1. Model structure creation: ~51s
2. NIXL transfer (162 GB/rank, 4 ranks parallel): ~6-11 min (TCP)
3. Checksum validation: ~10s

### 3.4 Deployment files

| File | Purpose |
|------|---------|
| `examples/p2p_transfer_trtllm/Dockerfile.ph3-gcp-gb200` | ARM64 image (layered) |
| `trtllm_patches/v1.3.0rc5/apply_patches.py` | Patch script for TRT-LLM 1.3.0rc5 |
| `examples/p2p_transfer_trtllm/deploy/gcp/mx-infra.yaml` | MX server + Redis |
| `examples/p2p_transfer_trtllm/deploy/gcp/kimi-source.yaml` | Kimi source (EP=4) |
| `examples/p2p_transfer_trtllm/deploy/gcp/kimi-target-standalone.yaml` | Standalone P2P validation target |
| `examples/p2p_transfer_trtllm/deploy/gcp/kimi-target-agg-mx.yaml` | Target DGD (prepared, not yet tested) |

---

## 4. Cluster Architecture

### 4.1 Node pools

| Pool | Arch | Taints | GPU | Reachable from GPU? |
|------|------|--------|-----|---------------------|
| `system-cpu` | amd64 | `system-workload` | No | **No** (network isolated) |
| `customer-cpu` | amd64 | `user-workload` | No | **Yes** |
| `customer-gpu-*` | arm64 | `user-workload` + `nvidia.com/gpu` + `kubernetes.io/arch` | 4x GB200 | Yes |

### 4.2 GPU allocation model

This cluster uses **DRA (Dynamic Resource Allocation)** with ComputeDomain:
- Some GPU nodes in `customer-gpu-9pb` show `nvidia.com/gpu: 0` (DRA only)
- Most nodes in pools `customer-gpu-mh2`, `customer-gpu-o7v`, `customer-gpu-w0e`
  expose GPUs via traditional `nvidia.com/gpu: 4`
- karenc's DGDs use `resourceClaims` with `karenc-compute-domain-channel`
- Our pods use traditional `nvidia.com/gpu: 4` — works on most GPU nodes

### 4.3 NVLink / MNNVL topology

Each node has 4x GB200 GPUs connected via **NV18 (18-lane NVLink)** intra-node.
Nodes within the same **clique** (`nvidia.com/gpu.clique` label) are connected
via **MNNVL (Multi-Node NVLink)** through NVSwitch at ~956 GB/s per GPU.

Each node also has **4x mlx5 NICs** (400 Gb/sec HDR Ethernet, RoCE-capable):
- Visible in `/sys/class/infiniband/mlx5_0` through `mlx5_3`
- IB verbs devices exist (`uverbs0-3` in sysfs)
- But `/dev/infiniband/` character device nodes are NOT created on the host
- UCX cannot access verbs transports (rc, dc) without device nodes

See Section 10 for RDMA investigation details.

### 4.3 Teleport access

```bash
tsh kube login dynamo-gcp-dev-01
# Context: nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-01
# Sessions expire frequently — re-login as needed
```

---

## 5. Reference Deployment (karenc's recipes)

### 5.1 Source of truth

GitLab: [`karenc/dynamo-workflows` — `k8s/gcp_gb200_setup/recipes/`](https://gitlab-master.nvidia.com/karenc/dynamo-workflows/-/tree/main/k8s/gcp_gb200_setup/recipes?ref_type=heads)

### 5.2 Working recipes

| File | Mode | Image | Status |
|------|------|-------|--------|
| `kimi-agg-rr.yaml` | Aggregated, round-robin | `karenc:dynamo-trtllm-v1.0.0-a9b6f95` | Working (DUMMY) |
| `kimi-disagg-kv-wideep.yaml` | Disagg (prefill + decode) | Same | Working (DUMMY) |

### 5.3 Key engine config

```yaml
tensor_parallel_size: 8
moe_expert_parallel_size: 8
enable_attention_dp: true
load_format: DUMMY       # << Replace with P2P
allreduce_strategy: MNNVL
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.75
model_kwargs:
  num_hidden_layers: 61
moe_config:
  backend: TRTLLM
  use_low_precision_moe_combine: true
trust_remote_code: true
```

---

## 6. Bugs Fixed During Validation

### 6.1 torch.distributed init hang (TRT-LLM 1.3.0rc5)

TRT-LLM 1.3.0rc5 model initialization triggers NCCL collectives that require
`torch.distributed`. On 1.3.0rc1 (H200), MPI alone was sufficient.

**Fix**: Add `torch.distributed.init_process_group(backend="nccl")` before
model creation:
```python
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
```

### 6.2 NCCL heartbeat abort during sleep

After publishing, the source enters `time.sleep(3600)`. The NCCL heartbeat
monitor runs a background thread that checks the TCPStore. When the main
thread sleeps, the TCPStore server becomes unresponsive, and NCCL aborts
with signal 6.

**Fix**: Destroy torch.distributed after publishing:
```python
dist.destroy_process_group()
```
NCCL is only needed for model creation. After publishing, we just hold
GPU memory for RDMA reads.

### 6.3 MX server worker overwrite (v0.2.2)

The deployed MX server (`modelexpress-server:latest` v0.2.2) predates the
Lua atomic merge fix (PR #135). Each rank's `PublishMetadata` call
overwrites the previous one instead of merging. Only the last rank's
metadata survives.

**Fix**: Gather all 4 workers' metadata on rank 0 via `comm.gather()`,
then publish ALL workers in a single gRPC `PublishMetadata` call:
```python
my_worker_bytes = my_worker.SerializeToString()
all_worker_bytes = comm.gather(my_worker_bytes, root=0)
if rank == 0:
    # Single publish with all workers
    request = PublishMetadataRequest(model_name=..., workers=all_workers)
    stub.PublishMetadata(request)
comm.Barrier()
```

## 7. Next Steps

### 7.1 Enable RDMA transport

Two paths being investigated (see Section 10):

**Path A: RoCE via IB verbs** — Need `/dev/infiniband/uverbs*` device nodes
created on GPU hosts. Asked cluster admin. Would give UCX `rc`/`dc` transports
over the 400 Gbps mlx5 NICs.

**Path B: MNNVL via IMEX daemon** — Need NVIDIA IMEX daemon running on GPU
nodes for cross-node CUDA IPC over NVSwitch. Asked cluster admin. Would give
NVLink speeds (~956 GB/s per GPU) via UCX `cuda_ipc` transport.

### 7.2 DGD target integration

Deploy `kimi-target-agg-mx.yaml` as a DynamoGraphDeployment with
`--model-express-url`. Requires Dynamo platform in `kavin` namespace
(etcd + NATS + operator).

### 7.3 TRT-LLM PyTorch backend P2P integration

See Section 11 for implementation plan.

---

## 8. Prior Art (What's Already Validated)

| Model | TP | Transfer | Bandwidth | Cluster | Status |
|-------|-----|----------|-----------|---------|--------|
| Qwen 0.5B (v1.3.0rc3) | 1 | 1.26 GB | 151.9 Gbps | Nebius H200 | PASSED |
| Qwen 0.5B (v1.2.0rc6) | 1 | 1.26 GB | 78.8 Gbps | Nebius H200 | PASSED |
| Llama 70B (v1.2.0rc6) | 8 | 141.1 GB | 368 Gbps | Nebius H200 | PASSED |
| DSV3.2 NVFP4 EP=8 (H200) | 8 | 547.6 GB | 2,227 Gbps | Nebius H200 | PASSED |

---

## 9. nscale B200 Status (Paused)

Paused due to:
1. Source pod crash-looping (47 restarts, MPI rank 4 exit code 1)
2. Slow NIXL registration (45+ min due to 12 IB devices)
3. Python error from rank 4 lost in UCX debug log overflow

See `TRTLLM_TESTING_CONTEXT.md` for full resume context.

---

## 10. Differences from Previous Attempts

| Aspect | nscale B200 | GCP GB200 |
|--------|-------------|-----------|
| Architecture | x86_64 | ARM64 |
| GPU | B200 (Blackwell) | GB200 (Blackwell NVL36) |
| GPUs/node | 8 | 4 |
| Interconnect | 12x mlx5 InfiniBand | MNNVL (NVLink) |
| TRT-LLM | 1.3.0rc1 | 1.3.0rc5 |
| Deployment | Raw k8s Deployments | Raw + DGD planned |
| Model | DSV3.2 NVFP4 (337 GB) | Kimi 2.5 NVFP4 (589 GB) |
| GPU allocation | Standard `nvidia.com/gpu` | DRA/ComputeDomain (mostly) |
| Namespace | `kavin` (own) | `kavin` (own, separate from karenc) |
| Current status | Crash-looping | P2P validated (TCP) |

---

## 11. RDMA Transport Investigation

### 11.1 Current state

UCX in the container only discovers `tcp`, `cuda_copy`, `cuda_ipc`, and basic
shared-memory transports. No IB verbs (`rc`, `dc`) transports available.

### 11.2 RoCE path (IB verbs)

Hardware exists but device nodes are not created:
- `/sys/class/infiniband/mlx5_0-3` — 4 devices, ACTIVE, 400 Gb/sec HDR Ethernet
- `/sys/class/infiniband_verbs/uverbs0-3` — verbs devices in sysfs
- `/dev/infiniband/` — does NOT exist on host
- libibverbs installed in container

To enable: need `/dev/infiniband/uverbs*` character device nodes on the GPU
hosts (via MOFED, rdma-core, or NVIDIA GPU Operator RDMA plugin). Once
available, mount as hostPath and UCX will use `rc`/`dc` verbs over RoCE.

Asked cluster admin.

### 11.3 MNNVL path (cross-node CUDA IPC)

Both source and target are in the same NVLink clique. NVLink active at
18 lanes x 53.125 GB/s = ~956 GB/s per GPU.

UCX supports MNNVL via `UCX_CUDA_IPC_ENABLE_MNNVL=y` (default: `try`).
Tested with `ucx_perftest` — still falls back to TCP. Root cause:
**NVIDIA IMEX daemon is not running** on the GPU nodes.

IMEX is required for cross-node CUDA IPC handle exchange over NVSwitch.
Without it, UCX `cuda_ipc` is limited to intra-node.

Note: karenc's NCCL-based MNNVL allreduce works because NCCL accesses
NVLink through the NVIDIA driver directly, bypassing UCX.

Asked cluster admin for IMEX daemon enablement.

### 11.4 karenc's DGD comparison

Inspected karenc's running worker pods:
- Also NO `/dev/infiniband/`
- UCX sees same transports (tcp, cuda_ipc only)
- MNNVL works via NCCL (driver-level, not UCX)
- Uses DRA `resourceClaim: compute-domain-channel` for GPU allocation

---

## 12. TRT-LLM PyTorch Backend P2P Implementation Plan

### 12.1 Context

The current P2P implementation uses a standalone source script that manually
creates a TRT-LLM model via the PyTorch backend (`AutoModelForCausalLM`,
`HfCheckpointLoader`, `MetaInitMode`). This works but is separate from the
Dynamo TRT-LLM worker (`python3 -m dynamo.trtllm`).

The goal is to integrate P2P into the Dynamo TRT-LLM worker so both source
and target run as DGDs, mirroring the vLLM integration (PR #6186).

### 12.2 vLLM pattern (for reference)

In vLLM, both source and target are DGDs running `python3 -m dynamo.vllm`:
- Source: `--load-format mx-source` loads from disk, registers with NIXL,
  publishes, then serves inference normally
- Target: `--load-format mx-target` creates dummy weights, receives via RDMA,
  then serves inference normally
- Both are fully functional inference endpoints

### 12.3 TRT-LLM PyTorch backend architecture

TRT-LLM's PyTorch backend (`_torch/`) differs from vLLM:
- Model loading: `ModelLoader.load()` in `_torch/pyexecutor/model_loader.py`
- Checkpoint system: `HfCheckpointLoader` + `LoadFormat` enum
- Weight injection: `LoadFormat.PRESHARDED` skips TP sharding, passes `model`
  to `load_weights()`
- Worker parallelism: MPI-based (not Python multiprocessing like vLLM)

### 12.4 Source implementation

**Option A: Source as DGD (recommended long-term)**

The source runs as a normal Dynamo TRT-LLM worker that loads weights from
disk AND publishes them via NIXL. After `ModelLoader.load()` completes,
weights are already in GPU memory. A post-load hook:
1. Iterates `model.named_parameters()`
2. Registers with NIXL
3. Publishes metadata to MX server

Triggered by `--model-express-url` on the Dynamo TRT-LLM worker. The source
serves inference AND holds weights for RDMA (warm source pool).

**Option B: Standalone source (current approach)**

Keep the current standalone script. Simpler, but source doesn't serve
inference.

### 12.5 Target implementation

Already partially implemented via `MxLiveCheckpointLoader`:

1. Dynamo TRT-LLM worker starts with `--model-express-url`
2. `_setup_modelexpress_loader()` injects `MxLiveCheckpointLoader`
3. `LoadFormat.PRESHARDED` is set
4. `ModelLoader.load()` calls `MxLiveCheckpointLoader.load_weights(model=model)`
5. `MxLiveWeightLoader` does NIXL RDMA directly into model param buffers
6. Returns `{}` — weights already in place, skip `model.load_weights()`
7. Engine compilation proceeds with transferred weights

### 12.6 Key differences from vLLM integration

| Aspect | vLLM | TRT-LLM PyTorch |
|--------|------|-----------------|
| Loader plugin | `@register_model_loader("mx-target")` | `checkpoint_loader=MxLiveCheckpointLoader()` |
| Weight format | Raw tensors (pre-FP8 processing) | Fused, TP-sharded, TRT-LLM format |
| Worker spawn | Python multiprocess (`ModelExpressWorker`) | MPI (NIXL works natively) |
| Post-transfer | `process_weights_after_loading()` | Engine compilation |
| TRT-LLM patches | None | 4 files: `llm_args.py`, `model_loader.py`, `linear.py`, `worker.py` |

---

## 13. Phase 2.5b: DGD Source Publish Fix (March 8-9, 2026)

### 13.1 Problem: Source crash-loop with TRT-LLM RPC executor

The Dynamo engine (`python3 -m dynamo.trtllm --model-express-role source`) crashed because `LLM()` with TP>1 creates `GenerationExecutorProxy` (MPI proxy). The proxy has no `model_engine` — the real torch model lives in MPI worker processes. `MxLiveSource._get_torch_model()` always failed, retries exhausted, `/live` returned 503, pod restarted.

### 13.2 Fix: Worker-side publish with MPI gather

Implemented across 3 repos:

- **modelexpress (`trtllm_live_transfer.py`):** New `publish_from_worker(worker)` — accesses `worker.engine.model_engine.model`, registers with NIXL, uses `MPI.COMM_WORLD.gather()` to collect all workers on rank 0, then rank 0 makes ONE `PublishMetadata` gRPC call with all workers.
- **TRT-LLM (`worker.py` via `apply_patches.py`):** After worker creation in `worker_main()`, checks `MODEL_EXPRESS_SOURCE` env var and calls `publish_from_worker(worker)`.
- **Dynamo engine (`engine.py`):** `_setup_modelexpress_source()` sets `MODEL_EXPRESS_SOURCE=1` env before `LLM()` init. After init, `_verify_workers_published()` polls MX server (300s timeout).

### 13.3 Image and deployment

**Image**: `nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx-v1.5.0`

**TRT-LLM patches** (`trtllm_patches/v1.3.0rc5/apply_patches.py`): `llm_args.py`, `model_loader.py`, `linear.py`, `worker.py`

**YAMLs** (DGD + plain Deployment fallbacks):

| File | Type | Role |
|------|------|------|
| `deploy/gcp/kimi-source-dgd.yaml` | DGD | Source (operator) |
| `deploy/gcp/kimi-source-deploy.yaml` | Deployment | Source (no operator) |
| `deploy/gcp/kimi-target-agg-mx.yaml` | DGD | Target (operator) |
| `deploy/gcp/kimi-target-deploy.yaml` | Deployment | Target (no operator) |

**Critical env vars** (must be in pod spec for MPI propagation):
`MODEL_EXPRESS_SOURCE`, `MODEL_EXPRESS_URL`, `MODEL_NAME`, `WORLD_SIZE`, `NATS_SERVER`, `ETCD_ENDPOINTS`

### 13.4 Source publish validated

All 4 workers published via MPI gather + single gRPC call:
- 4 × 1,815 params, 4 × 162.09 GB
- Redis: 4 workers, 7,260 tensors, 4 × 486,722 bytes NIXL metadata
- `_verify_workers_published`: confirmed all 4 on first poll

### 13.5 Target UCX connectivity — RESOLVED

Target UCX connection to source failed with `Connection refused` on the NIXL UCX port.

**Root cause:** `publish_from_worker()` created `NixlTransferManager` as a local variable. When the function returned, Python GC destroyed it, shutting down the UCX agent and closing the listening port. Metadata in Redis still referenced the dead port.

**Fix:** `worker._mx_nixl_manager = nixl_mgr` — keep reference alive for the process lifetime.

### 13.6 NCCL cluster-wide failures — WORKAROUND

After the UCX fix, source pods crash during autotuning/CUDA graph warmup with NCCL `unhandled system error`. All 4 ranks fail in MoE allgather. Cluster-wide GPU fabric issue — not node-specific (failed on both `o7v-6b99` and `o7v-2x2c`).

**Why NCCL runs:** `LLM()` does autotuning + CUDA graph warmup with real forward passes that use NCCL for TP. The standalone MPI source never runs forward passes so it never hits NCCL.

**Workaround:** Disable autotuner and CUDA graphs in source config:
```yaml
enable_autotuner: false
cuda_graph_config:
  max_batch_size: 0
```

### 13.7 Fixes discovered during debugging

**NCCL `/dev/shm`:** Plain Deployments need explicit `/dev/shm` volume (16+ Gi). DGD operator auto-injects it. Without it, `preallocateNCCLWindowBuffer` fails (NCCL needs ~34 MB per shm segment, K8s default is 64 MB).

**NIXL agent lifetime:** `publish_from_worker()` must store `nixl_mgr` on the worker object (`worker._mx_nixl_manager`). Otherwise Python GC destroys the agent, closing the UCX listening port.

**`cuda_ipc` breaks GB200:** `cuda_ipc` in `UCX_TLS` causes `cuIpcOpenMemHandle() failed: invalid device context` → `NIXL_ERR_NOT_ALLOWED`. Remove `cuda_ipc` to use host-staged RoCE RDMA instead.

### 13.8 RoCE P2P transfer — VALIDATED

**Qwen 0.5B TP=2 end-to-end via Dynamo engine on GCP GB200:**

| Rank | Params | Data | Time | Speed | Transport |
|------|--------|------|------|-------|-----------|
| 0 | 171/171 | 0.63 GB | 0.20s | **24.8 Gbps** | rc_mlx5 (RoCE) |
| 1 | 171/171 | 0.63 GB | 0.15s | **33.2 Gbps** | rc_mlx5 (RoCE) |

UCX bonded across `mlx5_0:1` + `mlx5_1:1` for zero-copy rendezvous.

**Critical config:**
```yaml
securityContext:
  privileged: true
env:
  UCX_TLS: "rc_v,rc_x,rc,dc_x,dc,cuda_copy,tcp"   # NO cuda_ipc
  UCX_RNDV_SCHEME: "get_zcopy"
  UCX_RNDV_THRESH: "0"
volumes:
  /dev/shm (emptyDir, 16Gi+)
  /dev/infiniband (hostPath)
resourceClaims:
  compute-domain-channel (for GPU allocation via DRA)
affinity:
  topologyKey: nvidia.com/gpu.clique
```

### 13.9 Speed comparison

| Config | UCX Transport | Speed |
|--------|--------------|-------|
| `runAsUser: 0` + computeDomain | TCP | 3.5 Gbps |
| `privileged` + `cuda_ipc` in UCX_TLS | Fails (`cuIpcOpenMemHandle` error) | N/A |
| `privileged` + NO `cuda_ipc` | **RoCE (`rc_mlx5`)** | **25-33 Gbps** |

### 13.10 Next steps

1. Deploy Kimi K2.5 TP=4 with RoCE config — validate at scale (162 GB/rank)
2. Enable coalesced transfers for higher throughput
3. Test DGD operator path (fix webhook first)
4. Upstream: fix `cuda_ipc` on GB200 in NIXL for GPU-direct RDMA
5. Upstream: Dynamo operator `kube-rbac-proxy` registry fix

### 13.11 Operator improvements

See [OPERATOR_LLM.md](./OPERATOR_LLM.md) for suggestions.
