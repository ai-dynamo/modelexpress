# Disaggregated TRT-LLM Serving with ModelExpress P2P

**Status**: Phase 1 (same TP) validated, Phase 2 (mixed TP) planned
**Date**: March 10, 2026

---

## 1. Overview

Disaggregated serving splits LLM inference into separate prefill and decode workers.
Each worker type can be independently scaled and optimized. ModelExpress P2P enables
new workers to load model weights via RDMA in seconds instead of from disk (~75 min
for Kimi K2.5).

This document covers:
- Phase 1: Same TP for prefill and decode (validated)
- Phase 2: Mixed TP with dual MX sources (near-term)
- Phase 3: Re-sharded P2P transfers (future optimization)

---

## 2. Phase 1: Same TP (Validated)

Both prefill and decode use TP=4, loading from a single MX source.

```
                    ┌──→ Prefill (TP=4, MX target) ──→ KV cache ──┐
MX Source (TP=4) ───┤                                              ├──→ Frontend
                    └──→ Decode  (TP=4, MX target) ──→ tokens ────┘
```

**Results (March 10, 2026):**

| Worker | Data | Time | Speed |
|--------|------|------|-------|
| Prefill (all ranks) | 648 GB | 5.09s | 255 Gbps |
| Decode (all ranks) | 648 GB | 5.53s | 234 Gbps |
| **Total concurrent** | **1.3 TB** | **~5.7s** | — |

YAML: `deploy/gcp/kimi-disagg-mx-dgd.yaml`

---

## 3. Phase 2: Mixed TP with Dual MX Sources

Production disagg configs use different TP sizes for prefill and decode.
For example, Karen's wideep recipe:

| Worker | TP | EP | Nodes | GPUs/node | MoE Backend |
|--------|-----|-----|-------|-----------|-------------|
| Prefill | 4 | 4 | 1 | 4 | TRTLLM |
| Decode | 8-32 | 8-32 | 2-8 | 4 | WIDEEP |

MX P2P requires source TP == target TP (tensor names match but per-rank sizes
differ at different TP). The simplest approach: **run two MX sources**, one per TP size.

### 3.1 Architecture

```
MX Source A (TP=4, 1 node)                        MX Server A
  • Loads Kimi K2.5 from disk (~75 min)    ──→     (gRPC + Redis)
  • Publishes 4 workers × 162 GB                       │
  • Holds GPU memory for RDMA reads                    │
                                                       ├──→ Prefill Workers (TP=4, MX target A)
                                                       │      • subComponentType: prefill
                                                       │      • DGDSA: kimi-disagg-prefill
                                                       │      • Scale: 1..N replicas
                                                       │
MX Source B (TP=8, 2 nodes)                       MX Server B
  • Loads Kimi K2.5 from disk (~75 min)    ──→     (gRPC + Redis)
  • Publishes 8 workers × 81 GB                        │
  • multinode MPI across 2 nodes                       │
                                                       └──→ Decode Workers (TP=8, MX target B)
                                                              • subComponentType: decode
                                                              • DGDSA: kimi-disagg-decode
                                                              • multinode: nodeCount: 2
                                                              • Scale: 1..N replicas
```

### 3.2 Infrastructure Requirements

| Component | TP=4 Source | TP=8 Source |
|-----------|-----------|-----------|
| GPUs | 4 (1 node) | 8 (2 nodes) |
| GPU Memory | 648 GB | 648 GB |
| MX Server | modelexpress-server-prefill:8001 | modelexpress-server-decode:8001 |
| Redis | redis-prefill | redis-decode |
| Load time | ~75 min | ~75 min |

Total source GPU overhead: 12 GPUs (4 for prefill source + 8 for decode source).

### 3.3 Multinode Support Plan

The decode service at TP=8 requires 2 nodes × 4 GPUs. The DGD operator handles
multinode via MPI/SSH — it wraps the worker command in `mpirun -n 8 -H host1,host2`
on the leader pod, while worker pods run SSH daemons.

#### 3.3.1 How multinode interacts with MX P2P

**Source side (publish)**: `publish_from_worker()` uses `MPI.COMM_WORLD.gather()`
which works correctly across nodes — MPI collectives are node-transparent. The
source publishes 8 workers with ranks 0-7, each with their NIXL metadata.

**Target side (receive)**: Each rank creates a NIXL agent, queries the MX server,
and does RDMA from the matching source rank. UCX `rc`/`dc` transports work
across nodes over RoCE. No `cuda_ipc` (intra-node only) needed.

**Operator wrapping**: The operator wraps the full command (including
`--model-express-role target` and `--model-express-url`) inside `mpirun` via
`trtllm-llmapi-launch`. No special handling of MX args needed.

#### 3.3.2 Code changes required in ModelExpress

**Bug: multinode rank mapping in `MxLiveWeightLoader.load_weights()`**

The current code uses `torch.cuda.current_device()` to determine the rank:

```python
device_id = torch.cuda.current_device()
my_workers = [w for w in source_meta.workers if w.worker_rank == device_id]
```

In multinode, rank 4 runs on node B but sees local GPU 0 →
`torch.cuda.current_device()` returns 0, not 4. This causes rank 4 to receive
rank 0's weights (wrong data).

**Fix** in `trtllm_live_transfer.py` (`MxLiveWeightLoader.load_weights`, ~line 472):

```python
# Use MPI rank for multinode correctness, fall back to GPU index for single-node
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except Exception:
    rank = torch.cuda.current_device()
device_id = torch.cuda.current_device()

my_workers = [w for w in source_meta.workers if w.worker_rank == rank]
```

Similarly, the NIXL agent must use `device_id` (local GPU) for GPU memory
operations but `rank` (MPI rank) for source worker matching.

**Same fix needed in `NixlTransferManager`**: the agent name and tensor
registration use `device_id` (correct — local GPU), but the source worker
lookup must use MPI rank.

#### 3.3.3 Infrastructure requirements for multinode

| Requirement | How | Notes |
|-------------|-----|-------|
| SSH key pair | Auto-generated by DGD operator | Secret `mpirun-ssh-key-<dgd-name>` |
| `HOME=/root` | Add to pod env | Image has `USER dynamo`, sshd expects `~/` = `/root` |
| Inter-node RDMA | `UCX_TLS=rc_v,rc_x,...` + `/dev/infiniband` | Same as single-node RoCE config |
| Pod affinity | `topologyKey: nvidia.com/gpu.clique` | All nodes in same RoCE/NVLink domain |
| MPI bootstrap | `OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=tcp,self,vader` | Avoid UCX UD timeout on GB200 |
| ComputeDomain | `resourceClaimTemplateName` per pod | GPU allocation via DRA |

#### 3.3.4 Multinode source deployment

The TP=8 source also needs multinode (2 nodes). It's deployed as a DGD with
`multinode: nodeCount: 2`. The operator creates a leader + worker pod.
`publish_from_worker` runs on all 8 MPI ranks, gathers on rank 0 (leader),
and publishes all 8 workers in one gRPC call.

#### 3.3.5 Implementation steps

1. **Fix rank mapping bug** in `trtllm_live_transfer.py` — use MPI rank instead
   of `torch.cuda.current_device()` for source worker lookup
2. **Add `HOME=/root`** to decode source and target DGD env vars
3. **Validate source publish** — deploy TP=8 source DGD, verify 8 workers in Redis
4. **Validate target receive** — deploy TP=8 target, verify all 8 ranks receive
   correct shards from correct source ranks
5. **End-to-end disagg** — deploy full phase 2 DGD, test inference through
   KV-aware frontend with prefill TP=4 + decode TP=8
6. **DGDSA scale test** — scale decode replicas, verify new multinode pod group
   loads via P2P

### 3.4 Scale-Up Workflow

**Scaling prefill (TP=4, single node):**

1. DGDSA patched: `kubectl scale dgdsa/kimi-disagg-prefill --replicas=2`
2. Operator creates new prefill pod (1 node, 4 GPUs)
3. Pod queries MX Server A → NIXL RDMA from Source A
4. Transfer: 648 GB in ~3.5s at ~370 Gbps
5. Autotuning (~30s) → ready to serve
6. **Total: ~35s to ready**

**Scaling decode (TP=8, multinode):**

1. DGDSA patched: `kubectl scale dgdsa/kimi-disagg-decode --replicas=2`
2. Operator creates new decode pod group (2 nodes, 8 GPUs via MPI)
   - Leader pod runs `mpirun -n 8` across both nodes
   - Worker pod runs SSH daemon (port 2222)
3. Pod queries MX Server B → NIXL RDMA from Source B
4. Transfer: 648 GB in ~3.5s at ~370 Gbps (8 ranks × 81 GB each)
5. Autotuning + CUDA graphs (~5 min) → ready to serve
6. **Total: ~6 min to ready**

### 3.4 DGD Configuration

Each worker service points at its own MX source via `MODEL_EXPRESS_URL`:

```yaml
# Prefill — reads from Source A (TP=4)
prefill:
  componentType: worker
  subComponentType: prefill
  scalingAdapter:
    enabled: true
  extraPodSpec:
    mainContainer:
      args:
        - --model-express-url
        - modelexpress-server-prefill.kavin.svc.cluster.local:8001
        - --model-express-role
        - target
        - --disaggregation-mode
        - prefill
        - --tensor-parallel-size
        - "4"
      env:
        - name: MODEL_EXPRESS_URL
          value: "modelexpress-server-prefill.kavin.svc.cluster.local:8001"
        - name: WORLD_SIZE
          value: "4"
  replicas: 1
  resources:
    limits:
      gpu: "4"

# Decode — reads from Source B (TP=8)
decode:
  componentType: worker
  subComponentType: decode
  scalingAdapter:
    enabled: true
  multinode:
    nodeCount: 2
  extraPodSpec:
    mainContainer:
      args:
        - --model-express-url
        - modelexpress-server-decode.kavin.svc.cluster.local:8001
        - --model-express-role
        - target
        - --disaggregation-mode
        - decode
        - --tensor-parallel-size
        - "8"
      env:
        - name: MODEL_EXPRESS_URL
          value: "modelexpress-server-decode.kavin.svc.cluster.local:8001"
        - name: WORLD_SIZE
          value: "8"
  replicas: 1
  resources:
    limits:
      gpu: "4"   # per node; 2 nodes × 4 GPUs = TP=8
```

### 3.5 Source Deployment

Each source is a plain Deployment (not part of the DGD) that loads the model
from disk at the target TP size, publishes via NIXL, and holds GPU memory.

```yaml
# Source A: TP=4 for prefill
kimi-source-prefill-deploy.yaml:
  args: [--tensor-parallel-size, "4"]
  env:
    - MODEL_EXPRESS_SOURCE: "1"
    - MODEL_EXPRESS_URL: modelexpress-server-prefill:8001
    - WORLD_SIZE: "4"
  resources:
    limits:
      gpu: "4"

# Source B: TP=8 for decode (multinode)
kimi-source-decode-deploy.yaml:
  args: [--tensor-parallel-size, "8"]
  env:
    - MODEL_EXPRESS_SOURCE: "1"
    - MODEL_EXPRESS_URL: modelexpress-server-decode:8001
    - WORLD_SIZE: "8"
  resources:
    limits:
      gpu: "4"   # per node, 2 nodes
  # Requires multinode MPI (mpirun across 2 nodes)
```

### 3.6 Validation Steps

1. Deploy MX infra: 2 MX servers + 2 Redis instances
2. Deploy Source A (TP=4) → wait ~75 min for publish
3. Deploy Source B (TP=8, 2 nodes) → wait ~75 min for publish
4. Deploy disagg DGD with prefill pointing at Server A, decode at Server B
5. Verify both workers load via P2P
6. Test inference through KV-aware frontend
7. Scale decode via DGDSA → verify new TP=8 worker loads from Source B

---

## 4. Phase 3: Re-sharded P2P Transfers (Future)

The dual-source approach works but has overhead:
- 12 source GPUs (4 for TP=4 + 8 for TP=8)
- Two separate ~75 min source loading cycles
- Per-TP-config infrastructure duplication

Re-sharding eliminates this by allowing a single TP=4 source to serve TP=8 targets:

```
MX Source (TP=4)
  │
  ├── 1:1 RDMA ──→ Prefill (TP=4)         # same TP, direct transfer
  │
  └── re-shard ──→ Decode (TP=8)           # split TP=4 shards into TP=8
                    Rank 0,1 ← Source Rank 0
                    Rank 2,3 ← Source Rank 1
                    Rank 4,5 ← Source Rank 2
                    Rank 6,7 ← Source Rank 3
```

### 4.1 How Re-sharding Works

TRT-LLM shards weights along one dimension per TP:
- Column-parallel (q_proj, gate_proj): split `out_features` → `[out/TP, in]`
- Row-parallel (o_proj, down_proj): split `in_features` → `[out, in/TP]`

For TP=4 → TP=8, each source shard splits into 2 sub-shards along the same dimension.
Target rank N maps to source rank `N * 4 // 8`:

| Target Rank | Source Rank | Sub-shard Index |
|-------------|-------------|-----------------|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 1 | 0 |
| 3 | 1 | 1 |
| 4 | 2 | 0 |
| 5 | 2 | 1 |
| 6 | 3 | 0 |
| 7 | 3 | 1 |

### 4.2 Implementation Changes Required

Changes to `modelexpress_client/python/modelexpress/trtllm_live_transfer.py`:

**Source rank mapping** (currently line 492):
```python
# Current: 1:1 mapping
my_workers = [w for w in source_meta.workers if w.worker_rank == device_id]

# Re-sharded: N:1 mapping
source_tp = len(source_meta.workers)
target_tp = int(os.environ.get("WORLD_SIZE", str(source_tp)))
source_rank = device_id * source_tp // target_tp
my_workers = [w for w in source_meta.workers if w.worker_rank == source_rank]
```

**Tensor matching** (currently line 518):
```python
# Current: exact size match only
if src_size == dst_size:
    matched.append(...)

# Re-sharded: detect TP ratio, match sub-shard size
tp_ratio = target_tp // source_tp
if src_size == dst_size:
    matched.append(...)              # unsharded (norms, embeds)
elif src_size == dst_size * tp_ratio:
    reshard.append(...)              # TP-sharded, needs split
```

**Two-phase transfer:**
- Phase A: direct RDMA for unsharded params (~5% of weights)
- Phase B: RDMA full source shard into staging buffer, `torch.chunk()` along
  TP dim, copy sub-shard into target param

**Additional changes:**
- `nixl_transfer.py`: staging buffer support for per-tensor RDMA
- `p2p.proto`: populate `shape` field in `TensorDescriptor` for TP dim detection
- TP dim heuristic: compare source vs target shapes to find the split dimension

### 4.3 Trade-offs

| Aspect | Dual Sources (Phase 2) | Re-sharding (Phase 3) |
|--------|----------------------|---------------------|
| Source GPUs | 12 (4 + 8) | 4 |
| Code changes | None (config only) | trtllm_live_transfer.py, nixl_transfer.py, p2p.proto |
| Load time | 2 × 75 min (parallel) | 1 × 75 min |
| Transfer speed | Same as Phase 1 | Slight overhead from staging + chunk |
| Complexity | Infrastructure (2 MX servers) | Code (re-sharding logic) |
| TP ratios | Any (each source matches its targets) | Power-of-2 ratios only (4→8, 4→32) |
| Risk | Low (proven path) | Medium (correctness of shard splitting) |

**Recommendation:** Start with dual sources (Phase 2) for production use.
Implement re-sharding (Phase 3) as an optimization when GPU budget for sources
becomes a concern or when frequent TP changes are needed.

---

## 5. Engine Config Reference

### Prefill (TP=4, single node)

```yaml
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 16384
max_num_tokens: 4096
enable_chunked_prefill: true
disable_overlap_scheduler: true
enable_attention_dp: true
kv_cache_config:
  dtype: fp8
  enable_block_reuse: true
  free_gpu_memory_fraction: 0.5
  tokens_per_block: 32
max_batch_size: 4
max_seq_len: 16384
tensor_parallel_size: 4
moe_expert_parallel_size: 4
moe_config:
  backend: TRTLLM
trust_remote_code: true
```

### Decode (TP=4, single node — Phase 1)

```yaml
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 16384
max_num_tokens: 8192
cuda_graph_config:
  max_batch_size: 8
  enable_padding: true
disable_overlap_scheduler: true
enable_attention_dp: true
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.5
  tokens_per_block: 32
max_batch_size: 8
max_seq_len: 16384
tensor_parallel_size: 4
moe_expert_parallel_size: 4
moe_config:
  backend: TRTLLM
  use_low_precision_moe_combine: true
trust_remote_code: true
```

### Decode (TP=8, 2 nodes — Phase 2)

```yaml
allreduce_strategy: MNNVL
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 16384
max_num_tokens: 8192
cuda_graph_config:
  max_batch_size: 8
  enable_padding: true
enable_attention_dp: true
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.5
  tokens_per_block: 32
max_batch_size: 8
max_seq_len: 16384
tensor_parallel_size: 8
moe_expert_parallel_size: 8
moe_config:
  backend: TRTLLM
  use_low_precision_moe_combine: true
trust_remote_code: true
```

### Decode (TP=32, 8 nodes — Wideep production)

```yaml
allreduce_strategy: MNNVL
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 240000
max_num_tokens: 8192
cuda_graph_config:
  max_batch_size: 8
  enable_padding: true
enable_attention_dp: true
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.6
  tokens_per_block: 32
max_batch_size: 8
max_seq_len: 230400
tensor_parallel_size: 32
moe_expert_parallel_size: 32
moe_config:
  backend: WIDEEP
  use_low_precision_moe_combine: true
num_postprocess_workers: 8
stream_interval: 10
trust_remote_code: true
```

---

## 6. Required Pod Configuration (GB200)

All worker pods on GCP GB200 need these settings (validated in Phase 1):

```yaml
securityContext:
  privileged: true         # for /dev/infiniband access

env:
  UCX_TLS: "rc_v,rc_x,rc,dc_x,dc,cuda_copy,tcp"   # NO cuda_ipc
  UCX_RNDV_SCHEME: "get_zcopy"
  UCX_RNDV_THRESH: "0"
  OMPI_MCA_pml: "ob1"                               # avoid UCX UD timeout
  OMPI_MCA_btl: "tcp,self,vader"

volumes:
  /dev/infiniband (hostPath)        # RoCE NICs
  # /dev/shm is auto-injected by DGD operator — do NOT add manually

resourceClaims:
  compute-domain-channel            # GPU allocation via DRA

affinity:
  topologyKey: nvidia.com/gpu.clique   # same RoCE fabric as source
```
