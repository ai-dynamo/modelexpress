# TRT-LLM Dynamo P2P Integration: Testing Context

**Last updated**: March 3, 2026
**Purpose**: Resume context for DeepSeek-V3.2 DEP8x2 P2P testing on nscale B200

---

## Push Status

Both repos are fully pushed:

| Repo | Branch | Commit | Status |
|------|--------|--------|--------|
| **modelexpress** | `kavink/trtllm` | `03be4ed` | Pushed to origin |
| **dynamo** | `kavink/trtllm-p2p` | `8c19dfc85` | Pushed to origin |

Local uncommitted changes in modelexpress (not needed for Phase 1 presentation):
- `docs/TRTLLM_DYNAMO.md` — minor updates
- `trtllm_live_transfer.py` — model_files proto compat fix
- `state.rs` — double-checked locking
- `trtllm_patches/v1.3.0rc3/*.py` — patch updates

---

## Current Cluster: nscale B200 (Blackwell)

### Access
```bash
tsh kube login dynamo-nscale-dev-cluster
# Verify: kubectl get nodes
```

Cluster name in Teleport: `dynamo-nscale-dev-cluster`

### Why nscale B200 (not H200 Nebius)
- H200 (Hopper) has a **FlashMLA kernel bug**: asserts `kv.dtype() == torch::kBFloat16` but DSV3.2 NVFP4 uses FP8 KV cache
- This is a known TRT-LLM incompatibility on Hopper GPUs
- B200 (Blackwell) should have this fixed

### Infrastructure Deployed (namespace `kavin`)

| Pod | Status | Purpose |
|-----|--------|---------|
| `dynamo-platform-etcd-0` | Running | etcd for Dynamo worker discovery |
| `dynamo-platform-nats-0` | Running | NATS for Dynamo messaging |
| `modelexpress-server` | Running | gRPC metadata coordinator |
| `redis` | Running | Persistent metadata store |
| `trtllm-source-dsv32` | CrashLoopBackOff (47 restarts) | **THE ISSUE** |

Infrastructure YAMLs:
- `examples/p2p_transfer_trtllm/deploy/dynamo-platform.yaml` — etcd + NATS
- MX server + Redis — already deployed from prior testing

### Secrets Required
- `hf-token-secret` — HuggingFace token with access to `nvidia/DeepSeek-V3.2-NVFP4`
- `nvcr-imagepullsecret` — NGC image pull secret

### PVC
- `dsv32-nvfp4-weights` — 500Gi, `local-path` storage class, holds the DSV3.2 model (~337 GB)
- Model is already downloaded (HF snapshot_download completed on earlier runs)

---

## Current Blocker: Source Pod Crash-Looping

### Symptom
`trtllm-source-dsv32` pod has 47 restarts over 8 hours. Each cycle:
1. Pod starts, 8 MPI ranks load weights from PVC (~2 min)
2. NIXL `register_dlist()` begins for all 8 ranks
3. Registration takes **~45-50 minutes** (see below)
4. **MPI rank 4 crashes with exit code 1**
5. MPI kills all ranks, pod restarts

### Why Registration is Slow (12 IB Devices)
The nscale B200 node has **12 mlx5 InfiniBand devices** (mlx5_0 through mlx5_11). Each tensor must be registered on all 12 devices via UCX KSM atomic-key memory registration:
- ~2,718 tensors per rank × 12 IB devices = ~32,616 registrations per rank
- Each registration cycle (12 devices) takes ~1 second
- Total: ~45 minutes per rank

This is not a hang — all registrations return `"Success"`. It's inherently slow with 12 IB ports. On H200 (fewer IB devices) this took ~5 minutes.

### Root Cause Unknown
The actual Python error from rank 4 is **lost** because:
1. UCX DEBUG logging produces millions of lines (one per KSM registration)
2. Kubernetes log buffer overflows, pushing Python-level logs out
3. By the time we check, only the most recent UCX debug lines remain

### Last Action Taken (March 3)
Redeployed with:
- `NIXL_LOG_LEVEL=INFO` (was DEBUG)
- `UCX_LOG_LEVEL=WARN` (was DEBUG)
- Added `--output-filename /tmp/mpi_logs` to mpirun for per-rank log files
- Added `tee /tmp/source_all.log` for combined output capture
- Used `python3 -u` for unbuffered output

Updated YAML: `examples/p2p_transfer_trtllm/deploy/trtllm-source-dsv32-ph3.yaml`

Key change in args:
```bash
# Before:
exec mpirun --allow-run-as-root -np ${WORLD_SIZE} python3 /tmp/source_dsv32.py

# After:
mpirun --allow-run-as-root -np ${WORLD_SIZE} \
  --output-filename /tmp/mpi_logs \
  python3 -u /tmp/source_dsv32.py 2>&1 | tee /tmp/source_all.log
```

### Resume Steps
1. Re-auth: `tsh kube login dynamo-nscale-dev-cluster`
2. Check pod status: `kubectl get pods -n kavin`
3. Check if registration completed: `kubectl logs -n kavin -l app=trtllm-source-dsv32 --tail=30`
4. If crashed again, exec into pod and read per-rank logs:
   ```bash
   kubectl exec -n kavin -l app=trtllm-source-dsv32 -- ls /tmp/mpi_logs/
   kubectl exec -n kavin -l app=trtllm-source-dsv32 -- cat /tmp/mpi_logs/1/rank.4/stderr
   ```
5. If logs are gone (pod restarted), check combined log:
   ```bash
   kubectl exec -n kavin -l app=trtllm-source-dsv32 -- tail -100 /tmp/source_all.log
   ```
   Note: `/tmp` is ephemeral — lost on pod restart. If still crash-looping, mount a PVC for logs.

### Possible Fixes to Try
1. **Mount log PVC**: Add a small PVC for `/tmp/mpi_logs` so logs survive restarts
2. **Reduce IB devices**: Set `UCX_NET_DEVICES=mlx5_0:1` to register on only 1 IB device (reduces registration time from 45min to ~4min, may limit bandwidth but useful for debugging)
3. **Isolate rank 4**: Add rank-specific error handling in the source Python script
4. **Check GPU memory**: Rank 4 might be OOM — DSV3.2 NVFP4 EP=8 should fit in 80GB but edge case with 12 IB registrations could add overhead

---

## Environment Details

### nscale B200 Node
- GPU: NVIDIA B200 (Blackwell)
- NVIDIA driver: 590.48.01 (CUDA 13.1)
- IB devices: 12x mlx5 (ConnectX)
- No `LD_LIBRARY_PATH` compat override needed (driver matches container CUDA)

### Docker Image
`nvcr.io/nvidian/dynamo-dev/modelexpress-trtllm-client:ph3-dynamo0.9`
- Base: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3` (same as Dynamo 0.9.0)
- Includes: system NIXL, ModelExpress client, 3 TRT-LLM patches

### Source Script Key Details
The source Python script (`/tmp/source_dsv32.py` inside the pod) does:
1. Rank 0 downloads/resolves model path, broadcasts to all ranks
2. Each rank creates `Mapping(world_size=8, tp_size=8, rank=rank, moe_ep_size=8, moe_tp_size=1, enable_attention_dp=True)`
3. Creates model via `AutoModelForCausalLM.from_config()` with `MetaInitMode()`
4. Materializes meta tensors → CUDA
5. Loads weights via `HfCheckpointLoader` + `weight_mapper`
6. Creates `MxLiveSource(model, model_name, mx_server)` and calls `source.publish()`
7. `publish()` → NIXL `register_dlist()` (THE SLOW PART) → gRPC `PublishMetadata`
8. `comm.Barrier()` → sleep forever

### UCX Environment
```yaml
UCX_TLS: "rc_x,rc,dc_x,dc,cuda_copy"
UCX_RNDV_SCHEME: "get_zcopy"
UCX_RNDV_THRESH: "0"
```

---

## Previous Clusters Tried

### Nebius H200 (dynamo-nebius-1)
- **Blocked by**: FlashMLA kernel bug (BF16 KV cache assertion on Hopper)
- Small models (Qwen 0.5B, Llama 70B) work fine
- Cleaned up all deployments

### Why Not AWS GB200
- ARM64 architecture requires multi-arch Docker builds
- EFA networking (not InfiniBand) requires different RDMA resource names
- Saved for later if nscale B200 doesn't work

---

## Validated Results (Ready for Presentation)

### Phase 1: Dynamo TRT-LLM P2P Integration — VALIDATED

See `docs/TRTLLM_DYNAMO_PHASE1.md` for full details.

| Model | TP | Params | Data | Time | Bandwidth | Status |
|-------|-----|--------|------|------|-----------|--------|
| Qwen 0.5B (v1.3.0rc3) | 1 | 171/171 | 1.26 GB | 0.07s | 151.9 Gbps | PASSED |
| Qwen 0.5B (v1.2.0rc6) | 1 | 171/171 | 1.26 GB | 0.13s | 78.8 Gbps | PASSED |
| Llama 70B (v1.2.0rc6) | 8 | 483/rank | 141.1 GB | 3.08s | 368 Gbps | PASSED |

### What's Been Proven
1. Phase 3 live GPU-to-GPU param transfer works end-to-end
2. Dynamo TRT-LLM worker integration works (`--model-express-url` flag)
3. `LoadFormat.PRESHARDED` patches work on both TRT-LLM 1.2.0rc6 and 1.3.0rc3
4. 100% param name matching between source and target (same model architecture)
5. Inference passes after P2P weight loading

### What's Not Yet Proven
1. DeepSeek-V3.2 MoE expert-parallel (EP=8) P2P transfer
2. NVFP4 quantized weight transfer
3. Multi-node DEP8x2 deployment
4. Dynamo worker inference with DSV3.2 on Blackwell

---

## File Map

### Source YAML
`examples/p2p_transfer_trtllm/deploy/trtllm-source-dsv32-ph3.yaml`

### Dynamo Platform YAML
`examples/p2p_transfer_trtllm/deploy/dynamo-platform.yaml`

### Phase 1 Results Doc
`docs/TRTLLM_DYNAMO_PHASE1.md`

### Dynamo Integration Plan
`docs/TRTLLM_DYNAMO.md`

### Phase 3 Detailed Design
`docs/TRTLLM_PLAN_PHASE3.md`
