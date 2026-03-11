# Disagg Multinode Inference Issues — GB200

**Date**: March 11, 2026
**Cluster**: `dynamo-gcp-dev-01` (GCP GB200 NVL36, ARM64)

---

## Summary

MX P2P weight transfer is fully validated for multinode disagg (TP=4 prefill + TP=8 decode at 345-610 Gbps). The remaining blockers are all **TRT-LLM multinode inference** issues on GB200 — not ModelExpress.

---

## Issue 1: MPI_ERR_TRUNCATE with ob1 PML

**Symptom**: During inference, `tp_gather` → `comm.allgather` fails with `MPI_ERR_TRUNCATE: message truncated` on the decode TP=8 multinode worker.

**Call stack**:
```
py_executor._executor_loop
  → _handle_first_token_response
    → _enqueue_responses
      → dist.tp_gather
        → safe_gather → comm.allgather
          → MPI_ERR_TRUNCATE
```

**Root cause**: `OMPI_MCA_pml=ob1` + `OMPI_MCA_btl=tcp,self,vader` forces MPI to use TCP BTL for cross-node communication. TCP BTL has a fixed max message size (~128KB default). TRT-LLM's `allgather` sends pickled Python response objects that exceed this limit.

**Why we need ob1**: Without it, UCX PML discovers IB/UD devices via `privileged: true` and hits a **UCX UD endpoint timeout** during MPI bootstrap on GB200 — a known hardware/driver issue.

**What we tried**:

| Config | Result |
|--------|--------|
| `ob1` + `tcp,self,vader` | P2P works (370+ Gbps), inference truncates |
| `ob1` + `tcp,self,vader` + `btl_tcp_max_send_size=1MB` | MPI session hung |
| No OMPI overrides + `privileged: true` | MPI init hangs (UCX UD timeout) |
| No OMPI overrides + `runAsUser: 0` | MPI works, NIXL TCP only (~3 Gbps) |
| `UCX_TLS=tcp,self,sm` + `privileged: true` | MpiSession hangs |
| `UCX_TLS=tcp,cuda_ipc,cuda_copy,self,sm` + `privileged: true` | MpiSession hangs |
| No env overrides, let operator handle MPI | `btl_tcp_endpoint` process ID mismatch |

**How Karen avoids this**: Uses `runAsUser: 0` (not privileged) — UCX never sees IB devices, NCCL handles TP collectives via NVIDIA driver, MPI only does small bootstrap messages.

---

## Issue 2: UCX_TLS conflict between MPI and NIXL

**Symptom**: Setting `UCX_TLS=tcp` globally (to fix MPI) also restricts NIXL to TCP, dropping P2P from 370 Gbps to ~3 Gbps.

**Root cause**: `UCX_TLS` is a process-wide env var. Both MPI's UCX PML and NIXL's UCX agent read it. We need MPI on TCP but NIXL on RoCE.

**Fix implemented (v1.6.0)**: In `nixl_transfer.py`, NIXL temporarily removes `UCX_TLS=tcp` before creating its agent, then restores it. Supports `NIXL_UCX_TLS` env var for explicit override.

```python
# In NixlTransferManager.initialize():
saved_ucx_tls = os.environ.get("UCX_TLS")
if saved_ucx_tls == "tcp":
    os.environ.pop("UCX_TLS", None)  # let NIXL auto-detect RoCE
try:
    self._agent = NixlAgent(self._agent_name, config)
finally:
    os.environ["UCX_TLS"] = saved_ucx_tls  # restore for MPI
```

**Status**: Built in v1.6.0 image, deployed but pending cluster scheduling. Untested end-to-end.

---

## Issue 3: KV Cache Transceiver UCX Connection

**Symptom**: `CacheReceiver` can't connect to prefill's UCX port for KV cache transfer:
```
Error in addConnection(ip) for rank 1 ip: 10.0.15.220 port: 33879
Error in UcxConnection constructor: Request canceled
```

**Root cause**: The `cache_transceiver_config: backend: UCX` creates its own UCX connections between prefill and decode pods for KV cache transfer. With our config, UCX connectivity between the prefill (w0e) and decode (w0e) pods isn't establishing properly.

**Karen's working config**: Both prefill and decode in the same DGD, same clique, `runAsUser: 0`, no UCX_TLS overrides. UCX auto-detects and connects.

**Status**: Not investigated in depth. Secondary to MPI issue.

---

## Issue 4: Compute Domain / DRA Scheduling

**Symptom**: Target pods stuck in `Pending` — "cannot allocate all claims" or "node is unschedulable".

**Root cause**: Each node's IMEX channel can only serve one compute domain (`allocationMode: Single`). With Karen's pods, Sara's NCCL test, and our source pods consuming channels, not enough free nodes remain in the clique.

**Workarounds**:
- Move targets to w0e pool (partially worked — 4 free nodes)
- Remove clique affinity between source and target (cross-clique RoCE works at 345-379 Gbps)
- Ask other users to clean up idle compute domains

---

## Issue 5: NFS File Lock (ESTALE) on Multinode

**Symptom**: Ranks 4-7 on node B crash with `[Errno 116] Stale file handle` during model config loading.

**Root cause**: TRT-LLM's `config_file_lock()` uses Python `filelock` on the shared NFS PVC. Cross-node NFS locking (NFSv3 NLM) is unreliable.

**Fix**: `HF_MODULES_CACHE=/tmp/hf_modules` moves the lock file to local disk. From coworker's investigation — 100% failure rate with filelock on NFS, 0% on local disk.

**Status**: Fixed.

---

## Issue 6: Multinode SSH / sshd

**Symptom**: Worker pod CrashLoopBackOff — `sshd: no hostkeys available` or `Missing privilege separation directory`.

**Root cause**: DGD operator's SSH key gen script uses `~` which resolves to `/home/dynamo` (image USER), but sshd runs as root and looks in `/root/.ssh/`.

**Fixes**:
- `HOME=/root` env var
- `/run/sshd` directory with `chmod 0755` + `chown root:root` in Dockerfile

**Status**: Fixed in v1.6.0 image.

---

## What Works

| Scenario | P2P Speed | Inference | Status |
|----------|-----------|-----------|--------|
| Aggregated TP=4 (single node) | 369 Gbps | Works | **VALIDATED** |
| DGDSA scale 1→2 (single node) | 371-390 Gbps | Works | **VALIDATED** |
| Disagg same-TP TP=4+TP=4 | 234-538 Gbps | KV transceiver issue | P2P validated |
| Mixed TP prefill TP=4 (single node) | 360-393 Gbps | Works (standalone) | **VALIDATED** |
| Mixed TP decode TP=8 (multinode) | 345-479 Gbps | MPI_ERR_TRUNCATE | P2P validated |
| Cross-clique RoCE (o7v→w0e) | 345-379 Gbps | — | **Works** |
| TP=8 source multinode publish | 8 workers published | — | **VALIDATED** |

---

## Approaches to Fix Multinode Inference

### Option A: `runAsUser: 0` for decode (immediate workaround)
- Use Karen's config: `runAsUser: 0`, no OMPI/UCX overrides
- MPI works, NCCL handles TP via NVIDIA driver, KV cache NIXL works
- P2P falls back to TCP (~3 Gbps) — 90.75 GB in ~4 min (still 8x faster than disk)
- **Pros**: Zero code changes, proven by Karen
- **Cons**: No RoCE for decode P2P. Prefill (single-node) still gets RoCE.
- **Status**: Ready to deploy

### Option B: Add `safe_allgather` to TRT-LLM (recommended upstream fix)
- **The actual bug**: `tp_allgather` uses raw `comm.allgather(obj)` without chunking.
  `safe_gather` already exists with `chunk_size=4MB`, but `tp_allgather` bypasses it.
- **File**: `tensorrt_llm/_torch/distributed/communicator.py` line 545
- **Current code**: `return self.tp_comm.allgather(obj)` — no size limit
- **Fix**: Route through chunked allgather (like `safe_gather`), or use
  `torch.distributed.all_gather_object` which handles large messages natively
- **Pros**: Minimal change, fixes ob1 truncation without changing executor
- **Cons**: Requires TRT-LLM upstream PR
- **Status**: Not implemented. Filed as suggestion.

```python
# Suggested fix in MPIDist (communicator.py):
def tp_allgather(self, obj):
    # Use safe chunked allgather instead of raw comm.allgather
    return safe_allgather(self.tp_comm, obj, chunk_size=64 * 1024)
```

### Option C: `TLLM_DISABLE_MPI=1` (needs Ray)
- TRT-LLM has `TLLM_DISABLE_MPI=1` → switches from `MPIDist` to `TorchDist` (NCCL)
- **Problem**: Also switches executor from MPI-based to **Ray-based**
- Requires Ray installed + Ray-based launch (not compatible with operator's `mpirun`)
- **Tested**: Crashes with `ModuleNotFoundError: Cannot import Ray`
- **Pros**: Would completely solve MPI issues
- **Cons**: Different launch paradigm, needs Ray in image + operator support
- **Status**: Tested, not viable with current operator

### Option D: NIXL UCX_TLS override (implemented)
- `UCX_TLS=tcp` globally for MPI
- NIXL temporarily removes it during agent creation → auto-detects RoCE
- `IPC_LOCK` + `SYS_RESOURCE` capabilities (not privileged)
- **Problem**: operator's `mpirun -x UCX_TLS=tcp` re-sets env in worker processes,
  so NIXL override doesn't take effect in multinode workers
- **Pros**: Clean concept, works for single-node
- **Cons**: Doesn't work with `mpirun -x` env propagation
- **Status**: Built in v1.6.0, doesn't work for multinode

### Option E: NIXL per-context UCX config (NIXL team)
- NIXL C++ backend supports `engine_config` with `TLS=rc_v,rc_x,...`
- Python API (`nixl_agent_config`) doesn't expose this yet
- **Fix**: Add `ucx_engine_config` param to Python `nixl_agent_config`:
  ```python
  config = nixl_agent_config(
      backends=["UCX"],
      ucx_engine_config="TLS=rc_v,rc_x,rc,dc_x,dc,cuda_copy,tcp"
  )
  ```
- This would override UCX_TLS at the NIXL context level, immune to `mpirun -x`
- **Pros**: Proper isolation, works with any MPI config
- **Cons**: Requires NIXL Python API change (C++ already supports it)
- **Status**: Not implemented. File as NIXL feature request.

### Option F: Fix UCX UD timeout on GB200
- Root cause fix in UCX/driver for IB UD transport timeout
- Would allow default UCX PML to work with `privileged: true`
- **Pros**: Fixes everything at the source
- **Cons**: Hardware/driver issue, not in our control
- **Status**: Known issue, no ETA

### Option G: Sara's pattern — `IPC_LOCK` + `UCX_TLS=tcp` + NCCL built-in IB
- From [k8s-nccl-test B200](https://github.com/sara4dev/k8s-nccl-test/tree/b200)
- `UCX_TLS=tcp` for MPI, NCCL uses built-in IB transport (not UCX)
- `IPC_LOCK` + `SYS_RESOURCE` capabilities for RDMA memory registration
- **Tested**: MPI works, but NIXL also gets `UCX_TLS=tcp` (no RoCE)
- Same issue as Option D — global UCX_TLS affects NIXL
- **Status**: Tested, NIXL falls back to TCP

---

## Recommendation Priority

1. **Immediate**: Option A (`runAsUser: 0`) — get end-to-end inference working today
2. **Short-term**: Option B (TRT-LLM `safe_allgather`) — one PR to TRT-LLM, fixes everything
3. **Medium-term**: Option E (NIXL `ucx_engine_config`) — proper RoCE isolation for NIXL
4. **Long-term**: Option F (fix UCX UD on GB200) — root cause, eliminates all workarounds

---

## Configuration Reference

### What works for P2P weight transfer (all scenarios):
```yaml
securityContext:
  privileged: true
env:
  UCX_TLS: "rc_v,rc_x,rc,dc_x,dc,cuda_ipc,cuda_copy,tcp"
  OMPI_MCA_pml: "ob1"
  OMPI_MCA_btl: "tcp,self,vader"
```

### What works for single-node inference (TP=4):
```yaml
securityContext:
  privileged: true
env:
  UCX_TLS: "rc_v,rc_x,rc,dc_x,dc,cuda_copy,tcp"  # NO cuda_ipc
  OMPI_MCA_pml: "ob1"
  OMPI_MCA_btl: "tcp,self,vader"
```

### What works for multinode inference (Karen's approach, no RoCE P2P):
```yaml
securityContext:
  runAsUser: 0
# No UCX_TLS, no OMPI_MCA overrides
# NCCL handles TP collectives via NVIDIA driver
# P2P falls back to TCP (~3 Gbps)
```

### What works for multinode P2P + startup (not inference):
```yaml
securityContext:
  privileged: true
env:
  UCX_TLS: "rc_v,rc_x,rc,dc_x,dc,cuda_ipc,cuda_copy,tcp"
  OMPI_MCA_pml: "ob1"
  OMPI_MCA_btl: "tcp,self,vader"
# P2P at 370-610 Gbps, autotuning works, inference hits MPI_ERR_TRUNCATE
```

### What we tried and failed:

| Config | P2P | MPI Init | Inference | Why it failed |
|--------|-----|----------|-----------|---------------|
| `privileged` + `ob1` + TCP BTL | RoCE 370+ Gbps | Works | `MPI_ERR_TRUNCATE` | TCP BTL can't handle large allgather |
| `privileged` + no OMPI overrides | N/A | Hangs | N/A | UCX UD timeout on GB200 |
| `runAsUser: 0` + no overrides | TCP ~3 Gbps | Works | Works (Karen) | No RoCE without IB devices |
| `IPC_LOCK` + `UCX_TLS=tcp` | TCP ~3 Gbps | Works | Untested | NIXL also gets TCP |
| `TLLM_DISABLE_MPI=1` | N/A | N/A | Ray not installed | Switches to Ray executor |
| `UCX_TLS=tcp,cuda_ipc,...` + `privileged` | N/A | Hangs | N/A | UCX confused by visible IB |
| No env overrides + operator mpirun | N/A | `btl_tcp` errors | N/A | Process ID mismatch |
