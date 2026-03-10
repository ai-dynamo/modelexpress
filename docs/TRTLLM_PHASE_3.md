# TRT-LLM Phase 3: Disaggregated Serving with MX P2P

**Status**: Phase 1 (same TP) validated, Phase 2 (mixed TP) in progress
**Date**: March 10, 2026
**Branch (modelexpress)**: `kavink/trtllm`
**Branch (dynamo)**: `kavink/trtllm-p2p`
**Cluster**: `dynamo-gcp-dev-01` (GCP GB200 NVL36, ARM64)
**Namespace**: `kavin`

---

## 1. What's Been Validated

### 1.1 Aggregated mode + MX P2P (Phase 2.5b — fully done)

- Kimi K2.5 TP=4 P2P at **369 Gbps** RoCE, 648 GB in 3.51s
- DGDSA scale test: second worker loads via P2P at 371-390 Gbps, ready in ~7.5 min
- Inference confirmed via Dynamo frontend
- See [TRTLLM_DYNAMO_PHASE2_5.md](./TRTLLM_DYNAMO_PHASE2_5.md) §13

### 1.2 Disagg same-TP P2P (Phase 3a — validated)

Both prefill and decode workers (TP=4) load from single MX source concurrently:

| Worker | Data | Time | Speed |
|--------|------|------|-------|
| Prefill | 648 GB | 5.09s | 255 Gbps |
| Decode | 648 GB | 5.53s | 234 Gbps |
| **Total concurrent** | **1.3 TB** | **~5.7s** | — |

YAML: `deploy/gcp/kimi-disagg-mx-dgd.yaml`

### 1.3 Planner integration

- AggPlanner `component_type` bug found and fixed (pushed to `kavink/trtllm-p2p`)
- DisaggPlanner (`mode: disagg`) avoids this bug entirely
- Manual DGDSA scaling validated: `kubectl scale dgdsa/... --replicas=2`
- Per-worker load metrics pipeline needs planner team investigation

---

## 2. Phase 2 Mixed TP: Current State

### 2.1 Architecture

```
MX Source A (TP=4, 1 node)  ──→  modelexpress-server      ──→ Prefill (TP=4)
MX Source B (TP=8, 2 nodes) ──→  modelexpress-server-decode ──→ Decode  (TP=8)
```

### 2.2 Infrastructure deployed

| Component | Status | Notes |
|-----------|--------|-------|
| `modelexpress-server` + `redis` | Running | Existing, for prefill source |
| `modelexpress-server-decode` + `redis-decode` | Running | New, for decode source |
| `kimi-source-deploy` (TP=4 prefill) | Running | Plain Deployment, loading model |
| `kimi-source-decode` (TP=8 decode) | **CrashLoopBackOff** | DGD multinode, SSH key path issue |

### 2.3 Blocker: Multinode SSH key path mismatch

The DGD operator's multinode worker pod generates SSH keys using `~` (shell home
directory). With our image, `~` resolves to `/home/dynamo` (from the base image's
`USER dynamo` directive), but sshd runs as root and looks for keys in `/root/.ssh/`.

```
Key generation: /home/dynamo/.ssh/host_keys/ssh_host_*_key
sshd expects:   /root/.ssh/host_keys/ssh_host_*_key
→ "Unable to load host key" → "sshd: no hostkeys available -- exiting."
```

Karen's image works because it has `HOME=/root` set (or the user is root natively).

**Fix**: Add `HOME=/root` to the decode source DGD's env vars so `~` resolves
consistently. One-line change in `kimi-source-decode-dgd.yaml`.

### 2.4 YAMLs created (ready to deploy after SSH fix)

| File | Purpose |
|------|---------|
| `deploy/gcp/mx-infra-decode.yaml` | Second MX server + Redis for decode source |
| `deploy/gcp/kimi-source-decode-dgd.yaml` | TP=8 decode source (DGD, multinode: 2) |
| `deploy/gcp/kimi-disagg-phase2-dgd.yaml` | Disagg DGD: prefill→Server A, decode→Server B |

### 2.5 Resume checklist

1. `tsh kube login dynamo-gcp-dev-01`
2. Fix SSH: Add `- name: HOME` / `value: "/root"` to `kimi-source-decode-dgd.yaml` env
3. Delete and redeploy decode source:
   ```bash
   kubectl -n kavin delete dgd kimi-source-decode
   kubectl -n kavin apply -f .../deploy/gcp/kimi-source-decode-dgd.yaml
   ```
4. Wait ~75 min for both sources to load and publish
5. Verify both sources published:
   ```bash
   kubectl -n kavin exec deploy/redis -- redis-cli KEYS '*'
   kubectl -n kavin exec deploy/redis-decode -- redis-cli KEYS '*'
   ```
6. Deploy disagg DGD:
   ```bash
   kubectl -n kavin apply -f .../deploy/gcp/kimi-disagg-phase2-dgd.yaml
   ```
7. Verify prefill loads from Server A (TP=4), decode from Server B (TP=8)
8. Test inference through KV-aware frontend
9. DGDSA scale test: `kubectl -n kavin scale dgdsa/kimi-disagg-p2-decode --replicas=2`

---

## 3. Bugs Found This Session

### 3.1 AggPlanner `component_type` crash (FIXED)

`BasePlanner.__init__()` accesses `self.component_type` before `AggPlanner` sets it.
Fix: added `component_type` parameter to `BasePlanner.__init__()`.
Pushed to `kavink/trtllm-p2p`.

### 3.2 AggPlanner requires `subComponentType: decode`

Worker service must have `subComponentType: decode` for planner validation in agg mode.

### 3.3 DGD operator `/dev/shm` duplicate mount

Operator auto-injects `/dev/shm`. If extraPodSpec also defines it, PodCliqueSet
creation fails. Worker DGD specs must NOT include `/dev/shm`.

### 3.4 Multinode SSH key path mismatch

`~` resolves to `/home/dynamo` in our image but sshd runs as root and looks in `/root/.ssh/`.
Fix: `HOME=/root` env var. Karen's image works natively.

### 3.5 Planner metrics not flowing

Per-worker load metrics (`worker_active_decode_blocks`) require:
- Worker: `--publish-events-and-metrics` + `--request-plane nats`
- Frontend: `--request-plane nats`
- Metrics still not appearing on frontend `/metrics` after traffic. Needs planner team investigation.

### 3.6 Disagg KV cache transceiver

Prefill→decode KV cache transfer via `cache_transceiver_config` not working end-to-end.
Prefill processes requests but decode doesn't receive KV data. Likely UCX connectivity
or config issue between pods. Separate from MX P2P (weight loading works fine).

---

## 4. Commits This Session

### modelexpress (`kavink/trtllm`)

| Commit | Description |
|--------|-------------|
| `695c91c` | feat: Kimi K2.5 P2P validated at 369 Gbps RoCE |
| `24afccf` | fix: planner DGD — patch component_type bug, add subComponentType |
| `5820389` | fix: planner DGD — add NATS event plane and subComponentType |
| `7da23d4` | docs: update planner plan with DGDSA scale test results |
| `7df85f4` | feat: disagg DGD with MX P2P — prefill + decode as independent MX targets |
| `6d5e794` | docs: disagg TRT-LLM with MX P2P — dual sources for mixed TP |

### dynamo (`kavink/trtllm-p2p`)

| Commit | Description |
|--------|-------------|
| `cdc38f458` | fix: AggPlanner crash — component_type not set before BasePlanner.__init__ |
| `f99118d63` | fix: kube-rbac-proxy image registry — use registry.k8s.io |

---

## 5. Files Created This Session

| File | Purpose |
|------|---------|
| `deploy/gcp/kimi-planner-dgd.yaml` | Aggregated DGD with planner (Phase 2.5b) |
| `deploy/gcp/kimi-disagg-mx-dgd.yaml` | Disagg DGD, same TP=4 (Phase 3a) |
| `deploy/gcp/mx-infra-decode.yaml` | Second MX server + Redis for decode |
| `deploy/gcp/kimi-source-decode-dgd.yaml` | TP=8 decode source (DGD multinode) |
| `deploy/gcp/kimi-disagg-phase2-dgd.yaml` | Mixed TP disagg DGD (Phase 3b) |
| `docs/disagg_trtllm.md` | Disagg + MX P2P design doc |
| `docs/TRTLLM_PHASE_3.md` | This file |

---

## 6. Current Deployment

```
kavin namespace:
  dynamo-platform-etcd-0                    Running   (45h)
  dynamo-platform-nats-0                    Running   (45h)
  modelexpress-server                       Running   (45h)   ← prefill MX server
  redis                                     Running   (45h)   ← prefill Redis
  modelexpress-server-decode                Running   (13m)   ← decode MX server
  redis-decode                              Running   (13m)   ← decode Redis
  kimi-source-deploy (TP=4)                 Running   (10m)   ← prefill source, loading
  kimi-source-decode (TP=8, multinode)      CrashLoop          ← BLOCKED: SSH key path
```
