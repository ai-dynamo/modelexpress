<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CI Test Plan

Coverage matrix for the ModelExpress CI infrastructure. Reflects what's currently in `.github/workflows/modelexpress-ci-tests.yml` plus the engine integrations under `ci/k8s/client/`.

| Pri | Test Area | Engine(s) | Status | Key Blocker |
|---|---|---|---|---|
| 1 | P2P Weight Transfer (InfiniBand) | vLLM | **In CI** — matrix entry active. Verified by manual `kubectl` + `pytest` deploy against the Azure A100 cluster; GitHub Actions workflow not yet triggered on a PR | — |
| 2 | P2P Weight Transfer (InfiniBand) | TRT-LLM | **In CI** — matrix entry active. Verified end-to-end by manual `kubectl` + `pytest` deploy against the Azure A100 cluster on the Dynamo `tensorrtllm-runtime` base image; GitHub Actions workflow not yet triggered on a PR | — |
| 3 | P2P Weight Transfer (InfiniBand) | SGLang | **Scaffolded, gated** — Dockerfile + manifest in place under `ci/k8s/client/sgl/`, workflow matrix entries commented out | Upstream [sgl-project/sglang#24723](https://github.com/sgl-project/sglang/pull/24723) (adds `--remote-instance-weight-loader-backend modelexpress` and `--modelexpress-config`) not yet merged into any tagged `lmsysorg/sglang` image |
| 4.1 | Source Bootstrap — HuggingFace | vLLM, TRT-LLM, SGLang | **Covered implicitly by P1+P2** for vLLM and TRT-LLM (source pod always loads from HF before publishing). SGLang will fall under same coverage when P3 lands | — |
| 4.2 | Source Bootstrap — NGC | vLLM, TRT-LLM, SGLang | **Not started** | NGC model registry path with a CI-accessible API key + a small published model to pull |
| 4.3 | Source Bootstrap — GCS | vLLM, TRT-LLM, SGLang | **Not started** | GCS bucket reachable from K8s nodes (workload identity or a service-account key secret) |
| 5 | TP > 1 (multi-node preferred) | vLLM, TRT-LLM, SGLang | **Not started** | Multi-node K8s with cross-node RDMA is the preferred shape since it exercises the inter-host P2P fabric; fall back to single-node multi-GPU (NVLink) only when multi-node capacity isn't available. Start with vLLM and expand |
| 6 | Model Streamer — S3 *(direct streaming, no server cache)* | vLLM, SGLang | **Not started** — TRT-LLM has no modelexpress engine adapter, so no `build_model_streamer_weight_iter` path; runai-model-streamer covers only the two engines. GCS and Azure Blob variants exist in the same code path but are not currently in the test plan | Access to S3 bucket from K8s nodes; start with vLLM and expand to SGLang |
| 7 | Dynamo Integration (without Disaggregated Serving) | vLLM, TRT-LLM, SGLang | **Not started** | Start with vLLM and expand |
| 8 | Disaggregated Serving | vLLM, TRT-LLM, SGLang | **Not started** | K8s nodes with sufficient GPUs; start with vLLM and expand; TP × disagg combo is long-term |
| 9.1 | Metadata Backend — K8s CRD | — | **Covered implicitly** — exercised by all P1+P2 runs (mx-server uses `MX_METADATA_BACKEND=kubernetes`) | — |
| 9.2 | Metadata Backend — Redis | — | **Not started** | Redis deployed as in-cluster service |
| 9.3 | Metadata Backend — K8s Service / Peer-Direct (PR #251) | — | **Not started** — third shipped backend. No central mx-server in this topology: sources sit behind a K8s Service, clients open direct gRPC to the Service DNS name and compute `mx_source_id` client-side. Differs from Redis/CRD on live weight refit and rolling-update recovery. Example manifests live at [`examples/k8s_service_sources/`](../examples/k8s_service_sources/) | Multi-pod Deployments with scale/restart hooks; mx-server-less manifest variant |
| 10 | MLA Model Testing (e.g. DeepSeek) | vLLM, TRT-LLM, SGLang | **Not started** | K8s nodes with sufficient GPU memory for MLA models |
| 11 | GPUDirect Storage | vLLM, TRT-LLM, SGLang | **Not started** | K8s nodes with GDS-capable hardware (DGX A100/H100) + nvidia-fs kernel module |
| 12.1 | P2P Weight Transfer — EFA via NIXL libfabric plugin (`MX_NIXL_BACKEND=LIBFABRIC`) | vLLM, TRT-LLM, SGLang | **Not started** — libfabric talks to EFA directly through NIXL, bypassing UCX | AWS K8s nodes with EFA-enabled instances (e.g., p4d/p5); libfabric installed and discoverable by NIXL |
| 12.2 | P2P Weight Transfer — EFA via UCX over libfabric (`MX_NIXL_BACKEND=UCX`, UCX configured with libfabric underneath) | vLLM, TRT-LLM, SGLang | **Not started** — same fabric as 12.1 but routed through UCX with libfabric as its provider; tests the UCX integration path | Same hardware as 12.1; UCX built with the libfabric provider enabled |
| 12.3 | P2P Weight Transfer — RoCE via UCX (`MX_NIXL_BACKEND=UCX`) | vLLM, TRT-LLM, SGLang | **Not started** — UCX over Ethernet (RoCE) rather than InfiniBand; different transport than P1+P2 which use IB | K8s nodes with RoCE-capable NICs (RDMA over Converged Ethernet) |
| 13 | NIC Pinning / `MX_RDMA_NIC_PIN=auto` (PR #255 — workaround for [openucx/ucx#11259](https://github.com/openucx/ucx/issues/11259)) | — | **Not started** — multi-NIC NUMA-aware selection logic. Easy to regress silently because single-NIC nodes don't exercise it | Node with multiple IB HCAs at different NUMA distances |
| 14 | NIXL Fallback Chains (rank-matching, `ManifestMismatchError`) | vLLM, TRT-LLM, SGLang | **Not started** — 3-candidate retry then disk; covers behavior when the first source has a different rank manifest than the target. Related to the bug logged in 9 (`_query_source` picks `my_workers[0]` without liveness probe) | Way to inject deliberately-mismatched / stale sources; could publish a stale CR alongside a fresh one |
| 15 | FP8 Tensor Adoption (PR #241 — NVFP4 MoE, Kimi-K2.5) | vLLM | **Not started** — historically verified against vLLM 0.17.1 and 0.19.0. Engine-agnostic in code but only verified on vLLM | FP8/NVFP4-capable GPUs (Hopper+); Kimi-K2.5 weights in a CI-accessible registry |
| 16 | FP8 Disk Fallback / CompilationConfig Cleanup PR #244 | vLLM | **Not started** — regression target for the FP8 disk-fallback path | FP8 model + a way to force the disk fallback (e.g., suppress source publish) |
| 17 | Rolling-Update Revision-Mismatch Recovery (k8s-service handshake) | — | **Not started** — pairs with 9.3. Tests how the k8s-service backend's clients recover when a rolling update lands a new revision and old sources go away mid-pull | Multi-replica Deployment with scriptable scale/restart hooks |
| 18 | Stale Metadata After Redeploy (Redis + K8s CRD) | vLLM, TRT-LLM, SGLang | **Partial, with a real race window**. Server-side reaper (PR #182) is backend-agnostic — covers both Redis and K8s CRD identically: marks workers STALE after `MX_HEARTBEAT_TIMEOUT_SECS` (default 90s) and deletes after `MX_GC_TIMEOUT_SECS` (default 3600s). Client-side `status_filter=SOURCE_STATUS_READY` is set in the shared `rdma_strategy.py` (vLLM + SGLang). **But `status_filter=READY` only protects AFTER 90s** — if a source pod is killed and replaced within that window, the stale CR is still labeled READY and gets returned to clients alongside the live one. The CI workflow exercises this path end-to-end (no manual `kubectl delete modelmetadata` workaround), so any regression in reaper/filter behavior surfaces as a test failure. **TRT-LLM additionally**: `trtllm_live_transfer.py:_query_source` doesn't pass `status_filter` at all and picks `my_workers[0]` blindly — known bug; CI is the forcing function to land the fix | (1) Way to forcibly terminate source pods and assert client behavior against the stale record. (2) Long-term fix likely needs publish-time cleanup (delete prior CR with same identity on republish) and/or client-side liveness probe before `add_remote_agent`, not just heartbeat-based reaping |
| 19 | S3 Fallback | vLLM, TRT-LLM, SGLang | **Not started** — fallback activation path, distinct from Row 6 which is the direct-streaming primary path | S3 bucket reachable from K8s nodes (can share with Row 6) |
| 20 | Small Fleet (~15 workers, CRD + P2P metadata) | vLLM, TRT-LLM, SGLang | **Not started** — realistic small-fleet scale test on the CRD metadata backend with P2P sharing between all members | K8s cluster with capacity for ~15 GPU pods on the same RDMA fabric |
| 21 | Pool Registration / `MX_POOL_REG=1` | — | **Not started** — collapses per-tensor NIXL registrations into region-level ones via `cuMemGetAddressRange`. Big lever on `register_tensors` cost when many NICs are visible. Assertion: region-count stays bounded on large multi-tensor models (not linear in tensor count) | Multi-NIC node + a large multi-tensor model; need a way to introspect NIXL region count (e.g., scrape NIXL agent logs or expose a count via the manager) |

## Status legend

- **In CI** — matrix entry is uncommented, the test runs on every PR, currently passing.
- **Scaffolded, gated** — files exist in the repo, matrix entry is commented with an explicit blocker. Easy to enable when the blocker clears.
- **Partial** — some aspect of the area is covered as a side effect of another row, but no targeted test exists.
- **Covered implicitly** — no separate test, but the functionality is exercised by another test that would fail if it broke.
- **Not started** — no scaffold, no matrix entry, no dependency planning yet.

When a row moves status, update this table in the same PR so the doc stays the source of truth on what CI does and doesn't catch.
