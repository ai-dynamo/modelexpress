<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CI Test Plan

Coverage matrix for the ModelExpress CI infrastructure. Reflects what's currently in `.github/workflows/modelexpress-e2e-ci.yml` and its reusable inference, GPU, and RL workflows plus the engine integrations under `ci/k8s/client/`.

| Pri | Test Area | Engine(s) | Status | Key Blocker |
|---|---|---|---|---|
| 1 | P2P Weight Transfer (InfiniBand) | vLLM | **In CI** — matrix entry active. Verified by manual `kubectl` + `pytest` deploy against the Azure A100 cluster; GitHub Actions workflow not yet triggered on a PR | — |
| 2 | P2P Weight Transfer (InfiniBand) | TRT-LLM | **Not started** | Qualified upstream image containing the native `checkpoint_format="MX"` integration, plus a no-shards target so fallback cannot mask a transfer regression |
| 3 | P2P Weight Transfer (InfiniBand) | SGLang | **In CI** — matrix entry active. Uses the upstream SGLang delegation hook included in `lmsysorg/sglang:v0.5.13.post1`; the CI Dockerfile overlays the ModelExpress Python client without dependency resolution. Verified manually on nscale with `Qwen/Qwen2.5-0.5B-Instruct`: source loaded natively and published metadata, target transferred 170 tensors / 0.99 GB over NIXL in 0.050s and served a correct chat completion | — |
| 3.1 | P2P Weight Transfer — Mooncake TransferEngine transport (`transport=transfer_engine`) | SGLang | **In CI** — `build-client-sglang-mooncake` builds `ci/k8s/client/sgl/Dockerfile.mooncake`, and the `sglang-mooncake` partition of the unified `test` matrix runs `ci/k8s/client/sgl/manifest-azure-mooncake.yaml`. Uses the same `remote_instance` + `backend=modelexpress` command shape as row 3, with `modelexpress-config` selecting `{"transport": "transfer_engine"}`. The first replica finds no READY source, loads natively through SGLang, registers TransferEngine memory, publishes its `transfer_engine_session_id`, and heartbeats through ModelExpress; later replicas discover the source via mx-server and pull with `batch_transfer_sync_read`. Image starts from `lmsysorg/sglang:v0.5.13.post1` and installs `mooncake-transfer-engine-cuda13`. Transfer marker is `SGLang MxModelLoader transfer_engine COMPLETE`; the NIXL-only per-rank-agent assertion is skipped via `--transport transfer_engine` | Mooncake initializes fine at TP=1 on a single-GPU a100a node (auto-discovers the IB HCAs; `MOONCAKE_DEVICE` not needed) |
| 3.2 | P2P Target VRAM Parity | vLLM, SGLang | **In CI** — enabled on the dedicated full-GPU MLA model jobs. The shared action samples source load and target RDMA-transfer process GPU memory via `nvidia-smi --query-compute-apps=used_memory` from pod startup until each side's `/health` returns 200; pytest fails if either peak VRAM or final-after-initialization VRAM exceeds source by more than the fixed 0.5% tolerance. SGLang source and target additionally must report the configured 131,072-token KV-pool cap | Raw P2P jobs run on the A100a MIG pool, where in-container NVML memory accounting returns `[Insufficient Permissions]`; TRT-LLM VRAM parity remains a follow-up on a full-GPU pool |
| 4.1 | Source Bootstrap — HuggingFace | vLLM, TRT-LLM, SGLang | **Covered implicitly by P1+P3** for vLLM and SGLang. TRT-LLM will have the same coverage when P2 is re-enabled | — |
| 4.2 | Source Bootstrap — NGC | vLLM, TRT-LLM, SGLang | **Not started** | NGC model registry path with a CI-accessible API key + a small published model to pull |
| 4.3 | Source Bootstrap — GCS | vLLM, TRT-LLM, SGLang | **Not started** | GCS bucket reachable from K8s nodes (workload identity or a service-account key secret) |
| 5.1 | Parallelism > 1 — TP (multi-node preferred) | vLLM, SGLang | **Configured; new SGLang lane awaits GPU qualification** — single-node TP2 uses GLM for vLLM and DeepSeek-V2-Lite for SGLang on full GPUs. The `large` partition runs cross-node TP2 with two source and two target pods on a100a: vLLM uses Ray; SGLang uses native `--nnodes 2 --node-rank` and a headless rendezvous Service. Same-rank source/target anti-affinity requires cross-host RDMA; both lanes run MMLU score comparison | TRT-LLM; GPU qualification of new SGLang lane |
| 5.2 | Parallelism > 1 — DP | vLLM, SGLang | **Configured; new SGLang lane awaits GPU qualification** — The framework partitions run native DP2/TP1 on a100b, using two full GPUs per replica. Both DP cores share rank-0 model identity and pull the complete weights. SGLang uses `--dp-size 2 --load-balance-method round_robin`. Shared P2P transfer assertions and MMLU source/target scores apply | A single endpoint cannot address each internal DP core separately; TRT-LLM |
| 5.3 | Parallelism > 1 — EP | vLLM, SGLang | **Configured; new SGLang lane awaits GPU qualification** — The framework partitions run vLLM GLM-4.7-Flash and SGLang DeepSeek-V2-Lite with TP2/EP2 on a100b. SGLang uses `--tp 2 --ep-size 2`; vLLM uses `--tensor-parallel-size 2 --enable-expert-parallel`. Requires rank-paired RDMA and MMLU score comparison | Per-expert subset verification; cross-node TP × EP; TRT-LLM |
| 6 | Model Streamer — S3 *(direct streaming, no server cache)* | vLLM, SGLang | **In CI** — direct-streaming matrix entries run for vLLM and SGLang on `prod-modelexpress-tester-amd-gpu-v1` in a single Docker container with no vCluster or MX server. IRSA provides S3 access; the first run uploads safetensors to `s3://ai-dynamo-modelexpress-ci/models/...`. The shared composite action dispatches vLLM's `modelexpress` load format or SGLang's `remote_instance` + `backend=modelexpress` command, then verifies the MX ModelStreamer completion marker, server health, and text inference. vLLM uses `--language-model-only` because the staged S3 model assets do not include a vision processor. After loading, a separate 300-second readiness budget covers compilation and CUDA-graph warmup; each health request is limited to 5 seconds. TRT-LLM remains gated because it has no `build_model_streamer_weight_iter` adapter ([MX-299](https://linear.app/nvidia/issue/MX-299/trt-llm-adapter)). | — |
| 7 | Dynamo aggregated serving | vLLM, SGLang | **Configured; Dynamo 1.5.1 upgrade and new SGLang lane await GPU qualification** — both frameworks build from Dynamo 1.5.1 runtime images with matching frontends. The Dynamo partitions scale the serving worker from one to two, require MX transfer, and compare worker-pinned MMLU source/target scores | TRT-LLM; runtime compatibility with the existing CI operator |
| 8 | Dynamo disaggregated serving | vLLM, SGLang | **Configured; Dynamo 1.5.1 upgrade and new SGLang lane await GPU qualification** — separate prefill/decode workers use NIXL KV transfer. The shared action scales the framework-specific decode component and compares pinned original/new decode workers using MMLU | TRT-LLM; TP × disaggregation; runtime compatibility with the existing CI operator |
| 9.1 | Metadata Backend — K8s CRD | — | **Covered implicitly** — exercised by the active P1 and P3 runs (mx-server uses `MX_METADATA_BACKEND=kubernetes`) | — |
| 9.2 | Metadata Backend — Redis | — | **Covered implicitly** — exercised by row 18's stale-metadata test, which deploys mx-server with the Redis backend (sidecar pattern at `ci/k8s/server/mx-server-redis.yaml`, `MX_METADATA_BACKEND=redis` + `REDIS_URL=redis://localhost:6379`). Confirms the Redis backend round-trips metadata correctly under the same publish + list code paths a full P2P workload would use. If we ever need explicit Redis coverage on a full source/target P2P run (vs the source-only staleness test), `run-mx-p2p-test` could grow a `backend` input to switch between mx-server manifests | — |
| 9.3 | Metadata Backend — K8s Service / Peer-Direct (PR #251) | — | **Not started** — third shipped backend. No central mx-server in this topology: sources sit behind a K8s Service, clients open direct gRPC to the Service DNS name and compute `mx_source_id` client-side. Differs from Redis/CRD on live weight refit and rolling-update recovery. Example manifests live at [`examples/k8s_service_sources/`](../examples/k8s_service_sources/) | Multi-pod Deployments with scale/restart hooks; mx-server-less manifest variant |
| 10 | MLA Model Testing (e.g. DeepSeek) | vLLM, TRT-LLM, SGLang | **In CI (vLLM + SGLang); TRT-LLM follow-up** — dedicated `P2P MLA model test (vllm TP=2)` and `P2P MLA model test (sglang TP=2)` jobs on the full-GPU A100b pool run `zai-org/GLM-4.7-Flash` for vLLM and `deepseek-ai/DeepSeek-V2-Lite` for SGLang, both MLA + MoE models, with TP=2 and VRAM parity assertions for peak and final-after-initialization memory. vLLM pins a fixed 1 GiB KV cache per GPU via `--kv-cache-memory-bytes` + `--max-model-len 1024`; SGLang fixes the pool at 131,072 tokens with `--max-total-tokens` and verifies that source and target report that capacity. The SGLang path gates the DeepSeek MLA `w_kc`/`w_vc` hidden-tensor capture (invisible to `named_parameters()`; corrupts the target if not captured + transferred) via `ci/k8s/client/sgl/manifest-azure-tp2.yaml`. | TRT-LLM variant only; TP=8 `deepseek-ai/DeepSeek-V2` requires two available full 8-GPU A100b nodes and did not schedule reliably in CI |
| 11 | GPUDirect Storage | vLLM, TRT-LLM, SGLang | **Not started** | K8s nodes with GDS-capable hardware (DGX A100/H100) + nvidia-fs kernel module |
| 12.1 | P2P Weight Transfer — EFA via NIXL libfabric plugin (`MX_NIXL_BACKEND=LIBFABRIC`) | vLLM, TRT-LLM, SGLang | **Not started** — libfabric talks to EFA directly through NIXL, bypassing UCX | AWS K8s nodes with EFA-enabled instances (e.g., p4d/p5); libfabric installed and discoverable by NIXL |
| 12.2 | P2P Weight Transfer — EFA via UCX over libfabric (`MX_NIXL_BACKEND=UCX`, UCX configured with libfabric underneath) | vLLM, TRT-LLM, SGLang | **Not started** — same fabric as 12.1 but routed through UCX with libfabric as its provider; tests the UCX integration path | Same hardware as 12.1; UCX built with the libfabric provider enabled |
| 12.3 | P2P Weight Transfer — RoCE via UCX (`MX_NIXL_BACKEND=UCX`) | vLLM, TRT-LLM, SGLang | **Not started** — UCX over Ethernet (RoCE) rather than InfiniBand; different transport than P1+P2 which use IB | K8s nodes with RoCE-capable NICs (RDMA over Converged Ethernet) |
| 13 | NIC Pinning / `MX_RDMA_NIC_PIN=auto` (PR #255 — workaround for [openucx/ucx#11259](https://github.com/openucx/ucx/issues/11259)) | — | **Not started** — multi-NIC NUMA-aware selection logic. Easy to regress silently because single-NIC nodes don't exercise it | Node with multiple IB HCAs at different NUMA distances |
| 14 | NIXL Fallback Chains (rank-matching, `ManifestMismatchError`) | vLLM, TRT-LLM, SGLang | **Not started** — 3-candidate retry then disk; covers behavior when the first source has a different rank manifest than the target. Related to the bug logged in 9 (`_query_source` picks `my_workers[0]` without liveness probe) | Way to inject deliberately-mismatched / stale sources; could publish a stale CR alongside a fresh one |
| 15 | FP8 Tensor Adoption (PR #241 — NVFP4 MoE, Kimi-K2.5) | vLLM | **Not started** — historically verified against vLLM 0.17.1 and 0.19.0. Engine-agnostic in code but only verified on vLLM | FP8/NVFP4-capable GPUs (Hopper+); Kimi-K2.5 weights in a CI-accessible registry |
| 16 | FP8 Disk Fallback / CompilationConfig Cleanup PR #244 | vLLM | **Not started** — regression target for the FP8 disk-fallback path | FP8 model + a way to force the disk fallback (e.g., suppress source publish) |
| 17 | Rolling-Update Revision-Mismatch Recovery (k8s-service handshake) | — | **Not started** — pairs with 9.3. Tests how the k8s-service backend's clients recover when a rolling update lands a new revision and old sources go away mid-pull | Multi-replica Deployment with scriptable scale/restart hooks |
| 18 | Stale Metadata After Redeploy (Redis + K8s CRD) | vLLM, TRT-LLM, SGLang | **Post-heartbeat path In CI (vLLM, SGLang); within-window race still open; TRT-LLM gated** — stale-metadata cases in the `vllm` and `sglang` partitions of the unified `test` matrix deploy a single source against mx-server (Redis backend, compressed reaper knobs threaded through the action so manifest + pytest stay aligned), force-kill it, and assert `ListSources(status_filter=READY)` returns 0 instances after the heartbeat window. The cases are backend-agnostic by construction (rolls up row 9.2). Catches reaper-scheduling / status-transition / status_filter regressions. Doesn't catch the within-window race (kill + replace inside `MX_HEARTBEAT_TIMEOUT_SECS` returns stale alongside live). TRT-LLM coverage remains gated | (1) Within-window race needs publish-time cleanup OR client-side liveness probe — independent of this test. (2) Add TRT-LLM after its MX CI entry is enabled |
| 19 | S3 Fallback | vLLM, TRT-LLM, SGLang | **In CI (vLLM, SGLang), gated (TRT-LLM)** — the `model-streamer-vllm` and `model-streamer-sglang` jobs each include a fallback case that points `MX_MODEL_URI` at a run-specific missing S3 prefix while keeping the Hugging Face model as the default-loader source; each verifies ModelStreamer fails, the default loader succeeds, the server becomes healthy, and inference works | TRT-LLM has no ModelStreamer adapter ([MX-299](https://linear.app/nvidia/issue/MX-299/trt-llm-adapter)) |
| 20 | Small Fleet (~15 workers, CRD + P2P metadata) | vLLM | **In CI** — the 15-worker fleet case is in the `large` partition of the unified `test` matrix. Raw Deployment (no Dynamo), no anti-affinity (pack a100a MIG slices), NIXL over TCP (`NIXL_UCX_TLS=tcp,cuda_copy`) because MIG slices GPU but not RDMA — each a100a node has one IB NIC, so requesting `rdma/ib: 1` per pod caps schedulable replicas to the node count. Scales in waves (1→5→10→15), each wave gated on workers reaching Ready so failures localise to a wave. Asserts ≥ fleet_size Ready CRs, permits retained Stale generations, rejects other non-Ready states and ≥ fleet_size − 1 P2P transfers. Inference correctness from P2P-loaded weights is delegated to the row 1/7 tests (`test_p2p_k8s.py`, `test_dynamo_p2p.py`); a single fleet-Service request would hit one pod non-deterministically and add no scale-specific signal. TRT-LLM/SGLang variants are follow-ups | — |
| 21 | Rolling-Update Fleet (5 workers, CRD + P2P metadata) | vLLM, SGLang | **In CI** — vLLM and SGLang rolling-update cases are in the `large` partition of the unified `test` matrix. All five large cases share a repository-wide concurrency group, so only one runs at a time. Each framework scales directly from 1→5, waits for all five workers to become Ready, then rolls with `maxUnavailable: 1`, `maxSurge: 0`, a 10-minute progress deadline, and an application `/health` readiness probe. Each replacement starts after one source terminates while four live sources remain. The full rollout has a 20-minute timeout, including scheduling and startup of all five replacements. Artifact transfer is enabled for both frameworks; vLLM pins its compile-cache directory. Readiness counts exclude cache artifact sources. Assertions require exactly 5 live Ready generations, every pre-roll Ready generation to become Stale, and a P2P completion marker with no disk fallback in every replacement pod | TRT-LLM variant |
| 22 | Pool Registration / `MX_POOL_REG=1` | — | **Not started** — collapses per-tensor NIXL registrations into region-level ones via `cuMemGetAddressRange`. Big lever on `register_tensors` cost when many NICs are visible. Assertion: region-count stays bounded on large multi-tensor models (not linear in tensor count) | Multi-NIC node + a large multi-tensor model; need a way to introspect NIXL region count (e.g., scrape NIXL agent logs or expose a count via the manager) |
| 23.1 | Cache Artifact Transfer — install path (`MX_ARTIFACT_TRANSFER=1`) | vLLM, SGLang | **Enabled across supported P2P, MLA/TP, multi-node TP, DP/EP, Mooncake, fleet, Dynamo, stale-metadata and Python RL refit workloads; install assertions remain partial** — `test_p2p_k8s.py::test_artifact_transfer` runs wherever `--require-artifact-transfer` is passed; it asserts the CRD lists ready artifact sources and the target logged `artifact install complete` for each expected type. `test_p2p_artifacts.py` self-tests the assertion logic against captured log shapes without a cluster | Full-manifest metadata mode, standalone ModelStreamer and Vime Rust serving cannot use these artifact hooks. Enabling transfer alone does not prove reuse. Install assertions only prove the bytes arrived, not that anything was gained; no wall-clock or recompile-time comparison against a no-artifact control run |
| 23.2 | Cache Artifact Transfer — effectiveness (engine actually reuses the cache) | vLLM | **In CI** — `test_artifact_transfer` fails if the target logs `was not reused and the engine recompiled`, and requires the effectiveness line whenever a `torch_compile_cache` install happened. It also requires all targets to resolve one `mx_source_id` per artifact type, catching non-deterministic identity construction. Guards the failure mode where a cache compiled under a different configuration installs cleanly and is then ignored — see [Pairing workers by compile configuration](../docs/DEPLOYMENT.md#pairing-workers-by-compile-configuration) | (a) No positive test that two *deliberately* differently-configured workers stay in separate pools — needs a manifest pair differing only in `--max-num-batched-tokens` plus distinct `MX_ARTIFACT_COMPILE_CONFIG_DIGEST` values. (b) SGLang has no equivalent check: the loader cannot observe which Inductor cache directory SGLang selected |
| 24 | RL Delta Refit — SeaweedFS object storage | Vime/Megatron, Dynamo/vLLM | **In CI** — every trusted PR uses the CPU deploy runner's rotating tbot kubeconfig and the shared example manifests to create a TP2 Vime trainer pod and a TP1 Dynamo/vLLM rollout worker on `dynamo-aws-dev-02`, bounded by a three-H100 namespace quota. The rollout uses the example's DynamoGraphDeployment and requires a healthy, compatible `v1beta1` Dynamo operator. CPU pods provide the Dynamo frontend, SeaweedFS, and Redis-backed ModelExpress, and a download Job places the pinned Qwen revision in the central FSx-backed `shared-model-cache`. The smoke requires all ten updates, verifies catalog-only `v0` is `FULL_TENSOR`, verifies updates 4 and 8 are full HF checkpoints, and verifies all other updates are XOR deltas | — |
| 25 | Multi-Token Prediction (MTP) speculative decoding | vLLM, SGLang | **Configured; new SGLang lane awaits GPU qualification** — Qwen3.5-0.8B uses native vLLM MTP or SGLang EAGLE with the same-checkpoint MTP head. Both require main and draft RDMA transfers plus MMLU scores and two 128-token long-decode requests per worker. SGLang pins matching main/draft checkpoint revisions, sets `remote_instance` for both loaders, and disables radix caching for the pinned runtime's default Mamba scheduler | Larger MTP models; TP > 1 MTP; GPU qualification of SGLang lane |

The shared inference CI model is `Qwen/Qwen3.5-0.8B`, including P2P,
ModelStreamer, Dynamo, fleet, rolling-update, and stale-metadata tests.
MLA TP2 and expert-parallel cases use `zai-org/GLM-4.7-Flash` for vLLM
and `deepseek-ai/DeepSeek-V2-Lite` for SGLang.
Both model migrations and the AIPerf checks require GPU qualification across these lanes;
the historical Qwen2.5 measurements above describe the original runs.
Standalone vLLM uses 0.19.1 with Transformers 5.5.1 so `glm4_moe_lite`
configuration is recognized at image build time. SGLang uses its default
attention backend; native source accuracy remains a required gate. The SGLang GLM migration is deferred until the streamed-weight
ownership fix is qualified separately.

The shared P2P test runs AIPerf 0.13.0's built-in MMLU loader and
multiple-choice grader: eight subjects, the first 16 test questions per subject,
128 questions total, one-shot prompting, chat requests with thinking disabled,
greedy decoding (`temperature=0`, `seed=0`), concurrency one, and an eight-token answer cap and an explicit single-letter system instruction. Each subject uses a separate native AIPerf invocation to
preserve equal representation without replacing its dataset loader or grader.
The standalone action captures and validates the source baseline before
deploying a target, then reuses those records for the final comparison. An
unusable source therefore fails without paying for target startup, and its
results remain available if target infrastructure cannot start.

Source and target must receive identical messages and ground truths. The score
gate uses a one-sided exact paired test: among questions whose correctness
changes, reject a target regression with `p < 0.01`. Wording, answer-length,
and per-question grade differences remain diagnostic. This is a regression
smoke test, not proof of score equivalence: seven losses without gains trigger
failure at 128 questions, while six do not. No absolute accuracy floor is imposed.
At least 95% of each worker's answers must parse as A/B/C/D; grader regex
fallbacks are counted separately. Profiling stops early once completed subjects
exceed the whole-run invalid-answer allowance. Failed, empty, duplicate, or missing responses
and invalid/missing grades fail independently of the score.

`timing.json` includes client startup and dataset loading per subject. Native
AIPerf exports retain inference durations. Server-metrics discovery and GPU
telemetry are disabled for these accuracy runs. Normal lanes run the
128-question source/target pair. An explicit `--accuracy-pilot` additionally
measures a 64-question source run and repeats the 128-question native baseline.
GPU calibration in run `39879750920` measured 58.6 seconds for 64 questions and
68.7 seconds for 128 on a warm vLLM source, with identical 60/128 scores across
the two 128-question source runs. The initial vLLM source/target pair took
327 seconds; SGLang Mooncake, DP2, and MTP pairs took 178–186 seconds, with
62/128 scores on both workers and all answers parsed. These timings include
AIPerf startup and dataset loading, but exclude deployment and model startup.
MTP lanes also execute two 128-token chat
requests per worker with EOS stopping disabled; this checks sustained decoding
without requiring identical wording.
Instrumented vLLM base/MTP workers separately compare SHA-256 hashes of their
registered main/draft transfer catalogs. Temporary draft embeddings replaced
by vLLM's main-model sharing are excluded by the native MX catalog. This gate
does not claim byte verification for uninstrumented worker images. Each tensor
uses a short independent log record to avoid interleaving large JSON lines.
The builtin AIPerf tokenizer avoids model downloads on the control runner;
server-reported token counts are used, and benchmark messages are sent unchanged.

Dynamo captures the source baseline before scaling, discovers both serving
worker IDs, and pins each target request using `nvext.backend_instance_id` and
`nvext.decode_worker_id`. Every raw response must report that target ID.
Before profiling a discovered target, a pinned chat request must succeed and
report its ID. Only the specific unknown-worker HTTP 400 is retried, for up to
180 seconds; a wrong serving worker or another HTTP error fails immediately.
Source-only routing fails. The existing metadata and RDMA checks remain required.
Both worker images prefetch the complete non-weight HF metadata before engine
startup. This includes the safetensors index, which a cold native load would
otherwise add only to the source cache. Dynamo 1.5.1 hashes these sibling files
into deployment-card compatibility, so differing inventories can exclude an
otherwise discovered target from routing. The action saves worker discovery
cards in `worker-discovery.json` to diagnose any remaining mismatch.
Native vLLM DP cores behind a single API endpoint are not individually selected.

Both composite actions upload `inference-<namespace>` artifacts on success or
failure, containing AIPerf commands, logs, raw exports, and comparison reports,
plus native accuracy summaries and per-example grades. `dataset.json` records
the benchmark, dataset/subjects/split, selected example IDs, and hashes of their
messages and ground truths; `inputs.json` retains the inputs. Hugging Face
caches downloads. AIPerf's built-in loader resolves `lighteval/mmlu` at `main`,
so the dataset revision is not pinned, but source/target input drift fails.
There is no absolute accuracy threshold for these small models. This subset
does not exhaustively activate experts or qualify quantized paths.
GLM's MTP head is not exercised by the ordinary MLA/EP lanes.

## Status legend

- **In CI** — matrix entry is uncommented, the test runs on every PR, currently passing.
- **Partial** — some aspect of the area is covered as a side effect of another row, but no targeted test exists.
- **Covered implicitly** — no separate test, but the functionality is exercised by another test that would fail if it broke.
- **Not started** — no scaffold, no matrix entry, no dependency planning yet.

When a row moves status, update this table in the same PR so the doc stays the source of truth on what CI does and doesn't catch.

The existing Vime smoke uses a digest-pinned SeaweedFS S3 fixture, shared with
the benchmark CI workflow.

## Comment-triggered model benchmark CI

`ci/bench/` contains registered Nemotron and Kimi-K2.7-Code profiles. Offline
contracts run on every PR in `RL refit harness tests`. `modelexpress-rl-ci.yml` accepts `/bench [--model PROFILE]
[--scenario delta] [--sha FULL_SHA]` from repository writers after copy-pr-bot mirrors the current
head. Every trusted PR also runs the one-GPU `RL generator-only delta refit (nemotron)`
S3 delta-weight refit through
`modelexpress-rl-ci.yml`; `CI status check` requires its build, GPU test, and cleanup to succeed.
Automatic and comment-triggered CI use the Vime example's SeaweedFS setup in an isolated
namespace and upload the pinned seed snapshot before measurement; they need no AWS IAM role.
Kimi remains opt-in. The reusable workflow independently validates the current
approved mirror before building the requested SHA. Only delta is registered; no reshard CI case
is introduced. GPU runs use profile-derived quotas; Nemotron defaults to a pinned
copy of the existing CI vLLM runtime, with a repository override available. Kimi
requires a per-model runtime override. No live qualification of
the new Nemotron benchmark has been performed; Kimi refit compatibility is
not established. Peer coverage stays manual. See
[usage and limitations](../docs/DEPLOYMENT.md#modelexpress-benchmark-ci-harness).

Benchmark success requires both weight/checkpoint validation and complete latency
measurements. Reports retain per-rank times and the slowest rank; failed trials are excluded from successful summaries. This is independent
of the existing Vime/Dynamo E2E smoke and does not enforce performance thresholds.

## Workflow responsibilities

`modelexpress-e2e-ci.yml` calls independent `modelexpress-inference-ci.yml` and
`modelexpress-rl-ci.yml` workflows. The existing required `CI status check` requires
both to succeed. Each child checks all of its test and cleanup results.
Inference owns image builds, ModelStreamer, and the reusable GPU test workflow.
RL owns Vime and generator-only refit, including their builds and cleanup.
One RL `GPU cleanup` fallback job handles both workloads after both tests finish;
each cleanup path runs only when its test needs it, and Vime cleanup still runs
if refit cleanup fails. Normal inline cleanup remains in each test job.
`modelexpress-rl-refit-gpu-ci.yml` and `modelexpress-rl-vime-gpu-ci.yml`
contain one GPU job each, avoiding skipped jobs for the other suite. One
workflow-level group in `modelexpress-e2e-ci.yml` cancels superseded PR runs
while repository-wide child groups retain GPU slots. Cleanup jobs use `always()`
so cancellation still runs independent fallback cleanup. Inference
and RL skip it when the GPU tests and their inline cleanup succeed; failures and
cancellations retain fallback cleanup. Status checks accept a skipped fallback
only when its corresponding tests succeeded.
The RL validation job also handles `/bench` authorization with the existing
writer and approved-mirror checks; comment runs execute only the selected refit profile.
Ordinary comments do not start RL builds, GPU tests, or Vime jobs.

`nightly-ci.yml` reuses the core, inference and RL workflows on upstream `main`.
Scheduled and manual runs authorize only their exact event SHA and use the run ID
for GPU slot selection. They remain outside PR supersession, while the GPU
capacity queues are shared with PR tests. The nightly status gate
requires all selected test suites and accepts skipping them only when
`run_tests=false`. RL evidence uploads are disabled for nightly calls.

## CI build caches

Image builds import and export registry caches in their existing NVCR image
repositories. Each image exports one `buildcache-shared` tag so a new approved PR
can reuse matching layers from other PRs. Builds import only that shared cache.
Concurrent exports replace the shared cache;
BuildKit still checks input keys before reusing a layer. Vime trainer and worker
build concurrently on the same runner with separate caches, and both must succeed.
`mode=max` preserves intermediate build layers;
cache export failure does not fail an otherwise successful build. The pinned
Dynamo sidecar builds before copying ModelExpress sources so ModelExpress-only
edits can reuse that compile layer. The first shared-cache build is cold;
subsequent PRs can reuse matching layers without changing GPU allocations.
Inference GPU checkouts abort transfers stalled below one byte per second for
60 seconds, allowing the checkout action's existing fetch retries to recover.

## vCluster connectivity recovery

Python port-forward diagnostics suppress routine forwarding and connection messages
while retaining tunnel errors and restart messages.

CI supervises each vCluster API tunnel and pytest service tunnel, restarting an
exited `kubectl port-forward` on the same local port. Existing job cleanup stops
the supervisor and its child. vCluster diagnostics include its timestamped tunnel
log; pytest prints service tunnel logs. Dynamo metadata polling preserves API
errors and retries failed reads within the existing readiness deadline. Worker
liveness polling refreshes active pods on every pass, excludes terminating pods,
and checks the DGD replica count within one shared deadline.

Pytest API `get` calls retry connection refusal, resets, and unexpected EOF up to
three times, including API request and TLS handshake timeouts. Completion requests retry connection refusal or reset twice, with a
fresh request through the reconnecting tunnel. HTTP errors, malformed responses,
and model correctness assertions still fail. These measures recover brief tunnel
interruptions; they do not establish the cause of the host API resets.

## GPU test admission and readiness budgets

Dynamo aggregated, prefill, and decode CI workers use a 1 GiB KV cache,
1,024-token context and batch limits, and at most 16 concurrent sequences to
leave GPU memory headroom for sampler warmup on A100a MIG slices. These smoke
tests exercise short requests, scaling, P2P weights, and disaggregated KV transfer.

Inference worker builds have independent jobs for vLLM, SGLang, Mooncake, and
Dynamo vLLM. Each framework's GPU tests depend only on its worker image, the
server image, and vCluster setup. Model Streamer depends only on its worker image.
Each framework call expands one matrix containing only its actual test cases,
with shared setup, diagnostics and GPU queues; no unrelated test jobs are created.
One fallback cleanup waits for all framework calls; the final status check requires
every framework build and test to succeed.

Replacement GPU tests become eligible after their image builds and setup finish.
The E2E caller cancels a superseded PR run as one unit; its `always()` cleanup
jobs continue after cancellation. Nightly calls are not superseded.

The parent workflow calls inference GPU tests per framework without additional
supersession groups. A100a defaults to 8 repository-wide test
slots; A100b defaults to one concurrent test requiring four full GPUs at peak.
`MX_CI_A100B_CONCURRENCY` can override its limit to 1–4, and the inference `MX_CI_GPU_CONCURRENCY` limit also applies. Inference cases have distinct indices rotated by PR number and reduced
modulo the configured limit; RL refit and Vime use PR-based slots per Kubernetes
context. One complete workload occupies each slot through execution and cleanup.
Builds and independent cleanup remain outside these queues. GitHub's `queue: max`
retains pending jobs. After all 15 small Kubernetes GPU cases in the same run succeed, five large cases
become eligible: multi-node TP2, Dynamo disaggregated, the 15-worker fleet, and the
vLLM and SGLang 5-worker rolling tests. Their required builds and vCluster setup
must have succeeded. These cases share the separate `modelexpress-gpu-a100a-large-workload`
group and `large-workload` cleanup lane. Only one of these tests runs across PRs at a time. Ordinary A100a tests keep their
eight-slot limit; small tests from other PRs may still run alongside a large test.
Model Streamer and RL use separate pools and do not gate this phase.

`MX_CI_GPU_CONCURRENCY` overrides the default of 8 and accepts integers from 1 to 16; 1
serializes ordinary tests in each inference pool through `slot-0`; the shared
large-workload group remains separate. Eight is an experimental default, not a
capacity reservation; slot collisions or runner availability may reduce actual
concurrency. Drain current runs (or cancel and finish cleanup) before changing
limits, because changing slot assignments does not move already queued or
running jobs. The cleanup guard streams cluster JSON to jq so large pod lists
do not exceed operating-system command argument limits.

P2P sources and targets, the fleet seed, and every fleet scale wave wait for all
controller-desired pods to be scheduled before starting their startup deadline.
P2P jobs allow one replacement when kubelet admission fails because the assigned
node has no healthy `rdma/ib` device, during scheduling or startup. Kubernetes
owns the retry; failed attempts remain available for diagnostics and are excluded
from active-pod selection. A second occurrence exhausts the job retry;
other admission failures, crashes, and OOMs still fail immediately.
Scheduling and startup each retain the configured readiness budget; each GPU
test job also has a two-hour limit. Both MLA framework variants use 900 seconds
per scheduling/startup phase instead of 3,600 seconds. Source/target topology, fleet size, metadata,
artifact and inference assertions are unchanged. Shell polling preserves API
stderr and retries transient read interruptions three times, with bounded API
requests; permanent read errors fail. A rollout retains its existing total
rollout budget rather than extending it for each replacement.

## Capacity recovery and qualification

GPU test namespaces include the run ID and attempt in their names and carry a
`ci.modelexpress.nvidia.com/run-id=<run>-<attempt>` owner label. Cleanup refuses
unowned namespaces, stops the owned workload controllers, requests normal
namespace deletion, and verifies absence through successful API reads. It also
waits for the corresponding host pods to disappear. API errors cannot masquerade
as successful cleanup. Cleanup preserves namespace/metadata finalizers and reports
stalled finalization. The parent `GPU test cleanup` job independently reconnects
after the test workflow finishes or is cancelled, retries remaining namespaces
owned by the attempt, and checks surviving host pods. Fallback cleanup is skipped
when tests succeeded or were skipped before starting; failed or cancelled tests
retain it. The umbrella status check
requires cleanup success. Legacy namespaces without ownership labels are not
automatically deleted.

Fleet and rolling workers request 2 CPU cores by default (30 cores for a complete
15-worker fleet, formerly 60). `MX_CI_FLEET_CPU` overrides that integer request;
use 4 to restore the previous reservation if live startup measurements regress.
These are scheduling reservations; CPU usage and performance still require live
qualification.

SGLang CI manifests select ModelStreamer through `MX_MODEL_URI` for the first
replica when no compatible P2P source exists. The pinned SGLang loader downloads
safetensors from the configured Hugging Face model and streams the local files;
no object-store credentials are required. Later replicas still try P2P first.
This applies to NIXL P2P, TP/MLA, rolling-update, and stale-metadata cases.
Mooncake retains its separate native-first loading path.
The existing P2P and inference assertions remain in place. Live GPU qualification
of this CI configuration remains pending.

## GPU admission during cleanup

Before a slot starts a new workload, it waits up to 900 seconds for previous CI
namespaces and physical GPU pods in that slot to disappear, including after
cancellation. Namespace cleanup allows 600 seconds for vCluster pod removal
and namespace finalization, then verifies host pod removal. A DP2 run needed
about eight minutes to finalize after deletion, exceeding the previous 180-second
budget. Ownership checks and finalizers remain intact. The guard only reads
namespaces in the same labeled slot; inference also checks vCluster host pods
using their mirrored pool/slot labels, including after virtual namespace deletion.
Reads use server-side label selectors and the host namespace instead of listing
all cluster pods. RL namespace finalization already covers its directly hosted
pods. Other slots, unlabeled legacy workloads, and previous concurrency modes
are left to Kubernetes scheduling. API errors fail closed after bounded retries,
including timeouts while reading the response body. GitHub capacity concurrency
groups remain the queue. Normal cleanup stays inside test jobs, and `always()`
fallback cleanup continues when a PR workflow is cancelled.

RL GPU jobs default to 4 repository-wide slots per Kubernetes context,
configured independently with `MX_CI_RL_GPU_CONCURRENCY` (1–16). Refit
runs choose PR number modulo the configured limit; Vime uses PR number plus one
modulo that limit. This bounds concurrency across PRs without a reservation
ledger. One-job mode serializes both workloads. Normal namespace deletion finishes before
releasing a slot. Independent cleanup remains outside the queues. Refit jobs have
a two-hour budget including admission, setup, execution and cleanup.

The 8/1/4 defaults for A100a/A100b/RL need live qualification of CPU, memory,
GPU/MIG and RDMA headroom.
Lower the limit when scheduling delays or resource contention appear. Source/target,
artifact, metadata and model assertions remain unchanged. These limits neither
reserve cluster-wide capacity nor guarantee placement against other workloads.
