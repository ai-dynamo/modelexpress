# K8s-Service Artifact Transfer Test Results

Last run: 2026-09-08 (America/Los_Angeles)

This file records the end-to-end GPU validation of artifact transfer through
the `k8s-service` metadata backend. It is an environment-specific test record,
not a performance guarantee.

## Environment

- Branch: `feature/k8s-service-artifact-transfer`
- Commit: `05203a02562e9ac869c7ccbf3ab582656de65be5`
- Kubernetes context: `nebius-2`
- GPUs: NVIDIA H200, up to 8 GPUs per node
- Transport resources: one `rdma/shared_ib` resource per GPU
- Test image:
  `nvcr.io/nvidian/dynamo-dev/zhongdongmin@sha256:0d68f867b9d9c1429ab8918cdc31640f3e8bb4c012538574648ff354e3d8e42b`
- vLLM: 0.17.1
- NIXL: 0.10.1, UCX backend

The cluster image contains the artifact routing implementation under test. It
predates the final one-line change that classifies an absent optional artifact
as gRPC `NOT_FOUND` instead of `FAILED_PRECONDITION`; that exact change is
covered by the focused Python tests on the recorded commit.

Every topology used an isolated namespace. The source became Ready before the
target was created, so the Service initially contained only valid source Pods.
Source and target were placed on different nodes for TP1, TP2, TP4, and TP8.

## Results

| Case | Model | GPU topology | Result | Artifact verification | Weight verification | Inference |
|---|---|---:|---|---|---|---|
| TP1 cross-node | Qwen2.5-0.5B-Instruct | 1 source + 1 target | PASS | Triton marker transferred; 4.82 MiB in 0.252 s | 0.99 GB in 0.029 s, 269.2 Gbps | PASS |
| TP2 cross-node | Qwen2.5-0.5B-Instruct | 2 source + 2 target | PASS | Triton 4.95 MiB in 0.218 s; FlashInfer 0.02 MiB in 0.197 s | Rank 0: 0.50 GB at 237.8 Gbps; rank 1: 0.50 GB at 156.1 Gbps | PASS |
| TP4 cross-node | Qwen3-0.6B | 4 source + 4 target | PASS | Triton 4.95 MiB in 0.235 s; FlashInfer 0.02 MiB in 0.215 s; source marker matched | Four rank-local transfers of 0.31 GB, 3.2-6.9 Gbps | PASS |
| TP8 cross-node | Qwen3-0.6B | 8 source + 8 target | PASS | Triton 4.95 MiB in 0.240 s; FlashInfer 0.02 MiB in 0.235 s; source marker matched | Eight rank-local transfers of 0.16 GB, 68.7-109.1 Gbps | PASS |
| Mixed Service | 1 compatible + 7 incompatible sources, 2 concurrent targets | 10 GPUs total | PASS with explicit high retry limit | Both targets rejected incompatible endpoints, resolved the compatible Pod IP, transferred the compatible marker, and installed Triton and FlashInfer | Both targets loaded 1.20 GB; 275.7 and 302.8 Gbps | PASS on both targets |

No case was skipped for insufficient capacity. These throughput numbers use a
small model and include no controlled benchmark warmup, so they are evidence of
the working RDMA path rather than comparative performance data.

## Multi-rank behavior observed

- Weight transfer is rank-local. TP8 produced one successful
  `GetTensorManifest` and one NIXL transfer for each rank 0 through 7. The
  Service exposed ports 6555 through 6562, mapping to the corresponding source
  worker in the source Pod.
- Artifacts are pod-local, not rank-local. All target ranks may enter artifact
  installation, but a shared file lock permits one transfer per artifact per
  Pod. The winning rank is not fixed: in TP8, rank 0 installed the Triton cache
  while rank 7 installed the FlashInfer cache.
- Initial artifact discovery went through the Service. The returned manifest
  header contained the source Pod's direct endpoint, and all chunk/lease RPCs
  remained pinned to that endpoint. For TP8 this was
  `10.53.12.24:6555`.
- Only device 0 published artifacts. The other seven TP8 workers logged the
  expected owner-device skip.

## Mixed-Service selection result

The mixed test deliberately put eight Ready endpoints behind one Service:

- seven Qwen2.5 endpoints that could not satisfy the Qwen3 identities;
- one compatible Qwen3 endpoint at `10.53.65.160:6555`;
- two Qwen3 targets starting concurrently on separate nodes.

Fresh-channel retries did move across kube-proxy backends. Both targets
eventually selected the sole compatible endpoint, and the incompatible Pods
successfully served zero target manifests or artifact chunks after the targets
started. Observed successful attempts were:

| Target | Triton artifact | FlashInfer artifact | Weights |
|---|---:|---:|---:|
| target 1 | attempt 6 | attempt 2 | attempt 13 |
| target 2 | attempt 2 | attempt 5 | attempt 1 |

This test also defines the operational boundary: the default configuration has
five retries, or six total attempts. It is therefore not sufficient to make an
arbitrarily heterogeneous Service reliable. A production Service selector must
form a compatible pool (same model/revision/topology for weights and compatible
runtime identity for artifacts). Retries handle transient rollout overlap and
occasional stale routing; they are not a replacement for a metadata index.

## Local regression coverage

- Focused Python artifact and k8s-service tests: 125 passed
- Python CPU suite: 1,092 passed, 1 skipped
- Rust artifact tests: 9 passed
- Go package build/tests: passed
- `cargo fmt --check`: passed

The full local Python run additionally found 17 collection errors caused by the
local environment not having the optional `transformers` package; the same
1,092 runnable tests passed.

## Cleanup

The TP4, TP8, and mixed-Service namespaces were deleted after evidence was
collected. TP1 and TP2 namespaces had already been deleted after their runs.
No test Service, Deployment, Pod, Secret, or PVC was left running.
