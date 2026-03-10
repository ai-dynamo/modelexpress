# GCP GB200 Node Status — dynamo-gcp-dev-01, kavin namespace

Last updated: March 9, 2026

## Node pools

- **customer-gpu-o7v**: ARM64 GPU nodes (GB200 NVL36)
- **customer-gpu-w0e**: ARM64 GPU nodes (GB200 NVL36)
- **customer-gpu-mh2**: ARM64 GPU nodes (GB200 NVL36)
- **customer-cpu**: CPU-only nodes (for MX server, Redis, etcd, NATS)

## Node status from this session

### Known good (earlier, now unreliable)

| Node | Pool | IP | Evidence |
|------|------|----|----------|
| `o7v-408414f9-2x2c` | o7v | 10.0.0.91 | DGD source worked earlier (published all 4 ranks). Later crashed with NCCL errors during autotuning — cluster-wide issue suspected. |

### Known bad (NCCL failures)

| Node | Pool | IP | Issue |
|------|------|----|-------|
| `o7v-408414f9-6b99` | o7v | 10.0.0.52 | Repeated NCCL `unhandled system error` during executor init/autotuning. 2 consecutive crashes. |
| `o7v-408414f9-2x2c` | o7v | 10.0.0.91 | NCCL failure during autotuning (after 78 min of successful loading). Was good earlier — suggests cluster-wide GPU fabric degradation over time. |

### Cluster-wide NCCL issue (March 9, 2026)

**Every node tested fails** with `NCCL error: unhandled system error` for any TP>1 workload. Tested on both `o7v` and `w0e` node pools with both Kimi K2.5 (TP=4) and Qwen 0.5B (TP=2). The error occurs at `preallocateNCCLWindowBuffer` during basic NCCL init — not autotuning or CUDA graphs.

TP=1 works fine (Qwen 0.5B published successfully on `o7v-2x2c`).

Earlier today (3+ hours prior), Kimi K2.5 TP=4 ran successfully on `o7v-2x2c` with full autotuning + CUDA graphs. The NCCL issue started after that run.

**Nodes tested with TP>1 (all failed):**
- `o7v-408414f9-6b99` (o7v pool) — Kimi TP=4, our image
- `o7v-408414f9-2x2c` (o7v pool) — Kimi TP=4, our image (was good earlier)
- `o7v-408414f9-mflg` (o7v pool) — Kimi TP=4, our image
- `o7v-408414f9-b2bd` (o7v pool) — Kimi TP=4, our image
- `o7v-408414f9-9lmw` (o7v pool) — Qwen TP=2, karenc's unpatched base image
- `w0e-59ca5514-z02x` (w0e pool) — Qwen TP=2, our image

**Action needed:** Report to cluster admins (`dynamo-gcp-dev-01`). NCCL GPU fabric is down cluster-wide.

### Untested

| Node | Pool | IP |
|------|------|----|
| `o7v-408414f9-9lmw` | o7v | 10.0.0.98 |
| `o7v-408414f9-b2bd` | o7v | 10.0.15.197 |
| `o7v-408414f9-bnkw` | o7v | 10.0.0.37 |
| `o7v-408414f9-bpts` | o7v | 10.0.0.41 |
| `o7v-408414f9-cl6v` | o7v | 10.0.15.208 |
| `o7v-408414f9-gv96` | o7v | 10.0.15.218 |
| `o7v-408414f9-jxt8` | o7v | 10.0.0.96 |
| `w0e-59ca5514-0qvv` | w0e | 10.0.0.107 |
| `w0e-59ca5514-3g4s` | w0e | 10.0.15.196 |
| `w0e-59ca5514-b7mp` | w0e | 10.0.0.64 |
| `w0e-59ca5514-bcsj` | w0e | 10.0.0.80 |
| `w0e-59ca5514-bm29` | w0e | 10.0.0.88 |
| `w0e-59ca5514-ckjt` | w0e | 10.0.0.104 |
| `w0e-59ca5514-crbt` | w0e | 10.0.0.67 |
| `w0e-59ca5514-cwmt` | w0e | 10.0.15.203 |
| `w0e-59ca5514-hhss` | w0e | 10.0.0.106 |

## Notes

- The DGD source that worked used `o7v-2x2c` (10.0.0.91). Target was on `o7v-6b99` (10.0.0.52) — UCX connection refused to source.
- `o7v-6b99` has persistent NCCL issues — all ranks fail during allgather in MoE autotuning.
- When deploying, exclude `o7v-6b99` via `kubernetes.io/hostname NotIn` affinity.
- Previous standalone MPI source also ran on `o7v` pool nodes successfully.
- The `w0e` pool nodes were only used for frontend pods (no GPU workload tested).
