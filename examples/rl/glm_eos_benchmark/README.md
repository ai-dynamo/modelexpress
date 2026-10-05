# GLM-5 on Eos: reproduce the October 2 benchmark

This branch packages the configuration and launch tools for the Python-only
DIRECT installation plus trainer COPY_TO_HOST run. It is based on measured MX
commit `2ec839b32bb983f02fff08d7c762fbf08486cf38`, not current main. No PR is
required to use it. The scripts launch actual training and weight updates.

## Configuration and recorded result

| Setting | Value |
| --- | --- |
| Cluster | Eos, Slurm batch, Pyxis/Enroot, H100 |
| Total | 36 nodes, 288 GPUs |
| Trainer | 32 nodes / 256 GPUs; PP1, TP1, EP32, FSDP256 |
| Generator | 4 nodes / 32 GPUs; PP1, TP32, EP32, one replica |
| Model | zai-org/GLM-5, revision c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2 |
| Inference | BF16, no quantization; selected FP32 parameters preserved |
| Publication | COPY_TO_HOST, reusable pinned CPU staging |
| Installation | DIRECT, one GPU receive arena, 2 GiB budget |
| Workload | Batch 256, sequence length 512, 11 steps |
| Timing | Updates 2-11; reference replay and profiling disabled |

The recorded clean run (job 6140783) had receiver E2E median **4.950 s**,
installation median **2.748 s**, and framework broadcast median **6.171 s**.
These are historical results, not guarantees for a new allocation. The CSVs in
`results/` include per-update values, aggregate min/median/P95, and a glossary.
The receive arena metric was 1.77 GiB; it excludes installer scratch.

`config/` contains the configuration inputs and host-resolution scripts for
review. They are reference files, not a standalone launch directory: the renderer
uses the checksummed baseline bundle described below and applies the final MX
overlay. Do not submit `config/run.sbatch` directly.

## Required shared assets

This is an **Eos reproduction recipe using existing immutable assets**, not a
standalone Docker build for another cluster. You need permission to read the
following task root and create new run/output directories there:

```bash
export GLM_ASSETS=/lustre/fsw/coreai_tritoninference_triton3/kavink/glm-eos-investigation
```

| Path under GLM_ASSETS | Purpose |
| --- | --- |
| optimized-pr-completion-20260929-v1/runtime.sqsh | Pinned container with the tested Torch/vLLM/NIXL runtime |
| runs/glm030-clean-v2 | Checksummed full launch template, Prime overlay, model configuration and report tools |
| python-warm-final-20261002/gpu-gate-v1 | Frozen tested MX source and successful GPU gate receipts |
| python-direct-validation-20261001/server-build | Matching server binary and source/checksum receipts |
| models | Hugging Face snapshot and converted Prime training checkpoint |
| data | Workload data and output root |

Image SHA256: `56b4c754ec16c6a202482acffcfecb29dd412f7b0c2068ff59fe6729c65b8803`.
Server SHA256: `776a46ffccc469a836358bb5784cb6825ebc44c66b5aea0f8fb2fdbd21c23fce`.
The launcher validates these artifacts on compute nodes. Images, checkpoints,
source archives, credentials and raw logs are not included in this Git branch.

The full benchmark uses Prime `700c8c4979805eb53fd87e75d9895bc67bf5450e`
with its frozen benchmark overlay. The separate one-node integration gate uses
Prime `9aba3ffe3cd4a14595a462d30f2a80c72f7cfcd3`. These are distinct drivers.
The frozen MX source `22296a6` is runtime-identical to the published `2ec839b`.
The server was built from `abc75671aff09043874f825c88d11f8c05305dfa`; its
Rust/protobuf/Cargo inputs match this measured Python revision.

## Launch the measured configuration again

Check the active reservation with `scontrol show reservation`. Do not assume a
previous reservation or date remains valid. This full run needs 36 H100 nodes;
the small gate needs one node. Follow the site's login-node and session-lock rules.

```bash
git clone --branch kavink/glm-eos-reproduction-20261005 --single-branch \
  https://github.com/ai-dynamo/modelexpress.git modelexpress-glm
cd modelexpress-glm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GLM_ASSETS=/lustre/fsw/coreai_tritoninference_triton3/kavink/glm-eos-investigation
export GLM_RESERVATION=YOUR_ACTIVE_RESERVATION
export GLM_PREFIX=glm-repro-20261005a
export GLM_LAUNCH="$GLM_ASSETS/launchers/$GLM_PREFIX"

python3 examples/rl/glm_eos_benchmark/prepare.py \
  --assets "$GLM_ASSETS" --output "$GLM_LAUNCH" \
  --run-prefix "$GLM_PREFIX" --reservation "$GLM_RESERVATION"
cd "$GLM_LAUNCH"
sha256sum -c SHA256SUMS
sbatch --parsable --reservation="$GLM_RESERVATION" launch.sbatch correctness
```

Use a fresh prefix every time. `prepare.py` refuses existing run IDs and does not
submit jobs. Submit from the generated directory: nested jobs use
`SLURM_SUBMIT_DIR`. The reservation is propagated to child jobs. The account is
`coreai_tritoninference_triton3`; users need an association with that account.

The sequence is correctness -> clean performance -> report/export, with Slurm
`afterok` dependencies. Correctness performs initial checks and 96 warm
comparisons (32 receivers, updates 2-4). Its update-6 profile is diagnostic.
Performance requires a matching correctness receipt and disables profiling and
warm reference replay. Failed qualification stops the sequence.

The historical workload durations were about 31 minutes for correctness,
19 minutes for performance and 2 minutes for export, excluding queue time and
launcher jobs. Check job IDs in the generated launch directory and use a single
`squeue -u "$USER"` or `sacct -j JOB_ID` query when needed; do not poll in loops.

Outputs:

- Generated run bundles: `$GLM_ASSETS/runs/${GLM_PREFIX}-correct` and `-clean`.
- Raw workload logs: `$GLM_ASSETS/data/outputs/${GLM_PREFIX}-correct` and `-clean`.
- CSVs and log archives: `$GLM_ASSETS/evidence/${GLM_PREFIX}-clean`.
- Launcher job IDs and logs: `$GLM_LAUNCH`.

## Rebuild the Python overlay and rerun the one-node gate

The default launch reuses the already-qualified frozen gate. To validate a fresh
snapshot of this branch, first make clean MX and Prime checkouts. In a Prime
repository containing the gate commit, create a worktree at
`9aba3ffe3cd4a14595a462d30f2a80c72f7cfcd3`. Set `PRIME_GATE_SOURCE` to that
worktree and `MX_SOURCE` to this clean ModelExpress checkout. Choose a new gate
path on shared storage:

```bash
export GLM_GATE="$GLM_ASSETS/gates/glm-repro-20261005a"
python3 "$MX_SOURCE/examples/rl/glm_eos_benchmark/followup/recipe/freeze_snapshot.py" \
  --prime "$PRIME_GATE_SOURCE" --mx "$MX_SOURCE" \
  --pr749-sha b2dfe3eb4468344158316051e861cc3a66cdc06c \
  --output "$GLM_GATE"
cd "$GLM_GATE"
sbatch --parsable --reservation="$GLM_RESERVATION" run.sbatch
```

This installs the frozen MX Python packages into the pinned runtime on a compute
node, verifies dependencies and sources, and tests real changed-weight transfers.
It does not rebuild the entire container. Once the gate completes successfully,
pass `--gate "$GLM_GATE"` to `prepare.py` and launch the full sequence above.

For a different Rust/protobuf/Cargo revision, the freezer deliberately refuses
the old server. Build and qualify a matching server before adapting the recipe.
Likewise, changing vLLM, Torch, NIXL, model topology or the image requires fresh
qualification. The historical image build package is retained at
`$GLM_ASSETS/optimized-pr-completion-20260929-v1`; it depends on earlier frozen
build assets and is not a portable from-scratch build recipe.
