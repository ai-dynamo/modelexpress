#!/bin/bash
#SBATCH --job-name=coreai_tritoninference_triton3-mx.python-warm-followup
#SBATCH --account=coreai_tritoninference_triton3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PACKAGE=$PWD
source ./run.env
export SBATCH_RESERVATION
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c SHA256SUMS
MODE=${1:?correctness or performance}
COMMON=(--baseline "$BASELINE"
 --gate "$GATE"
 --server-build "$SERVER_BUILD"
 --task-root "$TASK_ROOT" --warm-helper "$PACKAGE/mx_749_warm_reference.py")
case "$MODE" in
 correctness)
  RUN="$TASK_ROOT/runs/$CORRECTNESS_RUN"
  python3 recipe/render_glm.py "${COMMON[@]}" --run-id "$CORRECTNESS_RUN" --arm correctness    --install-profile-step 6 --install-profile-cprofile    --install-profile-helper "$PACKAGE/recipe/profile_installation_alias.py" > "$PACKAGE/correctness-render.log"
  cd "$RUN"
  JOB=$(sbatch --parsable run.sbatch benchmark)
  printf '%s\n' "$JOB" > "$PACKAGE/correctness-job-id.txt"
  cd "$PACKAGE"
  NEXT=$(sbatch --parsable --dependency="afterok:$JOB" --kill-on-invalid-dep=yes launch.sbatch performance)
  printf '%s\n' "$NEXT" > "$PACKAGE/performance-launcher-job-id.txt"
  ;;
 performance)
  LOGS="$TASK_ROOT/data/outputs/$CORRECTNESS_RUN"
  test "$(cat "$LOGS/batch-exit.txt")" = 0
  test "$(cat "$LOGS/workload-exit.txt")" = 0
  RUN="$TASK_ROOT/runs/$PERFORMANCE_RUN"
  python3 recipe/render_glm.py "${COMMON[@]}" --run-id "$PERFORMANCE_RUN" --arm performance    --qualification "$LOGS/recovered-verification/cache-arm-check/result.json" > "$PACKAGE/performance-render.log"
  cd "$RUN"
  JOB=$(sbatch --parsable run.sbatch benchmark)
  printf '%s\n' "$JOB" > "$PACKAGE/performance-job-id.txt"
  REPORT=$(sbatch --parsable --dependency="afterok:$JOB" --kill-on-invalid-dep=yes report/report.sbatch "$RUN" "$JOB")
  printf '%s\n' "$REPORT" > "$PACKAGE/report-job-id.txt"
  ;;
 *) echo 'Unknown mode' >&2; exit 2;;
esac
