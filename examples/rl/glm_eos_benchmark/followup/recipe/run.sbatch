#!/bin/bash
#SBATCH --job-name=coreai_tritoninference_triton3-mx.python-warm-gate
#SBATCH --account=coreai_tritoninference_triton3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen validation directory}"
source ./deployment.env
sha256sum -c SHA256SUMS > source-sha256-check.log
printf '%s  %s\n' "$VALIDATION_IMAGE_SHA256" "$VALIDATION_IMAGE" | sha256sum -c -
test "$(cat "$SERVER_BUILD/evidence/exit-code.txt")" = 0
mkdir evidence
printf '%s\n' "$SLURM_JOB_ID" > evidence/job-id.txt
cp source-sha256-check.log evidence/
scontrol show job "$SLURM_JOB_ID" > evidence/slurm-job.txt
scontrol show hostnames "$SLURM_JOB_NODELIST" > evidence/hosts.txt
srun --container-image="$VALIDATION_IMAGE" --no-container-mount-home --cpu-bind=none \
    --container-workdir=/app \
    --container-mounts="$PWD:/validation:ro,$PWD/candidate:/candidate:ro,$PWD/mx-source:/mx-source:ro,$PWD/evidence:/evidence,$SERVER_BUILD/evidence:/server-build:ro" \
    /bin/bash /validation/run.sh
