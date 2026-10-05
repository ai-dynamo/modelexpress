#!/bin/bash
set -euo pipefail
cd /app
export UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/app/.venv OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/generic-direct-uv-cache
export PYTHONPATH="/candidate/src:/candidate/packages/prime-rl-configs/src:/validation"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export MX_REFIT_PACK_MODULES=1 MX_POOL_REG=1 MX_RESHARD_MIN_GBPS=25
export MX_REFIT_REUSE_COMPLETE_PLAN=1 MX_REFIT_CACHE_RESOLVED_SOURCES=1
export MX_REFIT_CACHE_BOUNDED_PLANS=1 MX_REFIT_COPY_PLAN_KEY_ON_MISS=1
export MX_REFIT_TIMING=1 MX_REFIT_TIMING_STDOUT=1
if [[ -z ${UCX_NET_DEVICES:-} ]]; then export MX_RDMA_NIC_PIN=auto; fi
unset MX_REFIT_GLM_DIRECT MX_REFIT_STAGING_BYTES MX_VERIFY_INITIAL_REFIT
export NIXL_UCX_TLS="${NIXL_UCX_TLS:-rc,cuda_copy,cuda_ipc,sm,self}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn MODEL_EXPRESS_SECURITY_MODE=off
pids=()
cleanup() {
    local code=$?
    trap - EXIT
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    rm -rf /tmp/candidate-mx-client /tmp/generic-direct-uv-cache
    printf '%s\n' "$code" > /evidence/exit-code.txt
    exit "$code"
}
trap cleanup EXIT
cp /etc/primerl-pr3487-provenance /evidence/base-image-provenance.txt
cp /validation/snapshot.json /evidence/snapshot.json
sha256sum /server-build/modelexpress-server > /evidence/server-binary.sha256
uv pip freeze --python /app/.venv/bin/python > /evidence/base-packages.txt
uv run --no-sync python /validation/package_snapshot.py --output /evidence/base-packages.json
cp -a /mx-source/modelexpress_client/python /tmp/candidate-mx-client
mkdir -p /evidence/client-installer/scripts
cp /validation/install_modelexpress_client.sh /evidence/client-installer/scripts/
cp /candidate/pyproject.toml /evidence/client-installer/pyproject.toml
bash /evidence/client-installer/scripts/install_modelexpress_client.sh \
    /tmp/candidate-mx-client /app/.venv/bin/python > /evidence/install-mx.log 2>&1
uv pip freeze --python /app/.venv/bin/python > /evidence/runtime-packages.txt
uv run --no-sync python /validation/package_snapshot.py --output /evidence/runtime-packages.json --compare /evidence/base-packages.json
nvidia-smi -q > /evidence/nvidia-smi.txt
uv run --no-sync python /validation/verify_sources.py > /evidence/source-verification.log 2>&1
uv run --no-sync python -m pytest -q -o addopts='' -p no:cacheprovider \
    /mx-source/modelexpress_client/python/tests/test_refit_bulk_alias.py \
    /mx-source/modelexpress_client/python/tests/test_refit_vllm_installer.py \
    /mx-source/modelexpress_client/python/tests/test_pool_registration.py \
    /mx-source/modelexpress_client/python/tests/test_refit_streaming_ownership.py \
    /mx-source/modelexpress_client/python/tests/test_refit_fsdp_adapter.py \
    /mx-source/modelexpress_client/python/tests/test_refit_generator_runtime.py \
    /mx-source/modelexpress_client/python/tests/test_refit_nixl_staged_transfer.py \
    --junitxml=/evidence/focused-tests.xml > /evidence/focused-tests.log 2>&1
uv run --no-sync python /validation/probe_streaming.py --output /evidence/streaming.json \
    > /evidence/streaming.log 2>&1
redis_port=16379
server_port=18001
redis-server --port "$redis_port" --bind 127.0.0.1 --save '' --appendonly no \
    > /evidence/redis.log 2>&1 &
pids+=("$!")
for _ in {1..30}; do
    if redis-cli -p "$redis_port" ping >/dev/null 2>&1; then break; fi
    kill -0 "${pids[0]}"
    sleep 1
done
redis-cli -p "$redis_port" ping
env MX_METADATA_BACKEND=redis REDIS_URL="redis://127.0.0.1:$redis_port" \
    /server-build/modelexpress-server --host 127.0.0.1 --port "$server_port" --metrics-port 0 \
    > /evidence/mx-server.log 2>&1 &
pids+=("$!")
uv run --no-sync python - <<'PY'
import grpc
with grpc.insecure_channel('127.0.0.1:18001') as channel:
    grpc.channel_ready_future(channel).result(timeout=60)
PY
uv run --no-sync python /validation/probe_tied_worker.py \
    --server-url "127.0.0.1:$server_port" --work-dir /evidence/tied-worker \
    --output /evidence/tied-worker.json > /evidence/tied-worker.log 2>&1
uv run --no-sync python - <<'PY'
import json
import xml.etree.ElementTree as ET
from pathlib import Path
root = Path('/evidence')
checks = {name:json.loads((root/f'{name}.json').read_text()) for name in ('source-verification','streaming','tied-worker')}
assert all(check['passed'] for check in checks.values())
assert len(checks['streaming']['arms']) == 2
assert all(len(arm['records']) == 3 for arm in checks['streaming']['arms'])
assert len(checks['tied-worker']['updates']) == 3
suites = ET.parse(root/'focused-tests.xml').getroot().findall('testsuite')
assert suites and sum(int(s.get('tests',0)) for s in suites) > 0
assert all(int(s.get(k,0)) == 0 for s in suites for k in ('errors','failures','skipped'))
(root/'validation.json').write_text(json.dumps({'schema':'generic-direct-gpu-validation-v1','passed':True,'scope':'Python-only DIRECT and trainer COPY_TO_HOST with a verified server binary and unchanged Rust build inputs; not a full GLM benchmark.','tests':sum(int(s.get('tests',0)) for s in suites),'skips':sum(int(s.get('skipped',0)) for s in suites),'checks':checks},indent=2)+'\n')
PY
echo GENERIC_DIRECT_GPU_VALIDATION_PASS
