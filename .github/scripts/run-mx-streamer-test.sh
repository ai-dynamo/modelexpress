#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

populate_cache() {
  python3 - <<'EOF'
import boto3, os, pathlib, tempfile
from huggingface_hub import snapshot_download

bucket = os.environ["S3_BUCKET"]
region = os.environ["S3_REGION"]
model  = os.environ["MODEL"]
prefix = f"models/{model}/"

s3 = boto3.client("s3", region_name=region)

# Check if safetensors are already in S3.
resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
if any(obj["Key"].endswith(".safetensors") for obj in resp.get("Contents", [])):
    print(f"Safetensors already in s3://{bucket}/{prefix} — skipping upload.")
else:
    print(f"Downloading safetensors for {model} from HuggingFace...")
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_download(
            repo_id=model,
            local_dir=tmpdir,
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        )
        exts = (".safetensors", ".safetensors.index.json")
        files = [f for f in pathlib.Path(tmpdir).rglob("*")
                 if f.is_file() and any(f.name.endswith(e) for e in exts)]
        print(f"Uploading {len(files)} file(s) to s3://{bucket}/{prefix}")

        uploaded = []
        try:
            for f in files:
                key = prefix + f.relative_to(tmpdir).as_posix()
                print(f"  {f.name} -> {key}")
                s3.upload_file(str(f), bucket, key)
                uploaded.append((key, f.stat().st_size))
        except Exception as e:
            print(f"ERROR: upload failed mid-stream: {e}")
            print(f"Rolling back {len(uploaded)} partially-uploaded file(s) "
                  f"to avoid poisoning the S3 cache for future runs...")
            for key, _ in uploaded:
                try:
                    s3.delete_object(Bucket=bucket, Key=key)
                    print(f"  rolled back: {key}")
                except Exception as rb:
                    print(f"  WARN: failed to roll back {key}: {rb}")
            raise

        # Verify each uploaded object is actually in S3 with matching size.
        # Catches silent corruption (rare) and confirms the cache is usable.
        for key, expected_size in uploaded:
            head = s3.head_object(Bucket=bucket, Key=key)
            actual_size = head["ContentLength"]
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Size mismatch for s3://{bucket}/{key}: "
                    f"expected {expected_size}, got {actual_size}"
                )
        print(f"Upload verified: {len(uploaded)} file(s).")
EOF
}

prepare() {
  [[ "${PORT}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: port must be a positive integer, got '${PORT}'"; return 1; }
  [[ "${STREAMER_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: streamer_concurrency must be a positive integer, got '${STREAMER_CONCURRENCY}'"; return 1; }
  [[ "${FRAMEWORK}" = "vllm" || "${FRAMEWORK}" = "sglang" ]] || { echo "ERROR: unsupported framework '${FRAMEWORK}'"; return 1; }
  MODEL_S3_URI="s3://${S3_BUCKET}/models/${MODEL}"
  if [ "${TEST_MODEL_STREAMER_FALLBACK}" = "true" ]; then
    MODEL_S3_URI="s3://${S3_BUCKET}/mx-ci-s3-fallback/missing/${GITHUB_RUN_ID}/${GITHUB_RUN_ATTEMPT}/${MODEL}"
  fi

  STREAMER_LOG="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/model-streamer.log"
}

start_vllm() {
  MODEL_ARG="${MODEL_S3_URI}"
  if [ "${TEST_MODEL_STREAMER_FALLBACK}" = "true" ]; then
    MODEL_ARG="${MODEL}"
  fi
  MX_MODEL_URI="${MODEL_S3_URI}" VLLM_PLUGINS=modelexpress nohup python3 \
    -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_ARG}" \
      --served-model-name "${MODEL}" \
      --hf-config-path "${MODEL}" \
      --language-model-only \
      --tokenizer "${MODEL}" \
      --load-format modelexpress \
      --port "${PORT}" > "${STREAMER_LOG}" 2>&1 &
  STREAMER_PID=$!
}

start_sglang() {
  # Keep SGLang's model path on the Hugging Face id. Passing the S3 URI
  # here selects SGLang's native runai_streamer loader and bypasses MX.
  MX_MODEL_URI="${MODEL_S3_URI}" nohup python3 \
    -m sglang.launch_server \
      --model-path "${MODEL}" \
      --served-model-name "${MODEL}" \
      --load-format remote_instance \
      --remote-instance-weight-loader-backend modelexpress \
      --modelexpress-config '{"transport":"nixl"}' \
      --dtype bfloat16 \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph \
      --host 0.0.0.0 \
      --port "${PORT}" > "${STREAMER_LOG}" 2>&1 &
  STREAMER_PID=$!
}

wait_for_load() {
  [[ "${LOAD_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: load_timeout_seconds must be a positive integer, got '${LOAD_TIMEOUT}'"; return 1; }
  deadline=$((SECONDS + LOAD_TIMEOUT))
  while [ $SECONDS -lt $deadline ]; do
    LOGS="$(cat "${STREAMER_LOG}")"
    if [ "${TEST_MODEL_STREAMER_FALLBACK}" = "true" ]; then
      if echo "${LOGS}" | grep -q "Strategy model_streamer failed" \
        && echo "${LOGS}" | grep -q "Weights loaded from disk"; then
        echo "Model streamer fallback to default loader confirmed."
        return 0
      fi
    elif echo "${LOGS}" | grep -q "Model streamer weight loading complete"; then
      echo "Model streamer loading confirmed."
      return 0
    fi
    if ! kill -0 "${STREAMER_PID}" 2>/dev/null; then
      echo "ERROR: server exited before model loading completed."
      tail -80 "${STREAMER_LOG}"
      return 1
    fi
    echo "Still loading... (${SECONDS}s elapsed)"
    sleep 10
  done

  echo "ERROR: model loading did not complete within ${LOAD_TIMEOUT}s."
  tail -80 "${STREAMER_LOG}"
  return 1
}

wait_for_server() {
  [[ "${PORT}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: port must be a positive integer, got '${PORT}'"; return 1; }
  timeout 300 bash -c \
    "until curl -sf --max-time 5 http://localhost:${PORT}/health > /dev/null; do sleep 2; done"
  echo "Server ready on port ${PORT}."
}

cleanup() {
  if [ -n "${STREAMER_LOG:-}" ] && [ -f "${STREAMER_LOG}" ]; then
    echo "::group::Server logs (tail 200)"
    tail -200 "${STREAMER_LOG}" || true
    echo "::endgroup::"
  fi
  if [ -n "${STREAMER_PID:-}" ]; then
    kill "${STREAMER_PID}" 2>/dev/null || true
  fi
}

main() {
  : "${FRAMEWORK:?FRAMEWORK is required}"
  : "${S3_BUCKET:?S3_BUCKET is required}"
  export MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
  export S3_REGION="${S3_REGION:-us-west-2}"
  export PORT="${PORT:-18888}"
  export AWS_DEFAULT_REGION="${S3_REGION}"
  export RUNAI_STREAMER_CONCURRENCY="${STREAMER_CONCURRENCY:-16}"
  STREAMER_CONCURRENCY="${RUNAI_STREAMER_CONCURRENCY}"
  LOAD_TIMEOUT="${LOAD_TIMEOUT:-300}"
  TEST_MODEL_STREAMER_FALLBACK="${TEST_MODEL_STREAMER_FALLBACK:-false}"
  prepare
  trap cleanup EXIT
  if [ "${TEST_MODEL_STREAMER_FALLBACK}" = "false" ]; then
    populate_cache
  fi
  "start_${FRAMEWORK}"
  wait_for_load
  wait_for_server
  python3 .github/scripts/check_streamer_inference.py
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main
fi
