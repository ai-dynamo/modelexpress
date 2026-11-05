#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 || ( $1 != vllm && $1 != sglang ) ]]; then
  echo "Usage: bash $0 {vllm|sglang} [engine arguments...]" >&2
  exit 2
fi

engine=$1
shift

for arg in "$@"; do
  case $arg in
    --port|--port=*)
      echo "Set the engine port with MX_EXAMPLE_PORT, not --port, so artifact readiness uses the same port." >&2
      exit 2
      ;;
  esac
done

: "${MX_EXAMPLE_MODEL:?Set MX_EXAMPLE_MODEL to the model path or ID}"
: "${MX_EXAMPLE_RUN_DIR:?Set MX_EXAMPLE_RUN_DIR to a fresh, writable directory}"
: "${MX_ARTIFACT_MOONCAKE_NAMESPACE:?Set a namespace shared by cold and hit runs}"

port=${MX_EXAMPLE_PORT:-8000}
tp_size=${MX_EXAMPLE_TP_SIZE:-1}
mkdir -p "$MX_EXAMPLE_RUN_DIR"
run_dir=$(realpath "$MX_EXAMPLE_RUN_DIR")
cache_dir=$run_dir/cache

export MX_ARTIFACT_TRANSFER=1
export MX_ARTIFACT_BACKEND=mooncake
export MX_P2P_METADATA=${MX_P2P_METADATA:-0}
export MX_ARTIFACT_BUNDLE_ROOT=$run_dir/artifact-bundles
export MX_ARTIFACT_READY_URL=http://127.0.0.1:${port}/health

# Both engines can produce these file-backed cache artifacts. Each invocation
# uses its own roots, so a later run cannot accidentally reuse local files.
export TRITON_CACHE_DIR=$cache_dir/triton
export TILELANG_CACHE_DIR=$cache_dir/tilelang
export CUTE_DSL_CACHE_DIR=$cache_dir/cute_dsl
export FLASHINFER_WORKSPACE_BASE=$cache_dir/flashinfer_workspace
export CUDA_CACHE_PATH=$cache_dir/cuda
mkdir -p "$MX_ARTIFACT_BUNDLE_ROOT" "$CUDA_CACHE_PATH"

case $engine in
  vllm)
    export VLLM_CACHE_ROOT=$cache_dir/vllm
    export DG_JIT_CACHE_DIR=$VLLM_CACHE_ROOT/deep_gemm
    export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$VLLM_CACHE_ROOT/flashinfer_autotune_cache
    exec vllm serve "$MX_EXAMPLE_MODEL" \
      --host 0.0.0.0 --port "$port" \
      --tensor-parallel-size "$tp_size" \
      --load-format modelexpress "$@"
    ;;
  sglang)
    export SGLANG_CACHE_DIR=$cache_dir/sglang
    export SGLANG_DG_CACHE_DIR=$cache_dir/deep_gemm
    export TORCHINDUCTOR_CACHE_DIR=$cache_dir/torch_inductor
    export TVM_FFI_CACHE_DIR=$cache_dir/tvm_ffi
    exec python3 -m sglang.launch_server \
      --model-path "$MX_EXAMPLE_MODEL" \
      --host 0.0.0.0 --port "$port" --tp "$tp_size" \
      --load-format remote_instance \
      --remote-instance-weight-loader-backend modelexpress \
      --modelexpress-config '{"transport":"nixl"}' "$@"
    ;;
esac
