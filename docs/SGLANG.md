<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Using ModelExpress with SGLang

ModelExpress can serve as the remote-instance weight loader for SGLang,
streaming weights GPU-to-GPU over RDMA between SGLang processes instead
of loading from disk on every replica. The SGLang-side delegation hook was
added by upstream [sgl-project/sglang#24723](https://github.com/sgl-project/sglang/pull/24723):
it adds the `--modelexpress-config` flag and delegates ModelExpress loading to
the ModelExpress package.

With `remote_instance` + `backend=modelexpress`, SGLang does not run separate
source and target modes. Every server uses the same command, and
`modelexpress.engines.sglang.MxModelLoader` decides whether to load natively
and publish metadata or receive weights from an existing ModelExpress source.

## 1. Build an SGLang image

Use an SGLang image that contains the upstream ModelExpress delegation hook:

- **Pull the official image** — `lmsysorg/sglang:v0.5.13.post1` is a
  known-good release image that includes PR #24723.
- **Build from `main`** — follow SGLang's official install guide at
  [docs.sglang.io/docs/get_started/install](https://docs.sglang.io/docs/get_started/install).

Install the ModelExpress Python package into the SGLang image. The Kubernetes
examples provide a Dockerfile at
`examples/p2p_transfer_k8s/client/sglang/Dockerfile`.

```dockerfile
FROM lmsysorg/sglang:v0.5.13.post1

RUN python3 -m pip install --no-cache-dir --no-deps \
    "modelexpress @ git+https://github.com/ai-dynamo/modelexpress.git#subdirectory=modelexpress_client/python"
```

Use `--no-deps` inside SGLang images because the base image already owns the
CUDA, NIXL, Torch, gRPC, and protobuf stack. Letting pip resolve ModelExpress
dependencies can downgrade engine-provided runtime packages.

For Mooncake TransferEngine with the CUDA 13 SGLang image, install the
CUDA 13 Mooncake package into the same image.

```dockerfile
FROM lmsysorg/sglang:v0.5.13.post1

RUN python3 -m pip install --no-cache-dir mooncake-transfer-engine-cuda13
RUN python3 -m pip install --no-cache-dir --no-deps \
    "modelexpress @ git+https://github.com/ai-dynamo/modelexpress.git#subdirectory=modelexpress_client/python"
```

If you build SGLang from a local PR branch for e2e testing, install the local
ModelExpress package into that image:

```dockerfile
FROM sglang-source:modelexpress

COPY modelexpress_client/python /tmp/modelexpress_client_python
RUN python3 -m pip install --no-cache-dir --no-deps /tmp/modelexpress_client_python
```

Confirm the SGLang delegation flag is present before running:

```bash
python -m sglang.launch_server --help | grep modelexpress-config
```

For unreleased SGLang source builds:

```bash
cd /path/to/sglang

docker build --platform linux/amd64 \
  -f docker/Dockerfile \
  --target runtime \
  --build-arg BRANCH_TYPE=local \
  --build-arg CUDA_VERSION=13.0.1 \
  --build-arg BUILD_TYPE=all \
  -t sglang-source:modelexpress \
  .
```

Then use the local-package Dockerfile snippet above to install ModelExpress.

## 2. Start a ModelExpress server

ModelExpress server should be reachable at
`modelexpress-server:8001`. See [`DEPLOYMENT.md`](DEPLOYMENT.md) for how
to start one (Docker, Helm, or Kubernetes).

## 3. Launch SGLang

Use the same command for the first and later replicas. When no READY source is
available, the first replica continues through the configured storage-loading
strategies and publishes itself. Later replicas discover the source and load
via the selected ModelExpress transport.

```bash
export MX_SERVER_ADDRESS=modelexpress-server:8001

python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 --tp 8 --port 30000 \
  --load-format remote_instance \
  --remote-instance-weight-loader-backend modelexpress \
  --modelexpress-config '{"transport": "nixl"}'
```

`modelexpress-config` is intentionally small and only controls the SGLang
handoff into ModelExpress:

- `url` optionally overrides the ModelExpress server URL for this SGLang
  process. Prefer `MX_SERVER_ADDRESS` in deployments so endpoint configuration
  stays in environment variables.
- `transport` selects the ModelExpress package transport. Supported values are
  `nixl` and `transfer_engine`.

All other ModelExpress settings are environment variables, matching vLLM:
`MX_METADATA_BACKEND`, `MX_MODEL_REVISION`, `MX_P2P_METADATA`,
`MX_NIXL_BACKEND`, `MX_RDMA_NIC_PIN`, `MX_METADATA_PORT`,
`MX_WORKER_GRPC_PORT`, and `MODEL_EXPRESS_LOG_LEVEL`.

### Bootstrap with ModelStreamer

Set `MX_MODEL_URI` to select ModelStreamer inside the same ModelExpress loader. Keep `--model-path` as the model identity and configuration source; passing the object-storage URI as `--model-path` bypasses the ModelExpress strategy chain.

```bash
export MX_SERVER_ADDRESS=modelexpress-server:8001
export MX_MODEL_URI=s3://my-bucket/path/to/model
export MX_MS_DISTRIBUTED=1  # Enable for TP > 1

python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 --tp 8 --port 30000 \
  --load-format remote_instance \
  --remote-instance-weight-loader-backend modelexpress \
  --modelexpress-config '{"transport": "nixl"}'
```

The first replica attempts ModelStreamer when no compatible P2P source is ready. Later replicas prefer P2P RDMA from a serving peer. `MX_MODEL_URI` also accepts `gs://` and `az://` URIs or an absolute local path. `MX_MS_DISTRIBUTED=1` divides reads across CUDA tensor-parallel ranks when TP > 1 and is ignored for TP=1.

Weight-source publication waits for SGLang's health endpoint so another
instance cannot consume weights during warmup or CUDA graph capture. Readiness
defaults to `http://127.0.0.1:30000/health`; set `MX_ARTIFACT_READY_URL` when
using another server port.

Set `MX_ARTIFACT_TRANSFER=1` with the `nixl` transport to transfer compatible
JIT cache artifacts before SGLang initializes the model. The source publishes
cache artifacts after the SGLang `/health` endpoint is ready and the cache
directories have stopped changing briefly. Supported cache roots are
`TORCHINDUCTOR_CACHE_DIR` (or PyTorch Inductor's runtime `cache_dir()`),
`TRITON_CACHE_DIR`, `TVM_FFI_CACHE_DIR`, `SGLANG_DG_CACHE_DIR`, `TILELANG_CACHE_DIR`,
`CUTE_DSL_CACHE_DIR`, and `FLASHINFER_WORKSPACE_BASE`. This path requires
`MX_P2P_METADATA=1` and a central-coordinator metadata backend (`redis` or
`kubernetes`) for artifact discovery. Cache publication uses the same health
endpoint as weight publication.

For Mooncake TransferEngine, use the same command shape and change only the
transport. The SGLang image must include `mooncake-transfer-engine-cuda13`.
ModelExpress artifact transfer is currently implemented on the NIXL transport;
TransferEngine mode remains weight-only.

```bash
export MX_SERVER_ADDRESS=modelexpress-server:8001

python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 --tp 8 --port 30000 \
  --load-format remote_instance \
  --remote-instance-weight-loader-backend modelexpress \
  --modelexpress-config '{"transport": "transfer_engine"}'
```

## See also

- Upstream PR: [sgl-project/sglang#24723](https://github.com/sgl-project/sglang/pull/24723).
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — running the ModelExpress server, NIXL/UCX tuning, performance reference.
