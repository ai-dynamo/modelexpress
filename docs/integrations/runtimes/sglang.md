<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang

SGLang delegates its `remote_instance` weight loader to ModelExpress. ModelExpress then chooses the configured transport and loading strategy for the worker.

## Minimal P2P command

Use an SGLang image that includes the upstream ModelExpress delegation hook and install the ModelExpress Python client without dependency resolution:

```bash
export MX_SERVER_ADDRESS=modelexpress-server:8001
python -m sglang.launch_server --model-path my-org/my-model --tp 2 --load-format remote_instance --remote-instance-weight-loader-backend modelexpress --modelexpress-config '{"transport":"nixl"}'
```

The known-good example and CI image is `lmsysorg/sglang:v0.5.13.post1`, which includes upstream [sgl-project/sglang#24723](https://github.com/sgl-project/sglang/pull/24723). The full build, NIXL, Mooncake, artifact-transfer, and readiness instructions are in [`docs/SGLANG.md`](../../SGLANG.md).

## ModelStreamer

Keep `--model-path` as the model identity or local configuration path and set `MX_MODEL_URI` to `s3://`, `gs://`, `az://`, or an absolute local path. Passing an object-storage URI as `--model-path` bypasses the ModelExpress chain and selects SGLang's native loader. See the [SGLang ModelStreamer recipes](../../../examples/model_streamer_k8s/client/sglang/README.md).

## Transports

Set `{"transport":"nixl"}` for NIXL P2P and compatible artifact transfer. Set `{"transport":"transfer_engine"}` for Mooncake TransferEngine weight transfer. ModelExpress artifact transfer is currently implemented on the NIXL transport; TransferEngine mode is weight-only.

## Verification

The active CI matrix covers SGLang NIXL P2P, SGLang Mooncake TransferEngine P2P, TP and MLA coverage, rolling update, stale metadata, direct S3 ModelStreamer loading, and S3 fallback. GCS, Azure, and local/PVC are maintained examples rather than all being in the PR matrix.
