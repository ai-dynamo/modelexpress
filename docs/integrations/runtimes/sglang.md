<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang

SGLang delegates its `remote_instance` loader to ModelExpress.

```bash
export MX_SERVER_ADDRESS=modelexpress-server:8001
python -m sglang.launch_server --model-path my-org/my-model --tp 2 --load-format remote_instance --remote-instance-weight-loader-backend modelexpress --modelexpress-config '{"transport":"nixl"}'
```

Use an image with the upstream ModelExpress delegation hook. The checked-in examples use `lmsysorg/sglang:v0.5.13.post1`; full image, transport, and readiness details are in [Using ModelExpress with SGLang](../../SGLANG.md).

For ModelStreamer, keep `--model-path` on the model identity or local configuration path and set `MX_MODEL_URI` to the storage URI. Passing `s3://`, `gs://`, or `az://` as `--model-path` bypasses ModelExpress. See the [SGLang storage examples](../../../examples/model_streamer_k8s/client/sglang/README.md).

`{"transport":"nixl"}` supports NIXL weight transfer and compatible artifact transfer. `{"transport":"transfer_engine"}` selects Mooncake TransferEngine for weight transfer.

ModelExpress CI exercises SGLang NIXL and Mooncake P2P plus direct and fallback S3 ModelStreamer paths. Hardware and storage combinations outside that matrix require deployment-specific qualification.
