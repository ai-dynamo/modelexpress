<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelStreamer storage loading

ModelStreamer reads safetensor ranges concurrently and streams them into the runtime. It supports S3, GCS, Azure Blob Storage, and local filesystem paths. Direct ModelStreamer loading does not require a ModelExpress server, Redis, shared storage, or P2P resources.

## Configure a storage path

Set `MX_MODEL_URI` to the storage location. vLLM can use the same URI as its `--model` argument, as shown in the checked-in recipes. For SGLang, keep `--model-path` on the model identity or local configuration path; passing an object-storage URI there selects SGLang's native loader and bypasses the ModelExpress strategy chain.

```bash
export MX_MODEL_URI=s3://my-bucket/models/my-model
export AWS_DEFAULT_REGION=us-west-2
vllm serve s3://my-bucket/models/my-model --load-format modelexpress
```

The accepted URI forms are `s3://bucket/prefix`, `gs://bucket/prefix`, `az://container/prefix`, and an absolute local path. For a Hugging Face model ID or local path, the runtime's cache and configuration must resolve to files the streamer can read.

## Credentials

Credentials are consumed by the underlying storage SDK in the worker. They do not pass through ModelExpress gRPC.

| Storage | Typical credential path |
|---|---|
| S3 or S3-compatible | AWS default credential chain; `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`, and optionally `AWS_ENDPOINT_URL` |
| GCS | Application Default Credentials or `GOOGLE_APPLICATION_CREDENTIALS` |
| Azure Blob | Azure `DefaultAzureCredential`; commonly `AZURE_STORAGE_ACCOUNT_NAME`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID` |

## Tensor-parallel loading

`MX_MS_DISTRIBUTED=1` is the default. With CUDA tensor parallelism greater than one, it divides remote reads across ranks; with TP=1 it has no effect. Set it to `0` only when you intentionally want every rank to read the full checkpoint.

`RUNAI_STREAMER_CONCURRENCY` controls the number of concurrent read workers. `RUNAI_STREAMER_MEMORY_LIMIT` controls the bounded CPU staging buffer; a value of `0` uses a single-tensor buffer.

## Kubernetes examples

- [vLLM ModelStreamer recipes](../../../examples/model_streamer_k8s/client/vllm/README.md) cover Azure Blob, S3, and local/PVC manifests.
- [SGLang ModelStreamer recipes](../../../examples/model_streamer_k8s/client/sglang/README.md) cover Azure Blob, S3, and local/PVC manifests.
- [Top-level ModelStreamer examples](../../../examples/model_streamer_k8s/README.md) compare the storage variants.

The current CI workflow runs direct S3 streaming and an S3-failure fallback for vLLM and SGLang. Azure, GCS, and local/PVC paths have checked-in examples but are not all exercised by the PR matrix.
