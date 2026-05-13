# TensorRT-LLM ModelExpress Test Image

This directory contains only the minimum files needed to build a temporary
TRT-LLM image for ModelExpress checkpoint-loader validation.

The image starts from a TRT-LLM release image, installs the local ModelExpress
Python client, and expects the base TRT-LLM image to already include the
ModelExpress checkpoint-loader hooks.

The Dockerfile pins the default TRT-LLM base image to
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc11@sha256:d91c80ba8baf763782b1078267ed6b1e06363bebff4961094bf6e5679d371d04`
for reproducible validation builds. Override `TRTLLM_IMAGE` when testing a
TRT-LLM image that includes the ModelExpress checkpoint-loader hooks.

## Build

Run from the ModelExpress repo root:

```bash
docker buildx build --platform linux/amd64 \
    -f examples/p2p_transfer_k8s/client/trtllm/Dockerfile \
    --build-arg TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:<tag-or-digest> \
    -t <registry>/trtllm-modelexpress:<tag> \
    --push .
```

## Files

| File | Purpose |
| --- | --- |
| `Dockerfile` | Builds the temporary TRT-LLM + ModelExpress e2e image. |
| `install_modelexpress_client.sh` | Builds protobuf stubs and installs the local ModelExpress Python client. |
| `fix_nixl_runpath.py` | Makes the NIXL Python binding resolve TRT-LLM's system NIXL libraries. |

No Kubernetes manifests are kept here yet. The current TRT-LLM integration is
still under validation, and stale DGD manifests were intentionally removed.
