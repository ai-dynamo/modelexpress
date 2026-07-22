# SGLang ModelStreamer Through ModelExpress

These recipes start SGLang with the ModelExpress `remote_instance` loader and
stream safetensors from Azure Blob Storage, S3, or a local PVC. A ModelExpress
server and RDMA resources are not required for these storage-only paths.

| Storage source | Manifest | Credentials or volume |
|---|---|---|
| Azure Blob Storage | [`sglang-single-node-streamer-azure.yaml`](sglang-single-node-streamer-azure.yaml) | `azure-storage-creds` secret |
| S3 | [`sglang-single-node-streamer-s3.yaml`](sglang-single-node-streamer-s3.yaml) | `aws-creds` secret |
| Local PVC | [`sglang-single-node-streamer-local.yaml`](sglang-single-node-streamer-local.yaml) | `model-cache` PVC |

Build the image from the repository root:

```bash
docker buildx build --platform linux/amd64 \
  -f examples/model_streamer_k8s/client/sglang/Dockerfile \
  -t <your-registry>/modelexpress-sglang:latest --push .
```

Update the selected manifest's image, model, storage URI or path, credentials,
and resource settings, then deploy it. For example:

```bash
kubectl apply -f examples/model_streamer_k8s/client/sglang/sglang-single-node-streamer-s3.yaml
kubectl logs -f deployment/mx-sglang-s3 -c sglang
```

Expected log markers include:

- `Trying strategy: model_streamer`
- `Streaming weights from ...`
- `Model streamer weight loading complete`
- `Application startup complete`

Keep `--model-path` set to the Hugging Face model ID or a local configuration
path. Passing an object-storage URI there makes SGLang select its native
`runai_streamer` loader and bypasses the ModelExpress strategy chain. The
storage URI or local model path belongs in `MX_MODEL_URI`.

For tensor parallelism greater than one, keep `MX_MS_DISTRIBUTED=1`, set `--tp`
to the desired size, and request the matching number of GPUs.
