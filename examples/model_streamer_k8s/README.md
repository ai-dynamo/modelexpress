# ModelStreamer Kubernetes Examples

These examples deploy vLLM with ModelExpress `--load-format modelexpress` and stream model weights from storage through RunAI ModelStreamer. They do not require a ModelExpress server, RDMA resources, or a model PVC for object storage sources. The `mx` load format is kept as a backward-compatible alias.

For P2P RDMA weight transfer between vLLM pods, see [`../p2p_transfer_k8s/`](../p2p_transfer_k8s/).

## vLLM Examples

| Storage source | Manifest | Notes |
|---|---|---|
| Azure Blob Storage | [`client/vllm/vllm-single-node-streamer-azure.yaml`](client/vllm/vllm-single-node-streamer-azure.yaml) | Uses an `az://<container>/<model-prefix>` URI and Azure `DefaultAzureCredential`. |
| S3 | [`client/vllm/vllm-single-node-streamer-s3.yaml`](client/vllm/vllm-single-node-streamer-s3.yaml) | Uses an `s3://<bucket>/<model-prefix>` URI and AWS credentials. |
| Local PVC | [`client/vllm/vllm-single-node-streamer-local.yaml`](client/vllm/vllm-single-node-streamer-local.yaml) | Uses a PVC-mounted Hugging Face cache or local model path. |

For the Azure Blob end-to-end setup, see [`client/vllm/README.md`](client/vllm/README.md).

## Common Configuration

All manifests use:

- `--load-format modelexpress`
- `VLLM_PLUGINS=modelexpress`
- `MX_MODEL_URI` as the model path passed to vLLM

For tensor parallel deployments with TP > 1, set:

```bash
MX_MS_DISTRIBUTED=1
```

This enables vLLM's native distributed ModelStreamer path through the ModelExpress vLLM adapter. TP1 runs ignore this setting.

## Verify Startup

Check the vLLM logs:

```bash
kubectl logs deployment/<deployment-name> -c vllm
```

Expected signals:

- `Registered model loader ... MxModelLoader`
- `Trying strategy: model_streamer`
- `Streaming weights from ...`
- `Model streamer weight loading complete`
- `Application startup complete`
