# ModelStreamer Kubernetes Examples

These examples deploy vLLM or SGLang with a ModelExpress loader and stream model weights from storage through RunAI ModelStreamer. They do not require a ModelExpress server, RDMA resources, or a model PVC for object storage sources. The vLLM `mx` load format is kept as a backward-compatible alias.

For P2P RDMA weight transfer between vLLM pods, see [`../p2p_transfer_k8s/`](../p2p_transfer_k8s/).

## vLLM Examples

| Storage source | Manifest | Notes |
|---|---|---|
| Azure Blob Storage | [`client/vllm/vllm-single-node-streamer-azure.yaml`](client/vllm/vllm-single-node-streamer-azure.yaml) | Uses an `az://<container>/<model-prefix>` URI and Azure `DefaultAzureCredential`. |
| S3 | [`client/vllm/vllm-single-node-streamer-s3.yaml`](client/vllm/vllm-single-node-streamer-s3.yaml) | Uses an `s3://<bucket>/<model-prefix>` URI and AWS credentials. |
| Local PVC | [`client/vllm/vllm-single-node-streamer-local.yaml`](client/vllm/vllm-single-node-streamer-local.yaml) | Uses a PVC-mounted Hugging Face cache or local model path. |

For the Azure Blob end-to-end setup, see [`client/vllm/README.md`](client/vllm/README.md).

## SGLang Examples

| Storage source | Manifest | Notes |
|---|---|---|
| Azure Blob Storage | [`client/sglang/sglang-single-node-streamer-azure.yaml`](client/sglang/sglang-single-node-streamer-azure.yaml) | Uses an `az://<container>/<model-prefix>` URI and Azure `DefaultAzureCredential`. |
| S3 | [`client/sglang/sglang-single-node-streamer-s3.yaml`](client/sglang/sglang-single-node-streamer-s3.yaml) | Uses SGLang `remote_instance` with backend `modelexpress`; `MX_MODEL_URI` activates ModelStreamer. |
| Local PVC | [`client/sglang/sglang-single-node-streamer-local.yaml`](client/sglang/sglang-single-node-streamer-local.yaml) | Uses a PVC-mounted Hugging Face cache or local model path. |

For build and deployment instructions, see [`client/sglang/README.md`](client/sglang/README.md).

## Common Configuration

The vLLM manifests use:

- `--load-format modelexpress`
- `VLLM_PLUGINS=modelexpress`
- `MX_MODEL_URI` as the model path passed to vLLM

For tensor parallel deployments with TP > 1, set:

```bash
MX_MS_DISTRIBUTED=1
```

The SGLang manifests use `remote_instance` with backend `modelexpress`, while
`MX_MODEL_URI` selects ModelStreamer inside the shared MX strategy chain.

`MX_MS_DISTRIBUTED=1` enables the engine-native distributed ModelStreamer path
through the corresponding ModelExpress adapter. TP1 runs ignore this setting.

## Verify Startup

Check the inference server logs:

```bash
kubectl logs deployment/<deployment-name> -c <vllm-or-sglang>
```

Expected signals:

- `Trying strategy: model_streamer`
- `Streaming weights from ...`
- `Model streamer weight loading complete`
- `Application startup complete`
