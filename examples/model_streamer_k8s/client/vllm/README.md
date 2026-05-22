# vLLM ModelStreamer Recipes

## Azure Blob Storage

This recipe shows how to start a vLLM pod that loads model weights from Azure Blob Storage through ModelExpress ModelStreamer. It uses [`vllm-single-node-streamer-azure.yaml`](vllm-single-node-streamer-azure.yaml) as the deployment template.

This path does not require a model PVC. The pod streams weights directly from Azure Blob Storage at startup by using `--load-format mx` and an `az://` model URI.

### Prerequisites

- Kubernetes cluster with GPU nodes.
- A vLLM image with ModelExpress installed and vLLM `0.18.0` or newer.
- Azure Blob Storage account with Premium performance and block blob storage.
- Model weights uploaded to a blob container.
- Azure identity available to the pod through `DefaultAzureCredential`.
- The Azure identity must have `Storage Blob Data Reader` on the storage account or container.
- Image pull secret for the vLLM image registry, if required by your registry.

### Prepare Azure Blob Storage

Create a Premium block blob storage account and a container:

```bash
export RESOURCE_GROUP=<resource-group>
export LOCATION=<azure-region>
export STORAGE_ACCOUNT=<globally-unique-storage-account>
export CONTAINER=models

az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Premium_LRS \
  --kind BlockBlobStorage \
  --allow-blob-public-access false \
  --min-tls-version TLS1_2

az storage container create \
  --account-name "$STORAGE_ACCOUNT" \
  --name "$CONTAINER" \
  --auth-mode login
```

Download the model weights and upload them to the container. The uploader identity needs `Storage Blob Data Contributor` or equivalent write permissions.

```bash
export MODEL_NAME=deepseek-ai/DeepSeek-V3
export MODEL_PREFIX=models/$MODEL_NAME
export SNAPSHOT_DIR=/tmp/model-snapshot

huggingface-cli download "$MODEL_NAME" \
  --local-dir "$SNAPSHOT_DIR"

az storage blob upload-batch \
  --account-name "$STORAGE_ACCOUNT" \
  --destination "$CONTAINER" \
  --destination-path "$MODEL_PREFIX" \
  --source "$SNAPSHOT_DIR" \
  --auth-mode login \
  --overwrite true
```

The resulting ModelExpress URI is:

```bash
az://$CONTAINER/$MODEL_PREFIX
```

For the example above, that is:

```bash
az://models/models/deepseek-ai/DeepSeek-V3
```

### Provide Azure Credentials to the Pod

The deployment template reads Azure credentials from a Kubernetes secret named `azure-storage-creds`. For a service principal, create it like this:

```bash
kubectl create secret generic azure-storage-creds \
  --from-literal=AZURE_STORAGE_ACCOUNT_NAME="$STORAGE_ACCOUNT" \
  --from-literal=AZURE_CLIENT_ID="<client-id>" \
  --from-literal=AZURE_TENANT_ID="<tenant-id>" \
  --from-literal=AZURE_CLIENT_SECRET="<client-secret>"
```

The service principal needs `Storage Blob Data Reader` on the storage account or container:

```bash
az role assignment create \
  --assignee "<client-id>" \
  --role "Storage Blob Data Reader" \
  --scope "/subscriptions/<subscription-id>/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$STORAGE_ACCOUNT"
```

For AKS workload identity, use a service account and workload identity annotations instead of `AZURE_CLIENT_SECRET`. Keep `AZURE_STORAGE_ACCOUNT_NAME` available to the container.

### Configure the Manifest

Edit [`vllm-single-node-streamer-azure.yaml`](vllm-single-node-streamer-azure.yaml):

- Set the `image` to a vLLM image that includes ModelExpress and vLLM `0.18.0` or newer.
- Set `MODEL_NAME` to the Hugging Face model ID.
- Set `MX_MODEL_URI` to the `az://<container>/<model-prefix>` URI.
- Set `--tensor-parallel-size` and GPU requests to match the target model and node shape.
- Keep `VLLM_PLUGINS=modelexpress`.
- Keep `MX_MS_DISTRIBUTED=1` for tensor-parallel deployments with TP > 1. It is ignored for TP1.
- Tune `RUNAI_STREAMER_CONCURRENCY` for the storage account, network, and model size.

Optional: if the pod should publish its loaded weights as a P2P source for later pods, set `MX_SERVER_ADDRESS` and add RDMA resources and UCX settings. See [`vllm-single-node-p2p.yaml`](../../../p2p_transfer_k8s/client/vllm/vllm-single-node-p2p.yaml) for the RDMA resource pattern.

### Deploy

```bash
kubectl apply -f examples/model_streamer_k8s/client/vllm/vllm-single-node-streamer-azure.yaml
```

Check pod status:

```bash
kubectl get pods -l app=mx-vllm-azure
```

### Verify

Check the vLLM logs:

```bash
kubectl logs deployment/mx-vllm-azure -c vllm
```

Expected signals:

- `Registered model loader ... MxModelLoader`
- `Trying strategy: model_streamer`
- `Streaming weights from az://...`
- `Model streamer weight loading complete`
- `Model loading took ... seconds`
- `Application startup complete`

Check the HTTP API:

```bash
kubectl exec deployment/mx-vllm-azure -c vllm -- curl -sf http://localhost:8000/health
kubectl exec deployment/mx-vllm-azure -c vllm -- curl -s http://localhost:8000/v1/models
```

### Troubleshooting

- Use `az://<container>/<model-prefix>` for `MX_MODEL_URI`; do not use an HTTPS blob URL.
- Use vLLM `0.18.0` or newer for Azure Blob ModelStreamer support.
- If the pod cannot list or read blobs, verify the pod identity has `Storage Blob Data Reader` and that `AZURE_STORAGE_ACCOUNT_NAME` is set.
- If the upload command fails with missing permissions, the uploader identity needs `Storage Blob Data Contributor`.
- For TP > 1, keep `MX_MS_DISTRIBUTED=1` so ModelExpress passes distributed streaming to vLLM's native RunAI ModelStreamer loader.
