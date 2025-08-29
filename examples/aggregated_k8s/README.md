# Model Express with Aggregated Dynamo Inference on K8s


## Prerequisites

1. **Install Dynamo Cloud on your Cluster**
   - Follow the [Dynamo Cloud installation guide](https://github.com/ai-dynamo/dynamo/blob/a8cb6554779f8283edd0c62d50743f2cb58e989b/docs/guides/dynamo_deploy/quickstart.md)
   - Ensure Dynamo Cloud is running and healthy on your cluster

2. **Build and Push Model Express Image**
   - **IMPORTANT**: You must build the Model Express image and push it to a Docker registry that your Kubernetes cluster can access
   - The deployment expects the image to be available at `localhost:5000/model-express:latest` (for local development) or your registry URL
   
   ```bash
   # Build the Model Express image
   docker build -t model-express:latest .
   
   # Tag for your registry (replace with your registry)
   docker tag model-express:latest your-registry/model-express:latest
   
   # Push to your registry
   docker push your-registry/model-express:latest
   ```
   
   **Registry Options:**
   - **Docker Hub**: `docker tag model-express:latest yourusername/model-express:latest`
   - **Local Registry**: Use `localhost:5000/model-express:latest` (requires local registry setup)
   - **Private Registry**: Use your private registry URL

   **Update Image Reference**
   - **CRITICAL**: Update the image reference in `agg.yaml` to match your registry
   - Find `spec.services.ModelExpressServer.extraPodSpec.mainContainer.image` in `agg.yaml` and change:
     ```yaml
     image: model-express:latest
     ```
     to your registry URL, for example:
     ```yaml
     image: yourusername/model-express:latest
     ```

3. **Additional Requirements**
   - Kubernetes cluster with GPU support
   - `kubectl` configured to access your cluster
   - Docker registry accessible from your cluster

## Quick Start

1. **Deploy the configuration:**
   ```bash
   kubectl apply -f agg.yaml
   ```

2. **Monitor the deployment:**
   ```bash
   $ kubectl get po

   NAME                                                              READY   STATUS    RESTARTS      AGE
   dynamo-platform-dynamo-operator-controller-manager-54d48f4vdkh8   2/2     Running   8 (12d ago)   15d
   dynamo-platform-etcd-0                                            1/1     Running   4 (12d ago)   15d
   dynamo-platform-nats-0                                            2/2     Running   8 (12d ago)   15d
   dynamo-platform-nats-box-5dbf45c748-vstcm                         1/1     Running   4 (12d ago)   15d
   vllm-agg-frontend-569757b8f5-brnbq                                1/1     Running   0             23m
   vllm-agg-modelexpressserver-544b666cbc-2ll6d                      1/1     Running   0             23m
   vllm-agg-vllmdecodeworker-69dcddfc85-zcwd7                        1/1     Running   0             23m
   ```

## Monitoring the Deployment

### Monitor Model Express Server
```bash
kubectl logs -f vllm-agg-modelexpressserver-544b666cbc-2ll6d
```

Sample output:
```
Starting Model Express Server...
Server started with PID: 7
Setting up Model Express configuration...
Waiting for server to be ready...
Server is ready!
Cleaning up any stale lock files...
Downloading Qwen/Qwen3-0.6B model...
Model Download
  Model: Qwen/Qwen3-0.6B
  Provider: HuggingFace
  Strategy: SmartFallback
✅ SUCCESS
  Model 'Qwen/Qwen3-0.6B' downloaded successfully
Model download completed. Creating symlink for VLLM worker...
Created symlink: latest -> c1899de289a04d12100db370d81485cdf75e47ca
Model cache directory: models--Qwen--Qwen3-0.6B
```

### Monitor VLLM Worker
```bash
kubectl logs -f vllm-agg-vllmdecodeworker-69dcddfc85-zcwd7
```

Sample output:
```
Waiting for model to be ready...
Model is ready! Starting VLLM worker...
Model path: /model/.model-express/cache/models--Qwen--Qwen3-0.6B/snapshots/latest/
INFO 08-15 01:24:21 [__init__.py:235] Automatically detected platform cuda.
2025-08-15T01:24:24.377400Z  INFO dynamo_runtime::http_server: [spawn_http_server] binding to: 0.0.0.0:9090
2025-08-15T01:24:35.555894Z  INFO parallel_state.initialize_model_parallel: rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
2025-08-15T01:25:36.139394Z  INFO main.setup_vllm_engine: VllmWorker for /model/.model-express/cache/models--Qwen--Qwen3-0.6B/snapshots/latest/ has been initialized
```

## Configuration

The deployment includes:

- **Model Express Server**: Downloads and serves models
- **VLLM Worker**: Runs inference with GPU acceleration  
- **Frontend**: Provides HTTP API endpoints
- **Persistent Volume**: Shared storage for model cache

### Environment Variables

You can customize the deployment by modifying these environment variables in `agg.yaml`:

- `MODEL_NAME`: The HuggingFace model to download (default: "Qwen/Qwen3-0.6B")
- `MODEL_CACHE_PATH`: Path for model storage (default: "/root/.model-express/cache")


## Cleanup

To remove the deployment:
```bash
kubectl delete -f agg.yaml
```

This will remove all pods, services, and the persistent volume claim.
