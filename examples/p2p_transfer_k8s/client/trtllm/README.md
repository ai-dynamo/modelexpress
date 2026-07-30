# TensorRT-LLM native ModelExpress P2P

This example runs a scalable TensorRT-LLM deployment through the upstream
`checkpoint_format="MX"` interface. The first replica loads from the shared
model cache and publishes post-transform weights. Replicas added later receive
from a compatible ready source when one exists and retain storage fallback
when it does not. The current integration supports the `LlamaForCausalLM`
model family.

## Supported path

| Item | Example scope |
|---|---|
| Model | `LlamaForCausalLM` |
| Parallelism | TP=4 per replica |
| Transfer | NIXL RDMA |
| Metadata | External ModelExpress server |
| TensorRT-LLM API | Native `checkpoint_format="MX"` |
| Fallback | Hugging Face cache when no compatible source is ready |

## Build the image

The example Dockerfile starts from a TensorRT-LLM release image containing the
native MX checkpoint loader and installs the ModelExpress client from this
checkout without replacing the release image's dependencies:

```bash
docker build \
  --build-arg TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22 \
  -f examples/p2p_transfer_k8s/client/trtllm/Dockerfile \
  -t <your-registry>/trtllm-modelexpress:latest \
  .
```

Build from the repository root so `modelexpress_client/python` is in the build
context. Override `TRTLLM_IMAGE` with another qualified TensorRT-LLM release
when needed. The Dockerfile uses the release image's matching NIXL Python
binding and native libraries, so deployments do not need to configure
`PYTHONPATH` or `LD_LIBRARY_PATH`. It does not install a second NIXL wheel,
overlay TensorRT-LLM source, or apply patches.

## Run

Deploy the ModelExpress server in the target namespace so it is reachable at
`modelexpress-server:8001`. Create a PVC named `model-cache` containing the
supported model in Hugging Face cache layout, create the `hf-token-secret` and
`nvcr-imagepullsecret` secrets, then set the qualified worker image in
[`trtllm-single-node-p2p.yaml`](trtllm-single-node-p2p.yaml):

```bash
kubectl apply -f examples/p2p_transfer_k8s/client/trtllm/trtllm-single-node-p2p.yaml
kubectl rollout status deployment/mx-trtllm \
  --timeout=60m

# Scale only after the first replica is ready so new replicas have a source.
kubectl scale deployment/mx-trtllm --replicas=2
kubectl rollout status deployment/mx-trtllm \
  --timeout=60m
```

The manifest passes `checkpoint_format: MX` and the ModelExpress server
configuration to the standard `trtllm-serve --config` interface. A replica
immediately falls back to the shared model cache when no compatible source is
ready.

## Verify

| Stage | Expected log |
|---|---|
| Loader entry | `TRT-LLM MxModelLoader starting` |
| Strategy selection | `Eligible loaders` and `Trying strategy` |
| Source cold load | `Loading weights from disk` |
| Target transfer | `RDMA transfer complete` |
| Loader exit | `TRT-LLM MxModelLoader.load_model() COMPLETE` |

```bash
kubectl logs -l app=mx-trtllm -c trtllm --prefix |
  grep "RDMA transfer complete"

kubectl port-forward service/mx-trtllm 8000:8000
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.1-70B-Instruct","prompt":"The capital of France is","max_tokens":8}'
```

For a quick TP=1 smoke test, change `TP_SIZE` and the GPU/RDMA resource counts
to `1`, then use a small supported `LlamaForCausalLM` checkpoint. Production
scale-out should place replicas on RDMA-capable nodes.
