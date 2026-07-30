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

## Image requirement

Use a qualified TensorRT-LLM image that already contains the native MX
checkpoint loader, then install the ModelExpress client into that image
without replacing the CUDA, NIXL, UCX, Torch, or protobuf stack. This example
does not provide a source overlay, patch bundle, or compatibility image.

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
configuration to the standard `trtllm-serve --config` interface. The
source-query timeout is zero, so a replica immediately falls back to the
shared model cache when no compatible source is ready.

## Verify

```bash
kubectl logs -l app=mx-trtllm -c trtllm --prefix |
  grep "MX P2P weight transfer succeeded"

kubectl port-forward service/mx-trtllm 8000:8000
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.1-70B-Instruct","prompt":"The capital of France is","max_tokens":8}'
```

For a quick TP=1 smoke test, change `TP_SIZE` and the GPU/RDMA resource counts
to `1`, then use a small supported `LlamaForCausalLM` checkpoint. Production
scale-out should place replicas on RDMA-capable nodes.
