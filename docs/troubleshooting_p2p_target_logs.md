# Troubleshooting: mx-target logs and RDMA grep

If this returns nothing:

```bash
microk8s kubectl logs -n metadata -l app=mx-target 2>&1 | grep -iE "RDMA|transfer|bandwidth|Gbps|gRPC|backend|dummy|received" | head -20
```

it usually means the **target container never reached the RDMA receive path**. Those strings are only logged after the target has started the NIXL receive step.

## 1. See what the target is actually logging

```bash
# Last 150 lines (no filter)
microk8s kubectl logs -n metadata -l app=mx-target --tail=150

# Only lines from our loader (modelexpress)
microk8s kubectl logs -n metadata -l app=mx-target 2>&1 | grep -i modelexpress

# Early target path: loader started and timing
microk8s kubectl logs -n metadata -l app=mx-target 2>&1 | grep -iE "MxTargetModelLoader|TIMING|wait_for_ready|get_metadata|NIXL|Waiting|source worker"
```

- If you see **`MxTargetModelLoader.load_model() STARTING`** → our loader ran; look for the next line (structure init, dummy weights, then RDMA hook). If it stops before RDMA, the failure is in metadata/ready wait or NIXL init.
- If you **don’t** see `MxTargetModelLoader` → the target isn’t using our loader (check `--load-format mx-target` and image).

## 2. Common reasons RDMA lines never appear

| Symptom | Likely cause |
|--------|----------------|
| No `modelexpress` lines at all | Wrong image or `--load-format` not `mx-target`; or Python env not loading our package. |
| `MxTargetModelLoader.load_model() STARTING` then nothing | Crash or hang in model structure init or dummy weights. |
| "Waiting for source" / "wait_for_ready" and then silence | Target waiting for source ready signal (metadata/gRPC or Redis); source may not have published ready. |
| "NIXL not available" | NIXL not installed or not loadable in the target image. |

## 3. Confirm end-to-end

After you see RDMA/TIMING lines and "load_model() COMPLETE":

```bash
# Call target inference (adjust service name/port if needed)
curl -s -X POST http://<target-svc>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B","prompt":"Hello","max_tokens":10}'
```
