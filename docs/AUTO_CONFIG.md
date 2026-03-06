# Auto-Configuration: Removing Manual Source/Target Designation

**Status**: Planning (local doc, not checked in)

---

## Current State: Manual Configuration

Today, the user must **explicitly** designate each vLLM instance as source or target:

### For vLLM (standalone K8s)

```bash
# Source — must set --load-format mx-source
vllm serve model --load-format mx-source --worker-cls modelexpress.vllm_worker.ModelExpressWorker

# Target — must set --load-format mx-target
vllm serve model --load-format mx-target --worker-cls modelexpress.vllm_worker.ModelExpressWorker
```

### For Dynamo (vLLM worker integration)

```python
# The Dynamo vLLM worker detects ModelExpress via environment:
# - MX_REGISTER_LOADERS=1 triggers loader registration
# - MODEL_EXPRESS_URL points to the MX server
# - But --load-format mx-source / mx-target is still needed on the vLLM side
```

### For TRT-LLM

```python
# Source: explicit MxTrtllmSourcePublisher
publisher = MxTrtllmSourcePublisher(model_name=..., mx_server=..., hf_model_path=...)
publisher.initialize()

# Target: explicit MxCheckpointLoader instance
loader = MxCheckpointLoader()
llm = LLM(model=..., checkpoint_loader=loader)
```

### What the User Must Know

1. **Which node has the model on disk** → that's the source
2. **Which node needs the model** → that's the target
3. **Set the right --load-format** on each
4. **Ensure source starts before target** (target polls until source publishes)

---

## Problems with Manual Configuration

1. **Operational complexity**: Deploying P2P requires two different YAML templates (or CLI args)
   for what is logically the same workload — "run inference on this model"

2. **Error-prone**: Setting `mx-target` on a node that has the model on disk wastes it.
   Setting `mx-source` on a node without the model fails silently.

3. **No elasticity**: Auto-scaling requires the scaler to know whether to launch a source or
   target. Kubernetes HPA can't distinguish between the two.

4. **Dynamo friction**: Dynamo's model deployment system wants a single "deploy this model"
   command, not "deploy a source here and targets there."

---

## Proposed: Automatic Source/Target Detection

### Core Idea

A single `--load-format mx` (or `--load-format modelexpress`) that **auto-detects** whether
to behave as source or target based on:

1. **Is the model already available locally?** (disk/PVC/cache)
2. **Is a source already registered with the MX server?**

### Decision Flow

```
vllm serve model --load-format mx
                    │
                    ▼
         ┌──────────────────────┐
         │ Query MX server:     │
         │ GetMetadata(model)   │
         └──────────┬───────────┘
                    │
            ┌───────┴────────┐
            │                │
       Source exists?    No source?
            │                │
            ▼                ▼
     ┌─────────────┐  ┌──────────────┐
     │ TARGET mode  │  │ Check local  │
     │ - Dummy init │  │ model path   │
     │ - Wait ready │  └──────┬───────┘
     │ - RDMA recv  │         │
     │ - FP8 proc   │    ┌────┴─────┐
     └─────────────┘    │          │
                    Model on    No model
                     disk?      on disk?
                        │          │
                        ▼          ▼
                 ┌────────────┐  ┌────────────┐
                 │ SOURCE mode│  │ FAIL:      │
                 │ - Load disk│  │ No source  │
                 │ - Register │  │ No local   │
                 │ - Publish  │  │ model      │
                 └────────────┘  └────────────┘
```

### Detection Logic (Python)

```python
@register_model_loader("mx")
class MxAutoModelLoader:
    """Auto-detects source vs target role."""

    def __init__(self, load_config):
        self._role = self._detect_role(load_config)
        if self._role == "source":
            self._delegate = MxSourceModelLoader(load_config)
        else:
            self._delegate = MxTargetModelLoader(load_config)

    def _detect_role(self, load_config) -> str:
        model_path = load_config.model  # or however the model path is accessed

        # Step 1: Check if a source already exists on MX server
        mx_server = os.environ.get("MODEL_EXPRESS_URL", "localhost:8001")
        try:
            client = MxClient(mx_server)
            metadata = client.get_metadata(model_name)
            if metadata and metadata.found and len(metadata.workers) > 0:
                logger.info(f"Source already registered for {model_name}, becoming TARGET")
                return "target"
        except Exception:
            pass  # MX server not reachable, fall through

        # Step 2: Check if model is available locally
        if self._model_exists_locally(model_path):
            logger.info(f"Model found locally at {model_path}, becoming SOURCE")
            return "source"

        # Step 3: No source and no local model — wait for a source to appear
        logger.info(f"No local model and no source yet, becoming TARGET (will wait)")
        return "target"

    def _model_exists_locally(self, path) -> bool:
        """Check if model weights exist on disk/PVC."""
        if not os.path.exists(path):
            return False
        # Check for actual weight files (safetensors, bin, etc.)
        for ext in ("*.safetensors", "*.bin", "*.pt"):
            if glob.glob(os.path.join(path, ext)):
                return True
        return False

    def load_model(self, *args, **kwargs):
        return self._delegate.load_model(*args, **kwargs)
```

### Race Condition: Two Nodes Both Have the Model

If multiple nodes have the model on disk and both start simultaneously:

```
Node A: Has model → queries MX → no source → becomes SOURCE → publishes
Node B: Has model → queries MX → no source → becomes SOURCE → publishes (overwrites A!)
```

**Solutions:**

1. **First-writer-wins**: MX server returns "already registered" on second publish.
   Second node sees the existing source and switches to target mode.

2. **Lease-based**: Source acquires a lease from MX server. If lease fails (someone else
   has it), fall back to target.

3. **Don't worry about it**: Two sources is fine — targets will pick one. The "extra"
   source just wastes GPU memory holding weights. Can be reclaimed later.

4. **Prefer target**: If the MX query races and both see "no source", the one that
   successfully publishes first wins. The other gets a conflict response and becomes target.

### Multi-Source / Fan-Out

With auto-detection, we naturally support fan-out:

```
Node A: Has model, first to start → SOURCE
Node B: Has model, starts later   → sees source → TARGET (receives from A)
Node C: No model                  → sees source → TARGET (receives from A or B)
```

Once Node B has the model loaded (received from A), it could optionally **re-register as
a secondary source** so Node C can receive from both A and B in parallel.

### Environment Variables

```bash
# Single unified format (replaces mx-source / mx-target)
--load-format mx

# Optional: force a specific role (override auto-detection)
MX_ROLE=source   # Force source mode
MX_ROLE=target   # Force target mode
MX_ROLE=auto     # Default: auto-detect

# Existing
MODEL_EXPRESS_URL=modelexpress-server:8001
```

---

## Integration Points

### vLLM

```bash
# Before (two different commands):
vllm serve model --load-format mx-source --worker-cls modelexpress.vllm_worker.ModelExpressWorker
vllm serve model --load-format mx-target --worker-cls modelexpress.vllm_worker.ModelExpressWorker

# After (one command):
vllm serve model --load-format mx --worker-cls modelexpress.vllm_worker.ModelExpressWorker
```

### TRT-LLM

```python
# Before (two different code paths):
if role == "source":
    publisher = MxTrtllmSourcePublisher(...)
elif role == "target":
    loader = MxCheckpointLoader()
    llm = LLM(checkpoint_loader=loader)

# After:
llm = LLM(model=model_name, checkpoint_format="mx")
# Auto-detects: local model → source; MX source exists → target
```

### Dynamo

```python
# Dynamo deploy command — no source/target distinction:
dynamo deploy model --model meta-llama/Llama-3.1-70B --replicas 4

# Under the hood:
# - First replica finds model on shared storage → SOURCE
# - Replicas 2-4 see source registered → TARGET
# - All 4 serve inference identically
```

### Kubernetes HPA

```yaml
# Auto-scaling works because all pods use the same spec:
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    name: llm-inference  # All pods identical — role auto-detected at startup
  minReplicas: 1
  maxReplicas: 8
```

---

## Implementation Phases

### Phase 1: `--load-format mx` with Auto-Detection

- Register `mx` as a new load format (alongside existing `mx-source` / `mx-target`)
- Implement `MxAutoModelLoader` with the detection logic above
- Keep `mx-source` / `mx-target` as explicit overrides
- Add `MX_ROLE` env var for forced role

### Phase 2: First-Writer-Wins on MX Server

- Add `TryPublishMetadata` RPC that returns conflict if source already registered
- Source retries → if conflict, switch to target mode
- Handles the race condition cleanly

### Phase 3: Secondary Source Re-Registration

- After a target receives weights and is serving, it can register as a secondary source
- New targets can receive from any available source (load balancing)
- Enables tree-based fan-out: 1 → 2 → 4 → 8

### Phase 4: Remove `mx-source` / `mx-target`

- Deprecate explicit source/target designation
- All deployments use `--load-format mx`
- Documentation updated

---

## Open Questions

1. **Timeout behavior**: If no source appears and no local model exists, how long should
   the target wait before failing? Currently 2 hours — is that right for auto-mode?

2. **Source eviction**: If the source pod is killed, should targets re-register as sources
   (they now have the weights in GPU memory)?

3. **Model version**: If the source updates to a new model version, how do targets detect
   the change? Session ID comparison?

4. **Mixed-version clusters**: What if some nodes have model v1 on disk and others have v2?
   Auto-detection could lead to inconsistent serving.
