# TRT-LLM PyTorch Backend P2P — Implementation Plan

**Status**: Planning
**Date**: March 4, 2026

---

## 1. What Exists Today

### Target side (implemented, validated)

- `--model-express-url` CLI arg in Dynamo TRT-LLM worker (`backend_args.py`)
- `_setup_modelexpress_loader()` in `engine.py` injects `MxLiveCheckpointLoader`
- Sets `LoadFormat.PRESHARDED` — 3 TRT-LLM patches handle it
- `MxLiveWeightLoader.load_weights(model=model)` does RDMA into params, returns `{}`
- Validated: Qwen 0.5B, Llama 70B, DSV3.2 EP=8, Kimi K2.5 EP=4

### Source side (standalone only)

- Manual script creates model via `AutoModelForCausalLM.from_config()`, loads
  weights, calls `MxLiveSource.publish()`
- Source does NOT serve inference — sleeps forever holding GPU memory
- Requires separate deployment (not a DGD)

---

## 2. Goal

Both source and target run as Dynamo TRT-LLM workers (`python3 -m dynamo.trtllm`).
Source loads weights normally, publishes via NIXL, AND serves inference. Target
receives via RDMA, then serves inference. Identical to the vLLM pattern
(PR #6186 / PR #140).

---

## 3. Approach A: Post-LLM Hook (Recommended)

Hook into the Dynamo TRT-LLM engine AFTER `self._llm = self._llm_cls(**self.engine_args)`
completes. At this point, the model is loaded, compiled, and ready. We register the
PyTorch model's parameters with NIXL and publish.

### 3.1 Engine flow

```
engine.py flow (source):
  1. _setup_modelexpress_source()      # Set env vars, prepare
  2. self._llm = self._llm_cls(**)     # Normal load from disk
  3. _publish_modelexpress_source()    # Register NIXL, publish metadata
  4. Serve inference normally

engine.py flow (target):
  1. _setup_modelexpress_loader()      # Inject MxLiveCheckpointLoader
  2. self._llm = self._llm_cls(**)     # RDMA into params via PRESHARDED
  3. Serve inference normally
```

### 3.2 Dynamo repo changes

**`components/src/dynamo/trtllm/backend_args.py`**
- Add `--model-express-role` arg (values: `source`, `target`, `auto`)
- Environment variable: `MX_ROLE`

```python
add_argument(
    g,
    flag_name="--model-express-role",
    env_var="MX_ROLE",
    default="target",
    help="ModelExpress role: 'source' (publish weights), 'target' (receive weights), 'auto' (detect).",
)
```

**`components/src/dynamo/trtllm/engine.py`**

Add source publishing method:

```python
def _setup_modelexpress_source(self) -> None:
    """Prepare source publishing (called before LLM init)."""
    os.environ["MODEL_EXPRESS_URL"] = self._model_express_url
    os.environ.setdefault("MODEL_NAME", self.engine_args.get("model", "unknown"))

def _publish_modelexpress_source(self) -> None:
    """Register model params with NIXL and publish to MX server."""
    from modelexpress.trtllm_live_transfer import MxLiveSource

    model_path = self.engine_args.get("model", "")
    source = MxLiveSource(
        self._llm,
        model_name=os.environ["MODEL_NAME"],
        mx_server=self._model_express_url,
        model_path=model_path,
    )
    source.publish()
```

In `initialize()`:

```python
if self._model_express_url:
    if self._model_express_role == "source":
        self._setup_modelexpress_source()
    else:
        self._setup_modelexpress_loader()  # existing target path

self._llm = self._llm_cls(**self.engine_args)

if self._model_express_url and self._model_express_role == "source":
    self._publish_modelexpress_source()
```

**`components/src/dynamo/trtllm/workers/llm_worker.py`**
- Pass `model_express_role=config.model_express_role` to `get_llm_engine()`

### 3.3 ModelExpress repo changes

None required for Approach A. `MxLiveSource._get_torch_model()` already handles
LLM wrapper introspection (tries `model._model`, `model.model`, `model._executor`,
`model._engine` attribute paths).

**One concern**: MPI gather-and-publish-once. The Dynamo TRT-LLM worker uses MPI
for multi-GPU. Each rank needs to register its own params with NIXL, then rank 0
gathers and publishes all workers. This pattern is validated in the standalone
source script but `MxLiveSource.publish()` currently publishes per-rank. Two options:
1. Fix the MX server to properly merge workers (PR #135 Lua merge, already on main)
2. Add gather-and-publish to `MxLiveSource` (MPI-aware publish)

Option 1 is cleaner and already merged on main. Just need to deploy a server
image built from main.

### 3.4 TRT-LLM patches

No additional patches needed for source. The existing 3 patches
(`llm_args.py`, `model_loader.py`, `linear.py`) are only needed on the target
side for `LoadFormat.PRESHARDED`.

---

## 4. Approach B: Source Checkpoint Loader

Register a source-side checkpoint loader that loads weights AND registers them
with NIXL during the load step. Similar to vLLM's `MxSourceModelLoader`.

```python
class MxSourceCheckpointLoader:
    def load_weights(self, checkpoint_dir, mapping, model, **kwargs):
        # 1. Load weights from disk normally (delegate to HfCheckpointLoader)
        weights = HfCheckpointLoader().load_weights(checkpoint_dir, mapping)

        # 2. Register loaded params with NIXL
        nixl_mgr = NixlTransferManager(...)
        nixl_mgr.register_tensors(param_tensors)

        # 3. Publish to MX server
        publish_metadata(...)

        # 4. Return weights normally (model proceeds with load_weights)
        return weights
```

**Pros**: Registers weights at the right lifecycle point (before engine compilation)
**Cons**: More complex, needs additional TRT-LLM patches or new checkpoint loader

**Not recommended** as the starting point — Approach A achieves the same result
with less code.

---

## 5. Key Differences from vLLM Integration

| Aspect | vLLM | TRT-LLM PyTorch |
|--------|------|-----------------|
| Loader plugin | `@register_model_loader("mx-source/target")` | `checkpoint_loader=MxLiveCheckpointLoader()` |
| Weight format | Raw tensors (pre-FP8 processing) | Fused, TP-sharded, TRT-LLM format |
| Worker spawn | Python multiprocess (`ModelExpressWorker`) | MPI (NIXL works natively) |
| Post-transfer | `process_weights_after_loading()` | Engine compilation |
| Source hook | `load_model()` override | Post-`LLM()` init hook |
| TRT-LLM patches | None | 3 files (target only) |
| Role selection | `--load-format mx-source/mx-target` | `--model-express-role source/target` |

---

## 6. Deployment Pattern

### Source DGD

```yaml
# Source: loads from disk, publishes weights, serves inference
python3 -m dynamo.trtllm \
    --model-path baseten-admin/Kimi-2.5-text-nvfp4-v3 \
    --model-express-url modelexpress-server:8001 \
    --model-express-role source \
    --extra-engine-args /config/engine.yaml
```

### Target DGD

```yaml
# Target: receives weights via RDMA, serves inference
python3 -m dynamo.trtllm \
    --model-path baseten-admin/Kimi-2.5-text-nvfp4-v3 \
    --model-express-url modelexpress-server:8001 \
    --model-express-role target \
    --extra-engine-args /config/engine.yaml
```

### Future: Auto-detection

```yaml
# Auto: detects role based on model availability and MX server state
python3 -m dynamo.trtllm \
    --model-path baseten-admin/Kimi-2.5-text-nvfp4-v3 \
    --model-express-url modelexpress-server:8001 \
    --model-express-role auto \
    --extra-engine-args /config/engine.yaml
```

Auto-detection logic:
1. Query MX server for existing source
2. If source exists → become target (RDMA receive)
3. If no source and model on disk → become source (load + publish)
4. If no source and no model → wait for source

---

## 7. Open Questions

1. Can the source serve inference while holding weights for RDMA? (GPU memory
   pressure — weights stay registered with NIXL alongside KV cache and engine)
2. Do PRESHARDED patches need upstreaming to TRT-LLM for long-term support?
3. Should we support both standalone source AND DGD source long-term?
4. How does `MxLiveSource._get_torch_model()` handle TRT-LLM 1.3.0rc5's LLM
   wrapper? (Validated on 1.2.0rc6 and 1.3.0rc3, needs testing on 1.3.0rc5)

---

## 8. Timeline

| Task | Effort | Depends on |
|------|--------|------------|
| Add `--model-express-role` to `backend_args.py` | 30 min | — |
| Add `_publish_modelexpress_source()` to `engine.py` | 1 hour | — |
| Handle MPI gather-and-publish in `MxLiveSource` | 2 hours | Standalone validation |
| Deploy MX server from main (with Lua merge fix) | 30 min | — |
| Test source DGD + target DGD end-to-end | 2 hours | RDMA transport working |
| Auto-detection (`role=auto`) | 2 hours | Source DGD working |
