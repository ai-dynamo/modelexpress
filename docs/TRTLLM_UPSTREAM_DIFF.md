# TRT-LLM Upstream Changes for Pre-Sharded Weight Loading

**Purpose**: Enable TP>1 zero-disk weight loading by allowing custom checkpoint loaders
to return per-rank sharded weights that skip TP re-slicing during loading.

**Total diff**: ~30 lines across 3 files (backward compatible).

---

## File 1: `tensorrt_llm/llmapi/llm_args.py`

Add `PRESHARDED` to the `LoadFormat` enum:

```diff
 class LoadFormat(Enum):
     AUTO = 0
     # Initialize all weights randomly.
     DUMMY = 1
     # Only load the multimodal(vision) encoder weights
     VISION_ONLY = 2
+    # Weights are already sharded per TP rank — skip TP slicing during loading.
+    # The weight mapper still handles name mapping and fusing (q+k+v → qkv),
+    # but load_weight_shard() returns weights as-is without TP slicing.
+    # Use case: RDMA P2P transfers where each worker receives its own shard.
+    PRESHARDED = 3
```

---

## File 2: `tensorrt_llm/_torch/pyexecutor/model_loader.py`

Add a `PRESHARDED` branch in `ModelLoader.load()`. It's identical to `AUTO` except it sets
a `_weights_presharded` flag on all Linear modules before weight assignment:

```diff
             if load_format == LoadFormat.AUTO:
                 if hasattr(model, 'llm_checkpoint_dir'):
                     weights = checkpoint_loader.load_weights(
                         model.llm_checkpoint_dir, mapping=self.mapping)
                 else:
                     weights = checkpoint_loader.load_weights(
                         checkpoint_dir, mapping=self.mapping)

                 self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                     model, config)
                 self._call_load_weights(model.load_weights, weights,
                                         self.weight_mapper)

+            elif load_format == LoadFormat.PRESHARDED:
+                # Same flow as AUTO, but tell modules weights are already per-rank sharded.
+                # load_weight_shard() will skip TP slicing; fusing (q+k+v → qkv) still happens.
+                if hasattr(model, 'llm_checkpoint_dir'):
+                    weights = checkpoint_loader.load_weights(
+                        model.llm_checkpoint_dir, mapping=self.mapping)
+                else:
+                    weights = checkpoint_loader.load_weights(
+                        checkpoint_dir, mapping=self.mapping)
+
+                # Set flag on all Linear modules to skip TP slicing
+                from tensorrt_llm._torch.modules.linear import Linear
+                for module in model.modules():
+                    if isinstance(module, Linear):
+                        module._weights_presharded = True
+
+                self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
+                    model, config)
+                self._call_load_weights(model.load_weights, weights,
+                                        self.weight_mapper)

             elif load_format == LoadFormat.DUMMY:
```

---

## File 3: `tensorrt_llm/_torch/modules/linear.py`

In the three weight loading helpers, check `_weights_presharded` and pass `tp_size=1`
to `load_weight_shard()` when true. This makes the existing early-return path kick in
(line 139: `if tensor_parallel_mode is None or tensor_parallel_size <= 1: return as-is`).

### 3a: `load_weights_vanilla_helper()`

```diff
 def load_weights_vanilla_helper(module: Linear,
                                 weights: List[Dict],
                                 weight_transform=lambda x: x,
                                 bias_transform=lambda x: x,
                                 allow_partial_loading: bool = False):
     assert len(weights) == 1
     if not allow_partial_loading:
         assert "weight" in weights[0]
         if module.bias is not None:
             assert "bias" in weights[0]
     device = torch.device('cuda')
+    # If weights are pre-sharded (e.g., from P2P RDMA), skip TP slicing
+    tp_size = 1 if getattr(module, '_weights_presharded', False) else module.tp_size

-    weight = load_weight_shard(weights[0]['weight'], module.tp_size,
+    weight = load_weight_shard(weights[0]['weight'], tp_size,
                                module.tp_rank, module.tp_mode,
                                device) if "weight" in weights[0] else None
     ...
-    bias = load_weight_shard(weights[0]['bias'], module.tp_size,
+    bias = load_weight_shard(weights[0]['bias'], tp_size,
                              module.tp_rank, module.tp_mode,
                              device) if "bias" in weights[0] else None
```

### 3b: `load_weights_fused_qkv_helper()`

```diff
 def load_weights_fused_qkv_helper(
     module: Linear,
     weights: List[Dict],
     ...
 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
     ...
     device = torch.device('cuda')
+    tp_size = 1 if getattr(module, '_weights_presharded', False) else module.tp_size

-    q_weight = load_weight_shard(weights[0]['weight'], module.tp_size,
+    q_weight = load_weight_shard(weights[0]['weight'], tp_size,
                                  module.tp_rank, module.tp_mode,
                                  device) if "weight" in weights[0] else None
-    k_weight = load_weight_shard(weights[1]['weight'], module.tp_size,
+    k_weight = load_weight_shard(weights[1]['weight'], tp_size,
                                  module.tp_rank, module.tp_mode,
                                  device) if "weight" in weights[1] else None
-    v_weight = load_weight_shard(weights[2]['weight'], module.tp_size,
+    v_weight = load_weight_shard(weights[2]['weight'], tp_size,
                                  module.tp_rank, module.tp_mode,
                                  device) if "weight" in weights[2] else None
     # NOTE: The cat/fusing of q+k+v → qkv STILL HAPPENS after this.
     # We only skip the TP slicing, not the fusing.

     # Same for bias:
-    q_bias = load_weight_shard(weights[0]['bias'], module.tp_size, ...)
+    q_bias = load_weight_shard(weights[0]['bias'], tp_size, ...)
     # ... etc for k_bias, v_bias
```

### 3c: `load_weights_fused_gate_up_helper()`

Same pattern:

```diff
+    tp_size = 1 if getattr(module, '_weights_presharded', False) else module.tp_size

-    gate_weight = load_weight_shard(weights[0]['weight'], module.tp_size, ...)
+    gate_weight = load_weight_shard(weights[0]['weight'], tp_size, ...)
-    up_weight = load_weight_shard(weights[1]['weight'], module.tp_size, ...)
+    up_weight = load_weight_shard(weights[1]['weight'], tp_size, ...)
     # Same for biases
```

---

## How ModelExpress Uses This

With this change, each MPI worker receives only its rank's shard (21 GB for 70B/TP=8)
instead of the full model (141 GB). The weight mapper fuses q+k+v → qkv with the
already-sharded tensors.

```python
class MxCheckpointLoader(BaseCheckpointLoader):
    def load_weights(self, checkpoint_dir, mapping):
        # Each worker only receives from its matching source rank
        my_rank = mapping.tp_rank
        weights = nixl_receive_from_rank(my_rank)  # 21 GB, HF names, per-rank sizes
        return weights

# Usage:
llm = LLM(
    model=model_name,
    checkpoint_loader=MxCheckpointLoader(),
    load_format="PRESHARDED",          # NEW: tells TRT-LLM weights are per-rank
    tensor_parallel_size=8,
)
```

---

## What Still Works (Backward Compatible)

| Scenario | Behavior |
|----------|----------|
| `LoadFormat.AUTO` (default) | Unchanged — loads full weights, mapper shards |
| `LoadFormat.DUMMY` | Unchanged — random init |
| `LoadFormat.PRESHARDED` with full weights | Works but wasteful — slicing skipped, full weights assigned |
| `LoadFormat.PRESHARDED` with per-rank weights | Correct — slicing skipped, fusing still works |
| Existing `HfCheckpointLoader` | Unchanged — uses AUTO |
| Custom loaders without PRESHARDED | Unchanged — use AUTO |

---

## Why This is Safe

1. **No change to `load_weight_shard()` signature** — the function is untouched.
   We change what `tp_size` the callers pass to it.

2. **`_weights_presharded` is an opt-in flag** — only set when `LoadFormat.PRESHARDED`
   is explicitly used. Default `getattr(module, '_weights_presharded', False)` is False.

3. **Fusing is preserved** — `load_weights_fused_qkv_helper` still concatenates q+k+v
   into qkv. Only the per-component TP slicing is skipped.

4. **No model architecture changes** — `module.tp_size` and `module.tp_rank` are untouched.
   Inference behavior is identical.
