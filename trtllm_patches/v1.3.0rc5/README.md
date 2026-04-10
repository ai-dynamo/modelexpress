# TRT-LLM 1.3.0rc5 PRESHARDED Patches

Patches for TRT-LLM 1.3.0rc5 (used in Dynamo v1.0.0, image `karenc:dynamo-trtllm-v1.0.0-a9b6f95`).

## File paths (inside container)

```text
/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/llmapi/llm_args.py
/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/pyexecutor/model_loader.py
/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/modules/linear.py
```

## Changes

### llm_args.py (line ~2818)
Add `PRESHARDED = 3` to `LoadFormat` enum.

### model_loader.py (line ~337, after DUMMY branch)
Add `elif load_format == LoadFormat.PRESHARDED:` branch that:
- Sets `_weights_presharded = True` on all Linear modules
- Calls `checkpoint_loader.load_weights(checkpoint_dir, mapping=self.mapping, model=model)`
- If result is empty dict, skips `model.load_weights()`

### linear.py (functions at lines ~175, ~211, ~264)
In `load_weights_vanilla_helper`, `load_weights_fused_qkv_helper`, and
`load_weights_fused_gate_up_helper`: override `module.tp_size` to 1 when
`getattr(module, '_weights_presharded', False)` is True.
