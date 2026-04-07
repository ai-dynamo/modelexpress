# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply PRESHARDED patches to TRT-LLM 1.3.0rc5.

Run inside container:
    python3 /tmp/apply_patches.py

Patches 3 files to add LoadFormat.PRESHARDED support for ModelExpress P2P.
"""
import importlib.util
import re
import sys
from pathlib import Path

spec = importlib.util.find_spec("tensorrt_llm")
SITE = Path(spec.submodule_search_locations[0]) if spec else Path("/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm")

def patch_llm_args():
    """Add PRESHARDED = 3 to LoadFormat enum."""
    p = SITE / "llmapi" / "llm_args.py"
    text = p.read_text()
    if "PRESHARDED" in text:
        print("llm_args.py: already patched")
        return
    text = text.replace(
        "    VISION_ONLY = 2",
        "    VISION_ONLY = 2\n"
        "    # Weights already sharded per-rank (e.g. via ModelExpress P2P RDMA)\n"
        "    PRESHARDED = 3",
    )
    p.write_text(text)
    print("llm_args.py: patched (added PRESHARDED = 3)")


def patch_model_loader():
    """Add PRESHARDED branch to ModelLoader.load()."""
    p = SITE / "_torch" / "pyexecutor" / "model_loader.py"
    text = p.read_text()
    if "PRESHARDED" in text:
        print("model_loader.py: already patched")
        return

    old = '''\
            elif load_format == LoadFormat.VISION_ONLY:
                # Vision weights are already loaded within the model.
                logger.info(
                    "LoadFormat.VISION_ONLY: skipping weight loading; using preloaded vision weights."
                )

            else:'''

    new = '''\
            elif load_format == LoadFormat.PRESHARDED:
                for module in model.modules():
                    if hasattr(module, 'tp_size'):
                        module._weights_presharded = True
                weights = checkpoint_loader.load_weights(
                    checkpoint_dir, mapping=self.mapping, model=model)
                if weights:
                    self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                        model, config)
                    self._call_load_weights(model.load_weights, weights,
                                            self.weight_mapper)
                else:
                    logger.info("PRESHARDED: weights injected directly, skipping load_weights()")

            elif load_format == LoadFormat.VISION_ONLY:
                # Vision weights are already loaded within the model.
                logger.info(
                    "LoadFormat.VISION_ONLY: skipping weight loading; using preloaded vision weights."
                )

            else:'''

    if old not in text:
        print("model_loader.py: ERROR — cannot find insertion point", file=sys.stderr)
        sys.exit(1)
    text = text.replace(old, new)
    p.write_text(text)
    print("model_loader.py: patched (added PRESHARDED branch)")


def patch_linear():
    """Override tp_size to 1 when _weights_presharded is True in 3 helper functions."""
    p = SITE / "_torch" / "modules" / "linear.py"
    text = p.read_text()
    if "_weights_presharded" in text:
        print("linear.py: already patched")
        return

    helpers = [
        ("load_weights_vanilla_helper", "assert len(weights) == 1"),
        ("load_weights_fused_qkv_helper", "if not allow_partial_loading:"),
        ("load_weights_fused_gate_up_helper", "if not allow_partial_loading:"),
    ]

    override_line = "    _tp_size = 1 if getattr(module, '_weights_presharded', False) else module.tp_size\n"
    count = 0

    for func_name, anchor in helpers:
        if func_name not in text:
            print(f"linear.py: WARNING — cannot find {func_name}", file=sys.stderr)
            continue
        func_idx = text.find(func_name)
        anchor_idx = text.find(anchor, func_idx)
        if anchor_idx == -1:
            print(f"linear.py: WARNING — cannot find anchor '{anchor}' in {func_name}", file=sys.stderr)
            continue
        line_start = text.rfind("\n", 0, anchor_idx) + 1
        text = text[:line_start] + override_line + text[line_start:]
        count += 1

    text = re.sub(
        r'(load_weight_shard\([^)]*?,\s*)module\.tp_size',
        r'\1_tp_size',
        text,
    )

    p.write_text(text)
    print(f"linear.py: patched ({count} helpers, replaced module.tp_size with _tp_size)")


def patch_worker_main():
    """Add publish_from_worker() call to worker_main() after worker creation."""
    p = SITE / "executor" / "worker.py"
    text = p.read_text()
    if "publish_from_worker" in text:
        print("worker.py: already patched")
        return

    anchor = '''\
    # Optionally disable GC (default: not disabled)
    if os.getenv("TRTLLM_WORKER_DISABLE_GC", "0") == "1":'''

    patch = '''\
    # ModelExpress source: publish this rank's model params via NIXL
    if os.environ.get("MODEL_EXPRESS_SOURCE"):
        try:
            from modelexpress.trtllm_live_transfer import publish_from_worker
            publish_from_worker(worker)
        except Exception as e:
            logger.warning("ModelExpress publish_from_worker failed on rank %d: %s", mpi_rank(), e)

    # Optionally disable GC (default: not disabled)
    if os.getenv("TRTLLM_WORKER_DISABLE_GC", "0") == "1":'''

    if anchor not in text:
        print("worker.py: ERROR — cannot find insertion point", file=sys.stderr)
        sys.exit(1)
    text = text.replace(anchor, patch)
    p.write_text(text)
    print("worker.py: patched (added publish_from_worker hook)")


if __name__ == "__main__":
    patch_llm_args()
    patch_model_loader()
    patch_linear()
    patch_worker_main()
    print("All patches applied successfully.")
