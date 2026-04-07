# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch model_loader.py for PRESHARDED P2P — Option 3 architecture.

Source publishes weights BEFORE post_load_weights (pre-processed state).
Target receives via P2P and runs full post_load_weights normally.
No _mx_p2p_weights_loaded flag needed — both source and target run
the same post_load_weights transforms.

apply_patches.py already adds the PRESHARDED block with load_weights skip.
This patch adds:
1. ModelExpress source publish hook BEFORE post_load_weights loop
2. Updates worker.py publish hook to skip if already published from model_loader
"""
import os
import sys

target = "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/pyexecutor/model_loader.py"

with open(target) as f:
    content = f.read()

# Patch 1: Insert source publish hook before post_load_weights loop
old1 = """            for module in model.modules():
                if hasattr(module, 'post_load_weights') and not getattr(
                        module, '_weights_removed', False):
                    module.post_load_weights()"""

new1 = """            # ModelExpress source: publish pre-processed weights BEFORE
            # post_load_weights so targets receive raw loaded state and can
            # run their own post_load_weights() transforms.
            if os.environ.get("MODEL_EXPRESS_SOURCE"):
                try:
                    from modelexpress.trtllm_live_transfer import publish_model_params
                    publish_model_params(model)
                    model._mx_source_published = True
                except Exception as e:
                    import logging
                    logging.getLogger("modelexpress").warning("ModelExpress publish failed: %s", e)

            for module in model.modules():
                if hasattr(module, 'post_load_weights') and not getattr(
                        module, '_weights_removed', False):
                    module.post_load_weights()"""

if old1 in content and "publish_model_params" not in content:
    content = content.replace(old1, new1)
    print("patch_model_loader: patch 1 (source publish hook) applied")
elif "publish_model_params" in content:
    print("patch_model_loader: patch 1 already applied")
else:
    print("patch_model_loader: WARNING — patch 1 target not found", file=sys.stderr)

# Ensure 'import os' exists at top of file
if "import os" not in content.split("\n")[0:20]:
    content = "import os\n" + content
    print("patch_model_loader: added 'import os'")

with open(target, "w") as f:
    f.write(content)
print("patch_model_loader: done")


# Patch 2: Update worker.py publish hook to skip if already published
worker_target = "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/executor/worker.py"
if os.path.exists(worker_target):
    with open(worker_target) as f:
        worker_content = f.read()

    old_worker = """    # ModelExpress source: publish this rank's model params via NIXL
    if os.environ.get("MODEL_EXPRESS_SOURCE"):
        try:
            from modelexpress.trtllm_live_transfer import publish_from_worker
            publish_from_worker(worker)
        except Exception as e:
            logger.warning("ModelExpress publish_from_worker failed on rank %d: %s", mpi_rank(), e)"""

    new_worker = """    # ModelExpress source: publish this rank's model params via NIXL.
    # Skip if already published from ModelLoader.load() (pre-post_load_weights).
    if os.environ.get("MODEL_EXPRESS_SOURCE"):
        model = getattr(getattr(getattr(worker, 'engine', None), 'model_engine', None), 'model', None)
        if model and getattr(model, '_mx_source_published', False):
            logger.info("ModelExpress: already published from model_loader, skipping worker publish")
        else:
            try:
                from modelexpress.trtllm_live_transfer import publish_from_worker
                publish_from_worker(worker)
            except Exception as e:
                logger.warning("ModelExpress publish_from_worker failed on rank %d: %s", mpi_rank(), e)"""

    if old_worker in worker_content:
        worker_content = worker_content.replace(old_worker, new_worker)
        with open(worker_target, "w") as f:
            f.write(worker_content)
        print("patch_model_loader: worker.py patch applied (skip duplicate publish)")
    elif "_mx_source_published" in worker_content:
        print("patch_model_loader: worker.py already patched")
    else:
        print("patch_model_loader: WARNING — worker.py patch target not found", file=sys.stderr)
else:
    print("patch_model_loader: worker.py not found at expected path")
