# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install an opt-in, CI-image-only audit of the loaded transfer catalog."""

import functools
import hashlib
import json
import os


def instrument_loader(loader_type):
    original = loader_type._load_model

    @functools.wraps(original)
    def load_model(self, vllm_config, model_config, prefix, ctx, is_speculative_draft, model=None):
        model = original(self, vllm_config, model_config, prefix, ctx, is_speculative_draft, model)
        if os.environ.get("MX_CI_WEIGHT_DIGESTS") == "1":
            import torch
            tensors = ctx.tensors
            assert tensors, "CI weight audit requires the registered transfer catalog"
            digests = {
                name: {
                    "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                    "sha256": hashlib.sha256(
                        tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
                    ).hexdigest(),
                }
                for name, tensor in tensors.items()
            }
            for name in sorted(digests):
                print("[MX-CI-WEIGHTS] " + json.dumps({name: digests[name]}, sort_keys=True), flush=True)
        return model

    loader_type._load_model = load_model


if __name__ == "__main__":
    import shutil
    import sysconfig
    from pathlib import Path

    directory = Path(sysconfig.get_path("purelib")) / "modelexpress/engines/vllm"
    shutil.copyfile(__file__, directory / "_ci_weight_diagnostics.py")
    with (directory / "loader.py").open("a") as loader:
        loader.write("\nfrom ._ci_weight_diagnostics import instrument_loader\ninstrument_loader(MxModelLoader)\n")
