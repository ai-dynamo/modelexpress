# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give cold-loaded and P2P workers identical metadata before MDC registration."""

import os
import sys

def launch(framework: str, args: list[str]) -> None:
    from huggingface_hub import snapshot_download

    model_flag = "--model" if framework == "vllm" else "--model-path"
    model = args[args.index(model_flag) + 1]
    snapshot_download(
        model,
        ignore_patterns=["*.safetensors", "*.bin", "*.gguf", "*.pt", "*.pth", "*.h5", "*.msgpack"],
    )
    os.execv(sys.executable, [sys.executable, "-m", f"dynamo.{framework}", *args])


if __name__ == "__main__":
    launch(sys.argv[1], sys.argv[2:])
