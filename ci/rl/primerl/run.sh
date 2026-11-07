#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /app
export PYTEST_OUTPUT_DIR="/models/mx-primerl-ci/${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}/outputs"
export USERNAME_CI=CI_RUNNER
mkdir -p "$PYTEST_OUTPUT_DIR"
python - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import tomllib

from huggingface_hub import snapshot_download
import torch

assert torch.cuda.device_count() == 2, "PrimeRL CI requires exactly two visible GPUs"
revision = "c97a910849ec6aa962add3dc253a0817d61c0210"
model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
snapshot = snapshot_download(model, revision=revision, local_dir=f"/models/primerl-reverse-text-{revision}")
path = Path("configs/ci/integration/reverse-text/start.toml")
text = path.read_text()
config = tomllib.loads(text)
assert config["model"]["name"] == model
assert config["max_steps"] == 15
assert config["ckpt"].get("output_dir") is None
text = text.replace('[ckpt]\n', '[ckpt]\noutput_dir = "/checkpoints/reverse-text-modelexpress"\n', 1)
# All roles receive the same pinned local checkpoint; retain the upstream renderer.
path.write_text(text.replace(f'name = "{model}"', f'name = "{snapshot}"') + '''

[deployment]
type = "single_node"
num_train_gpus = 1
num_infer_gpus = 1
''')
evidence = Path(os.environ["PYTEST_OUTPUT_DIR"]).parent
(evidence / "runtime.json").write_text(json.dumps({
    "primerl_tag": "v0.10.1.dev140",
    "primerl_sha": "0790ebb5d856860d916b270bda4884184ce9d786",
    "mx_sha": os.environ["MX_TEST_SHA"],
    "model": model, "model_revision": revision,
    "versions": {name: importlib.metadata.version(name) for name in ("modelexpress", "torch", "vllm", "nixl")},
}, indent=2))
(evidence / "start.toml").write_text(path.read_text())
PY
python -m pytest tests/integration/test_reverse_text_modelexpress.py -v \
    --junitxml="$(dirname "$PYTEST_OUTPUT_DIR")/junit.xml"
