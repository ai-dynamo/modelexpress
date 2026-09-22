# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one real MILES optimizer step and two ModelExpress M2N update rounds."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import miles.utils.external_utils.command_utils as U

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_TYPE = os.environ.get("MILES_MODEL_TYPE", "qwen2.5-0.5B")
CACHE_ROOT = Path(os.environ.get("MILES_CACHE_ROOT", "/models"))
ACTOR_GPUS = int(os.environ.get("ACTOR_GPUS", "2"))
PIPELINE_PARALLEL_SIZE = int(os.environ.get("PIPELINE_PARALLEL_SIZE", "2"))
ROLLOUT_GPUS = int(os.environ.get("ROLLOUT_GPUS", "2"))
ROLLOUT_GPUS_PER_ENGINE = int(os.environ.get("ROLLOUT_GPUS_PER_ENGINE", "1"))
TOTAL_GPUS = int(os.environ.get("TOTAL_GPUS", str(ACTOR_GPUS + ROLLOUT_GPUS)))


def prepare() -> tuple[Path, Path]:
    model_dir = CACHE_ROOT / "models" / MODEL_ID.rsplit("/", 1)[-1]
    dataset_root = CACHE_ROOT / "datasets"
    dataset_dir = dataset_root / "gsm8k"
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    if not (model_dir / "config.json").exists():
        U.exec_command_cpu(f"hf download {MODEL_ID} --local-dir {model_dir}")
    if not (dataset_dir / "train.parquet").exists():
        U.hf_download_dataset("zhuzilin/gsm8k", data_dir=str(dataset_root))
    return model_dir, dataset_dir


def execute(model_dir: Path, dataset_dir: Path) -> None:
    train_args = " ".join(
        (
            f"--hf-checkpoint {model_dir}",
            f"--prompt-data {dataset_dir / 'train.parquet'}",
            "--input-key messages",
            "--label-key label",
            "--apply-chat-template",
            "--rollout-shuffle",
            "--rm-type math",
            "--num-rollout 1",
            "--rollout-batch-size 8",
            "--n-samples-per-prompt 4",
            "--rollout-max-response-len 256",
            "--rollout-temperature 0.8",
            "--global-batch-size 32",
            "--advantage-estimator grpo",
            "--entropy-coef 0.00",
            "--eps-clip 0.2",
            "--eps-clip-high 0.28",
            "--optimizer adam",
            "--lr 1e-6",
            "--lr-decay-style constant",
            "--weight-decay 0.1",
            "--adam-beta1 0.9",
            "--adam-beta2 0.98",
            "--tensor-model-parallel-size 1",
            f"--pipeline-model-parallel-size {PIPELINE_PARALLEL_SIZE}",
            "--context-parallel-size 1",
            "--expert-model-parallel-size 1",
            "--expert-tensor-parallel-size 1",
            "--use-dynamic-batch-size",
            "--max-tokens-per-gpu 9216",
            f"--rollout-num-gpus {ROLLOUT_GPUS}",
            f"--rollout-num-gpus-per-engine {ROLLOUT_GPUS_PER_ENGINE}",
            "--sglang-mem-fraction-static 0.65",
            "--attention-dropout 0.0",
            "--hidden-dropout 0.0",
            "--accumulate-allreduce-grads-in-fp32",
            "--attention-softmax-in-fp32",
            "--attention-backend flash",
            "--actor-num-nodes 1",
            f"--actor-num-gpus-per-node {ACTOR_GPUS}",
            "--megatron-to-hf-mode bridge",
            "--update-weights-interval 1",
            "--update-weight-transfer-mode external",
            "--update-weight-transfer-protocol "
            "modelexpress_rl.collective.integrations.miles_protocol:build_protocol",
            "--skip-eval-before-train",
            "--ci-test",
        )
    )
    forwarded_environment = {
        key: os.environ[key]
        for key in (
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "MX_SERVER_ADDRESS",
            "NCCL_CUMEM_ENABLE",
            "SGLANG_NCCL_SO_PATH",
            "SGLANG_PLUGINS",
        )
    }
    for key in (
        "MX_MILES_RUN_ID",
        "MX_MILES_VERIFY_TENSOR_EQUALITY",
        "MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S",
        "MX_NCCL_REFIT_GROUP_TIMEOUT_S",
        "MX_NCCL_REFIT_NUM_STREAMS",
        "MX_NCCL_REFIT_POLL_INTERVAL_S",
        "MX_NCCL_REFIT_TRANSFER_TIMEOUT_S",
    ):
        if key in os.environ:
            forwarded_environment[key] = os.environ[key]
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=TOTAL_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
        extra_env_vars=forwarded_environment,
    )


if __name__ == "__main__":
    subprocess.run(
        ["python3", "/opt/modelexpress-miles-m2n/verify_runtime.py"],
        check=True,
    )
    model_path, data_path = prepare()
    execute(model_path, data_path)
