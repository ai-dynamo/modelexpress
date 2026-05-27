# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest


_LIVE_SERVER_ENV = "MX_LIVE_SERVER_URL"
_VERL_REPO_ENV = "MX_VERL_REPO_PATH"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_LIVE_SERVER_ENV) or not os.environ.get(_VERL_REPO_ENV),
    reason=f"{_LIVE_SERVER_ENV} and {_VERL_REPO_ENV} must be set",
)


def test_live_verl_checkpoint_manager_updates_weights_with_modelexpress(tmp_path):
    """Exercise veRL's real CheckpointEngineManager with the MX backend."""
    torch = pytest.importorskip("torch")
    ray = pytest.importorskip("ray")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("nixl._api")

    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")

    verl_repo = Path(os.environ[_VERL_REPO_ENV]).resolve()
    if not (verl_repo / "verl").is_dir():
        pytest.skip(f"{_VERL_REPO_ENV} does not point at a veRL checkout")
    sys.path.insert(0, str(verl_repo))

    mx_python_path = Path(__file__).resolve().parents[1]
    python_path = os.pathsep.join([str(verl_repo), str(mx_python_path)])
    model_path = _create_tiny_qwen2_checkpoint(tmp_path / "tiny-qwen2-verl-mx")

    from modelexpress.integrations import verl_checkpoint_engine  # noqa: F401
    from transformers import AutoModelForCausalLM
    from verl.checkpoint_engine import (
        CheckpointEngineManager,
        CheckpointEngineRegistry,
        CheckpointEngineWorker,
    )
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from verl.single_controller.ray.base import split_resource_pool
    from verl.utils.device import get_device_name
    from verl.utils.fs import copy_to_local
    from verl.utils.import_utils import import_external_libs
    from verl.workers.config import (
        CheckpointEngineConfig,
        FSDPEngineConfig,
        HFModelConfig,
        RolloutConfig,
        TrainingWorkerConfig,
    )
    from verl.workers.engine_workers import TrainingWorker

    class TrainingWorkerMX(TrainingWorker):
        def __init__(
            self,
            config: TrainingWorkerConfig,
            checkpoint_engine_config: CheckpointEngineConfig,
        ) -> None:
            super().__init__(config)
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend,
                bucket_size=bucket_size,
                **engine_kwargs,
            )

        @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
        async def update_weights(self, global_steps: int = None, mode: str = "auto"):
            per_tensor_param, _ = self.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(
                per_tensor_param,
                global_steps=global_steps,
            )

        @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
        def execute_checkpoint_engine(self, method: str, *args, **kwargs):
            return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    def create_trainer_worker_group(
        resource_pool: RayResourcePool,
        model_config: HFModelConfig,
        checkpoint_engine_config: CheckpointEngineConfig,
    ) -> RayWorkerGroup:
        engine_config = FSDPEngineConfig(
            forward_only=True,
            fsdp_size=resource_pool.world_size,
            strategy="fsdp",
            model_dtype="bf16",
            use_torch_compile=False,
        )
        trainer_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=model_config,
            engine_config=engine_config,
        )
        ray_cls_with_init = RayClassWithInitArgs(
            cls=ray.remote(TrainingWorkerMX),
            config=trainer_config,
            checkpoint_engine_config=checkpoint_engine_config,
        )
        ray_cls_with_init.update_options(
            {
                "runtime_env": {
                    "env_vars": {
                        "PYTHONPATH": python_path,
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    }
                }
            }
        )
        return RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            device_name=get_device_name(),
        )

    from verl.workers.rollout import BaseRollout, RolloutReplica

    class MockServerAdapter(BaseRollout):
        def __init__(
            self,
            config: RolloutConfig,
            model_config: HFModelConfig,
            check_allclose: bool = True,
        ) -> None:
            super().__init__(config, model_config, device_mesh=None)
            self.check_allclose = check_allclose
            self.model = None
            self.received_weights = {}

        async def resume(self, tags: list[str]):
            raise NotImplementedError

        async def release(self):
            raise NotImplementedError

        async def update_weights(self, weights, **kwargs):
            async for name, weight in weights:
                if self.check_allclose:
                    self.received_weights[name] = weight.clone()

        def check_weights(self):
            if not self.check_allclose:
                return
            if self.model is None:
                local_path = copy_to_local(self.model_config.path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                )
            for name, weight in self.model.state_dict().items():
                assert name in self.received_weights, f"weight {name} not received"
                received = self.received_weights[name]
                assert torch.allclose(weight.to(received.device), received), (
                    f"weight {name} not equal"
                )
            self.received_weights.clear()

    class MockReplica(RolloutReplica):
        async def init_hybrid(self, worker_group: RayWorkerGroup):
            start = self.world_size * self.replica_rank
            stop = self.world_size * (self.replica_rank + 1)
            self.workers = worker_group.workers[start:stop]

        def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
            raise NotImplementedError

        async def launch_servers(self):
            raise NotImplementedError

    class CheckpointEngineWorkerSmoke(CheckpointEngineWorker):
        def __init__(
            self,
            rollout_config: RolloutConfig,
            model_config: HFModelConfig,
            check_allclose: bool = True,
            *args,
            **kwargs,
        ) -> None:
            server_adapter = MockServerAdapter(
                rollout_config,
                model_config,
                check_allclose,
            )
            super().__init__(rollout_config, model_config, server_adapter, *args, **kwargs)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def check_weights(self):
            self.server_adapter.check_weights()

    async def create_rollout_worker_group(
        resource_pool: RayResourcePool,
        model_config: HFModelConfig,
        rollout_config: RolloutConfig,
        check_allclose: bool = True,
    ):
        ray_cls_with_init = RayClassWithInitArgs(
            cls=ray.remote(CheckpointEngineWorkerSmoke),
            model_config=model_config,
            rollout_config=rollout_config,
            check_allclose=check_allclose,
        )
        ray_cls_with_init.update_options(
            {
                "runtime_env": {
                    "env_vars": {
                        "PYTHONPATH": python_path,
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    }
                }
            }
        )
        worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            device_name=get_device_name(),
        )
        rollout_world_size = (
            rollout_config.tensor_model_parallel_size
            * rollout_config.data_parallel_size
            * rollout_config.pipeline_model_parallel_size
        )
        replicas = []
        for replica_rank in range(worker_group.world_size // rollout_world_size):
            replicas.append(
                MockReplica(
                    replica_rank=replica_rank,
                    config=rollout_config,
                    model_config=model_config,
                )
            )
        await asyncio.gather(*(replica.init_hybrid(worker_group) for replica in replicas))
        return worker_group, replicas

    async def run_update() -> None:
        ray.init(
            runtime_env={
                "env_vars": {
                    "NCCL_IB_DISABLE": "1",
                    "NCCL_DEBUG": "WARN",
                    "PYTHONPATH": python_path,
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "RAY_DEDUP_LOGS": "0",
                    "VERL_LOGGING_LEVEL": "DEBUG",
                }
            },
            include_dashboard=False,
            num_gpus=2,
        )

        checkpoint_engine_config = CheckpointEngineConfig(
            backend="modelexpress",
            update_weights_bucket_megabytes=16,
            custom_backend_module="modelexpress.integrations.verl_checkpoint_engine",
            engine_kwargs={
                "modelexpress": {
                    "server_url": os.environ[_LIVE_SERVER_ENV],
                    "model_name": f"tiny-qwen2-verl-mx-smoke-{int(time.time())}",
                    "dtype": "bfloat16",
                    "retain_latest_k": 1,
                    "timeout_seconds": 60.0,
                }
            },
        )
        model_config = HFModelConfig(
            path=str(model_path),
            use_remove_padding=False,
            override_config={"attn_implementation": "eager"},
            enable_gradient_checkpointing=False,
        )
        rollout_config = RolloutConfig(
            name="vllm",
            checkpoint_engine=checkpoint_engine_config,
            tensor_model_parallel_size=1,
            data_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        resource_pool = RayResourcePool(process_on_nodes=[2], max_colocate_count=3)
        trainer_pool, rollout_pool = split_resource_pool(resource_pool, [1, 1])
        trainer = create_trainer_worker_group(
            trainer_pool,
            model_config,
            checkpoint_engine_config,
        )
        trainer.reset()
        rollout, replicas = await create_rollout_worker_group(
            rollout_pool,
            model_config,
            rollout_config,
            check_allclose=True,
        )
        manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=trainer,
            replicas=replicas,
        )
        await manager.update_weights(global_steps=17)
        rollout.check_weights()

    try:
        asyncio.run(run_update())
    finally:
        ray.shutdown()


def _create_tiny_qwen2_checkpoint(path: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    path.mkdir(parents=True, exist_ok=True)
    vocab = {"<pad>": 0, "<unk>": 1, "<|im_start|>": 2, "<|im_end|>": 3}
    for index, token in enumerate(
        ["hello", "world", "test", "rl", "checkpoint", "engine", "model", "express"],
        start=4,
    ):
        vocab[token] = index
    for index in range(len(vocab), 64):
        vocab[f"tok{index}"] = index

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
    )
    fast_tokenizer.save_pretrained(path)

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    Qwen2ForCausalLM(config).save_pretrained(path, safe_serialization=True)
    return path

