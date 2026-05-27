# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
import os
import sys
import time
from pathlib import Path
from typing import Literal

import pytest


@dataclass(frozen=True)
class VerlRuntimeSmokeResult:
    backend: str
    model_name: str
    update_seconds: float
    check_seconds: float
    total_seconds: float
    failed: bool = False
    error_message: str = ""
    receive_success: bool = False
    requested_model_version: int | None = None
    resolved_model_version: int | None = None
    source_roles: tuple[str, ...] = ()
    attempt_roles: tuple[str, ...] = ()
    attempt_successes: tuple[bool, ...] = ()
    retry_count: int = 0
    bytes_transferred: int = 0
    tensor_count: int = 0
    attempt_lease_ids: tuple[str, ...] = ()
    report_lease_ids: tuple[str, ...] = ()
    matching_lease_statuses: tuple[int, ...] = ()
    missing_lease_ids: tuple[str, ...] = ()
    non_completed_lease_statuses: tuple[int, ...] = ()
    transfer_lease_discovery_supported: bool = False
    replica_events: tuple[str, ...] = ()
    recovery_update_seconds: float = 0.0
    recovery_check_seconds: float = 0.0
    recovery_requested_model_version: int | None = None
    recovery_resolved_model_version: int | None = None
    recovery_source_roles: tuple[str, ...] = ()
    recovery_attempt_roles: tuple[str, ...] = ()
    recovery_attempt_successes: tuple[bool, ...] = ()
    recovery_retry_count: int = 0
    recovery_bytes_transferred: int = 0
    recovery_tensor_count: int = 0
    recovery_attempt_lease_ids: tuple[str, ...] = ()
    recovery_success: bool = False


def run_verl_checkpoint_manager_update(
    *,
    backend: Literal["modelexpress", "nccl"],
    tmp_path: Path,
    verl_repo: Path,
    mx_python_path: Path,
    server_url: str | None = None,
    global_steps: int = 17,
    republish_received: bool = False,
    recover_latest_from_replica: bool = False,
    fail_trainer_transfer_before_recovery: bool = False,
    fail_refit_after_tensors: int | None = None,
    expect_update_failure: bool = False,
) -> VerlRuntimeSmokeResult:
    torch = pytest.importorskip("torch")
    ray = pytest.importorskip("ray")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    if backend == "modelexpress":
        pytest.importorskip("nixl._api")
    if backend == "nccl":
        pytest.importorskip("cupy")

    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")

    if not (verl_repo / "verl").is_dir():
        pytest.skip("MX_VERL_REPO_PATH does not point at a veRL checkout")

    for local_path in (mx_python_path, verl_repo):
        path_text = str(local_path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    python_path = os.pathsep.join([str(verl_repo), str(mx_python_path)])
    model_path = _create_tiny_qwen2_checkpoint(tmp_path / f"tiny-qwen2-verl-{backend}")

    if backend == "modelexpress":
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
    from verl.workers.rollout import BaseRollout, RolloutReplica

    class TrainingWorkerSmoke(TrainingWorker):
        def __init__(
            self,
            config: TrainingWorkerConfig,
            checkpoint_engine_config: CheckpointEngineConfig,
        ) -> None:
            super().__init__(config)
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            engine_backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = dict(checkpoint_engine_config.engine_kwargs.get(engine_backend, {}))
            if engine_backend == "nccl" and torch.distributed.get_rank() == 0:
                engine_kwargs["is_master"] = True
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                engine_backend,
                bucket_size=bucket_size,
                **engine_kwargs,
            )

        @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
        async def update_weights(self, global_steps: int = None, mode: str = "auto"):
            per_tensor_param, _ = self.engine.get_per_tensor_param()
            if backend == "modelexpress":
                await self.checkpoint_engine.send_weights(
                    per_tensor_param,
                    global_steps=global_steps,
                )
            else:
                await self.checkpoint_engine.send_weights(per_tensor_param)

        @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
        def execute_checkpoint_engine(self, method: str, *args, **kwargs):
            return getattr(self.checkpoint_engine, method)(*args, **kwargs)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
        def drop_published_nixl_managers_without_stale(self):
            transfer = getattr(self.checkpoint_engine, "_transfer", None)
            if transfer is None:
                raise RuntimeError("ModelExpress transfer state is unavailable")
            transfer.shutdown_nixl_manager()

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
            cls=ray.remote(TrainingWorkerSmoke),
            config=trainer_config,
            checkpoint_engine_config=checkpoint_engine_config,
        )
        ray_cls_with_init.update_options({"runtime_env": {"env_vars": _ray_env(python_path)}})
        return RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            device_name=get_device_name(),
        )

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
            received_count = 0
            async for name, weight in weights:
                received_count += 1
                if self.check_allclose:
                    self.received_weights[name] = weight.clone()
                if (
                    fail_refit_after_tensors is not None
                    and received_count >= fail_refit_after_tensors
                ):
                    raise RuntimeError("synthetic veRL refit failure after MX receive")

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
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.events: list[str] = []

        async def init_hybrid(self, worker_group: RayWorkerGroup):
            start = self.world_size * self.replica_rank
            stop = self.world_size * (self.replica_rank + 1)
            self.workers = worker_group.workers[start:stop]

        async def abort_all_requests(self):
            self.events.append("abort_all_requests")

        async def release_kv_cache(self):
            self.events.append("release_kv_cache")

        async def resume_kv_cache(self):
            self.events.append("resume_kv_cache")

        async def resume_generation(self):
            self.events.append("resume_generation")

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

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def transfer_lease_snapshot(self):
            summary = self.checkpoint_engine.transfer_lease_summary()
            return {
                "report_lease_ids": tuple(summary.report_lease_ids),
                "matching_lease_statuses": tuple(
                    int(lease.status) for lease in summary.matching_leases
                ),
                "missing_lease_ids": tuple(summary.missing_lease_ids),
                "non_completed_lease_statuses": tuple(
                    int(lease.status)
                    for lease in summary.non_completed_matching_leases
                ),
                "discovery_supported": summary.inventory.discovery_supported,
            }

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def receive_report_snapshot(self):
            report = self.checkpoint_engine.last_receive_report
            if report is None:
                return {}
            return {
                "requested_model_version": report.requested_model_version,
                "resolved_model_version": report.resolved_model_version,
                "success": report.success,
                "retry_count": report.retry_count,
                "source_roles": tuple(
                    attempt.role.value for attempt in report.attempts if attempt.success
                ),
                "attempt_roles": tuple(
                    attempt.role.value for attempt in report.attempts
                ),
                "attempt_successes": tuple(
                    attempt.success for attempt in report.attempts
                ),
                "bytes_transferred": sum(
                    attempt.bytes_transferred
                    for attempt in report.attempts
                    if attempt.success
                ),
                "tensor_count": sum(
                    attempt.tensor_count for attempt in report.attempts if attempt.success
                ),
                "attempt_lease_ids": tuple(
                    attempt.lease_id for attempt in report.attempts if attempt.lease_id
                ),
            }

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
        ray_cls_with_init.update_options({"runtime_env": {"env_vars": _ray_env(python_path)}})
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

    async def run_update() -> VerlRuntimeSmokeResult:
        model_name = f"tiny-qwen2-verl-{backend}-smoke-{int(time.time())}"
        ray.init(
            runtime_env={"env_vars": _ray_env(python_path)},
            include_dashboard=False,
            num_gpus=2,
        )

        checkpoint_engine_config = _checkpoint_engine_config(
            backend=backend,
            model_name=model_name,
            server_url=server_url,
            republish_received=republish_received,
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

        total_started = time.perf_counter()
        update_started = time.perf_counter()
        failed = False
        error_message = ""
        try:
            await manager.update_weights(global_steps=global_steps)
        except Exception as exc:
            failed = True
            error_message = str(exc)
            if not expect_update_failure:
                raise
        else:
            if expect_update_failure:
                raise AssertionError("expected veRL checkpoint update to fail")
        update_seconds = time.perf_counter() - update_started

        if failed:
            check_seconds = 0.0
        else:
            check_started = time.perf_counter()
            rollout.check_weights()
            check_seconds = time.perf_counter() - check_started
        receive_report = (
            _receive_report_snapshot(rollout) if backend == "modelexpress" else {}
        )
        recovery_update_seconds = 0.0
        recovery_check_seconds = 0.0
        recovery_report = {}
        if recover_latest_from_replica:
            if backend != "modelexpress":
                raise ValueError("latest replica recovery is only supported for ModelExpress")
            if failed:
                raise AssertionError("cannot recover latest after a failed initial update")
            if fail_trainer_transfer_before_recovery:
                ray.get(trainer.drop_published_nixl_managers_without_stale())
            else:
                ray.get(
                    trainer.execute_checkpoint_engine(
                        ["mark_current_source_stale"] * trainer.world_size
                    )
                )
            recovery_rollout, recovery_replicas = await create_rollout_worker_group(
                rollout_pool,
                model_config,
                rollout_config,
                check_allclose=True,
            )
            recovery_manager = CheckpointEngineManager(
                config=checkpoint_engine_config,
                trainer=trainer,
                replicas=recovery_replicas,
            )
            recovery_manager.build_process_group(recovery_rollout)
            recovery_started = time.perf_counter()
            try:
                ray.get(recovery_rollout.update_weights(global_steps=None))
            finally:
                recovery_manager.finalize_process_group(recovery_rollout)
            recovery_update_seconds = time.perf_counter() - recovery_started
            recovery_check_started = time.perf_counter()
            recovery_rollout.check_weights()
            recovery_check_seconds = time.perf_counter() - recovery_check_started
            recovery_report = _receive_report_snapshot(recovery_rollout)
        lease_snapshot = _lease_snapshot(rollout) if backend == "modelexpress" else {}
        return VerlRuntimeSmokeResult(
            backend=backend,
            model_name=model_name,
            update_seconds=update_seconds,
            check_seconds=check_seconds,
            total_seconds=time.perf_counter() - total_started,
            failed=failed,
            error_message=error_message,
            receive_success=bool(receive_report.get("success", False)),
            requested_model_version=receive_report.get("requested_model_version"),
            resolved_model_version=receive_report.get("resolved_model_version"),
            source_roles=tuple(receive_report.get("source_roles", ())),
            attempt_roles=tuple(receive_report.get("attempt_roles", ())),
            attempt_successes=tuple(receive_report.get("attempt_successes", ())),
            retry_count=int(receive_report.get("retry_count", 0)),
            bytes_transferred=int(receive_report.get("bytes_transferred", 0)),
            tensor_count=int(receive_report.get("tensor_count", 0)),
            attempt_lease_ids=tuple(receive_report.get("attempt_lease_ids", ())),
            report_lease_ids=tuple(lease_snapshot.get("report_lease_ids", ())),
            matching_lease_statuses=tuple(
                lease_snapshot.get("matching_lease_statuses", ())
            ),
            missing_lease_ids=tuple(lease_snapshot.get("missing_lease_ids", ())),
            non_completed_lease_statuses=tuple(
                lease_snapshot.get("non_completed_lease_statuses", ())
            ),
            transfer_lease_discovery_supported=bool(
                lease_snapshot.get("discovery_supported", False)
            ),
            replica_events=tuple(
                event for replica in replicas for event in replica.events
            ),
            recovery_update_seconds=recovery_update_seconds,
            recovery_check_seconds=recovery_check_seconds,
            recovery_requested_model_version=recovery_report.get(
                "requested_model_version"
            ),
            recovery_resolved_model_version=recovery_report.get(
                "resolved_model_version"
            ),
            recovery_source_roles=tuple(recovery_report.get("source_roles", ())),
            recovery_attempt_roles=tuple(recovery_report.get("attempt_roles", ())),
            recovery_attempt_successes=tuple(
                recovery_report.get("attempt_successes", ())
            ),
            recovery_retry_count=int(recovery_report.get("retry_count", 0)),
            recovery_bytes_transferred=int(
                recovery_report.get("bytes_transferred", 0)
            ),
            recovery_tensor_count=int(recovery_report.get("tensor_count", 0)),
            recovery_attempt_lease_ids=tuple(
                recovery_report.get("attempt_lease_ids", ())
            ),
            recovery_success=bool(recovery_report.get("success", False)),
        )

    try:
        return asyncio.run(run_update())
    finally:
        ray.shutdown()


def _ray_env(python_path: str) -> dict[str, str]:
    return {
        "NCCL_IB_DISABLE": "1",
        "NCCL_DEBUG": "WARN",
        "PYTHONPATH": python_path,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "RAY_DEDUP_LOGS": "0",
        "VERL_LOGGING_LEVEL": "DEBUG",
    }


def _checkpoint_engine_config(
    *,
    backend: Literal["modelexpress", "nccl"],
    model_name: str,
    server_url: str | None,
    republish_received: bool = False,
):
    from verl.workers.config import CheckpointEngineConfig

    if backend == "modelexpress":
        if server_url is None:
            raise ValueError("server_url is required for the ModelExpress backend")
        return CheckpointEngineConfig(
            backend="modelexpress",
            update_weights_bucket_megabytes=16,
            custom_backend_module="modelexpress.integrations.verl_checkpoint_engine",
            engine_kwargs={
                "modelexpress": {
                    "server_url": server_url,
                    "model_name": model_name,
                    "dtype": "bfloat16",
                    "retain_latest_k": 1,
                    "timeout_seconds": 60.0,
                    "republish_received": republish_received,
                    "retain_sources_on_finalize": True,
                }
            },
        )
    return CheckpointEngineConfig(
        backend="nccl",
        update_weights_bucket_megabytes=16,
        engine_kwargs={"nccl": {"rebuild_group": False}},
    )


def _lease_snapshot(rollout):
    snapshots = rollout.transfer_lease_snapshot()
    if isinstance(snapshots, list):
        return snapshots[0] if snapshots else {}
    return snapshots


def _receive_report_snapshot(rollout):
    snapshots = rollout.receive_report_snapshot()
    if isinstance(snapshots, list):
        return snapshots[0] if snapshots else {}
    return snapshots


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
