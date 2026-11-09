# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM startup-plan detection, compatibility keys and plan selection."""

import copy
import hashlib
import importlib
import json
import logging
import os
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.engines.vllm import adapter as adapter_module
from modelexpress.engines.vllm import framework_caches
from modelexpress.metadata.framework_cache import framework_cache_entries
from modelexpress.metadata.source_id import compute_mx_source_id


@pytest.fixture
def native(monkeypatch, tmp_path):
    @dataclass
    class CacheConfig:
        gpu_memory_utilization: float = 0.92
        kv_cache_memory_bytes: int | None = None

    @dataclass
    class ModelConfig:
        multimodal_config: object = None
        enforce_eager: bool = False
        use_fp64_gumbel: bool = False
        logprobs_mode: str = "raw_logprobs"
        convert: str = "none"
        hf_token: str = ""
        served_model_name: str = ""

        def compute_hash(self):
            return "native-main-model-hash"

    runtime = SimpleNamespace(
        VLLM_ENABLE_STARTUP_PLAN=True,
        VLLM_CACHE_ROOT=str(tmp_path / "vllm"),
        compile_factors=lambda: {},
    )
    # Native fingerprints are rank-specific; "stage" stands for the parts of
    # the live configuration that model construction mutates.
    stage = {"value": "preinit"}
    plan = SimpleNamespace(
        PLAN_SCHEMA_VERSION=1,
        compute_plan_fingerprint=Mock(
            side_effect=lambda config, rank, world_size: f"{stage['value']}-r{rank}-of{world_size}"
        ),
        _plan_path=lambda digest: str(
            Path(runtime.VLLM_CACHE_ROOT) / "startup_plan" / f"startup_plan_{digest}.json"
        ),
    )

    def save(worker, size):
        fingerprint = plan.compute_plan_fingerprint(worker.vllm_config, worker.rank, 2)
        path = Path(plan._plan_path(fingerprint))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fingerprint": fingerprint, "kv_cache_memory_bytes": size}))

    def apply(worker):
        fingerprint = plan.compute_plan_fingerprint(worker.vllm_config, worker.rank, 2)
        path = Path(plan._plan_path(fingerprint))
        if path.exists():
            worker.cache_config.kv_cache_memory_bytes = json.loads(path.read_text())[
                "kv_cache_memory_bytes"
            ]

    plan.maybe_save_startup_plan = save
    plan.maybe_apply_startup_plan = apply
    modules = {
        "vllm": SimpleNamespace(__version__="0.29.1.dev10+gexample"),
        "vllm.envs": runtime,
        "vllm.v1.worker.startup_plan": plan,
    }
    monkeypatch.setattr(framework_caches, "import_module", modules.__getitem__)
    monkeypatch.setattr(framework_caches, "_device_signature", lambda: {"name": "test GPU"})
    config = SimpleNamespace(
        cache_config=CacheConfig(),
        parallel_config=SimpleNamespace(world_size=2),
        model_config=ModelConfig(),
    )
    return SimpleNamespace(
        config=config, runtime=runtime, plan=plan, modules=modules, stage=stage
    )


@pytest.fixture
def speculative(native):
    @dataclass
    class DraftModelConfig:
        model: str = "draft-model"
        runner_type: str = "draft"
        dtype: object = torch.bfloat16
        enforce_eager: bool = False
        hf_token: str = ""
        served_model_name: str = ""
        hf_overrides: object = None
        hf_config: object = field(
            init=False,
            default_factory=lambda: SimpleNamespace(hidden_size=64, num_hidden_layers=2),
        )

        def compute_hash(self):
            return hashlib.sha256(json.dumps(vars(self.hf_config), sort_keys=True).encode()).hexdigest()

    @dataclass
    class DraftParallelConfig:
        tensor_parallel_size: int = 1
        pipeline_parallel_size: int = 1
        decode_context_parallel_size: int = 1
        prefill_context_parallel_size: int = 1
        disable_custom_all_reduce: bool = False
        master_port: int = 29501

        def compute_hash(self):
            factors = {
                name: value for name, value in vars(self).items()
                if name not in {"disable_custom_all_reduce", "master_port"}
            }
            return hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()

    @dataclass
    class SpeculativeConfig:
        method: str = "draft_model"
        model: str = "draft-model"
        num_speculative_tokens: int = 2
        num_speculative_tokens_per_batch_size: object = None
        kv_cache_dtype: str | None = None
        target_model_config: object = None
        target_parallel_config: object = None
        draft_model_config: object = field(default_factory=DraftModelConfig)
        draft_parallel_config: object = field(default_factory=DraftParallelConfig)
        draft_load_config: object = None

    native.config.speculative_config = SpeculativeConfig(
        target_model_config=native.config.model_config,
        target_parallel_config=native.config.parallel_config,
    )
    return native.config.speculative_config


@pytest.fixture
def adapter_factory(native, speculative, monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setattr(
        adapter_module.VllmAdapter, "_resolve_target_device", lambda self: torch.device("cpu")
    )
    monkeypatch.setattr(adapter_module, "accelerator_backend_for", lambda device: object())
    native.config.load_config = SimpleNamespace()
    native.config.model_config.model = "target-model"
    native.config.model_config.runner_type = "generate"
    draft = speculative.draft_model_config

    def create(*, draft_pass=False, config=None):
        config = config if config is not None else native.config
        model_config = draft if draft_pass else config.model_config
        return adapter_module.VllmAdapter(config, model_config)

    return create


def _worker(config, rank=0):
    return SimpleNamespace(vllm_config=config, cache_config=config.cache_config, rank=rank)


def _plan_dir(native) -> Path:
    return Path(native.runtime.VLLM_CACHE_ROOT) / "startup_plan"


def _archived_files(bundle) -> list[str]:
    with tarfile.open(bundle.tar_path) as archive:
        return sorted(Path(member.name).name for member in archive.getmembers() if member.isfile())


def _install(transfer, bundle) -> None:
    (staged,) = transfer.target_file_paths()
    staged.write_bytes(bundle.tar_path.read_bytes())
    received = p2p_pb2.ArtifactManifestFile()
    received.CopyFrom(bundle.manifest.files[0])
    received.path = str(staged)
    transfer.install(
        p2p_pb2.GetArtifactManifestHeaderResponse(
            artifact_id=bundle.artifact_id,
            mx_source_type=transfer.mx_source_type,
            files=[received],
            total_size=received.size,
        )
    )


def test_startup_plan_uses_native_directory_and_framework_identity(native, tmp_path):
    (cache,) = framework_caches.build_framework_cache_artifacts(native.config)
    transfer, identity = cache.entry("test/model", tmp_path / "bundle")

    assert cache.name == "startup_plan"
    assert cache.root == _plan_dir(native)
    assert transfer.roots[0].source_root == transfer.roots[0].target_root == cache.root
    assert transfer.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_FRAMEWORK_CACHE
    assert identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_VLLM
    assert identity.backend_framework_version == "0.29.1.dev10+gexample"
    assert identity.extra_parameters["framework_cache_schema"] == "1"
    native.plan.compute_plan_fingerprint.assert_called_once_with(native.config, 0, 2)


def test_native_startup_plan_functions_are_not_replaced(native):
    originals = {
        name: getattr(native.plan, name)
        for name in ("_plan_path", "maybe_apply_startup_plan", "maybe_save_startup_plan")
    }
    framework_caches.build_framework_cache_artifacts(native.config)
    assert {name: getattr(native.plan, name) for name in originals} == originals


def test_round_trip_installs_plans_that_native_apply_uses(native, tmp_path):
    (source,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.stage["value"] = "postinit"
    for rank, size in ((0, 900), (1, 800)):
        native.plan.maybe_save_startup_plan(_worker(native.config, rank), size)
    sender, sender_identity = source.entry("model", tmp_path / "sender-bundle")
    bundle = sender.prepare_source()

    native.runtime.VLLM_CACHE_ROOT = str(tmp_path / "other-pod")
    native.stage["value"] = "preinit"
    (target,) = framework_caches.build_framework_cache_artifacts(native.config)
    receiver, receiver_identity = target.entry("model", tmp_path / "receiver-bundle")
    _install(receiver, bundle)
    native.stage["value"] = "postinit"
    workers = [_worker(copy.deepcopy(native.config), rank) for rank in (0, 1)]
    for worker in workers:
        native.plan.maybe_apply_startup_plan(worker)

    assert compute_mx_source_id(sender_identity) == compute_mx_source_id(receiver_identity)
    assert [worker.cache_config.kv_cache_memory_bytes for worker in workers] == [900, 800]


def test_publication_selects_this_boots_plans_for_its_own_ranks(native, tmp_path):
    plan_dir = _plan_dir(native)
    plan_dir.mkdir(parents=True)
    stale = plan_dir / "startup_plan_postinit-r1-of2.json"
    stale.write_text("{}")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    (cache,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.stage["value"] = "postinit"
    native.plan.maybe_save_startup_plan(_worker(native.config, 0), 900)
    (plan_dir / "startup_plan_other-config-r0-of2.json").write_text("{}")
    (plan_dir / "nested").mkdir()
    (plan_dir / "nested" / "startup_plan_postinit-r0-of2.json").write_text("{}")
    transfer, _ = cache.entry("model", tmp_path / "bundle")

    assert _archived_files(transfer.prepare_source()) == ["startup_plan_postinit-r0-of2.json"]


def test_selection_uses_fingerprints_after_model_construction(native, tmp_path):
    (cache,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.plan.maybe_save_startup_plan(_worker(native.config, 0), 900)
    native.stage["value"] = "postinit"
    native.plan.maybe_save_startup_plan(_worker(native.config, 0), 700)
    transfer, _ = cache.entry("model", tmp_path / "bundle")

    assert _archived_files(transfer.prepare_source()) == ["startup_plan_postinit-r0-of2.json"]


def test_boot_without_saved_plans_publishes_nothing(native, tmp_path):
    (cache,) = framework_caches.build_framework_cache_artifacts(native.config)
    _plan_dir(native).mkdir(parents=True)
    (_plan_dir(native) / "startup_plan_unrelated.json").write_text("{}")
    transfer, _ = cache.entry("model", tmp_path / "bundle")

    with pytest.raises(LookupError):
        transfer.prepare_source()


def test_descriptor_identity_survives_native_config_mutation(native, tmp_path):
    (cache,) = framework_caches.build_framework_cache_artifacts(native.config)
    ctx = SimpleNamespace(
        adapter=SimpleNamespace(framework_cache_artifacts=(cache,)),
        identity=p2p_pb2.SourceIdentity(model_name="model"),
    )
    (before,) = framework_cache_entries(ctx, tmp_path / "bundle")
    native.stage["value"] = "postinit"
    (after,) = framework_cache_entries(ctx, tmp_path / "bundle")

    assert before[1] == after[1]
    native.plan.compute_plan_fingerprint.assert_called_once()


@pytest.mark.parametrize("dimension", [
    "hidden_size", "num_hidden_layers", "dtype", "enforce_eager",
    "tensor_parallel_size", "pipeline_parallel_size",
    "decode_context_parallel_size", "prefill_context_parallel_size",
    "disable_custom_all_reduce", "num_speculative_tokens",
    "num_speculative_tokens_per_batch_size", "kv_cache_dtype", "method",
])
def test_speculative_profiling_changes_partition_identity(
    native, speculative, tmp_path, dimension
):
    (source,) = framework_caches.build_framework_cache_artifacts(native.config)
    if dimension in {"hidden_size", "num_hidden_layers"}:
        setattr(speculative.draft_model_config.hf_config, dimension, 128)
    elif dimension in {"dtype", "enforce_eager"}:
        setattr(speculative.draft_model_config, dimension, torch.float32 if dimension == "dtype" else True)
    elif dimension in {
        "tensor_parallel_size", "pipeline_parallel_size",
        "decode_context_parallel_size", "prefill_context_parallel_size",
        "disable_custom_all_reduce",
    }:
        setattr(speculative.draft_parallel_config, dimension, True if dimension == "disable_custom_all_reduce" else 2)
    else:
        setattr(speculative, dimension, {
            "num_speculative_tokens": 4,
            "num_speculative_tokens_per_batch_size": [(1, 8, 4)],
            "kv_cache_dtype": "fp8",
            "method": "eagle",
        }[dimension])
    (target,) = framework_caches.build_framework_cache_artifacts(native.config)

    assert source.compatibility_key != target.compatibility_key
    assert compute_mx_source_id(source.identity("model")) != compute_mx_source_id(
        target.identity("model")
    )


def test_equivalent_speculative_config_ignores_credentials_and_launch_port(native, speculative):
    (source,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.config.speculative_config = copy.deepcopy(speculative)
    native.config.speculative_config.draft_model_config.hf_token = "other-token"
    native.config.speculative_config.draft_model_config.served_model_name = "other-service"
    native.config.speculative_config.draft_parallel_config.master_port = 12345
    (target,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert source.compatibility_key == target.compatibility_key


def test_nonportable_speculative_config_skips_only_startup_plan(native, speculative, caplog):
    speculative.draft_load_config = object()
    with caplog.at_level(logging.WARNING):
        assert framework_caches.build_framework_cache_artifacts(native.config) == ()
    assert "Skipping vLLM startup plan artifact" in caplog.text


def test_missing_draft_model_hash_skips_only_startup_plan(native, speculative):
    speculative.draft_model_config.compute_hash = lambda: ""
    assert framework_caches.build_framework_cache_artifacts(native.config) == ()


def test_resolved_draft_hf_override_does_not_disable_startup_plan(native, speculative):
    (original,) = framework_caches.build_framework_cache_artifacts(native.config)
    speculative.draft_model_config.hf_overrides = lambda config: config
    caches = framework_caches.build_framework_cache_artifacts(native.config)

    assert len(caches) == 1
    assert original.compatibility_key == caches[0].compatibility_key


@pytest.mark.parametrize("separate_config", [False, True])
def test_draft_adapter_does_not_register_startup_plans(native, adapter_factory, separate_config):
    target = adapter_factory()
    native.stage["value"] = "postinit"
    draft_config = copy.deepcopy(native.config) if separate_config else native.config
    draft = adapter_factory(draft_pass=True, config=draft_config)

    assert len(target.framework_cache_artifacts) == 1
    assert draft.framework_cache_artifacts == ()
    native.plan.compute_plan_fingerprint.assert_called_once()


def test_main_adapter_with_aliased_ngram_draft_keeps_startup_plan(native, adapter_factory):
    native.config.speculative_config.draft_model_config = native.config.model_config
    target = adapter_factory()

    assert len(target.framework_cache_artifacts) == 1
    native.plan.compute_plan_fingerprint.assert_called_once()


@pytest.mark.parametrize("enabled", [False, None])
def test_disabled_or_older_runtime_skips_startup_plan(native, enabled):
    if enabled is None:
        del native.runtime.VLLM_ENABLE_STARTUP_PLAN
    else:
        native.runtime.VLLM_ENABLE_STARTUP_PLAN = enabled
    assert framework_caches.build_framework_cache_artifacts(native.config) == ()
    native.plan.compute_plan_fingerprint.assert_not_called()


@pytest.mark.parametrize("attribute", [
    "compute_plan_fingerprint", "_plan_path", "maybe_apply_startup_plan",
    "maybe_save_startup_plan", "PLAN_SCHEMA_VERSION",
])
def test_partial_native_api_skips_only_startup_plan(native, attribute, caplog):
    delattr(native.plan, attribute)
    with caplog.at_level(logging.WARNING):
        assert framework_caches.build_framework_cache_artifacts(native.config) == ()
    assert "startup plan" in caplog.text.lower()


def test_missing_module_skips_only_startup_plan(native, monkeypatch):
    def load(name):
        if name.endswith("startup_plan"):
            raise ModuleNotFoundError(name)
        return native.modules[name]
    monkeypatch.setattr(framework_caches, "import_module", load)
    assert framework_caches.build_framework_cache_artifacts(native.config) == ()


def test_unknown_native_schema_is_not_assumed_compatible(native):
    native.plan.PLAN_SCHEMA_VERSION = 2
    assert framework_caches.build_framework_cache_artifacts(native.config) == ()


def test_explicit_kv_memory_does_not_transfer_unused_plans(native):
    native.config.cache_config.kv_cache_memory_bytes = 1 << 30
    assert framework_caches.build_framework_cache_artifacts(native.config) == ()


def test_native_key_failure_does_not_break_other_artifacts(native, caplog):
    native.plan.compute_plan_fingerprint.side_effect = RuntimeError("unsupported config")
    with caplog.at_level(logging.WARNING):
        assert framework_caches.build_framework_cache_artifacts(native.config) == ()
    assert "unsupported config" in caplog.text


def test_failing_provider_does_not_hide_other_providers(native, monkeypatch, tmp_path):
    def broken(vllm_config):
        raise RuntimeError("broken provider")

    monkeypatch.setattr(
        framework_caches, "_PROVIDERS", (broken, framework_caches.startup_plan_cache)
    )
    caches = framework_caches.build_framework_cache_artifacts(native.config)
    assert [cache.name for cache in caches] == ["startup_plan"]


def test_memory_utilization_partitions_plan_pool(native):
    (high,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.config.cache_config.gpu_memory_utilization = 0.5
    (low,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert high.compatibility_key != low.compatibility_key


def test_node_rank_partitions_plan_pool(native):
    (first,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.config.parallel_config.node_rank = 1
    (second,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert first.compatibility_key != second.compatibility_key


def test_device_partitions_plan_pool(native, monkeypatch):
    (first,) = framework_caches.build_framework_cache_artifacts(native.config)
    monkeypatch.setattr(framework_caches, "_device_signature", lambda: {"name": "other GPU"})
    (second,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert first.compatibility_key != second.compatibility_key



def test_relocated_cache_root_preserves_identity(native, tmp_path):
    (source,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.runtime.VLLM_CACHE_ROOT = str(tmp_path / "other-pod")
    (target,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert source.root != target.root
    assert source.entry("model", tmp_path / "bundle-a")[1] == target.entry("model", tmp_path / "bundle-b")[1]




def test_framework_version_partitions_plan_pool(native, tmp_path):
    (source,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.modules["vllm"].__version__ = "0.30.0"
    (other,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert compute_mx_source_id(source.entry("model", tmp_path / "a")[1]) != compute_mx_source_id(other.entry("model", tmp_path / "b")[1])


def test_multimodal_profiling_settings_partition_plan_pool(native):
    @dataclass
    class MMConfig:
        skip_mm_profiling: bool = False
        limit_per_prompt: dict | None = None

    native.config.model_config.multimodal_config = MMConfig()
    (first,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.config.model_config.multimodal_config.skip_mm_profiling = True
    (second,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert first.compatibility_key != second.compatibility_key


def test_compile_environment_partitions_plan_pool(native):
    (first,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.runtime.compile_factors = lambda: {"VLLM_USE_V2_MODEL_RUNNER": True}
    (second,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert first.compatibility_key != second.compatibility_key


@pytest.mark.parametrize("flag", ["enforce_eager", "use_fp64_gumbel"])
def test_model_profiling_flags_partition_plan_pool(native, flag):
    setattr(native.config.model_config, flag, False)
    (first,) = framework_caches.build_framework_cache_artifacts(native.config)
    setattr(native.config.model_config, flag, True)
    (second,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert first.compatibility_key != second.compatibility_key


def test_model_logprobs_mode_partitions_plan_pool(native):
    (first,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.config.model_config.logprobs_mode = "processed_logits"
    (second,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert first.compatibility_key != second.compatibility_key


def test_scheduler_profiling_settings_partition_plan_pool(native):
    @dataclass
    class SchedulerConfig:
        async_scheduling: bool = False

    native.config.scheduler_config = SchedulerConfig()
    (first,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.config.scheduler_config.async_scheduling = True
    (second,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert first.compatibility_key != second.compatibility_key


def test_credentials_and_served_name_do_not_partition_plan_pool(native):
    (first,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.config.model_config.hf_token = "different-credential"
    native.config.model_config.served_model_name = "different-deployment"
    (second,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert first.compatibility_key == second.compatibility_key


def test_worker_backend_changes_partition_plan_pool(native):
    (first,) = framework_caches.build_framework_cache_artifacts(native.config)
    native.config.parallel_config.disable_custom_all_reduce = True
    (second,) = framework_caches.build_framework_cache_artifacts(native.config)
    assert first.compatibility_key != second.compatibility_key




def test_nonportable_profiling_settings_skip_only_startup_plan(native):
    @dataclass
    class MMConfig:
        processor: object = object()

    native.config.model_config.multimodal_config = MMConfig()
    assert framework_caches.build_framework_cache_artifacts(native.config) == ()


def test_module_import_does_not_require_vllm():
    assert importlib.import_module(framework_caches.__name__) is framework_caches
