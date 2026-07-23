# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tensor utilities, metadata publishing, and loading strategies."""

import logging
import logging.handlers
import os
from unittest.mock import MagicMock, patch, call

import grpc
import pytest
import torch
import torch.nn as nn

from modelexpress import p2p_pb2
from modelexpress.adapter import EngineAdapter, StrategyFailed
from modelexpress.load_strategy.context import LoadResult
from modelexpress.nixl_transfer import NixlTransferManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loader():
    """Return an MxModelLoader with a fresh mock MxClient."""
    with patch("modelexpress.engines.vllm.loader.DefaultModelLoader"):
        load_config = MagicMock()
        load_config.load_format = "mx"
        load_config.device = None
        from modelexpress.engines.vllm.loader import MxModelLoader
        loader = MxModelLoader(load_config)
    loader._mx_client = MagicMock()
    return loader


def _make_identity(model_name="test-model"):
    return p2p_pb2.SourceIdentity(model_name=model_name)


def _make_worker(rank=0, n_tensors=3):
    tensors = [
        p2p_pb2.TensorDescriptor(name=f"t{i}", addr=0x1000 + i, size=1024, device_id=rank, dtype="bfloat16")
        for i in range(n_tensors)
    ]
    return p2p_pb2.WorkerMetadata(worker_rank=rank, tensors=tensors)


def _make_instance_ref(mx_source_id="abc123def456abcd", worker_id="inst-1", model_name="test-model", worker_rank=0):
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id,
        worker_id=worker_id,
        model_name=model_name,
        worker_rank=worker_rank,
    )


class _IdentitySelector:
    """Deterministic passthrough selector: preserves input candidate order.

    Replaces the old ``random.shuffle`` no-op patch so candidate ordering is
    stable for assertions that depend on attempt order.
    """

    name = "random"

    def order(self, candidates, context):
        return list(candidates)


class _FakeAdapter(EngineAdapter):
    def build_identity(self):
        return _make_identity()

    def get_worker_rank(self) -> int:
        return 0

    def get_device_id(self) -> int:
        return 0

    def discover_tensors(self, result: LoadResult):
        return {}

    def after_weight_iter_load(self, result: LoadResult):
        return result

    def after_native_load(self, result: LoadResult):
        return result

    def apply_weight_iter(self, result: LoadResult, weights_iter):
        if result.model is not None:
            result.model.load_weights(weights_iter)
        return result

    def load_via_native(self, result: LoadResult):
        if result.model is not None:
            result.model.load_weights([])
        return result

    def reinit_for_retry(self, result: LoadResult):
        return result


def _make_metadata_resp(found=True, rank=0, mx_source_id="abc123def456abcd", worker_id="inst-1"):
    worker = _make_worker(rank=rank) if found else None
    return p2p_pb2.GetMetadataResponse(
        found=found,
        worker=worker,
        mx_source_id=mx_source_id,
        worker_id=worker_id,
    )


def _make_load_context(**overrides):
    """Return a LoadContext with mocked dependencies."""
    from modelexpress.load_strategy import LoadContext
    defaults = dict(
        model_config=MagicMock(),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        device_id=0,
        identity=_make_identity(),
        mx_client=MagicMock(),
        worker_id="test-worker",
        adapter=_FakeAdapter(),
    )
    defaults.update(overrides)
    return LoadContext(**defaults)


class _FakeRpcError(grpc.RpcError):
    def __init__(self, status_code: grpc.StatusCode, details: str):
        super().__init__()
        self._status_code = status_code
        self._details = details

    def code(self):
        return self._status_code

    def details(self):
        return self._details

    def __str__(self):
        return self._details


# ---------------------------------------------------------------------------
# collect_module_tensors (tensor_utils)
# ---------------------------------------------------------------------------


class TestCollectModuleTensors:
    """Tests for the collect_module_tensors helper."""

    def test_empty_model(self):
        from modelexpress.tensor_utils import collect_module_tensors

        model = nn.Module()
        result = collect_module_tensors(model)
        assert result == {}

    def test_cpu_only_model(self):
        from modelexpress.tensor_utils import collect_module_tensors

        model = nn.Linear(4, 2, bias=False)
        result = collect_module_tensors(model)
        assert result == {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_model(self):
        from modelexpress.tensor_utils import collect_module_tensors

        model = nn.Linear(4, 2, bias=True).cuda()
        result = collect_module_tensors(model)
        assert len(result) == 2  # weight + bias
        assert "weight" in result
        assert "bias" in result
        for t in result.values():
            assert t.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_skips_non_contiguous(self):
        from modelexpress.tensor_utils import collect_module_tensors

        model = nn.Module()
        model.weight = nn.Parameter(torch.randn(4, 3, device="cuda"))
        model.weight_t = model.weight.data.T
        assert not model.weight_t.is_contiguous()

        result = collect_module_tensors(model)
        assert "weight" in result
        assert "weight_t" not in result

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_deduplicate_tied_weights(self):
        """Tied weights (same data_ptr) should only be registered once."""
        from modelexpress.tensor_utils import collect_module_tensors

        model = nn.Module()
        shared = nn.Parameter(torch.randn(4, 3, device="cuda"))
        embed = nn.Module()
        embed.weight = shared
        head = nn.Module()
        head.weight = shared
        model.embed_tokens = embed
        model.lm_head = head

        result = collect_module_tensors(model)
        ptrs = [t.data_ptr() for t in result.values()]
        assert len(ptrs) == len(set(ptrs)), "duplicate data_ptr found in result"
        assert len(result) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_deduplicate_bare_attr_alias(self):
        """A bare tensor attr sharing data_ptr with a parameter is skipped."""
        from modelexpress.tensor_utils import collect_module_tensors

        model = nn.Module()
        model.weight = nn.Parameter(torch.randn(4, 3, device="cuda"))
        model.__dict__["w_alias"] = model.weight.data

        result = collect_module_tensors(model)
        assert "weight" in result
        assert "w_alias" not in result
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Abstract method completeness
# ---------------------------------------------------------------------------


class TestAbstractMethodCompleteness:
    def test_instantiation_succeeds(self):
        assert _make_loader() is not None

    def test_no_remaining_abstract_methods(self):
        from modelexpress.engines.vllm.loader import MxModelLoader
        remaining = getattr(MxModelLoader, "__abstractmethods__", frozenset())
        assert remaining == frozenset()

    def test_download_model_delegates(self):
        loader = _make_loader()
        cfg = MagicMock()
        with patch("modelexpress.engines.vllm.loader.DefaultModelLoader") as mock_cls:
            loader.download_model(cfg)
            mock_cls.return_value.download_model.assert_called_once_with(cfg)

    def test_load_weights_delegates(self):
        loader = _make_loader()
        model, cfg = MagicMock(), MagicMock()
        with patch("modelexpress.engines.vllm.loader.DefaultModelLoader") as mock_cls:
            loader.load_weights(model, cfg)
            mock_cls.return_value.load_weights.assert_called_once_with(model, cfg)

    def test_load_model_clears_stale_nixl_manager_when_none_registered(self):
        from modelexpress.engines.vllm import loader as loader_mod

        loader = _make_loader()
        stale_manager = MagicMock()
        model = MagicMock()
        ctx = _make_load_context(device_id=3)
        ctx.tensors = {"w": MagicMock()}
        ctx.nixl_manager = None
        loader_mod._nixl_managers[3] = stale_manager

        try:
            with patch(
                "modelexpress.engines.vllm.loader.build_vllm_load_context",
                return_value=ctx,
            ), patch(
                "modelexpress.engines.vllm.loader.initialize_model",
                return_value=model,
            ), patch(
                "modelexpress.engines.vllm.loader.LoadStrategyChain.run",
                return_value=model,
            ):
                loaded = loader.load_model(MagicMock(), MagicMock(dtype=torch.float32))

            assert loaded is model.eval.return_value
            assert loader_mod._tensor_registry[3] == ctx.tensors
            assert 3 not in loader_mod._nixl_managers
        finally:
            loader_mod._nixl_managers.pop(3, None)
            loader_mod._tensor_registry.pop(3, None)

    @pytest.mark.parametrize(
        ("ready_url", "health_gated"),
        [("", False), ("http://127.0.0.1:8000/health", True)],
    )
    def test_load_model_installs_and_schedules_vllm_artifacts(
        self, ready_url, health_gated
    ):
        from modelexpress.engines.vllm import loader as loader_mod

        loader = _make_loader()
        model = MagicMock()
        ctx = _make_load_context(device_id=3)
        events = []

        def install(ctx_arg):
            assert ctx_arg is ctx
            events.append("install")

        def initialize_model(**_kwargs):
            events.append("initialize")
            return model

        def run(model_arg, ctx_arg):
            assert model_arg is model
            assert ctx_arg is ctx
            if health_gated:
                assert ctx_arg.source_ready_fn is loader_mod._vllm_health_ready
            else:
                assert ctx_arg.source_ready_fn is None
            events.append("load")
            return model_arg

        def schedule(ctx_arg):
            assert ctx_arg is ctx
            events.append("schedule")

        with patch.dict(os.environ, {"MX_ARTIFACT_READY_URL": ready_url}), patch(
            "modelexpress.engines.vllm.loader.build_vllm_load_context",
            return_value=ctx,
        ), patch(
            "modelexpress.engines.vllm.loader.install_vllm_cache_artifacts",
            side_effect=install,
        ), patch(
            "modelexpress.engines.vllm.loader.initialize_model",
            side_effect=initialize_model,
        ), patch(
            "modelexpress.engines.vllm.loader.LoadStrategyChain.run",
            side_effect=run,
        ), patch(
            "modelexpress.engines.vllm.loader.schedule_vllm_cache_artifact_publish",
            side_effect=schedule,
        ):
            loaded = loader.load_model(MagicMock(), MagicMock(dtype=torch.float32))

        assert loaded is model.eval.return_value
        assert events == ["install", "initialize", "load", "schedule"]

    def test_loader_import_does_not_register_mx(self, monkeypatch):
        import importlib
        import vllm.model_executor.model_loader as model_loader

        registry = dict(model_loader._LOAD_FORMAT_TO_MODEL_LOADER)
        registry.pop("mx", None)
        monkeypatch.setattr(model_loader, "_LOAD_FORMAT_TO_MODEL_LOADER", registry)

        importlib.import_module("modelexpress.engines.vllm.loader")

        assert "mx" not in model_loader._LOAD_FORMAT_TO_MODEL_LOADER

    def test_plugin_registration_preserves_native_modelexpress(self, monkeypatch):
        import importlib
        import vllm.model_executor.model_loader as model_loader

        sentinel = object()
        registry = dict(model_loader._LOAD_FORMAT_TO_MODEL_LOADER)
        registry["modelexpress"] = sentinel
        registry.pop("mx", None)
        monkeypatch.setattr(model_loader, "_LOAD_FORMAT_TO_MODEL_LOADER", registry)

        registration = importlib.import_module(
            "modelexpress.engines.vllm.registration"
        )
        registered = {}

        def fake_register_model_loader(load_format):
            def register(loader_cls):
                registered[load_format] = loader_cls
                return loader_cls

            return register

        monkeypatch.setattr(
            registration,
            "register_model_loader",
            fake_register_model_loader,
        )

        registration.register_plugin_model_loader()

        from modelexpress.engines.vllm.loader import MxModelLoader

        assert model_loader._LOAD_FORMAT_TO_MODEL_LOADER["modelexpress"] is sentinel
        assert registered["mx"] is MxModelLoader
        assert "modelexpress" not in registered

    def test_plugin_registration_registers_load_formats_without_native_vllm(
        self, monkeypatch
    ):
        import importlib
        import vllm.model_executor.model_loader as model_loader

        registry = dict(model_loader._LOAD_FORMAT_TO_MODEL_LOADER)
        registry.pop("mx", None)
        registry.pop("modelexpress", None)
        monkeypatch.setattr(model_loader, "_LOAD_FORMAT_TO_MODEL_LOADER", registry)

        registration = importlib.import_module(
            "modelexpress.engines.vllm.registration"
        )
        registered = {}

        def fake_register_model_loader(load_format):
            def register(loader_cls):
                registered[load_format] = loader_cls
                return loader_cls

            return register

        monkeypatch.setattr(
            registration,
            "register_model_loader",
            fake_register_model_loader,
        )

        registration.register_plugin_model_loader()

        from modelexpress.engines.vllm.loader import MxModelLoader

        assert registered["modelexpress"] is MxModelLoader
        assert registered["mx"] is MxModelLoader


class TestMtpDrafterSecondLoad:
    """vLLM loads MTP in two passes on one worker: the target, then the drafter
    via a second load_model whose draft ModelConfig has runner_type="draft" and
    reuses the target's device and model name. The test drives both on device 0."""

    def _load(self, loader, vllm_config, model_config, ctx):
        with patch(
            "modelexpress.engines.vllm.loader.build_vllm_load_context",
            return_value=ctx,
        ), patch(
            "modelexpress.engines.vllm.loader.install_vllm_cache_artifacts",
        ), patch(
            "modelexpress.engines.vllm.loader.initialize_model",
            return_value=MagicMock(),
        ), patch(
            "modelexpress.engines.vllm.loader.LoadStrategyChain.run",
            side_effect=lambda model, _ctx: model,
        ), patch(
            "modelexpress.engines.vllm.loader.schedule_vllm_cache_artifact_publish",
        ) as schedule:
            loader.load_model(vllm_config, model_config)
        return schedule

    def test_drafter_does_not_clobber_target(self):
        """The drafter's second load leaves the target's device registry and
        P2P publish untouched."""
        from modelexpress.engines.vllm import loader as loader_mod

        loader = _make_loader()
        target_ctx = _make_load_context(device_id=0)
        target_ctx.tensors = {"target.weight": MagicMock()}
        target_ctx.nixl_manager = MagicMock()
        target_config = MagicMock(dtype=torch.float32, runner_type="generate")
        self._load(loader, MagicMock(), target_config, target_ctx)

        draft_config = MagicMock(dtype=torch.float32, runner_type="draft")
        draft_vllm_config = MagicMock()
        draft_ctx = _make_load_context(device_id=0)
        draft_ctx.tensors = {"drafter.mtp": MagicMock()}
        draft_ctx.nixl_manager = MagicMock()
        try:
            schedule = self._load(loader, draft_vllm_config, draft_config, draft_ctx)
            assert loader_mod._tensor_registry[0] is target_ctx.tensors
            assert loader_mod._nixl_managers[0] is target_ctx.nixl_manager
            schedule.assert_not_called()
        finally:
            loader_mod._tensor_registry.pop(0, None)
            loader_mod._nixl_managers.pop(0, None)

    def test_is_speculative_draft(self):
        from modelexpress.engines.vllm.loader import _is_speculative_draft

        # No speculative decoding: never a draft.
        no_spec = MagicMock(speculative_config=None)
        assert _is_speculative_draft(no_spec, MagicMock(runner_type="generate")) is False

        # Draft model load under speculative decoding.
        spec = MagicMock()
        assert _is_speculative_draft(spec, MagicMock(runner_type="draft")) is True

        # Target load with spec on (e.g. ngram aliases draft to the target):
        # runner_type stays "generate", so the target keeps P2P.
        assert _is_speculative_draft(spec, MagicMock(runner_type="generate")) is False

    @patch("modelexpress.load_strategy.base.is_nixl_available", return_value=True)
    @patch("modelexpress.load_strategy.base._init_nixl_manager")
    def test_register_tensors_skips_when_p2p_disabled(self, mock_init, _avail):
        from modelexpress.load_strategy.base import register_tensors

        ctx = _make_load_context()
        ctx.p2p_enabled = False
        ctx.adapter.discover_tensors = MagicMock(return_value={"w": MagicMock()})
        with patch.dict(os.environ, {"MX_SERVER_ADDRESS": "localhost:8001"}):
            register_tensors(MagicMock(), ctx)

        mock_init.assert_not_called()
        ctx.adapter.discover_tensors.assert_not_called()
        assert ctx.nixl_manager is None

    @patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True)
    def test_rdma_unavailable_when_p2p_disabled(self, _mock):
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy

        ctx = _make_load_context()
        ctx.p2p_enabled = False
        with patch.dict("os.environ", {"MX_SERVER_ADDRESS": "server:8001"}):
            assert RdmaStrategy().is_available(ctx) is False


# ---------------------------------------------------------------------------
# register_tensors (load_strategy.base)
# ---------------------------------------------------------------------------


class TestInitNixlManager:
    """Verify _init_nixl_manager passes listen_port to NixlTransferManager."""

    def test_default_listen_port_is_zero(self):
        from modelexpress.load_strategy.base import _init_nixl_manager

        with patch.object(NixlTransferManager, "initialize"):
            mgr = _init_nixl_manager(global_rank=0, device_id=0, role="auto")

        assert mgr._listen_port == 0

    def test_explicit_listen_port(self):
        from modelexpress.load_strategy.base import _init_nixl_manager

        with patch.object(NixlTransferManager, "initialize"):
            mgr = _init_nixl_manager(global_rank=0, device_id=2, role="auto", listen_port=6002)

        assert mgr._listen_port == 6002


class TestRegisterTensorsErrorHandling:
    """Verify register_tensors never raises — failures are logged as warnings."""

    @patch("modelexpress.load_strategy.base.is_nixl_available")
    def test_skips_when_no_metadata_path_configured(self, mock_available):
        from modelexpress.load_strategy.base import register_tensors
        ctx = _make_load_context()
        model = MagicMock()
        register_tensors(model, ctx)
        mock_available.assert_not_called()
        assert ctx.nixl_manager is None

    @patch("modelexpress.load_strategy.base.is_nixl_available", return_value=False)
    def test_skips_when_nixl_unavailable(self, _mock):
        from modelexpress.load_strategy.base import register_tensors
        ctx = _make_load_context()
        model = MagicMock()
        with patch.dict(os.environ, {"MX_SERVER_ADDRESS": "localhost:8001"}):
            register_tensors(model, ctx)
        assert ctx.nixl_manager is None

    @patch("modelexpress.load_strategy.base.is_nixl_available", return_value=True)
    def test_requires_adapter_when_nixl_available(self, _mock):
        from modelexpress.load_strategy.base import register_tensors
        ctx = _make_load_context(adapter=None)
        with patch.dict(os.environ, {"MX_SERVER_ADDRESS": "localhost:8001"}):
            with pytest.raises(RuntimeError, match="engine adapter"):
                register_tensors(MagicMock(), ctx)

    @patch("modelexpress.load_strategy.base.is_nixl_available", return_value=True)
    @patch(
        "modelexpress.load_strategy.base._init_nixl_manager",
        side_effect=RuntimeError("NIXL_ERR_BACKEND"),
    )
    def test_nixl_init_failure_does_not_raise(self, _init, _avail):
        from modelexpress.load_strategy.base import register_tensors
        ctx = _make_load_context()
        model = MagicMock()
        with patch.dict(os.environ, {"MX_SERVER_ADDRESS": "localhost:8001"}):
            register_tensors(model, ctx)
        assert ctx.nixl_manager is None

    @patch("modelexpress.load_strategy.base.is_nixl_available", return_value=True)
    @patch("modelexpress.load_strategy.base._init_nixl_manager")
    def test_tensor_registration_failure_does_not_raise(self, mock_init, _avail):
        from modelexpress.load_strategy.base import register_tensors
        mock_mgr = MagicMock()
        mock_mgr.tensor_descriptors = []
        mock_mgr.register_tensors.side_effect = RuntimeError("memory registration failed")
        mock_init.return_value = mock_mgr
        ctx = _make_load_context()
        ctx.adapter.discover_tensors = MagicMock(return_value={"t": MagicMock()})
        model = MagicMock()
        with patch.dict(os.environ, {"MX_SERVER_ADDRESS": "localhost:8001"}):
            register_tensors(model, ctx)


class TestRegisterTensorsReuseDiscovered:
    """Verify the reuse_discovered flag skips adapter discovery.

    Used on wake / CRIU-restore paths where the weight set is unchanged
    since initial load. Skipping discovery avoids tripping on
    post-warmup artifacts (e.g. torch._dynamo AOT compiled functions).
    """

    @patch("modelexpress.load_strategy.base.is_nixl_available", return_value=True)
    @patch("modelexpress.load_strategy.base._init_nixl_manager")
    def test_reuse_skips_discovery_when_ctx_has_tensors(self, mock_init, _avail):
        from modelexpress.load_strategy.base import register_tensors
        mock_mgr = MagicMock()
        mock_mgr.tensor_descriptors = []
        mock_init.return_value = mock_mgr

        ctx = _make_load_context()
        ctx.adapter.discover_tensors = MagicMock()
        cached = {"weight_0": MagicMock(), "weight_1": MagicMock()}
        ctx.tensors = cached

        with patch.dict(os.environ, {"MX_SERVER_ADDRESS": "localhost:8001"}):
            register_tensors(MagicMock(), ctx, reuse_discovered=True)

        ctx.adapter.discover_tensors.assert_not_called()
        assert ctx.tensors is cached
        mock_mgr.register_tensors.assert_called_once_with(cached)

    @patch("modelexpress.load_strategy.base.is_nixl_available", return_value=True)
    @patch("modelexpress.load_strategy.base._init_nixl_manager")
    def test_reuse_falls_back_to_discovery_when_ctx_tensors_empty(
        self, mock_init, _avail,
    ):
        """If reuse requested but no cache, fall through to discovery."""
        from modelexpress.load_strategy.base import register_tensors
        mock_mgr = MagicMock()
        mock_mgr.tensor_descriptors = []
        mock_init.return_value = mock_mgr

        ctx = _make_load_context()
        fresh = {"fresh_w": MagicMock()}
        ctx.adapter.discover_tensors = MagicMock(return_value=fresh)
        # ctx.tensors defaults to {} in LoadContext.

        with patch.dict(os.environ, {"MX_SERVER_ADDRESS": "localhost:8001"}):
            register_tensors(MagicMock(), ctx, reuse_discovered=True)

        ctx.adapter.discover_tensors.assert_called_once()
        assert ctx.tensors == fresh

    @patch("modelexpress.load_strategy.base.is_nixl_available", return_value=True)
    @patch("modelexpress.load_strategy.base._init_nixl_manager")
    def test_default_behavior_unchanged(self, mock_init, _avail):
        """Default (reuse_discovered=False) must still run discovery."""
        from modelexpress.load_strategy.base import register_tensors
        mock_mgr = MagicMock()
        mock_mgr.tensor_descriptors = []
        mock_init.return_value = mock_mgr

        ctx = _make_load_context()
        fresh = {"w": MagicMock()}
        ctx.adapter.discover_tensors = MagicMock(return_value=fresh)
        # Pre-populate ctx.tensors so we can prove discovery still runs.
        ctx.tensors = {"stale": MagicMock()}

        with patch.dict(os.environ, {"MX_SERVER_ADDRESS": "localhost:8001"}):
            register_tensors(MagicMock(), ctx)

        ctx.adapter.discover_tensors.assert_called_once()
        assert ctx.tensors == fresh


class TestNixlTransferManagerDictOwnership:
    """Verify shutdown does not mutate the caller's tensor dict.

    The reuse_discovered=True path on wake relies on the caller's
    ``ctx.tensors`` surviving across ``shutdown()`` so the next
    ``register_tensors`` can re-register them. Earlier code aliased
    the caller's dict (``self._tensors = tensors``) and then called
    ``self._tensors.clear()`` in shutdown, silently wiping the
    caller's dict.
    """

    def _make_manager(self):
        mgr = NixlTransferManager(agent_name="test", device_id=0)
        mgr._agent = MagicMock()
        return mgr

    def test_register_takes_shallow_copy_of_caller_dict(self):
        mgr = self._make_manager()
        t = MagicMock()
        t.is_contiguous.return_value = True
        t.data_ptr.return_value = 0x1000
        t.numel.return_value = 1
        t.element_size.return_value = 1
        t.dtype = torch.float32
        caller = {"w": t}

        mgr._agent.get_local_md.return_value = b"meta"
        mgr.register_tensors(caller)

        assert mgr._tensors is not caller
        assert mgr._tensors == caller  # same tensor values

    def test_shutdown_does_not_clear_caller_dict(self):
        mgr = self._make_manager()
        t = MagicMock()
        t.is_contiguous.return_value = True
        t.data_ptr.return_value = 0x1000
        t.numel.return_value = 1
        t.element_size.return_value = 1
        t.dtype = torch.float32
        caller = {"w": t}

        mgr._agent.get_local_md.return_value = b"meta"
        mgr.register_tensors(caller)
        mgr.shutdown()

        assert caller == {"w": t}, "caller dict was mutated by shutdown"

    def test_shutdown_rebinds_internal_containers(self):
        """Defense in depth: even if a future caller aliases _tensors
        directly, shutdown should rebind rather than mutate."""
        mgr = self._make_manager()
        aliased = {"w": MagicMock()}
        mgr._tensors = aliased  # simulate hostile / future caller alias
        mgr._tensor_descriptors = [MagicMock()]

        mgr.shutdown()

        assert aliased == {"w": aliased["w"]}, (
            "shutdown mutated the externally-held dict via shared reference"
        )
        assert mgr._tensors == {}
        assert mgr._tensor_descriptors == []


class TestPublishMetadataErrorHandling:
    """Verify publish_metadata never raises."""

    def test_skips_when_no_nixl_manager(self):
        from modelexpress.load_strategy.base import publish_metadata
        ctx = _make_load_context()
        ctx.nixl_manager = None
        publish_metadata(ctx)

    @patch("modelexpress.load_strategy.base.publish_metadata_and_ready", side_effect=RuntimeError("gRPC fail"))
    def test_publish_failure_does_not_raise(self, _mock):
        from modelexpress.load_strategy.base import publish_metadata
        ctx = _make_load_context()
        ctx.nixl_manager = MagicMock()
        with patch.dict(os.environ, {"MX_SERVER_ADDRESS": "localhost:8001"}):
            publish_metadata(ctx)

    @patch("modelexpress.load_strategy.base.publish_metadata_and_ready")
    def test_publish_uses_accelerator_backend_name(
        self,
        mock_publish,
        mock_accelerator_backend_cls,
    ):
        from modelexpress.load_strategy.base import publish_metadata

        ctx = _make_load_context(
            accelerator_backend=mock_accelerator_backend_cls(name="xpu"),
        )
        ctx.nixl_manager = MagicMock()
        with patch.dict(os.environ, {"MX_SERVER_ADDRESS": "localhost:8001"}):
            publish_metadata(ctx)

        assert mock_publish.call_args.kwargs["accelerator"] == "xpu"

    def test_unpublish_uses_worker_rank_for_heartbeat_lifecycle(self):
        from modelexpress.load_strategy.base import unpublish_metadata
        from modelexpress.metadata.publish import _heartbeat_threads, _worker_servers

        ctx = _make_load_context(global_rank=4, worker_rank=0)
        heartbeat = MagicMock()
        _heartbeat_threads[ctx.worker_rank] = heartbeat
        try:
            unpublish_metadata(ctx)
        finally:
            _heartbeat_threads.pop(ctx.worker_rank, None)
            _worker_servers.pop(ctx.device_id, None)

        heartbeat.stop.assert_called_once()


class TestLoadStrategyChainRunErrorHandling:
    """Verify LoadStrategyChain.run catches exceptions from strategy.load()."""

    def test_unexpected_strategy_exception_rolls_back_before_next(self):
        from modelexpress.load_strategy import LoadStrategyChain

        call_order = []

        def exploding_load(self_or_model, *args, **kwargs):
            call_order.append("exploding")
            raise RuntimeError("unexpected crash")

        def rollback(self_or_ctx, *args, **kwargs):
            call_order.append("rollback")

        def fallback_load(self_or_model, *args, **kwargs):
            call_order.append("fallback")
            return args[0]

        ctx = _make_load_context()
        model = MagicMock()

        with patch(
            "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.is_available",
            return_value=False,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.is_available",
            return_value=True,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.load",
            exploding_load,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.rollback",
            rollback,
        ), patch(
            "modelexpress.load_strategy.gds_strategy.GdsStrategy.is_available",
            return_value=False,
        ), patch(
            "modelexpress.load_strategy.default_strategy.DefaultStrategy.is_available",
            return_value=True,
        ), patch(
            "modelexpress.load_strategy.default_strategy.DefaultStrategy.load",
            fallback_load,
        ):
            LoadStrategyChain.run(model, ctx)

        assert call_order == ["exploding", "rollback", "fallback"]

    def test_strategy_failed_runs_rollback_before_next(self):
        from modelexpress.load_strategy import LoadStrategyChain

        call_order = []

        def failed_load(self_or_model, *args, **kwargs):
            call_order.append("failed")
            raise StrategyFailed("expected miss")

        def rollback(self_or_ctx, *args, **kwargs):
            call_order.append("rollback")

        def fallback_load(self_or_model, *args, **kwargs):
            call_order.append("fallback")
            return args[0]

        ctx = _make_load_context()
        ctx.adapter.reinit_for_retry = MagicMock(side_effect=lambda result: result)
        model = MagicMock()

        with patch(
            "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.is_available",
            return_value=False,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.is_available",
            return_value=True,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.load",
            failed_load,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.rollback",
            rollback,
        ), patch(
            "modelexpress.load_strategy.gds_strategy.GdsStrategy.is_available",
            return_value=False,
        ), patch(
            "modelexpress.load_strategy.default_strategy.DefaultStrategy.is_available",
            return_value=True,
        ), patch(
            "modelexpress.load_strategy.default_strategy.DefaultStrategy.load",
            fallback_load,
        ):
            LoadStrategyChain.run(model, ctx)

        assert call_order == ["failed", "rollback", "fallback"]
        ctx.adapter.reinit_for_retry.assert_not_called()

    def test_strategy_failed_runs_rollback_and_reinit_when_mutated(self):
        from modelexpress.load_strategy import LoadStrategyChain

        call_order = []

        def failed_load(self_or_model, *args, **kwargs):
            call_order.append("failed")
            raise StrategyFailed("mutated failure", mutated=True)

        def rollback(self_or_ctx, *args, **kwargs):
            call_order.append("rollback")

        def reinit(result):
            call_order.append("reinit")
            return result

        def fallback_load(self_or_model, *args, **kwargs):
            call_order.append("fallback")
            return args[0]

        ctx = _make_load_context()
        ctx.adapter.reinit_for_retry = MagicMock(side_effect=reinit)
        model = MagicMock()

        with patch(
            "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.is_available",
            return_value=False,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.is_available",
            return_value=True,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.load",
            failed_load,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.rollback",
            rollback,
        ), patch(
            "modelexpress.load_strategy.gds_strategy.GdsStrategy.is_available",
            return_value=False,
        ), patch(
            "modelexpress.load_strategy.default_strategy.DefaultStrategy.is_available",
            return_value=True,
        ), patch(
            "modelexpress.load_strategy.default_strategy.DefaultStrategy.load",
            fallback_load,
        ):
            LoadStrategyChain.run(model, ctx)

        assert call_order == ["failed", "rollback", "reinit", "fallback"]

    def test_strategy_failed_runs_rollback_without_reinit_when_clean(self):
        from modelexpress.load_strategy import LoadStrategyChain

        call_order = []

        def failed_load(self_or_model, *args, **kwargs):
            call_order.append("failed")
            raise StrategyFailed("clean failure", mutated=False)

        def rollback(self_or_ctx, *args, **kwargs):
            call_order.append("rollback")

        def fallback_load(self_or_model, *args, **kwargs):
            call_order.append("fallback")
            return args[0]

        ctx = _make_load_context()
        ctx.adapter.reinit_for_retry = MagicMock(side_effect=lambda result: result)
        model = MagicMock()

        with patch(
            "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.is_available",
            return_value=False,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.is_available",
            return_value=True,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.load",
            failed_load,
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.rollback",
            rollback,
        ), patch(
            "modelexpress.load_strategy.gds_strategy.GdsStrategy.is_available",
            return_value=False,
        ), patch(
            "modelexpress.load_strategy.default_strategy.DefaultStrategy.is_available",
            return_value=True,
        ), patch(
            "modelexpress.load_strategy.default_strategy.DefaultStrategy.load",
            fallback_load,
        ):
            LoadStrategyChain.run(model, ctx)

        assert call_order == ["failed", "rollback", "fallback"]
        ctx.adapter.reinit_for_retry.assert_not_called()


class TestDefaultStrategy:
    @patch("modelexpress.load_strategy.default_strategy.register_tensors")
    def test_after_native_load_failure_is_mutated(self, mock_register):
        from modelexpress.load_strategy.default_strategy import DefaultStrategy

        ctx = _make_load_context()
        ctx.adapter.after_native_load = MagicMock(side_effect=RuntimeError("post load"))

        with pytest.raises(StrategyFailed, match="post load") as exc:
            DefaultStrategy().load(MagicMock(), ctx)

        assert exc.value.mutated is True
        mock_register.assert_not_called()


class TestRdmaStrategyAvailability:
    """Verify RdmaStrategy.is_available checks for MX server config."""

    def _make_strategy(self):
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
        return RdmaStrategy()

    @patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True)
    def test_unavailable_when_no_server_address(self, _mock):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        with patch.dict("os.environ", {}, clear=True):
            assert strategy.is_available(ctx) is False

    @patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True)
    def test_available_when_mx_server_address_set(self, _mock):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        with patch.dict("os.environ", {"MX_SERVER_ADDRESS": "server:8001"}):
            assert strategy.is_available(ctx) is True

    @patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True)
    def test_available_when_model_express_url_set(self, _mock):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        with patch.dict("os.environ", {"MODEL_EXPRESS_URL": "server:8001"}):
            assert strategy.is_available(ctx) is True

    @patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=False)
    def test_unavailable_when_nixl_not_available(self, _mock):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        assert strategy.is_available(ctx) is False

    @patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True)
    def test_unavailable_when_backend_lacks_rdma_p2p(
        self, _mock, mock_accelerator_backend_cls
    ):
        strategy = self._make_strategy()
        ctx = _make_load_context(
            accelerator_backend=mock_accelerator_backend_cls(rdma_p2p=False)
        )
        with patch.dict("os.environ", {"MX_SERVER_ADDRESS": "server:8001"}):
            assert strategy.is_available(ctx) is False

    @patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True)
    def test_available_when_backend_supports_rdma_p2p(
        self, _mock, mock_accelerator_backend_cls
    ):
        strategy = self._make_strategy()
        ctx = _make_load_context(
            accelerator_backend=mock_accelerator_backend_cls(rdma_p2p=True)
        )
        with patch.dict("os.environ", {"MX_SERVER_ADDRESS": "server:8001"}):
            assert strategy.is_available(ctx) is True

    @patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True)
    def test_available_when_decentralized_backend_and_no_server_address(self, _mock):
        # Decentralized backends (REQUIRES_P2P_METADATA=True) don't need
        # a central server, so the "no server configured" gate shouldn't
        # trip. Strict `is True` check means MagicMock's auto-attribute
        # in _make_load_context's default ctx does NOT accidentally
        # trigger this path - this test has to override explicitly.
        strategy = self._make_strategy()
        ctx = _make_load_context()
        ctx.mx_client.REQUIRES_P2P_METADATA = True
        with patch.dict("os.environ", {}, clear=True):
            assert strategy.is_available(ctx) is True


# ---------------------------------------------------------------------------
# RdmaStrategy._find_source_instances
# ---------------------------------------------------------------------------


class TestFindSourceInstances:
    def _make_strategy(self):
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
        return RdmaStrategy()

    def test_returns_empty_when_nixl_unavailable(self):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        with patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=False):
            assert strategy.is_available(ctx) is False

    def test_returns_empty_when_no_instances(self):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[])
        result = strategy._find_source_instances(ctx)
        assert result == []

    def test_returns_empty_when_list_sources_raises(self):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        ctx.mx_client.list_sources.side_effect = RuntimeError("server unreachable")
        result = strategy._find_source_instances(ctx)
        assert result == []

    def test_calls_list_sources_with_ready_filter(self):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[])
        strategy._find_source_instances(ctx)
        ctx.mx_client.list_sources.assert_called_once_with(
            identity=ctx.identity,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )

    def test_filters_by_worker_rank(self):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        insts = [
            _make_instance_ref(worker_id="w-0", worker_rank=0),
            _make_instance_ref(worker_id="w-1", worker_rank=1),
            _make_instance_ref(worker_id="w-2", worker_rank=0),
        ]
        ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=insts)
        with patch("modelexpress.load_strategy.rdma_strategy.get_configured_selector", return_value=_IdentitySelector()):
            result = strategy._find_source_instances(ctx)
        assert len(result) == 2
        assert all(r.worker_rank == 0 for r in result)

    def test_returns_source_instance_refs(self):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        inst = _make_instance_ref()
        ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[inst])
        with patch("modelexpress.load_strategy.rdma_strategy.get_configured_selector", return_value=_IdentitySelector()):
            result = strategy._find_source_instances(ctx)
        assert len(result) == 1
        assert result[0].mx_source_id == inst.mx_source_id
        assert result[0].worker_id == inst.worker_id


# ---------------------------------------------------------------------------
# RdmaStrategy._fetch_worker_metadata
# ---------------------------------------------------------------------------


class TestFetchWorkerMetadata:
    def _make_strategy(self):
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
        return RdmaStrategy()

    def test_returns_worker_when_found(self):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        ctx.mx_client.get_metadata.return_value = _make_metadata_resp(rank=0)
        result = strategy._fetch_worker_metadata(ctx, "src", "w-1")
        assert result is not None
        assert result.worker_rank == 0

    def test_returns_none_when_not_found(self):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        ctx.mx_client.get_metadata.return_value = _make_metadata_resp(found=False)
        result = strategy._fetch_worker_metadata(ctx, "src", "w-1")
        assert result is None

    def test_returns_none_when_worker_has_no_tensors(self):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        empty_worker = p2p_pb2.WorkerMetadata(worker_rank=0, tensors=[])
        ctx.mx_client.get_metadata.return_value = p2p_pb2.GetMetadataResponse(
            found=True, worker=empty_worker,
        )
        result = strategy._fetch_worker_metadata(ctx, "src", "w-1")
        assert result is None


# ---------------------------------------------------------------------------
# RdmaStrategy.load (candidate iteration)
# ---------------------------------------------------------------------------


class TestRdmaStrategyLoad:
    def _setup(self, ctx, candidates, metadata_side_effects, load_raises_for=None):
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy

        strategy = RdmaStrategy()
        ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
            instances=candidates
        )
        ctx.mx_client.get_metadata.side_effect = metadata_side_effects
        attempts = []

        def fake_load_as_target(_result, _ctx, _worker, _mx_id, worker_id):
            attempts.append(worker_id)
            if load_raises_for and worker_id in load_raises_for:
                raise StrategyFailed(f"transfer failed: {worker_id}", mutated=True)
            return _result

        strategy._load_as_target = fake_load_as_target
        return strategy, attempts

    def test_returns_result_on_first_success(self):
        ctx = _make_load_context()
        candidates = [_make_instance_ref(worker_id="w-1")]
        strategy, attempts = self._setup(ctx, candidates, [_make_metadata_resp(rank=0, worker_id="w-1")])

        with patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True), \
             patch("modelexpress.load_strategy.rdma_strategy.get_configured_selector", return_value=_IdentitySelector()):
            result = strategy.load(MagicMock(), ctx)

        assert isinstance(result, LoadResult)
        assert attempts == ["w-1"]

    def test_transfer_failure_reinitializes_and_tries_next_source(self):
        ctx = _make_load_context()
        ctx.adapter.reinit_for_retry = MagicMock(side_effect=lambda result: result)
        candidates = [
            _make_instance_ref(worker_id="w-1"),
            _make_instance_ref(worker_id="w-2"),
        ]
        strategy, attempts = self._setup(
            ctx, candidates,
            [_make_metadata_resp(rank=0, worker_id="w-1"),
             _make_metadata_resp(rank=0, worker_id="w-2")],
            load_raises_for={"w-1"},
        )

        with patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True), \
             patch("modelexpress.load_strategy.rdma_strategy.get_configured_selector", return_value=_IdentitySelector()):
            result = strategy.load(MagicMock(), ctx)

        assert isinstance(result, LoadResult)
        assert attempts == ["w-1", "w-2"]
        ctx.adapter.reinit_for_retry.assert_called_once()

    def test_raises_strategy_failed_when_no_candidates(self):
        ctx = _make_load_context()
        ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[])
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
        strategy = RdmaStrategy()

        with patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True):
            with pytest.raises(StrategyFailed, match="No RDMA source available") as exc:
                strategy.load(MagicMock(), ctx)
        assert exc.value.mutated is False

    def test_raises_strategy_failed_when_all_sources_fail(self):
        ctx = _make_load_context()
        candidates = [_make_instance_ref(worker_id=f"w-{i}") for i in range(3)]
        strategy, _ = self._setup(
            ctx, candidates,
            [
                p2p_pb2.GetMetadataResponse(found=False)
                for _ in range(3)
            ],
        )

        with patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True), \
             patch("modelexpress.load_strategy.rdma_strategy.get_configured_selector", return_value=_IdentitySelector()):
            with pytest.raises(StrategyFailed, match="No RDMA source succeeded") as exc:
                strategy.load(MagicMock(), ctx)

        assert exc.value.mutated is False

    def test_skips_mismatched_accelerator_before_target(
        self,
        mock_accelerator_backend_cls,
    ):
        ctx = _make_load_context(
            accelerator_backend=mock_accelerator_backend_cls(name="xpu"),
        )
        source_resp = _make_metadata_resp(rank=0, worker_id="w-1")
        source_resp.worker.accelerator = "cuda"
        candidates = [_make_instance_ref(worker_id="w-1")]
        strategy, attempts = self._setup(ctx, candidates, [source_resp])

        with patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True), \
             patch("modelexpress.load_strategy.rdma_strategy.get_configured_selector", return_value=_IdentitySelector()):
            with pytest.raises(StrategyFailed, match="No RDMA source succeeded") as exc:
                strategy.load(MagicMock(), ctx)

        assert exc.value.mutated is False
        assert attempts == []

    def test_accepts_matching_xpu_accelerator(
        self,
        mock_accelerator_backend_cls,
    ):
        ctx = _make_load_context(
            accelerator_backend=mock_accelerator_backend_cls(name="xpu"),
        )
        source_resp = _make_metadata_resp(rank=0, worker_id="w-1")
        source_resp.worker.accelerator = "xpu"
        candidates = [_make_instance_ref(worker_id="w-1")]
        strategy, attempts = self._setup(ctx, candidates, [source_resp])

        with patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True), \
             patch("modelexpress.load_strategy.rdma_strategy.get_configured_selector", return_value=_IdentitySelector()):
            result = strategy.load(MagicMock(), ctx)

        assert isinstance(result, LoadResult)
        assert attempts == ["w-1"]

    def test_load_as_target_marks_post_prepare_failure_as_mutated(self):
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy

        ctx = _make_load_context()
        result = LoadResult(value=MagicMock(), model=MagicMock())
        strategy = RdmaStrategy()
        source_worker = _make_worker()

        ctx.adapter.prepare_rdma_target = MagicMock(side_effect=lambda result: result)
        ctx.adapter.before_rdma_receive = MagicMock(
            side_effect=RuntimeError("post-prepare failure")
        )

        with pytest.raises(StrategyFailed, match="post-prepare failure") as exc:
            strategy._load_as_target(result, ctx, source_worker, "src", "worker")

        assert exc.value.mutated is True

    def test_rollback_shuts_down_nixl_manager(self):
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy

        ctx = _make_load_context()
        ctx.tensors = {"w": MagicMock()}
        manager = MagicMock()
        ctx.nixl_manager = manager

        RdmaStrategy().rollback(ctx)

        manager.shutdown.assert_called_once()
        assert ctx.nixl_manager is None
        assert ctx.tensors == {}


# ---------------------------------------------------------------------------
# publish_metadata_and_ready (metadata)
# ---------------------------------------------------------------------------


class TestPublishMetadataAndReady:
    def test_starts_publisher_with_publish_fn(self):
        from modelexpress.metadata.publish import publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.return_value = "abc123def456abcd"

        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"nixl-data"

        tensors = {}
        for i in range(3):
            t = MagicMock(spec=torch.Tensor)
            t.data_ptr.return_value = 0x1000 + i * 1024
            t.numel.return_value = 256
            t.element_size.return_value = 2
            t.dtype = torch.bfloat16
            tensors[f"layer.{i}.weight"] = t

        identity = _make_identity("my-model")
        ready_fn = MagicMock(return_value=False)
        mock_publisher = MagicMock()
        with patch.dict(os.environ, {"MX_P2P_METADATA": "0"}), \
             patch("modelexpress.metadata.publish.PublisherThread", return_value=mock_publisher) as publisher_cls:
            publish_metadata_and_ready(
                mx_client,
                nixl_manager,
                tensors,
                worker_rank=2,
                device_id=0,
                identity=identity,
                worker_id="inst-uuid",
                ready_fn=ready_fn,
            )

        mx_client.publish_metadata.assert_not_called()
        publisher_cls.assert_called_once()
        publisher_kwargs = publisher_cls.call_args.kwargs
        assert publisher_kwargs["mx_client"] is mx_client
        assert publisher_kwargs["worker_id"] == "inst-uuid"
        assert publisher_kwargs["worker_rank"] == 2
        assert publisher_kwargs["nixl_manager"] is nixl_manager
        assert callable(publisher_kwargs["publish_fn"])
        assert publisher_kwargs["ready_fn"] is ready_fn
        mock_publisher.start.assert_called_once()

        result = publisher_kwargs["publish_fn"]()
        assert result == "abc123def456abcd"
        mx_client.publish_metadata.assert_called_once()
        call_args = mx_client.publish_metadata.call_args
        assert call_args.args[0] is identity
        assert call_args.args[1].accelerator == "cuda"
        assert call_args.args[2] == "inst-uuid"

    def test_full_manifest_publish_dual_writes_tensor_fields(self):
        """On the full tensor-manifest publish path (MX_P2P_METADATA=0, source
        embeds the full manifest in the published WorkerMetadata), both the
        legacy `tensors` field and the newer `tensor_source` must be populated.
        Servers predating the tensor_source oneof read only `tensors`; dropping
        it leaves them with 0 tensors and targets fall back to disk load."""
        from modelexpress.metadata.publish import publish_metadata_and_ready

        mx_client = MagicMock()
        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"nixl-data"

        tensors = {}
        for i in range(3):
            t = MagicMock(spec=torch.Tensor)
            t.data_ptr.return_value = 0x1000 + i * 1024
            t.numel.return_value = 256
            t.element_size.return_value = 2
            t.dtype = torch.bfloat16
            tensors[f"layer.{i}.weight"] = t

        identity = _make_identity("my-model")
        mock_publisher = MagicMock()
        with patch.dict(os.environ, {"MX_P2P_METADATA": "0"}), \
             patch("modelexpress.metadata.publish.PublisherThread", return_value=mock_publisher) as publisher_cls:
            publish_metadata_and_ready(mx_client, nixl_manager, tensors, worker_rank=0, device_id=0, identity=identity, worker_id="inst-uuid")
            publisher_cls.call_args.kwargs["publish_fn"]()

        worker = mx_client.publish_metadata.call_args.args[1]
        # Legacy field — servers predating tensor_source read only from here.
        assert len(worker.tensors) == 3
        # New field — current servers read from here.
        assert len(worker.tensor_source.tensors) == 3
        assert {t.name for t in worker.tensors} == {
            "layer.0.weight", "layer.1.weight", "layer.2.weight",
        }
        assert {t.name for t in worker.tensor_source.tensors} == {
            "layer.0.weight", "layer.1.weight", "layer.2.weight",
        }

    def test_publish_fn_retries_publish(self):
        from modelexpress.metadata.publish import publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.side_effect = [
            _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "grpc error 1"),
            _FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "grpc error 2"),
            "abc123def456abcd",
        ]

        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"data"

        identity = _make_identity()
        mock_publisher = MagicMock()
        with patch.dict(os.environ, {"MX_P2P_METADATA": "0"}), \
             patch("modelexpress.metadata.publish.time.sleep") as sleep_mock, \
             patch("modelexpress.metadata.publish.PublisherThread", return_value=mock_publisher) as publisher_cls:
            publish_metadata_and_ready(
                mx_client,
                nixl_manager,
                {},
                worker_rank=0,
                device_id=0,
                identity=identity,
                worker_id="w-1",
            )
            publish_fn = publisher_cls.call_args.kwargs["publish_fn"]
            result = publish_fn()

        assert result == "abc123def456abcd"
        assert mx_client.publish_metadata.call_count == 3
        assert sleep_mock.call_args_list == [call(1.0), call(2.0)]
        mock_publisher.start.assert_called_once()

    def test_publish_fn_failure_after_retries_raises_runtime_error(self):
        from modelexpress.metadata.publish import publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.side_effect = [
            _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "grpc error 1"),
            _FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "grpc error 2"),
            _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "grpc error 3"),
        ]

        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"data"

        identity = _make_identity()
        mock_publisher = MagicMock()
        with patch.dict(os.environ, {"MX_P2P_METADATA": "0"}), \
             patch("modelexpress.metadata.publish.time.sleep") as sleep_mock, \
             patch("modelexpress.metadata.publish.PublisherThread", return_value=mock_publisher) as publisher_cls:
            publish_metadata_and_ready(
                mx_client,
                nixl_manager,
                {},
                worker_rank=0,
                device_id=0,
                identity=identity,
                worker_id="w-1",
            )
            publish_fn = publisher_cls.call_args.kwargs["publish_fn"]
            with pytest.raises(RuntimeError, match="Failed to publish metadata after 3 attempts"):
                publish_fn()

        assert mx_client.publish_metadata.call_count == 3
        assert sleep_mock.call_args_list == [call(1.0), call(2.0)]
        mock_publisher.start.assert_called_once()

    def test_publish_fn_non_retryable_grpc_failure_fails_immediately(self):
        from modelexpress.metadata.publish import publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.side_effect = _FakeRpcError(
            grpc.StatusCode.PERMISSION_DENIED,
            "permission denied",
        )

        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"data"

        identity = _make_identity()
        mock_publisher = MagicMock()
        with patch.dict(os.environ, {"MX_P2P_METADATA": "0"}), \
             patch("modelexpress.metadata.publish.time.sleep") as sleep_mock, \
             patch("modelexpress.metadata.publish.PublisherThread", return_value=mock_publisher) as publisher_cls:
            publish_metadata_and_ready(
                mx_client,
                nixl_manager,
                {},
                worker_rank=0,
                device_id=0,
                identity=identity,
                worker_id="w-1",
            )
            publish_fn = publisher_cls.call_args.kwargs["publish_fn"]
            with pytest.raises(_FakeRpcError, match="permission denied"):
                publish_fn()

        assert mx_client.publish_metadata.call_count == 1
        assert sleep_mock.call_args_list == []
        mock_publisher.start.assert_called_once()

    def test_p2p_mode_starts_grpc_server_before_publish(self):
        from modelexpress.metadata.publish import publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.return_value = "abc123def456abcd"

        nixl_manager = MagicMock()
        nixl_manager._listen_port = 5555
        nixl_manager.agent_name = "test-agent"

        mock_grpc_server = MagicMock()
        mock_grpc_server.start.return_value = 6555
        mock_publisher = MagicMock()

        with patch.dict(os.environ, {"MX_P2P_METADATA": "1", "MX_WORKER_HOST": "10.0.0.1"}), \
             patch("modelexpress.metadata.worker_server.WorkerGrpcServer", return_value=mock_grpc_server) as grpc_cls, \
             patch("modelexpress.metadata.publish.PublisherThread", return_value=mock_publisher) as publisher_cls:
            publish_metadata_and_ready(
                mx_client,
                nixl_manager,
                {},
                worker_rank=0,
                device_id=0,
                identity=_make_identity(),
                worker_id="w-1",
            )

            mx_client.publish_metadata.assert_not_called()
            grpc_cls.assert_called_once()
            mock_grpc_server.start.assert_called_once()

            publish_fn = publisher_cls.call_args.kwargs["publish_fn"]
            publish_fn()

        mx_client.publish_metadata.assert_called_once()
        mock_grpc_server.set_mx_source_id.assert_called_once_with("abc123def456abcd")


# ---------------------------------------------------------------------------
# storage_view / collect_module_tensors storage view tests
# ---------------------------------------------------------------------------

_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for storage view tests"
)


class TestStorageView:
    """Verify that storage_view creates usable flat byte views (CPU tensors OK)."""

    def test_contiguous_tensor(self):
        from modelexpress.tensor_utils import storage_view
        t = torch.randn(4, 8)
        view = storage_view(t)
        assert view.data_ptr() == t.data_ptr()
        assert view.numel() == t.numel() * t.element_size()

    def test_dense_noncontiguous(self):
        """A transposed tensor that densely covers its storage."""
        from modelexpress.tensor_utils import storage_view
        t = torch.randn(4, 8).t()
        assert not t.is_contiguous()
        view = storage_view(t)
        assert view.data_ptr() == t.data_ptr()

    def test_partial_view_gets_full_storage(self):
        """A partial view (like MLA's W_UV) gets the full storage block."""
        from modelexpress.tensor_utils import storage_view
        full = torch.randn(8, 8)
        partial = full[:4]
        view = storage_view(partial)
        assert view.data_ptr() == full.data_ptr()
        assert view.numel() == full.numel() * full.element_size()

    def test_shared_storage_same_data_ptr(self):
        """Two views into the same storage produce views with the same data_ptr."""
        from modelexpress.tensor_utils import storage_view
        full = torch.randn(8, 8)
        view_a = storage_view(full[:4])
        view_b = storage_view(full[4:])
        assert view_a.data_ptr() == view_b.data_ptr()


@_skip_no_cuda
class TestCollectModuleTensorsStorageViews:
    """Verify that collect_module_tensors handles non-contiguous tensors
    via storage views instead of silently dropping them."""

    def test_noncontiguous_tensor_included(self):
        from modelexpress.tensor_utils import collect_module_tensors

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                raw = torch.randn(4, 8, device="cuda")
                self.scale = nn.Parameter(raw.t())
        m = M()
        assert not m.scale.is_contiguous()
        result = collect_module_tensors(m)
        assert "scale.__storage" in result
        assert result["scale.__storage"].data_ptr() == m.scale.data_ptr()

    def test_contiguous_tensor_registered_directly(self):
        from modelexpress.tensor_utils import collect_module_tensors

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 8, device="cuda"))
        m = M()
        result = collect_module_tensors(m)
        assert "weight" in result
        assert "weight.__storage" not in result

    def test_shared_storage_deduped(self):
        """Two views into the same storage are registered once."""
        from modelexpress.tensor_utils import collect_module_tensors

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                full = torch.randn(8, 8, device="cuda")
                self.w_uv = nn.Parameter(full[:4].t())
                self.w_uk_t = nn.Parameter(full[4:].permute(1, 0))
        m = M()
        result = collect_module_tensors(m)
        storage_keys = [k for k in result if ".__storage" in k]
        assert len(storage_keys) == 1

    def test_source_target_name_symmetry(self):
        """Both source and target produce the same tensor names."""
        from modelexpress.tensor_utils import collect_module_tensors

        def make_model():
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = nn.Parameter(torch.randn(4, 8, device="cuda"))
                    self.scale = nn.Parameter(torch.randn(4, 8, device="cuda").t())
            return M()

        source = collect_module_tensors(make_model())
        target = collect_module_tensors(make_model())
        assert set(source.keys()) == set(target.keys())


# ---------------------------------------------------------------------------
# configure_vllm_logging
# ---------------------------------------------------------------------------


class TestConfigureVllmLogging:
    """Verify modelexpress loggers inherit vLLM handlers in EngineCore subprocess."""

    @pytest.fixture(autouse=True)
    def _isolate_log_level_env(self, monkeypatch):
        """Clear MODEL_EXPRESS_LOG_LEVEL so the logging tests stay hermetic."""
        # configure_vllm_logging() reads the var at call time, so an ambient
        # value (e.g. exported on a dev box) would take the explicit-level
        # branch instead of inheriting vLLM's level. Clear it so each test
        # controls the var explicitly; tests that set it do so in their block.
        monkeypatch.delenv("MODEL_EXPRESS_LOG_LEVEL", raising=False)

    def _reset_mx_logger(self):
        """Clear any handlers/level from the modelexpress root logger."""
        mx_root = logging.getLogger("modelexpress")
        mx_root.handlers.clear()
        mx_root.setLevel(logging.NOTSET)

    def _simulate_vllm_enginecore_logging(self):
        """Reproduce vLLM 0.19.0 EngineCore: only "vllm" gets a handler."""
        self._reset_mx_logger()

        vllm_logger = logging.getLogger("vllm")
        self._saved_vllm_handlers = list(vllm_logger.handlers)
        self._saved_vllm_level = vllm_logger.level
        self._saved_vllm_propagate = vllm_logger.propagate

        vllm_logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        vllm_logger.addHandler(handler)
        vllm_logger.setLevel(logging.DEBUG)
        vllm_logger.propagate = False
        return vllm_logger, handler

    def _cleanup(self, vllm_logger):
        self._reset_mx_logger()
        vllm_logger.handlers.clear()
        for h in self._saved_vllm_handlers:
            vllm_logger.addHandler(h)
        vllm_logger.setLevel(self._saved_vllm_level)
        vllm_logger.propagate = self._saved_vllm_propagate

    def test_child_loggers_visible_after_configure(self):
        from modelexpress import configure_vllm_logging

        vllm_logger, handler = self._simulate_vllm_enginecore_logging()
        try:
            configure_vllm_logging()

            mx_root = logging.getLogger("modelexpress")
            assert len(mx_root.handlers) == 1
            assert mx_root.handlers[0] is handler

            child = logging.getLogger("modelexpress.metadata")
            assert child.getEffectiveLevel() == logging.DEBUG
        finally:
            self._cleanup(vllm_logger)

    def test_no_duplicate_handlers_on_repeated_calls(self):
        from modelexpress import configure_vllm_logging

        vllm_logger, _handler = self._simulate_vllm_enginecore_logging()
        try:
            configure_vllm_logging()
            configure_vllm_logging()

            mx_root = logging.getLogger("modelexpress")
            assert len(mx_root.handlers) == 1
        finally:
            self._cleanup(vllm_logger)

    def test_log_output_actually_captured(self):
        from modelexpress import configure_vllm_logging

        vllm_logger, _ = self._simulate_vllm_enginecore_logging()
        try:
            buf = logging.handlers.MemoryHandler(capacity=100)
            vllm_logger.handlers.clear()
            vllm_logger.addHandler(buf)

            self._reset_mx_logger()
            configure_vllm_logging()

            child = logging.getLogger("modelexpress.metadata.publisher")
            child.info("Heartbeat started")

            assert len(buf.buffer) == 1
            assert "Heartbeat started" in buf.buffer[0].getMessage()
            assert buf.buffer[0].name == "modelexpress.metadata.publisher"
        finally:
            vllm_logger.removeHandler(buf)
            self._cleanup(vllm_logger)

    def test_model_express_log_level_overrides_vllm(self):
        from modelexpress import configure_vllm_logging

        vllm_logger, _handler = self._simulate_vllm_enginecore_logging()
        try:
            assert vllm_logger.level == logging.DEBUG
            with patch.dict("os.environ", {"MODEL_EXPRESS_LOG_LEVEL": "WARNING"}):
                configure_vllm_logging()
            mx_root = logging.getLogger("modelexpress")
            assert mx_root.level == logging.WARNING
        finally:
            self._cleanup(vllm_logger)

    def test_falls_back_to_vllm_level_when_env_unset(self):
        from modelexpress import configure_vllm_logging

        vllm_logger, _handler = self._simulate_vllm_enginecore_logging()
        try:
            with patch.dict("os.environ", {}, clear=False):
                os.environ.pop("MODEL_EXPRESS_LOG_LEVEL", None)
                configure_vllm_logging()
            mx_root = logging.getLogger("modelexpress")
            assert mx_root.level == logging.DEBUG
        finally:
            self._cleanup(vllm_logger)
