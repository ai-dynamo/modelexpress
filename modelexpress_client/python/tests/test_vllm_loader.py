# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tensor utilities, metadata publishing, and loading strategies."""

from unittest.mock import MagicMock, patch, call

import grpc
import pytest
import torch
import torch.nn as nn

from modelexpress import p2p_pb2
from modelexpress.nixl_transfer import NixlTransferManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loader():
    """Return an MxModelLoader with a fresh mock MxClient."""
    with patch("modelexpress.vllm_loader.DefaultModelLoader"):
        load_config = MagicMock()
        load_config.load_format = "mx"
        load_config.device = None
        from modelexpress.vllm_loader import MxModelLoader
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
        vllm_config=MagicMock(),
        model_config=MagicMock(),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=0,
        device_id=0,
        identity=_make_identity(),
        mx_client=MagicMock(),
        worker_id="test-worker",
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
        from modelexpress.vllm_loader import MxModelLoader
        remaining = getattr(MxModelLoader, "__abstractmethods__", frozenset())
        assert remaining == frozenset()

    def test_download_model_delegates(self):
        loader = _make_loader()
        cfg = MagicMock()
        with patch("modelexpress.vllm_loader.DefaultModelLoader") as mock_cls:
            loader.download_model(cfg)
            mock_cls.return_value.download_model.assert_called_once_with(cfg)

    def test_load_weights_delegates(self):
        loader = _make_loader()
        model, cfg = MagicMock(), MagicMock()
        with patch("modelexpress.vllm_loader.DefaultModelLoader") as mock_cls:
            loader.load_weights(model, cfg)
            mock_cls.return_value.load_weights.assert_called_once_with(model, cfg)


# ---------------------------------------------------------------------------
# register_tensors (load_strategy.base)
# ---------------------------------------------------------------------------


class TestInitNixlManager:
    """Verify _init_nixl_manager computes listen_port from MX_P2P_METADATA."""

    def test_centralized_mode_skips_listen_port(self):
        from modelexpress.load_strategy.base import _init_nixl_manager

        with patch.dict("os.environ", {"MX_P2P_METADATA": "0"}, clear=False), \
             patch.object(NixlTransferManager, "initialize"):
            mgr = _init_nixl_manager(global_rank=0, device_id=0, role="auto")

        assert mgr._listen_port is None

    def test_p2p_mode_sets_listen_port_with_device_offset(self):
        from modelexpress.load_strategy.base import _init_nixl_manager

        with patch.dict(
            "os.environ",
            {"MX_P2P_METADATA": "1", "MX_METADATA_PORT": "6000"},
            clear=False,
        ), \
             patch.object(NixlTransferManager, "initialize"):
            mgr = _init_nixl_manager(global_rank=0, device_id=2, role="auto")

        assert mgr._listen_port == 6002  # base port 6000 + device_id 2


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
        with patch("modelexpress.load_strategy.rdma_strategy.random.shuffle"):
            result = strategy._find_source_instances(ctx)
        assert len(result) == 2
        assert all(r.worker_rank == 0 for r in result)

    def test_returns_source_instance_refs(self):
        strategy = self._make_strategy()
        ctx = _make_load_context()
        inst = _make_instance_ref()
        ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[inst])
        with patch("modelexpress.load_strategy.rdma_strategy.random.shuffle"):
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
        from modelexpress.load_strategy import SourceTransferError

        strategy = RdmaStrategy()
        ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
            instances=candidates
        )
        ctx.mx_client.get_metadata.side_effect = metadata_side_effects
        attempts = []

        original_load_as_target = strategy._load_as_target

        def fake_load_as_target(_m, _ctx, _worker, _mx_id, worker_id):
            attempts.append(worker_id)
            if load_raises_for and worker_id in load_raises_for:
                raise SourceTransferError(f"transfer failed: {worker_id}")

        strategy._load_as_target = fake_load_as_target
        return strategy, attempts

    def test_returns_true_on_first_success(self):
        ctx = _make_load_context()
        candidates = [_make_instance_ref(worker_id="w-1")]
        strategy, attempts = self._setup(ctx, candidates, [_make_metadata_resp(rank=0, worker_id="w-1")])

        with patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True), \
             patch("modelexpress.load_strategy.rdma_strategy.random.shuffle"):
            result = strategy.load(MagicMock(), ctx)

        assert result is True
        assert attempts == ["w-1"]

    def test_tries_next_on_source_transfer_error(self):
        ctx = _make_load_context()
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
             patch("modelexpress.load_strategy.rdma_strategy.random.shuffle"):
            result = strategy.load(MagicMock(), ctx)

        assert result is True
        assert attempts == ["w-1", "w-2"]

    def test_returns_false_when_no_candidates(self):
        ctx = _make_load_context()
        ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[])
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
        strategy = RdmaStrategy()

        with patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True):
            result = strategy.load(MagicMock(), ctx)
        assert result is False

    def test_returns_false_when_all_fail(self):
        ctx = _make_load_context()
        candidates = [_make_instance_ref(worker_id=f"w-{i}") for i in range(3)]
        strategy, _ = self._setup(
            ctx, candidates,
            [_make_metadata_resp(rank=0, worker_id=f"w-{i}") for i in range(3)],
            load_raises_for={"w-0", "w-1", "w-2"},
        )

        with patch("modelexpress.load_strategy.rdma_strategy.is_nixl_available", return_value=True), \
             patch("modelexpress.load_strategy.rdma_strategy.random.shuffle"):
            result = strategy.load(MagicMock(), ctx)

        assert result is False


# ---------------------------------------------------------------------------
# start_metadata_publisher (metadata)
# ---------------------------------------------------------------------------


class TestStartMetadataPublisher:
    def test_starts_heartbeat_with_publish_fn(self):
        """start_metadata_publisher should build a publish callback
        and pass it to HeartbeatThread without calling publish itself."""
        from modelexpress.metadata import start_metadata_publisher

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
        mock_hb = MagicMock()
        with patch.dict("os.environ", {"MX_CONTIGUOUS_REG": "0", "MX_P2P_METADATA": "0"}), \
             patch("modelexpress.metadata.HeartbeatThread", return_value=mock_hb) as hb_cls:
            start_metadata_publisher(
                mx_client, nixl_manager, tensors,
                global_rank=2, device_id=0, identity=identity, worker_id="inst-uuid",
            )

        mx_client.publish_metadata.assert_not_called()

        hb_cls.assert_called_once()
        hb_kwargs = hb_cls.call_args.kwargs
        assert hb_kwargs["mx_client"] is mx_client
        assert hb_kwargs["worker_id"] == "inst-uuid"
        assert hb_kwargs["worker_rank"] == 2
        assert hb_kwargs["nixl_manager"] is nixl_manager
        assert callable(hb_kwargs["publish_fn"])
        mock_hb.start.assert_called_once()

    def test_publish_fn_calls_publish_metadata(self):
        """The publish_fn callback should call mx_client.publish_metadata
        and return the mx_source_id."""
        from modelexpress.metadata import start_metadata_publisher

        mx_client = MagicMock()
        mx_client.publish_metadata.return_value = "abc123def456abcd"
        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"data"

        identity = _make_identity()
        mock_hb = MagicMock()
        with patch.dict("os.environ", {"MX_CONTIGUOUS_REG": "0", "MX_P2P_METADATA": "0"}), \
             patch("modelexpress.metadata.HeartbeatThread", return_value=mock_hb) as hb_cls:
            start_metadata_publisher(
                mx_client, nixl_manager, {}, global_rank=0,
                device_id=0, identity=identity, worker_id="w-1",
            )

        publish_fn = hb_cls.call_args.kwargs["publish_fn"]
        result = publish_fn()
        assert result == "abc123def456abcd"
        mx_client.publish_metadata.assert_called_once()

    def test_p2p_mode_starts_grpc_server_before_heartbeat(self):
        """In P2P mode, the gRPC server should start immediately (before
        any publish attempt), and publish_fn should set mx_source_id on it."""
        from modelexpress.metadata import start_metadata_publisher

        mx_client = MagicMock()
        mx_client.publish_metadata.return_value = "abc123def456abcd"
        nixl_manager = MagicMock()
        nixl_manager._listen_port = 5555
        nixl_manager.agent_name = "test-agent"

        mock_grpc_server = MagicMock()
        mock_grpc_server.start.return_value = 6555
        mock_hb = MagicMock()

        with patch.dict("os.environ", {"MX_P2P_METADATA": "1", "MX_CONTIGUOUS_REG": "0", "MX_WORKER_HOST": "10.0.0.1"}), \
             patch("modelexpress.worker_server.WorkerGrpcServer", return_value=mock_grpc_server) as grpc_cls, \
             patch("modelexpress.metadata.HeartbeatThread", return_value=mock_hb) as hb_cls:
            start_metadata_publisher(
                mx_client, nixl_manager, {}, global_rank=0,
                device_id=0, identity=_make_identity(), worker_id="w-1",
            )

        grpc_cls.assert_called_once()
        mock_grpc_server.start.assert_called_once()
        mx_client.publish_metadata.assert_not_called()

        publish_fn = hb_cls.call_args.kwargs["publish_fn"]
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
