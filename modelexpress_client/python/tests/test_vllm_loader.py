# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared helpers and MxModelLoader detection logic."""

from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn

from modelexpress import p2p_pb2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loader():
    """Return an MxModelLoader with a fresh mock MxClient."""
    with patch("modelexpress.vllm_loader.DefaultModelLoader"), \
         patch("modelexpress.vllm_loader.DummyModelLoader"):
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


# ---------------------------------------------------------------------------
# _collect_module_tensors
# ---------------------------------------------------------------------------


class TestCollectModuleTensors:
    """Tests for the _collect_module_tensors helper."""

    def test_empty_model(self):
        from modelexpress.vllm_loader import _collect_module_tensors

        model = nn.Module()
        result = _collect_module_tensors(model)
        assert result == {}

    def test_cpu_only_model(self):
        from modelexpress.vllm_loader import _collect_module_tensors

        model = nn.Linear(4, 2, bias=False)
        # Parameters default to CPU -- not CUDA, so should be empty
        result = _collect_module_tensors(model)
        assert result == {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_model(self):
        from modelexpress.vllm_loader import _collect_module_tensors

        model = nn.Linear(4, 2, bias=True).cuda()
        result = _collect_module_tensors(model)
        assert len(result) == 2  # weight + bias
        assert "weight" in result
        assert "bias" in result
        for t in result.values():
            assert t.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_skips_non_contiguous(self):
        from modelexpress.vllm_loader import _collect_module_tensors

        model = nn.Module()
        model.weight = nn.Parameter(torch.randn(4, 3, device="cuda"))
        model.weight_t = model.weight.data.T
        assert not model.weight_t.is_contiguous()

        result = _collect_module_tensors(model)
        assert "weight" in result
        assert "weight_t" not in result

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_deduplicate_tied_weights(self):
        """Tied weights (same data_ptr) should only be registered once."""
        from modelexpress.vllm_loader import _collect_module_tensors

        model = nn.Module()
        shared = nn.Parameter(torch.randn(4, 3, device="cuda"))
        embed = nn.Module()
        embed.weight = shared
        head = nn.Module()
        head.weight = shared
        model.embed_tokens = embed
        model.lm_head = head

        result = _collect_module_tensors(model)
        ptrs = [t.data_ptr() for t in result.values()]
        assert len(ptrs) == len(set(ptrs)), "duplicate data_ptr found in result"
        assert len(result) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_deduplicate_bare_attr_alias(self):
        """A bare tensor attr sharing data_ptr with a parameter is skipped."""
        from modelexpress.vllm_loader import _collect_module_tensors

        model = nn.Module()
        model.weight = nn.Parameter(torch.randn(4, 3, device="cuda"))
        model.__dict__["w_alias"] = model.weight.data

        result = _collect_module_tensors(model)
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

    def test_download_model_delegates_to_default_loader(self):
        loader = _make_loader()
        cfg = MagicMock()
        loader.download_model(cfg)
        loader._default_loader.download_model.assert_called_once_with(cfg)

    def test_load_weights_delegates_to_default_loader(self):
        loader = _make_loader()
        model, cfg = MagicMock(), MagicMock()
        loader.load_weights(model, cfg)
        loader._default_loader.load_weights.assert_called_once_with(model, cfg)


# ---------------------------------------------------------------------------
# _find_source_instances
# ---------------------------------------------------------------------------


class TestFindSourceInstances:
    def test_returns_empty_when_nixl_unavailable(self):
        loader = _make_loader()
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=False):
            result = loader._find_source_instances(_make_identity(), global_rank=0)
        assert result == []
        loader._mx_client.list_sources.assert_not_called()

    def test_returns_empty_when_no_instances(self):
        loader = _make_loader()
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[])
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            result = loader._find_source_instances(_make_identity(), global_rank=0)
        assert result == []

    def test_returns_empty_when_list_sources_raises(self):
        loader = _make_loader()
        loader._mx_client.list_sources.side_effect = RuntimeError("server unreachable")
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            result = loader._find_source_instances(_make_identity(), global_rank=0)
        assert result == []

    def test_calls_list_sources_with_ready_filter(self):
        loader = _make_loader()
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[])
        identity = _make_identity()
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            loader._find_source_instances(identity, global_rank=0)
        loader._mx_client.list_sources.assert_called_once_with(
            identity=identity,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )

    def test_does_not_call_get_metadata(self):
        """Metadata is fetched on demand by the caller, not here."""
        loader = _make_loader()
        inst = _make_instance_ref()
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[inst])
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            loader._find_source_instances(_make_identity(), global_rank=0)
        loader._mx_client.get_metadata.assert_not_called()

    def test_filters_by_worker_rank(self):
        loader = _make_loader()
        insts = [
            _make_instance_ref(worker_id="w-0", worker_rank=0),
            _make_instance_ref(worker_id="w-1", worker_rank=1),
            _make_instance_ref(worker_id="w-2", worker_rank=0),
        ]
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=insts)
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True), \
             patch("modelexpress.vllm_loader.random.shuffle"):
            result = loader._find_source_instances(_make_identity(), global_rank=0)
        assert len(result) == 2
        assert all(r.worker_rank == 0 for r in result)

    def test_returns_source_instance_refs(self):
        loader = _make_loader()
        inst = _make_instance_ref()
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[inst])
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True), \
             patch("modelexpress.vllm_loader.random.shuffle"):
            result = loader._find_source_instances(_make_identity(), global_rank=0)
        assert len(result) == 1
        assert result[0].mx_source_id == inst.mx_source_id
        assert result[0].worker_id == inst.worker_id


# ---------------------------------------------------------------------------
# _fetch_worker_metadata
# ---------------------------------------------------------------------------


class TestFetchWorkerMetadata:
    def test_returns_worker_when_found(self):
        loader = _make_loader()
        loader._mx_client.get_metadata.return_value = _make_metadata_resp(rank=0)
        result = loader._fetch_worker_metadata("src", "w-1", global_rank=0)
        assert result is not None
        assert result.worker_rank == 0

    def test_returns_none_when_not_found(self):
        loader = _make_loader()
        loader._mx_client.get_metadata.return_value = _make_metadata_resp(found=False)
        result = loader._fetch_worker_metadata("src", "w-1", global_rank=0)
        assert result is None

    def test_returns_none_when_worker_has_no_tensors(self):
        loader = _make_loader()
        empty_worker = p2p_pb2.WorkerMetadata(worker_rank=0, tensors=[])
        loader._mx_client.get_metadata.return_value = p2p_pb2.GetMetadataResponse(
            found=True, worker=empty_worker,
        )
        result = loader._fetch_worker_metadata("src", "w-1", global_rank=0)
        assert result is None


# ---------------------------------------------------------------------------
# _try_load_from_candidates
# ---------------------------------------------------------------------------


class TestTryLoadFromCandidates:
    def _setup(self, loader, candidates, metadata_side_effects, load_raises_for=None):
        from modelexpress.vllm_loader import SourceTransferError

        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
            instances=candidates
        )
        loader._mx_client.get_metadata.side_effect = metadata_side_effects
        attempts = []

        def fake_load_as_target(_m, _mc, _td, _gr, _did, _ident, _worker, _mx_id, worker_id):
            attempts.append(worker_id)
            if load_raises_for and worker_id in load_raises_for:
                raise SourceTransferError(f"transfer failed: {worker_id}")

        loader._load_as_target = fake_load_as_target
        loader._load_as_source = MagicMock()
        return attempts

    def test_returns_true_on_first_success(self):
        loader = _make_loader()
        candidates = [_make_instance_ref(worker_id="w-1")]
        attempts = self._setup(loader, candidates, [_make_metadata_resp(rank=0, worker_id="w-1")])

        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True), \
             patch("modelexpress.vllm_loader.random.shuffle"):
            result = loader._try_load_from_candidates(
                MagicMock(), MagicMock(), MagicMock(), 0, 0, _make_identity()
            )

        assert result is True
        assert attempts == ["w-1"]

    def test_marks_stale_on_source_transfer_error_and_tries_next(self):
        loader = _make_loader()
        candidates = [
            _make_instance_ref(worker_id="w-1"),
            _make_instance_ref(worker_id="w-2"),
        ]
        attempts = self._setup(
            loader, candidates,
            [_make_metadata_resp(rank=0, worker_id="w-1"),
             _make_metadata_resp(rank=0, worker_id="w-2")],
            load_raises_for={"w-1"},
        )

        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True), \
             patch("modelexpress.vllm_loader.random.shuffle"):
            result = loader._try_load_from_candidates(
                MagicMock(), MagicMock(), MagicMock(), 0, 0, _make_identity()
            )

        assert result is True
        assert attempts == ["w-1", "w-2"]
        # Target no longer marks sources STALE — that's owned by heartbeat + reaper
        loader._mx_client.update_status.assert_not_called()

    def test_returns_false_when_no_candidates(self):
        loader = _make_loader()
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=False):
            result = loader._try_load_from_candidates(
                MagicMock(), MagicMock(), MagicMock(), 0, 0, _make_identity()
            )
        assert result is False

    def test_returns_false_when_all_fail(self):
        loader = _make_loader()
        candidates = [_make_instance_ref(worker_id=f"w-{i}") for i in range(3)]
        self._setup(
            loader, candidates,
            [_make_metadata_resp(rank=0, worker_id=f"w-{i}") for i in range(3)],
            load_raises_for={"w-0", "w-1", "w-2"},
        )

        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True), \
             patch("modelexpress.vllm_loader.random.shuffle"):
            result = loader._try_load_from_candidates(
                MagicMock(), MagicMock(), MagicMock(), 0, 0, _make_identity()
            )

        assert result is False
        # Target no longer marks sources STALE — that's owned by heartbeat + reaper
        loader._mx_client.update_status.assert_not_called()

    def test_get_metadata_not_called_after_first_success(self):
        """Only one GetMetadata RPC is issued when the first worker succeeds."""
        loader = _make_loader()
        candidates = [
            _make_instance_ref(worker_id="w-1"),
            _make_instance_ref(worker_id="w-2"),
        ]
        self._setup(loader, candidates, [_make_metadata_resp(rank=0, worker_id="w-1")])

        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True), \
             patch("modelexpress.vllm_loader.random.shuffle"):
            loader._try_load_from_candidates(
                MagicMock(), MagicMock(), MagicMock(), 0, 0, _make_identity()
            )

        loader._mx_client.get_metadata.assert_called_once()


# ---------------------------------------------------------------------------
# _publish_metadata_and_ready
# ---------------------------------------------------------------------------


class TestPublishMetadataAndReady:
    def test_calls_publish_and_starts_heartbeat(self):
        from modelexpress.vllm_loader import _publish_metadata_and_ready

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
        mock_hb = MagicMock()
        with patch.dict("os.environ", {"MX_CONTIGUOUS_REG": "0"}), \
             patch("modelexpress.vllm_loader.HeartbeatThread", return_value=mock_hb) as hb_cls:
            _publish_metadata_and_ready(mx_client, nixl_manager, tensors, global_rank=2, device_id=0, identity=identity, worker_id="inst-uuid")

        mx_client.publish_metadata.assert_called_once()
        call_args = mx_client.publish_metadata.call_args
        assert call_args.args[0] is identity
        assert call_args.args[2] == "inst-uuid"

        hb_cls.assert_called_once_with(
            mx_client=mx_client,
            mx_source_id="abc123def456abcd",
            worker_id="inst-uuid",
            worker_rank=2,
            nixl_manager=nixl_manager,
        )
        mock_hb.start.assert_called_once()

    def test_publish_failure_does_not_start_heartbeat(self):
        """If publish_metadata raises, heartbeat should not be started."""
        from modelexpress.vllm_loader import _publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.side_effect = RuntimeError("grpc error")

        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"data"

        identity = _make_identity()
        mock_hb = MagicMock()
        with patch.dict("os.environ", {"MX_CONTIGUOUS_REG": "0"}), \
             patch("modelexpress.vllm_loader.HeartbeatThread", return_value=mock_hb) as hb_cls:
            with pytest.raises(RuntimeError, match="grpc error"):
                _publish_metadata_and_ready(mx_client, nixl_manager, {}, global_rank=0, device_id=0, identity=identity, worker_id="w-1")

        hb_cls.assert_not_called()


# ---------------------------------------------------------------------------
# _storage_view / _collect_module_tensors tests
# ---------------------------------------------------------------------------

_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for storage view tests"
)


class TestStorageView:
    """Verify that _storage_view creates usable flat byte views (CPU tensors OK)."""

    def test_contiguous_tensor(self):
        from modelexpress.vllm_loader import _storage_view
        t = torch.randn(4, 8)
        view = _storage_view(t)
        assert view.data_ptr() == t.data_ptr()
        assert view.numel() == t.numel() * t.element_size()

    def test_dense_noncontiguous(self):
        """A transposed tensor that densely covers its storage."""
        from modelexpress.vllm_loader import _storage_view
        t = torch.randn(4, 8).t()
        assert not t.is_contiguous()
        view = _storage_view(t)
        assert view.data_ptr() == t.data_ptr()

    def test_partial_view_gets_full_storage(self):
        """A partial view (like MLA's W_UV) gets the full storage block."""
        from modelexpress.vllm_loader import _storage_view
        full = torch.randn(8, 8)
        partial = full[:4]
        view = _storage_view(partial)
        assert view.data_ptr() == full.data_ptr()
        assert view.numel() == full.numel() * full.element_size()

    def test_shared_storage_same_data_ptr(self):
        """Two views into the same storage produce views with the same data_ptr."""
        from modelexpress.vllm_loader import _storage_view
        full = torch.randn(8, 8)
        view_a = _storage_view(full[:4])
        view_b = _storage_view(full[4:])
        assert view_a.data_ptr() == view_b.data_ptr()


@_skip_no_cuda
class TestCollectModuleTensorsStorageViews:
    """Verify that _collect_module_tensors handles non-contiguous tensors
    via storage views instead of silently dropping them."""

    def test_noncontiguous_tensor_included(self):
        from modelexpress.vllm_loader import _collect_module_tensors

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                raw = torch.randn(4, 8, device="cuda")
                self.scale = nn.Parameter(raw.t())
        m = M()
        assert not m.scale.is_contiguous()
        result = _collect_module_tensors(m)
        assert "scale.__storage" in result
        assert result["scale.__storage"].data_ptr() == m.scale.data_ptr()

    def test_contiguous_tensor_registered_directly(self):
        from modelexpress.vllm_loader import _collect_module_tensors

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 8, device="cuda"))
        m = M()
        result = _collect_module_tensors(m)
        assert "weight" in result
        assert "weight.__storage" not in result

    def test_shared_storage_deduped(self):
        """Two views into the same storage are registered once."""
        from modelexpress.vllm_loader import _collect_module_tensors

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # Simulate MLA: two views from same dequantized intermediate
                full = torch.randn(8, 8, device="cuda")
                self.w_uv = nn.Parameter(full[:4].t())
                self.w_uk_t = nn.Parameter(full[4:].permute(1, 0))
        m = M()
        result = _collect_module_tensors(m)
        # Both share storage, so only one __storage entry
        storage_keys = [k for k in result if ".__storage" in k]
        assert len(storage_keys) == 1

    def test_source_target_name_symmetry(self):
        """Both source and target produce the same tensor names."""
        from modelexpress.vllm_loader import _collect_module_tensors

        def make_model():
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = nn.Parameter(torch.randn(4, 8, device="cuda"))
                    self.scale = nn.Parameter(torch.randn(4, 8, device="cuda").t())
            return M()

        source = _collect_module_tensors(make_model())
        target = _collect_module_tensors(make_model())
        assert set(source.keys()) == set(target.keys())
