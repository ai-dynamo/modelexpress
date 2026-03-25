# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared helpers and MxModelLoader detection logic."""

from unittest.mock import MagicMock, patch

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


def _make_worker(rank=0, worker_grpc_endpoint="10.0.0.1:60051"):
    return p2p_pb2.WorkerMetadata(
        worker_rank=rank,
        worker_grpc_endpoint=worker_grpc_endpoint,
        metadata_endpoint="10.0.0.1:5555",
        agent_name="mx-auto-worker0-abc12345",
    )


def _make_instance_ref(mx_source_id="abc123def456abcd", worker_id="inst-1", model_name="test-model", worker_rank=0):
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id,
        worker_id=worker_id,
        model_name=model_name,
        worker_rank=worker_rank,
    )


def _make_metadata_resp(found=True, rank=0, mx_source_id="abc123def456abcd", worker_id="inst-1"):
    worker = _make_worker(rank=rank, worker_grpc_endpoint="10.0.0.1:60051") if found else None
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

    def test_returns_none_when_worker_has_no_grpc_endpoint(self):
        loader = _make_loader()
        empty_worker = p2p_pb2.WorkerMetadata(worker_rank=0, worker_grpc_endpoint="")
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
        loader._mx_client.update_status.assert_called_once_with(
            mx_source_id=candidates[0].mx_source_id,
            worker_id="w-1",
            worker_rank=0,
            status=p2p_pb2.SOURCE_STATUS_STALE,
        )

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
        assert loader._mx_client.update_status.call_count == 3

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
    def test_calls_publish_then_update_status_ready(self):
        from modelexpress.vllm_loader import _publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.return_value = "abc123def456abcd"
        mx_client.update_status.return_value = True

        nixl_manager = MagicMock()
        nixl_manager.agent_name = "mx-auto-worker2-abc12345"

        tensors = {}
        for i in range(3):
            t = MagicMock(spec=torch.Tensor)
            t.data_ptr.return_value = 0x1000 + i * 1024
            t.numel.return_value = 256
            t.element_size.return_value = 2
            t.dtype = torch.bfloat16
            tensors[f"layer.{i}.weight"] = t

        identity = _make_identity("my-model")
        with patch("modelexpress.vllm_loader.WorkerGrpcServer") as mock_server_cls:
            mock_server = MagicMock()
            mock_server_cls.return_value = mock_server
            _publish_metadata_and_ready(
                mx_client, nixl_manager, tensors, global_rank=2, device_id=0,
                identity=identity, worker_id="inst-uuid",
                worker_grpc_endpoint="10.0.0.1:6555",
            )

        # Worker gRPC server should be started
        mock_server.start.assert_called_once()

        mx_client.publish_metadata.assert_called_once()
        call_args = mx_client.publish_metadata.call_args
        assert call_args.args[0] is identity
        assert call_args.args[2] == "inst-uuid"
        # Published worker should have worker_grpc_endpoint, not tensors
        worker_proto = call_args.args[1]
        assert worker_proto.worker_grpc_endpoint == "10.0.0.1:6555"

        mx_client.update_status.assert_called_once_with(
            mx_source_id="abc123def456abcd",
            worker_id="inst-uuid",
            worker_rank=2,
            status=p2p_pb2.SOURCE_STATUS_READY,
        )

    def test_update_status_failure_is_logged_not_raised(self):
        """update_status returning False should not propagate as an exception."""
        from modelexpress.vllm_loader import _publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.return_value = "abc123def456abcd"
        mx_client.update_status.return_value = False  # server rejected

        nixl_manager = MagicMock()
        nixl_manager.agent_name = "mx-auto-worker0-abc12345"

        identity = _make_identity()
        with patch("modelexpress.vllm_loader.WorkerGrpcServer"):
            _publish_metadata_and_ready(
                mx_client, nixl_manager, {}, global_rank=0, device_id=0,
                identity=identity, worker_id="w-1",
            )

    def test_publish_failure_raises(self):
        """publish_metadata raising should propagate out."""
        from modelexpress.vllm_loader import _publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.side_effect = RuntimeError("grpc error")

        nixl_manager = MagicMock()
        nixl_manager.agent_name = "mx-auto-worker0-abc12345"

        identity = _make_identity()
        with patch("modelexpress.vllm_loader.WorkerGrpcServer"):
            with pytest.raises(RuntimeError, match="grpc error"):
                _publish_metadata_and_ready(
                    mx_client, nixl_manager, {}, global_rank=0, device_id=0,
                    identity=identity, worker_id="w-1",
                )


# ---------------------------------------------------------------------------
# fetch_remote_and_wait
# ---------------------------------------------------------------------------


class TestFetchRemoteAndWait:
    """Tests for NixlTransferManager.fetch_remote_and_wait."""

    def test_timeout_raises(self):
        from modelexpress.nixl_transfer import NixlTransferManager

        manager = NixlTransferManager.__new__(NixlTransferManager)
        agent = MagicMock()
        agent.check_remote_metadata.return_value = False
        manager._agent = agent

        with pytest.raises(TimeoutError, match="Timed out"):
            manager.fetch_remote_and_wait(
                agent_name="remote-agent",
                ip="10.0.0.1",
                port=5555,
                timeout=0.05,
            )

        agent.fetch_remote_metadata.assert_called_once_with("remote-agent", "10.0.0.1", 5555)

    def test_successful_poll_returns_agent_name(self):
        from modelexpress.nixl_transfer import NixlTransferManager

        manager = NixlTransferManager.__new__(NixlTransferManager)
        agent = MagicMock()
        # Succeed on the second check
        agent.check_remote_metadata.side_effect = [False, True]
        manager._agent = agent

        result = manager.fetch_remote_and_wait(
            agent_name="remote-agent",
            ip="10.0.0.1",
            port=5555,
            timeout=5.0,
        )
        assert result == "remote-agent"


# ---------------------------------------------------------------------------
# _get_worker_host
# ---------------------------------------------------------------------------


class TestGetWorkerHost:
    """Tests for _get_worker_host env var priority."""

    def test_mx_worker_address_takes_priority(self):
        from modelexpress.vllm_loader import _get_worker_host

        with patch.dict("os.environ", {
            "MX_WORKER_ADDRESS": "explicit-host",
            "POD_IP": "10.0.0.99",
        }):
            assert _get_worker_host() == "explicit-host"

    def test_pod_ip_fallback(self):
        from modelexpress.vllm_loader import _get_worker_host

        env = {"POD_IP": "10.0.0.99"}
        with patch.dict("os.environ", env, clear=False):
            # Remove MX_WORKER_ADDRESS if present
            import os
            os.environ.pop("MX_WORKER_ADDRESS", None)
            assert _get_worker_host() == "10.0.0.99"

    @patch("socket.getfqdn", return_value="worker-node.local")
    def test_fqdn_fallback(self, _mock_fqdn):
        from modelexpress.vllm_loader import _get_worker_host

        import os
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("MX_WORKER_ADDRESS", None)
            os.environ.pop("POD_IP", None)
            assert _get_worker_host() == "worker-node.local"

    @patch("socket.getfqdn", return_value="localhost")
    def test_localhost_fqdn_raises(self, _mock_fqdn):
        from modelexpress.vllm_loader import _get_worker_host

        import os
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("MX_WORKER_ADDRESS", None)
            os.environ.pop("POD_IP", None)
            with pytest.raises(RuntimeError, match="routable worker address"):
                _get_worker_host()

    @patch("socket.getfqdn", return_value="localhost.localdomain")
    def test_localhost_localdomain_raises(self, _mock_fqdn):
        from modelexpress.vllm_loader import _get_worker_host

        import os
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("MX_WORKER_ADDRESS", None)
            os.environ.pop("POD_IP", None)
            with pytest.raises(RuntimeError, match="routable worker address"):
                _get_worker_host()
