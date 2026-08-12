# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TrainerPullStrategy eligibility.

The strategy sits at the head of the chain and blocks for
MX_TRAINER_SYNC_TIMEOUT seconds waiting for a TrainerTable, so declaring
itself eligible in a deployment that has no trainer starves every strategy
behind it. Eligibility must therefore key on the trainer sync configuration,
not only on transport capability.
"""

from unittest.mock import MagicMock, patch

import torch

from modelexpress import p2p_pb2
from modelexpress.adapter import EngineAdapter
from modelexpress.load_strategy.context import LoadResult


_MODULE = "modelexpress.load_strategy.trainer_pull_strategy"


class _FakeAdapter(EngineAdapter):
    def discover_tensors(self, result: LoadResult):
        return {}

    def is_cuda_alike(self) -> bool:
        return True


def _make_load_context(**overrides):
    """Return a LoadContext with mocked dependencies."""
    from modelexpress.load_strategy import LoadContext

    accelerator_backend = MagicMock()
    accelerator_backend.supports_rdma_p2p.return_value = True

    defaults = dict(
        model_config=MagicMock(),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        device_id=0,
        identity=p2p_pb2.SourceIdentity(
            model_name="test-model",
            tensor_parallel_size=1,
        ),
        mx_client=MagicMock(),
        worker_id="test-worker",
        adapter=_FakeAdapter(),
        accelerator_backend=accelerator_backend,
    )
    defaults.update(overrides)
    return LoadContext(**defaults)


def _make_strategy():
    from modelexpress.load_strategy.trainer_pull_strategy import TrainerPullStrategy

    return TrainerPullStrategy()


class TestTrainerPullIsAvailable:
    """Transport is available in every case here; only the config varies."""

    def _assert_availability(self, env, expected):
        ctx = _make_load_context()
        strategy = _make_strategy()
        with patch.dict("os.environ", env, clear=True):
            with patch(f"{_MODULE}.is_nixl_available", return_value=True):
                assert strategy.is_available(ctx) is expected

    def test_unavailable_when_no_trainer_sync_configured(self):
        """The CI case: NIXL and RDMA present, no trainer anywhere."""
        self._assert_availability({}, False)

    def test_available_with_weight_sync_server(self):
        self._assert_availability({"MX_WEIGHT_SYNC_SERVER": "mx-server:8080"}, True)

    def test_available_with_trainer_table_key(self):
        """LocalPlanner deployments set no server, only the table key."""
        self._assert_availability({"MX_TRAINER_TABLE_KEY": "mx:trainer_table:m"}, True)

    def test_unavailable_when_weight_sync_server_disabled(self):
        """A falsy MX_WEIGHT_SYNC_SERVER is not a trainer sync configuration."""
        self._assert_availability({"MX_WEIGHT_SYNC_SERVER": "0"}, False)

    def test_unavailable_when_trainer_table_key_blank(self):
        self._assert_availability({"MX_TRAINER_TABLE_KEY": "   "}, False)

    def test_unavailable_without_nixl_even_when_configured(self):
        ctx = _make_load_context()
        strategy = _make_strategy()
        with patch.dict("os.environ", {"MX_WEIGHT_SYNC_SERVER": "mx:8080"}, clear=True):
            with patch(f"{_MODULE}.is_nixl_available", return_value=False):
                assert strategy.is_available(ctx) is False

    def test_unavailable_without_mx_client_even_when_configured(self):
        ctx = _make_load_context(mx_client=None)
        strategy = _make_strategy()
        with patch.dict("os.environ", {"MX_WEIGHT_SYNC_SERVER": "mx:8080"}, clear=True):
            with patch(f"{_MODULE}.is_nixl_available", return_value=True):
                assert strategy.is_available(ctx) is False


class TestTrainerPullRequiresRegistration:
    """register_tensors is best-effort at every other call site, and that is right
    there: a failure only costs P2P serving.  On this path the registered region is
    the destination of an inbound READ, so a silent skip means every transfer fails
    at prep with NIXL_ERR_NOT_FOUND and falls through to disk looking successful.
    """

    def _load(self, tensor_descriptors):
        ctx = _make_load_context()
        ctx.nixl_manager = MagicMock()
        ctx.nixl_manager.tensor_descriptors = tensor_descriptors
        strategy = _make_strategy()
        result = LoadResult(value=MagicMock(), model=MagicMock())

        with (
            patch.object(strategy, "_fetch_table", return_value=MagicMock()),
            patch(f"{_MODULE}.register_tensors"),
            patch(f"{_MODULE}.PullRole") as pull_role,
            patch(f"{_MODULE}.LocalPlanner"),
            patch.dict("os.environ", {}, clear=True),
        ):
            return strategy.load(result, ctx), pull_role

    def test_load_fails_when_registration_left_no_descriptors(self):
        """The whole point: refuse rather than read into unregistered memory."""
        from modelexpress.adapter import StrategyFailed

        import pytest

        with pytest.raises(StrategyFailed) as excinfo:
            self._load({})

        assert excinfo.value.mutated is False

    def test_load_fails_when_nixl_manager_is_absent(self):
        from modelexpress.adapter import StrategyFailed

        import pytest

        ctx = _make_load_context()
        ctx.nixl_manager = None
        strategy = _make_strategy()
        result = LoadResult(value=MagicMock(), model=MagicMock())

        with (
            patch.object(strategy, "_fetch_table", return_value=MagicMock()),
            patch(f"{_MODULE}._init_nixl_manager", return_value=None),
            patch(f"{_MODULE}.register_tensors"),
            patch(f"{_MODULE}.PullRole"),
            patch(f"{_MODULE}.LocalPlanner"),
            patch.dict("os.environ", {}, clear=True),
        ):
            with pytest.raises(StrategyFailed):
                strategy.load(result, ctx)

    def test_load_proceeds_once_descriptors_exist(self):
        """The guard must not fire on the healthy path."""
        _result, pull_role = self._load({"weight": object()})
        assert pull_role.called
