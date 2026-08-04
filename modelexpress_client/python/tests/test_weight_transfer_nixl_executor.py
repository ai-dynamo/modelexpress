# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Characterization tests for NixlExecutor, pinned at the NIXL agent boundary.

These assert the exact prep_xfer_dlist / make_prepped_xfer calls that reach the
agent, rather than anything about how NixlExecutor builds them. That is
deliberate: the executor is scheduled to be reimplemented on top of
NixlTransferManager.post_read_batch, and asserting at the agent boundary is what
makes these tests survive that change and prove it was faithful.

Two things here are load-bearing and enforced nowhere else in the codebase:
  - remote descriptors carry device id 0, hardcoded, because RdmaDescriptor has
    no field for the remote device;
  - local descriptors carry the device id the *executor* was built with, which
    equals the manager's own device id today only because TrainerPullStrategy
    passes the same ctx.device_id to both.

No NIXL, no GPU. The agent is a MagicMock throughout.

Run: pytest tests/test_weight_transfer_nixl_executor.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.weight_transfer.protocol.types import RdmaDescriptor
from modelexpress.weight_transfer.transport.nixl_executor import NixlExecutor


def _desc(agent_index: int, src: int, dst: int, nbytes: int) -> RdmaDescriptor:
    return RdmaDescriptor(
        agent_index=agent_index, src_addr=src, dst_addr=dst, nbytes=nbytes
    )


def _executor(mgr, device_id=0, remote_agents=None) -> NixlExecutor:
    return NixlExecutor(
        nixl_manager=mgr,
        remote_agents={0: "trainer0"} if remote_agents is None else remote_agents,
        device_id=device_id,
    )


@pytest.fixture
def backend(mock_accelerator_backend_cls):
    return mock_accelerator_backend_cls()


@pytest.fixture
def mgr(backend):
    manager = NixlTransferManager(
        agent_name="test", device_id=0, accelerator_backend=backend
    )
    manager._agent = MagicMock()
    manager._agent.check_xfer_state.return_value = "DONE"
    return manager


class TestReadCallArgs:
    def test_prep_and_xfer_args_are_exact(self, mgr, backend):
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]
        mgr._agent.make_prepped_xfer.return_value = "handle"

        total, _elapsed = _executor(mgr).execute(
            [_desc(0, 0x1000, 0x9000, 64), _desc(0, 0x2000, 0xA000, 128)],
            operation="READ",
        )

        assert total == 192
        assert mgr._agent.prep_xfer_dlist.call_args_list == [
            call(
                agent_name="trainer0",
                xfer_list=[(0x1000, 64, 0), (0x2000, 128, 0)],
                mem_type=backend.nixl_mem_type,
                backends=mgr._backends,
            ),
            call(
                agent_name="",
                xfer_list=[(0x9000, 64, 0), (0xA000, 128, 0)],
                mem_type=backend.nixl_mem_type,
                backends=mgr._backends,
            ),
        ]
        assert mgr._agent.make_prepped_xfer.call_args_list == [
            call(
                operation="READ",
                local_xfer_side="dst",
                local_indices=[0, 1],
                remote_xfer_side="src",
                remote_indices=[0, 1],
                backends=mgr._backends,
            )
        ]
        mgr._agent.transfer.assert_called_once_with("handle")
        mgr._agent.release_xfer_handle.assert_called_once_with("handle")

    def test_local_descriptors_use_the_executor_device_id(self, mgr):
        """The substitution a post_read_batch swap would silently make."""
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]

        _executor(mgr, device_id=3).execute([_desc(0, 0x1000, 0x9000, 64)])

        local_call = mgr._agent.prep_xfer_dlist.call_args_list[1]
        assert local_call.kwargs["xfer_list"] == [(0x9000, 64, 3)]

    def test_remote_descriptors_hardcode_device_zero(self, mgr):
        """RdmaDescriptor carries no remote device field; 0 is assumed."""
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]

        _executor(mgr, device_id=3).execute([_desc(0, 0x1000, 0x9000, 64)])

        remote_call = mgr._agent.prep_xfer_dlist.call_args_list[0]
        assert remote_call.kwargs["xfer_list"] == [(0x1000, 64, 0)]

    def test_device_is_synchronized_once_after_the_batch(self, mgr, backend):
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]

        _executor(mgr, device_id=3).execute([_desc(0, 0x1000, 0x9000, 64)])

        assert backend.synchronize_calls == [3]


class TestGroupingByRemoteAgent:
    def test_one_transfer_per_remote_agent(self, mgr):
        mgr._agent.prep_xfer_dlist.side_effect = ["s0", "d0", "s1", "d1"]
        mgr._agent.make_prepped_xfer.side_effect = ["h0", "h1"]

        total, _ = _executor(
            mgr, remote_agents={0: "trainer0", 1: "trainer1"}
        ).execute(
            [
                _desc(0, 0x1000, 0x9000, 64),
                _desc(1, 0x3000, 0xB000, 32),
                _desc(0, 0x2000, 0xA000, 16),
            ]
        )

        assert total == 112
        assert mgr._agent.make_prepped_xfer.call_count == 2
        # Descriptors keep their per-agent order, and indices are per batch.
        assert mgr._agent.prep_xfer_dlist.call_args_list[0].kwargs["xfer_list"] == [
            (0x1000, 64, 0),
            (0x2000, 16, 0),
        ]
        assert mgr._agent.prep_xfer_dlist.call_args_list[2].kwargs["xfer_list"] == [
            (0x3000, 32, 0)
        ]
        assert mgr._agent.make_prepped_xfer.call_args_list[0].kwargs[
            "local_indices"
        ] == [0, 1]
        assert mgr._agent.make_prepped_xfer.call_args_list[1].kwargs[
            "local_indices"
        ] == [0]

    def test_descriptor_for_unloaded_agent_is_skipped(self, mgr, caplog):
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]

        with caplog.at_level("WARNING"):
            total, _ = _executor(mgr, remote_agents={0: "trainer0"}).execute(
                [_desc(0, 0x1000, 0x9000, 64), _desc(7, 0x3000, 0xB000, 999)]
            )

        assert total == 64
        assert mgr._agent.make_prepped_xfer.call_count == 1
        assert "agent_index 7" in caplog.text


class TestDegenerateAndFailurePaths:
    def test_empty_descriptor_list_touches_no_agent(self, mgr):
        assert _executor(mgr).execute([]) == (0, 0.0)

        mgr._agent.prep_xfer_dlist.assert_not_called()
        mgr._agent.transfer.assert_not_called()

    def test_uninitialized_agent_raises(self, mgr):
        mgr._agent = None

        with pytest.raises(RuntimeError, match="NIXL agent not initialized"):
            _executor(mgr).execute([_desc(0, 0x1000, 0x9000, 64)])

    def test_every_handle_is_released_when_a_transfer_fails(self, mgr):
        mgr._agent.prep_xfer_dlist.side_effect = ["s0", "d0", "s1", "d1"]
        mgr._agent.make_prepped_xfer.side_effect = ["h0", "h1"]
        mgr._agent.transfer.side_effect = [None, RuntimeError("nixl exploded")]

        with pytest.raises(RuntimeError, match="nixl exploded"):
            _executor(mgr, remote_agents={0: "trainer0", 1: "trainer1"}).execute(
                [_desc(0, 0x1000, 0x9000, 64), _desc(1, 0x3000, 0xB000, 32)]
            )

        released = [c.args[0] for c in mgr._agent.release_xfer_handle.call_args_list]
        assert sorted(released) == ["h0", "h1"]

    def test_error_status_from_the_agent_raises(self, mgr):
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]
        mgr._agent.check_xfer_state.return_value = "ERR"

        with pytest.raises(RuntimeError, match="failed with status ERR"):
            _executor(mgr).execute([_desc(0, 0x1000, 0x9000, 64)])

        mgr._agent.release_xfer_handle.assert_called_once()
