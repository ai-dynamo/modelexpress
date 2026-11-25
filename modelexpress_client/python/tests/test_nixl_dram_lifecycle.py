# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DRAM READ ownership must outlive failed waits and failed cleanup."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress import nixl_transfer as nt
from modelexpress.load_strategy.base import clear_exception_tracebacks


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setenv("MX_NIXL_BACKEND", "UCX")
    value = nt.NixlTransferManager("dram-test", 0, accelerator_backend=MagicMock())
    value._agent = MagicMock()
    value._agent.get_agent_metadata.return_value = b"metadata"
    value._agent.check_xfer_state.return_value = "DONE"
    value._metadata = b"metadata"
    return value


def _receive(manager, buffer, **kwargs):
    return manager.receive_dram_into_buffer("source", 123, buffer, buffer.numel(), **kwargs)


@pytest.mark.parametrize("state", ["PROC", "ERR"])
def test_timeout_retains_resources_after_tracebacks_are_cleared(manager, state):
    buffer = torch.empty(8, dtype=torch.uint8)
    buffer_ref = weakref.ref(buffer)
    agent = manager._agent
    agent.check_xfer_state.return_value = state
    with pytest.raises(TimeoutError) as failure:
        _receive(manager, buffer, timeout_seconds=0)
    clear_exception_tracebacks(failure.value)
    del buffer
    gc.collect()

    assert buffer_ref() is not None
    agent.release_xfer_handle.assert_not_called()
    agent.release_dlist_handle.assert_not_called()
    with pytest.raises(nt.NixlReadCleanupError, match="incomplete"):
        manager.shutdown()
    assert manager._agent is agent
    agent.deregister_memory.assert_not_called()
    agent.remove_remote_agent.assert_not_called()

    agent.check_xfer_state.return_value = "DONE"
    manager.cleanup_dram_reads()
    gc.collect()
    assert buffer_ref() is None
    agent.release_xfer_handle.assert_called_once()
    assert agent.release_dlist_handle.call_count == 2


def test_success_releases_request_before_prepared_descriptors(manager):
    buffer = torch.empty(8, dtype=torch.uint8)
    _receive(manager, buffer)

    calls = [call[0] for call in manager._agent.mock_calls]
    assert calls.index("release_xfer_handle") < calls.index("release_dlist_handle")
    manager.cleanup_dram_reads()
    assert manager._agent.release_xfer_handle.call_count == 1


def test_release_failure_is_retained_and_retryable_after_completion(manager):
    buffer = torch.empty(8, dtype=torch.uint8)
    buffer_ref = weakref.ref(buffer)
    agent = manager._agent
    agent.release_xfer_handle.side_effect = RuntimeError("release refused")
    with pytest.raises(nt.NixlReadCleanupError, match="release refused") as failure:
        _receive(manager, buffer)
    clear_exception_tracebacks(failure.value)
    del buffer
    gc.collect()
    assert buffer_ref() is not None
    agent.release_dlist_handle.assert_not_called()

    with pytest.raises(nt.NixlReadCleanupError, match="release refused") as cleanup:
        manager.cleanup_dram_reads()
    clear_exception_tracebacks(cleanup.value)
    agent.release_xfer_handle.side_effect = None
    manager.cleanup_dram_reads()
    gc.collect()
    assert buffer_ref() is None


class NixlNotPostedError(Exception):
    pass


@pytest.mark.parametrize(
    "state", ["PROC", "ERR", RuntimeError("NIXL_ERR_NOT_POSTED"), NixlNotPostedError()]
)
def test_submit_exception_uses_native_request_state(manager, monkeypatch, state):
    monkeypatch.setattr(nt, "_NIXL_NOT_POSTED_ERRORS", (NixlNotPostedError,))
    agent = manager._agent
    agent.transfer.side_effect = RuntimeError("submit failed")
    if isinstance(state, Exception):
        agent.check_xfer_state.side_effect = state
    else:
        agent.check_xfer_state.return_value = state
    with pytest.raises(RuntimeError, match="submit failed"):
        _receive(manager, torch.empty(8, dtype=torch.uint8))
    agent.release_xfer_handle.assert_not_called()

    if isinstance(state, NixlNotPostedError):
        manager.cleanup_dram_reads()
        agent.release_xfer_handle.assert_called_once()
        assert agent.release_dlist_handle.call_count == 2

        agent.transfer.side_effect = None
        agent.check_xfer_state.side_effect = None
        agent.check_xfer_state.return_value = "DONE"
        _receive(manager, torch.empty(8, dtype=torch.uint8))
        assert agent.transfer.call_count == 2
    else:
        with pytest.raises(nt.NixlReadCleanupError):
            manager.cleanup_dram_reads()
        agent.release_xfer_handle.assert_not_called()
        agent.release_dlist_handle.assert_not_called()
        assert len(manager._pending_dram_reads) == 1


def test_partial_preparation_releases_only_created_descriptors(manager):
    descriptor = object()
    agent = manager._agent
    agent.prep_xfer_dlist.side_effect = [descriptor, RuntimeError("prepare failed")]
    with pytest.raises(RuntimeError, match="prepare failed"):
        _receive(manager, torch.empty(8, dtype=torch.uint8))
    manager.cleanup_dram_reads()
    agent.release_dlist_handle.assert_called_once_with(descriptor)
    agent.release_xfer_handle.assert_not_called()
    assert manager.is_healthy()


def test_descriptor_release_failure_does_not_rerelease_request(manager):
    agent = manager._agent
    agent.release_dlist_handle.side_effect = RuntimeError("descriptor busy")
    with pytest.raises(nt.NixlReadCleanupError, match="descriptor busy"):
        _receive(manager, torch.empty(8, dtype=torch.uint8))
    agent.release_dlist_handle.side_effect = None
    manager.cleanup_dram_reads()
    agent.release_xfer_handle.assert_called_once()


def test_registration_metadata_failure_keeps_buffer_for_shutdown(manager):
    buffer = torch.empty(8, dtype=torch.uint8)
    buffer_ref = weakref.ref(buffer)
    agent = manager._agent
    agent.get_agent_metadata.side_effect = RuntimeError("metadata failed")
    with pytest.raises(RuntimeError, match="metadata failed") as failure:
        manager.register_dram_buffer(buffer)
    clear_exception_tracebacks(failure.value)
    agent.reset_mock()
    del buffer
    gc.collect()
    assert buffer_ref() is not None

    manager.shutdown()
    agent.deregister_memory.assert_called_once_with(agent.register_memory.return_value)
    assert manager._agent is None
    gc.collect()
    assert buffer_ref() is None


def test_completed_deregistration_survives_metadata_refresh_failure(manager):
    buffer = torch.empty(8, dtype=torch.uint8)
    registered = manager.register_dram_buffer(buffer)
    agent = manager._agent
    agent.get_agent_metadata.side_effect = RuntimeError("metadata refresh failed")

    manager.deregister_memory(registered)

    assert not manager._dram_registrations
    assert manager.nixl_metadata == b""
    manager.shutdown()
    agent.deregister_memory.assert_called_once_with(registered)


def test_pending_read_blocks_registered_buffer_deregistration(manager):
    buffer = torch.empty(8, dtype=torch.uint8)
    registered = manager.register_dram_buffer(buffer)
    with pytest.raises(TimeoutError):
        _receive(manager, buffer, timeout_seconds=0)
    with pytest.raises(nt.NixlReadCleanupError, match="incomplete"):
        manager.deregister_memory(registered)
    manager._agent.deregister_memory.assert_not_called()


def test_deregistration_failure_keeps_agent_and_buffer(manager):
    buffer = torch.empty(8, dtype=torch.uint8)
    buffer_ref = weakref.ref(buffer)
    agent = manager._agent
    manager.register_dram_buffer(buffer)
    agent.reset_mock()
    del buffer
    agent.deregister_memory.side_effect = RuntimeError("deregister failed")
    with pytest.raises(nt.NixlReadCleanupError, match="deregister failed") as failure:
        manager.shutdown()
    clear_exception_tracebacks(failure.value)
    gc.collect()
    assert manager._agent is agent
    assert buffer_ref() is not None

    agent.deregister_memory.side_effect = None
    manager.shutdown()
    gc.collect()
    assert buffer_ref() is None


def test_failed_read_blocks_submissions_until_cleanup(manager):
    agent = manager._agent
    with pytest.raises(TimeoutError):
        _receive(manager, torch.empty(8, dtype=torch.uint8), timeout_seconds=0)

    with pytest.raises(nt.NixlReadCleanupError, match="timed out"):
        _receive(manager, torch.empty(8, dtype=torch.uint8))

    agent.transfer.assert_called_once()
    manager.cleanup_dram_reads()
    assert not manager.is_healthy()

    _receive(manager, torch.empty(8, dtype=torch.uint8))
    assert agent.transfer.call_count == 2
    assert manager.is_healthy()


def test_cleanup_checks_all_pending_reads_and_keeps_only_unfinished(manager, monkeypatch):
    posted = Barrier(2)
    handles = [object(), object()]
    agent = manager._agent
    agent.make_prepped_xfer.side_effect = handles
    original_wait = manager._wait_for_xfer

    def wait(handle, timeout, label):
        posted.wait(timeout=5)
        original_wait(handle, 0, label)

    monkeypatch.setattr(manager, "_wait_for_xfer", wait)
    with ThreadPoolExecutor(2) as executor:
        tasks = [
            executor.submit(_receive, manager, torch.empty(8, dtype=torch.uint8))
            for _ in handles
        ]
        for task in tasks:
            with pytest.raises(TimeoutError):
                task.result(timeout=5)

    agent.check_xfer_state.side_effect = ["PROC", "DONE"]
    with pytest.raises(nt.NixlReadCleanupError, match="incomplete"):
        manager.cleanup_dram_reads()
    assert agent.check_xfer_state.call_count == 2
    agent.release_xfer_handle.assert_called_once()
    agent.check_xfer_state.side_effect = None
    manager.cleanup_dram_reads()
    assert {call.args[0] for call in agent.release_xfer_handle.call_args_list} == set(handles)
