# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for raw-address GDS read transfers."""

from unittest.mock import MagicMock, call, patch

import pytest

from modelexpress.gds_transfer import (
    GdsReadRequest,
    GdsReadTransfer,
    GdsTransferManager,
)


def _make_manager(agent=None):
    manager = GdsTransferManager(
        agent_name="test-gds-agent",
        accelerator_backend=MagicMock(),
    )
    if agent is None:
        agent = MagicMock()
    agent.name = "test-gds-agent"
    manager._agent = agent
    return manager


def _request(**overrides):
    values = {
        "fd": 10,
        "file_offset": 4096,
        "dst_addr": 0x1000,
        "byte_count": 4096,
        "device": 0,
        "label": "allocation-0",
    }
    values.update(overrides)
    return GdsReadRequest(**values)


def _context_request(**overrides):
    return _request(
        label=(
            "allocation_id=allocation-0 "
            "file_path=/snapshot/shard-0.bin"
        ),
        **overrides,
    )


def _assert_request_context(error):
    message = str(error)
    for expected in (
        "allocation_id=allocation-0",
        "file_path=/snapshot/shard-0.bin",
        "file_offset=4096",
        "byte_count=4096",
        "dst_addr=4096",
        "device=0",
    ):
        assert expected in message


class FatalTransferError(BaseException):
    pass


def _configure_prepare(agent):
    file_descs = MagicMock(name="file_descs")
    file_descs.trim.return_value = "trimmed-file-descs"
    vram_descs = MagicMock(name="vram_descs")
    vram_descs.trim.return_value = "trimmed-vram-descs"
    handle = object()
    agent.register_memory.side_effect = [file_descs, vram_descs]
    agent.initialize_xfer.return_value = handle
    return file_descs, vram_descs, handle


def _prepare_context_transfer(manager, agent):
    _configure_prepare(agent)
    return manager.prepare_read([_context_request()], label="snapshot-file")


class TestInitialize:
    def test_constructs_gds_agent_after_selecting_accelerator_device(
        self, monkeypatch
    ):
        events = []
        backend = MagicMock()

        def current_device():
            events.append("current_device")
            return 3

        config = object()
        agent = object()

        def make_config(*, backends, num_threads):
            events.append(("config", backends, num_threads))
            return config

        def make_agent(name, agent_config):
            events.append(("agent", name, agent_config))
            return agent

        backend.current_device.side_effect = current_device
        monkeypatch.setenv("MX_GDS_THREADS", "13")
        manager = GdsTransferManager(
            agent_name="test-gds-agent",
            accelerator_backend=backend,
        )

        with (
            patch("modelexpress.gds_transfer.NIXL_AVAILABLE", True),
            patch(
                "modelexpress.gds_transfer.NixlAgentConfig",
                side_effect=make_config,
            ) as config_factory,
            patch(
                "modelexpress.gds_transfer.NixlAgent",
                side_effect=make_agent,
            ) as agent_factory,
        ):
            manager.initialize()

        backend.current_device.assert_called_once_with()
        config_factory.assert_called_once_with(
            backends=["GDS_MT"],
            num_threads=13,
        )
        agent_factory.assert_called_once_with("test-gds-agent", config)
        assert events == [
            "current_device",
            ("config", ["GDS_MT"], 13),
            ("agent", "test-gds-agent", config),
        ]
        assert manager._device_id == 3
        assert manager._agent is agent


class TestPrepareRead:
    def test_requires_initialized_agent(self):
        manager = GdsTransferManager(
            agent_name="test-gds-agent",
            accelerator_backend=MagicMock(),
        )

        with pytest.raises(RuntimeError, match="GDS agent not initialized"):
            manager.prepare_read([_request()])

    def test_rejects_empty_requests(self):
        manager = _make_manager()

        with pytest.raises(ValueError, match="requests must not be empty"):
            manager.prepare_read([])

    def test_registration_error_includes_full_request_context(self):
        agent = MagicMock()
        manager = _make_manager(agent)
        agent.register_memory.side_effect = RuntimeError("registration failed")

        with pytest.raises(RuntimeError, match="registration failed") as exc_info:
            manager.prepare_read([_context_request()])

        _assert_request_context(exc_info.value)

    def test_registers_multiple_file_and_vram_descriptors(self):
        agent = MagicMock()
        manager = _make_manager(agent)
        file_descs, vram_descs, handle = _configure_prepare(agent)
        requests = [
            _request(),
            _request(
                fd=11,
                file_offset=12288,
                dst_addr=0x3000,
                byte_count=8192,
                device=1,
                label="allocation-1",
            ),
        ]

        transfer = manager.prepare_read(requests, label="snapshot-file")

        agent.register_memory.assert_has_calls(
            [
                call(
                    [
                        (4096, 4096, 10, ""),
                        (12288, 8192, 11, ""),
                    ],
                    "FILE",
                ),
                call(
                    [
                        (0x1000, 4096, 0, ""),
                        (0x3000, 8192, 1, ""),
                    ],
                    "VRAM",
                ),
            ]
        )
        agent.initialize_xfer.assert_called_once_with(
            "READ",
            "trimmed-vram-descs",
            "trimmed-file-descs",
            "test-gds-agent",
        )
        assert transfer.handle is handle
        assert transfer.file_descs is file_descs
        assert transfer.vram_descs is vram_descs
        assert transfer.label == "snapshot-file"
        assert transfer.request_count == 2
        assert transfer.total_bytes == 12288
        assert transfer.state == "INIT"
        assert transfer.released is False

    def test_releases_file_registration_when_vram_registration_fails(self):
        agent = MagicMock()
        manager = _make_manager(agent)
        file_descs = MagicMock(name="file_descs")
        agent.register_memory.side_effect = [
            file_descs,
            RuntimeError("VRAM registration failed"),
        ]

        with pytest.raises(RuntimeError, match="VRAM registration failed"):
            manager.prepare_read([_request()])

        agent.release_xfer_handle.assert_not_called()
        agent.deregister_memory.assert_called_once_with(file_descs)

    def test_releases_registrations_when_initialize_raises_base_exception(self):
        agent = MagicMock()
        manager = _make_manager(agent)
        file_descs = MagicMock(name="file_descs")
        vram_descs = MagicMock(name="vram_descs")
        agent.register_memory.side_effect = [file_descs, vram_descs]
        agent.initialize_xfer.side_effect = FatalTransferError(
            "initialize interrupted"
        )

        with pytest.raises(FatalTransferError, match="initialize interrupted"):
            manager.prepare_read([_request()])

        agent.release_xfer_handle.assert_not_called()
        agent.deregister_memory.assert_has_calls(
            [call(file_descs), call(vram_descs)]
        )


class TestReadTransferLifecycle:
    def test_start_posts_one_transfer_for_multiple_requests(self):
        agent = MagicMock()
        manager = _make_manager(agent)
        _, _, handle = _configure_prepare(agent)
        agent.transfer.return_value = "PROC"
        transfer = manager.prepare_read(
            [_request(), _request(file_offset=8192, dst_addr=0x2000)]
        )

        manager.start(transfer)

        agent.transfer.assert_called_once_with(handle)
        assert transfer.request_count == 2
        assert transfer.state == "PROC"

    def test_start_accepts_immediate_done_and_wait_does_not_poll(self):
        agent = MagicMock()
        manager = _make_manager(agent)
        transfer = _prepare_context_transfer(manager, agent)
        agent.transfer.return_value = "DONE"

        manager.start(transfer)
        manager.wait(transfer)

        assert transfer.state == "DONE"
        agent.check_xfer_state.assert_not_called()

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ("ERR", "failed to start"),
            ("UNKNOWN", "unexpected"),
        ],
    )
    def test_start_rejects_error_and_unexpected_states(self, state, expected):
        agent = MagicMock()
        manager = _make_manager(agent)
        transfer = _prepare_context_transfer(manager, agent)
        agent.transfer.return_value = state

        with pytest.raises(RuntimeError) as exc_info:
            manager.start(transfer)

        assert expected in str(exc_info.value)
        assert transfer.state != "INIT"

    def test_start_wraps_agent_exception_with_context_and_is_not_retryable(self):
        agent = MagicMock()
        manager = _make_manager(agent)
        transfer = _prepare_context_transfer(manager, agent)
        agent.transfer.side_effect = RuntimeError("post failed")

        with pytest.raises(RuntimeError, match="post failed") as exc_info:
            manager.start(transfer)

        assert "label=snapshot-file" in str(exc_info.value)
        assert transfer.state != "INIT"

        with pytest.raises(RuntimeError):
            manager.start(transfer)
        assert agent.transfer.call_count == 1

    def test_wait_polls_started_transfer_to_completion(self):
        agent = MagicMock()
        manager = _make_manager(agent)
        handle = object()
        transfer = GdsReadTransfer(
            handle=handle,
            file_descs=MagicMock(),
            vram_descs=MagicMock(),
            label="snapshot-file",
            request_count=2,
            total_bytes=8192,
            state="PROC",
        )
        agent.check_xfer_state.side_effect = ["PROC", "DONE"]

        manager.wait(transfer)

        assert agent.check_xfer_state.call_args_list == [call(handle), call(handle)]
        assert transfer.state == "DONE"

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ("ERR", "failed"),
            ("UNKNOWN", "unexpected"),
        ],
    )
    def test_wait_rejects_error_and_unexpected_states(self, state, expected):
        agent = MagicMock()
        manager = _make_manager(agent)
        transfer = _prepare_context_transfer(manager, agent)
        agent.transfer.return_value = "PROC"
        manager.start(transfer)
        agent.check_xfer_state.return_value = state

        with pytest.raises(RuntimeError) as exc_info:
            manager.wait(transfer)

        assert expected in str(exc_info.value)
        assert transfer.state != "PROC"

    def test_wait_wraps_agent_exception_with_context(self):
        agent = MagicMock()
        manager = _make_manager(agent)
        transfer = _prepare_context_transfer(manager, agent)
        agent.transfer.return_value = "PROC"
        manager.start(transfer)
        agent.check_xfer_state.side_effect = RuntimeError("poll failed")

        with pytest.raises(RuntimeError, match="poll failed") as exc_info:
            manager.wait(transfer)

        assert "label=snapshot-file" in str(exc_info.value)

    def test_wait_timeout(self, monkeypatch):
        agent = MagicMock()
        manager = _make_manager(agent)
        transfer = _prepare_context_transfer(manager, agent)
        agent.transfer.return_value = "PROC"
        manager.start(transfer)
        monkeypatch.setattr("modelexpress.gds_transfer._MX_GDS_TIMEOUT", 0.5)

        with (
            patch(
                "modelexpress.gds_transfer.time.perf_counter",
                side_effect=[10.0, 11.0],
            ),
            pytest.raises(TimeoutError, match=r"timeout after 0\.5s"),
        ):
            manager.wait(transfer)

        agent.check_xfer_state.assert_not_called()

    def test_release_releases_handle_and_registrations_once(self):
        agent = MagicMock()
        manager = _make_manager(agent)
        handle = object()
        file_descs = object()
        vram_descs = object()
        transfer = GdsReadTransfer(
            handle=handle,
            file_descs=file_descs,
            vram_descs=vram_descs,
            label="snapshot-file",
            request_count=2,
            total_bytes=8192,
        )

        manager.release(transfer)
        manager.release(transfer)

        agent.release_xfer_handle.assert_called_once_with(handle)
        assert agent.mock_calls == [
            call.release_xfer_handle(handle),
            call.deregister_memory(file_descs),
            call.deregister_memory(vram_descs),
        ]
        assert transfer.handle is None
        assert transfer.file_descs is None
        assert transfer.vram_descs is None
        assert transfer.released is True

    @pytest.mark.parametrize("operation", ["start", "wait"])
    def test_transfer_operations_reject_use_after_release(self, operation):
        agent = MagicMock()
        manager = _make_manager(agent)
        transfer = _prepare_context_transfer(manager, agent)
        manager.release(transfer)

        with pytest.raises(RuntimeError, match="released"):
            getattr(manager, operation)(transfer)

        agent.transfer.assert_not_called()
        agent.check_xfer_state.assert_not_called()
