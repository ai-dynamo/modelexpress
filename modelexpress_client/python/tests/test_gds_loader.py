# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GDS loader and transfer manager."""

import json
import os
import struct
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from modelexpress.accelerators import CudaAcceleratorBackend
from modelexpress.adapter import EngineAdapter, StrategyFailed
from modelexpress.load_strategy.context import LoadResult


# ---------------------------------------------------------------------------
# GDS availability detection
# ---------------------------------------------------------------------------


class TestIsGdsAvailable:
    """Tests for system-level GDS detection."""

    @patch("modelexpress.gds_transfer.NIXL_AVAILABLE", False)
    def test_nixl_not_installed(self):
        from modelexpress.gds_transfer import is_gds_available
        assert is_gds_available() is False

    @patch("modelexpress.gds_transfer.NIXL_AVAILABLE", True)
    @patch("modelexpress.gds_transfer._nvidia_fs_loaded", return_value=False)
    def test_no_nvidia_fs_module(self, _mock_fs):
        from modelexpress.gds_transfer import is_gds_available
        assert is_gds_available() is False

    @patch("modelexpress.gds_transfer.NIXL_AVAILABLE", True)
    @patch("modelexpress.gds_transfer._nvidia_fs_loaded", return_value=True)
    @patch("modelexpress.gds_transfer._cufile_loadable", return_value=False)
    def test_no_libcufile(self, _mock_cufile, _mock_fs):
        from modelexpress.gds_transfer import is_gds_available
        assert is_gds_available() is False

    @patch("modelexpress.gds_transfer.NIXL_AVAILABLE", True)
    @patch("modelexpress.gds_transfer._nvidia_fs_loaded", return_value=True)
    @patch("modelexpress.gds_transfer._cufile_loadable", return_value=True)
    def test_all_present(self, _mock_cufile, _mock_fs):
        from modelexpress.gds_transfer import is_gds_available
        assert is_gds_available() is True


# ---------------------------------------------------------------------------
# GdsTransferManager
# ---------------------------------------------------------------------------


class TestGdsTransferManager:
    """Tests for the GdsTransferManager class."""

    def test_not_available_raises(self):
        with patch("modelexpress.gds_transfer.NIXL_AVAILABLE", False):
            from modelexpress.gds_transfer import GdsTransferManager
            mgr = GdsTransferManager(
                agent_name="test", accelerator_backend=CudaAcceleratorBackend()
            )
            with pytest.raises(RuntimeError, match="not available"):
                mgr.initialize()

# ---------------------------------------------------------------------------
# GMS snapshot restore
# ---------------------------------------------------------------------------


class TestGmsSnapshotRestore:
    """Tests for restoring GMS file ranges into existing GPU addresses."""

    @staticmethod
    def _source(
        allocation_id="allocation-0",
        file_path="/snapshot/shard-0.bin",
        file_offset=0,
        byte_count=16,
    ):
        from modelexpress.gds_loader import MxFileReadSource

        return MxFileReadSource(
            allocation_id=allocation_id,
            file_path=file_path,
            file_offset=file_offset,
            byte_count=byte_count,
        )

    @staticmethod
    def _target(
        allocation_id="allocation-0",
        va=0x1000,
        device=0,
        byte_count=16,
    ):
        from modelexpress.gds_loader import MxDeviceReadTarget

        return MxDeviceReadTarget(
            allocation_id=allocation_id,
            va=va,
            device=device,
            byte_count=byte_count,
        )

    @staticmethod
    def _loader_with_manager():
        from modelexpress.gds_loader import MxGdsLoader

        loader = MxGdsLoader(CudaAcceleratorBackend())
        manager = MagicMock()
        loader._gds_manager = manager
        return loader, manager

    @staticmethod
    def _grouped_sources(sources, targets):
        grouped = {}
        for source in sources:
            grouped.setdefault(source.file_path, []).append(
                (source, targets[source.allocation_id])
            )
        for pairs in grouped.values():
            pairs.sort(key=lambda pair: pair[0].file_offset)
        return grouped

    def _restore(self, loader, *, sources, targets, device=None, **kwargs):
        if device is None:
            device = next(iter(targets.values())).device if targets else 0
        return loader.restore_gms_snapshot(
            grouped_sources=self._grouped_sources(sources, targets),
            device=device,
            **kwargs,
        )

    @patch("modelexpress.gds_loader.is_gds_available", return_value=False)
    def test_empty_sources_return_clamped_stats_without_gds_probe(
        self, mock_available
    ):
        from modelexpress.gds_loader import MxGdsLoader

        loader = MxGdsLoader(CudaAcceleratorBackend())

        stats = self._restore(
            loader,
            sources=[],
            targets={},
            max_inflight_batches=0,
        )

        assert stats == {
            "total_bytes": 0,
            "elapsed_s": 0.0,
            "selected_strategy": "gds",
            "source_count": 0,
            "file_count": 0,
            "max_inflight_batches": 1,
        }
        mock_available.assert_not_called()

    def test_validates_positive_byte_count(self):
        loader, manager = self._loader_with_manager()
        source = self._source(byte_count=0)
        target = self._target(byte_count=0)

        with pytest.raises(RuntimeError, match="byte_count must be positive"):
            loader._build_read_requests(
                file_path=source.file_path,
                fd=10,
                file_size=4096,
                pairs=[(source, target)],
                chunk_size_bytes=None,
            )

        manager.prepare_read.assert_not_called()

    @pytest.mark.parametrize(
        ("source_overrides", "target_overrides", "message"),
        [
            ({"file_offset": 1}, {}, "file_offset"),
            ({}, {"va": 0x1001}, "target VA"),
        ],
    )
    def test_validates_gds_alignment(
        self,
        source_overrides,
        target_overrides,
        message,
    ):
        loader, manager = self._loader_with_manager()
        source = self._source(**source_overrides)
        target = self._target(**target_overrides)

        with pytest.raises(RuntimeError, match=message):
            loader._build_read_requests(
                file_path=source.file_path,
                fd=10,
                file_size=4096,
                pairs=[(source, target)],
                chunk_size_bytes=None,
            )

        manager.prepare_read.assert_not_called()

    def test_validates_positive_target_va(self):
        loader, manager = self._loader_with_manager()
        source = self._source()
        target = self._target(va=0)

        with pytest.raises(RuntimeError, match="target VA must be positive"):
            loader._build_read_requests(
                file_path=source.file_path,
                fd=10,
                file_size=4096,
                pairs=[(source, target)],
                chunk_size_bytes=None,
            )

        manager.prepare_read.assert_not_called()

    @pytest.mark.parametrize(
        ("chunk_size_bytes", "message"),
        [
            (0, "chunk_size_bytes must be positive"),
            (4097, "chunk_size_bytes must be a multiple of 4096"),
        ],
    )
    def test_validates_chunk_size_bytes(self, chunk_size_bytes, message):
        loader, manager = self._loader_with_manager()
        source = self._source()
        target = self._target()

        with pytest.raises(ValueError, match=message):
            loader._build_read_requests(
                file_path=source.file_path,
                fd=10,
                file_size=4096,
                pairs=[(source, target)],
                chunk_size_bytes=chunk_size_bytes,
            )

        manager.prepare_read.assert_not_called()

    def test_validates_non_negative_file_offset(self):
        loader, manager = self._loader_with_manager()
        source = self._source(file_offset=-4096)
        target = self._target()

        with pytest.raises(RuntimeError, match="file_offset must be non-negative"):
            loader._build_read_requests(
                file_path=source.file_path,
                fd=10,
                file_size=4096,
                pairs=[(source, target)],
                chunk_size_bytes=None,
            )

        manager.prepare_read.assert_not_called()

    @patch("modelexpress.gds_loader.torch.cuda.set_device")
    @patch("modelexpress.gds_loader.is_gds_available", return_value=True)
    def test_rejects_negative_device_argument_before_scanning_targets(
        self,
        _mock_available,
        mock_set_device,
    ):
        loader, manager = self._loader_with_manager()
        source = self._source()
        target = self._target()

        with (
            patch.object(loader, "_open_gds_file") as mock_open_file,
            pytest.raises(
                RuntimeError,
                match="GMS transfer device must be non-negative: -1",
            ),
        ):
            self._restore(
                loader,
                sources=[source],
                targets={source.allocation_id: target},
                device=-1,
            )

        mock_set_device.assert_not_called()
        mock_open_file.assert_not_called()
        manager.prepare_read.assert_not_called()

    @patch("modelexpress.gds_loader.torch.cuda.set_device")
    @patch("modelexpress.gds_loader.is_gds_available", return_value=True)
    def test_rejects_mixed_target_devices_before_starting_io(
        self,
        _mock_available,
        mock_set_device,
    ):
        loader, manager = self._loader_with_manager()
        sources = [
            self._source("allocation-0", "/snapshot/a.bin"),
            self._source("allocation-1", "/snapshot/b.bin"),
        ]
        targets = {
            "allocation-0": self._target("allocation-0", device=0),
            "allocation-1": self._target("allocation-1", device=1),
        }

        with (
            patch.object(loader, "_open_gds_file") as mock_open_file,
            pytest.raises(RuntimeError, match="expected=0 got=1"),
        ):
            self._restore(loader, sources=sources, targets=targets, device=0)

        mock_set_device.assert_not_called()
        mock_open_file.assert_not_called()
        manager.prepare_read.assert_not_called()

    @patch("modelexpress.gds_loader.os.close")
    @patch("modelexpress.gds_loader.os.fstat")
    @patch("modelexpress.gds_loader.torch.cuda.set_device")
    @patch("modelexpress.gds_loader.is_gds_available", return_value=True)
    def test_rejects_read_beyond_eof_and_closes_file(
        self,
        _mock_available,
        _mock_set_device,
        mock_fstat,
        mock_close,
    ):
        loader, manager = self._loader_with_manager()
        mock_fstat.return_value.st_size = 8191

        with (
            patch.object(loader, "_open_gds_file", return_value=10),
            pytest.raises(RuntimeError, match="GDS read beyond EOF") as exc_info,
        ):
            self._restore(
                loader,
                sources=[self._source(file_offset=4096, byte_count=4096)],
                targets={
                    "allocation-0": self._target(byte_count=4096),
                },
            )

        message = str(exc_info.value)
        assert "allocation-0" in message
        assert "path=/snapshot/shard-0.bin" in message
        assert "end=8192" in message
        assert "file_size=8191" in message
        manager.prepare_read.assert_not_called()
        mock_close.assert_called_once_with(10)

    @patch("modelexpress.gds_loader.os.close")
    @patch("modelexpress.gds_loader.os.fstat")
    @patch("modelexpress.gds_loader.torch.cuda.set_device")
    @patch("modelexpress.gds_loader.is_gds_available", return_value=True)
    def test_success_constructs_manager_after_selecting_target_device(
        self,
        _mock_available,
        mock_torch_set_device,
        mock_fstat,
        _mock_close,
    ):
        from modelexpress.gds_loader import MxGdsLoader
        from modelexpress.gds_transfer import GdsReadTransfer

        events = []
        backend = MagicMock()
        loader = MxGdsLoader(backend)
        manager = MagicMock()
        transfer = GdsReadTransfer(
            handle=object(),
            file_descs=object(),
            vram_descs=object(),
            label="gms:shard-0.bin",
            request_count=1,
            total_bytes=16,
        )
        manager.prepare_read.return_value = transfer
        mock_fstat.return_value.st_size = 4096

        def select_device(device_id):
            events.append(("select-device", device_id))

        def construct_manager(*, agent_name, accelerator_backend):
            events.append(("construct-manager", loader._device_id))
            assert agent_name.startswith("mx-gds-2-")
            assert accelerator_backend is backend
            return manager

        backend.set_device.side_effect = select_device
        mock_torch_set_device.side_effect = select_device
        manager.initialize.side_effect = lambda: events.append(
            ("initialize-manager", loader._device_id)
        )

        with (
            patch(
                "modelexpress.gds_loader.GdsTransferManager",
                side_effect=construct_manager,
            ) as mock_manager_class,
            patch.object(loader, "_open_gds_file", return_value=10),
        ):
            stats = self._restore(
                loader,
                sources=[self._source()],
                targets={"allocation-0": self._target(device=2)},
            )

        assert events[:3] == [
            ("select-device", 2),
            ("construct-manager", 2),
            ("initialize-manager", 2),
        ]
        assert backend.set_device.call_count + mock_torch_set_device.call_count == 1
        mock_manager_class.assert_called_once()
        manager.prepare_read.assert_called_once()
        manager.start.assert_called_once_with(transfer)
        manager.wait.assert_called_once_with(transfer)
        manager.release.assert_called_once_with(transfer)
        assert stats["selected_strategy"] == "gds"

    @patch("modelexpress.gds_loader.os.close")
    @patch("modelexpress.gds_loader.os.fstat")
    @patch("modelexpress.gds_loader.torch.cuda.set_device")
    @patch("modelexpress.gds_loader.is_gds_available", return_value=True)
    def test_chunks_large_ranges_when_chunk_size_is_set(
        self,
        _mock_available,
        _mock_set_device,
        mock_fstat,
        _mock_close,
    ):
        loader, manager = self._loader_with_manager()
        manager.prepare_read.return_value = MagicMock(label="gms:shard-0.bin")
        mock_fstat.return_value.st_size = 16384

        with patch.object(loader, "_open_gds_file", return_value=10):
            self._restore(
                loader,
                sources=[
                    self._source(file_offset=4096, byte_count=10240),
                ],
                targets={
                    "allocation-0": self._target(
                        va=0x1000,
                        byte_count=10240,
                    ),
                },
                chunk_size_bytes=4096,
            )

        requests = manager.prepare_read.call_args.args[0]
        assert [request.file_offset for request in requests] == [4096, 8192, 12288]
        assert [request.dst_addr for request in requests] == [
            0x1000,
            0x2000,
            0x3000,
        ]
        assert [request.byte_count for request in requests] == [4096, 4096, 2048]

    @patch("modelexpress.gds_loader.os.close")
    @patch("modelexpress.gds_loader.os.fstat")
    @patch("modelexpress.gds_loader.torch.cuda.set_device")
    @patch("modelexpress.gds_loader.is_gds_available", return_value=True)
    def test_opens_one_transfer_per_file_with_bounded_inflight_window(
        self,
        _mock_available,
        _mock_set_device,
        mock_fstat,
        mock_close,
    ):
        loader, manager = self._loader_with_manager()
        mock_fstat.return_value.st_size = 8192
        events = []
        transfers = []

        def prepare(requests, *, label):
            transfer = MagicMock(label=label)
            transfer.requests = requests
            transfers.append(transfer)
            return transfer

        manager.prepare_read.side_effect = prepare
        manager.start.side_effect = lambda transfer: events.append(
            ("start", transfer.label)
        )
        manager.wait.side_effect = lambda transfer: events.append(
            ("wait", transfer.label)
        )
        sources = [
            self._source("a0", "/snapshot/a.bin", 0, 16),
            self._source("a1", "/snapshot/a.bin", 4096, 16),
            self._source("b0", "/snapshot/b.bin", 0, 16),
            self._source("c0", "/snapshot/c.bin", 0, 16),
        ]
        targets = {
            source.allocation_id: self._target(
                source.allocation_id,
                va=0x1000 + index * 4096,
            )
            for index, source in enumerate(sources)
        }

        with patch.object(
            loader,
            "_open_gds_file",
            side_effect=[10, 11, 12],
        ) as mock_open_file:
            stats = self._restore(
                loader,
                sources=sources,
                targets=targets,
                max_inflight_batches=2,
            )

        assert events == [
            ("start", "gms:a.bin"),
            ("start", "gms:b.bin"),
            ("wait", "gms:a.bin"),
            ("start", "gms:c.bin"),
            ("wait", "gms:b.bin"),
            ("wait", "gms:c.bin"),
        ]
        mock_open_file.assert_has_calls(
            [
                call("/snapshot/a.bin"),
                call("/snapshot/b.bin"),
                call("/snapshot/c.bin"),
            ]
        )
        assert manager.prepare_read.call_count == 3
        assert len(transfers[0].requests) == 2
        assert manager.release.call_args_list == [
            call(transfer) for transfer in transfers
        ]
        assert mock_close.call_args_list == [call(10), call(11), call(12)]
        assert stats["elapsed_s"] >= 0
        assert {key: value for key, value in stats.items() if key != "elapsed_s"} == {
            "total_bytes": 64,
            "selected_strategy": "gds",
            "source_count": 4,
            "file_count": 3,
            "max_inflight_batches": 2,
        }

    @patch("modelexpress.gds_loader.os.close")
    @patch("modelexpress.gds_loader.os.fstat")
    @patch("modelexpress.gds_loader.torch.cuda.set_device")
    @patch("modelexpress.gds_loader.is_gds_available", return_value=True)
    def test_waits_then_releases_transfer_before_closing_file(
        self,
        _mock_available,
        _mock_set_device,
        mock_fstat,
        mock_close,
    ):
        loader, manager = self._loader_with_manager()
        transfer = MagicMock(label="gms:shard-0.bin")
        manager.prepare_read.return_value = transfer
        mock_fstat.return_value.st_size = 1024
        events = []
        manager.wait.side_effect = lambda current: events.append(("wait", current))
        manager.release.side_effect = lambda current: events.append(
            ("release", current)
        )
        mock_close.side_effect = lambda fd: events.append(("close", fd))

        with patch.object(loader, "_open_gds_file", return_value=10):
            self._restore(
                loader,
                sources=[self._source()],
                targets={"allocation-0": self._target()},
            )

        assert events == [
            ("wait", transfer),
            ("release", transfer),
            ("close", 10),
        ]

    @patch("modelexpress.gds_loader.os.close")
    @patch("modelexpress.gds_loader.os.fstat")
    @patch("modelexpress.gds_loader.torch.cuda.set_device")
    @patch("modelexpress.gds_loader.is_gds_available", return_value=True)
    def test_prepare_failure_closes_current_file_and_drains_pending_transfer(
        self,
        _mock_available,
        _mock_set_device,
        mock_fstat,
        mock_close,
    ):
        loader, manager = self._loader_with_manager()
        pending = MagicMock(label="gms:a.bin")
        manager.prepare_read.side_effect = [pending, RuntimeError("prepare failed")]
        mock_fstat.return_value.st_size = 1024
        events = []
        manager.wait.side_effect = lambda transfer: events.append(("wait", transfer))
        manager.release.side_effect = lambda transfer: events.append(
            ("release", transfer)
        )
        mock_close.side_effect = lambda fd: events.append(("close", fd))
        sources = [
            self._source("a0", "/snapshot/a.bin"),
            self._source("b0", "/snapshot/b.bin"),
        ]
        targets = {
            source.allocation_id: self._target(source.allocation_id)
            for source in sources
        }

        with (
            patch.object(loader, "_open_gds_file", side_effect=[10, 11]),
            pytest.raises(RuntimeError, match="prepare failed"),
        ):
            self._restore(
                loader,
                sources=sources,
                targets=targets,
                max_inflight_batches=2,
            )

        assert events == [
            ("close", 11),
            ("release", pending),
            ("close", 10),
        ]

    @patch("modelexpress.gds_loader.os.close")
    @patch("modelexpress.gds_loader.os.fstat")
    @patch("modelexpress.gds_loader.torch.cuda.set_device")
    @patch("modelexpress.gds_loader.is_gds_available", return_value=True)
    def test_releases_current_and_pending_transfers_on_start_failure(
        self,
        _mock_available,
        _mock_set_device,
        mock_fstat,
        mock_close,
    ):
        loader, manager = self._loader_with_manager()
        mock_fstat.return_value.st_size = 1024
        transfers = [MagicMock(label=f"gms:{index}.bin") for index in range(2)]
        manager.prepare_read.side_effect = transfers
        manager.start.side_effect = [None, RuntimeError("start failed")]
        sources = [
            self._source(f"a{index}", f"/snapshot/{index}.bin")
            for index in range(2)
        ]
        targets = {
            source.allocation_id: self._target(source.allocation_id)
            for source in sources
        }

        with (
            patch.object(loader, "_open_gds_file", side_effect=[10, 11]),
            pytest.raises(RuntimeError, match="start failed"),
        ):
            self._restore(
                loader,
                sources=sources,
                targets=targets,
                max_inflight_batches=2,
            )

        manager.wait.assert_not_called()
        assert manager.release.call_args_list == [
            call(transfers[1]),
            call(transfers[0]),
        ]
        assert mock_close.call_args_list == [call(11), call(10)]

    @patch("modelexpress.gds_loader.os.close")
    @patch("modelexpress.gds_loader.os.fstat")
    @patch("modelexpress.gds_loader.torch.cuda.set_device")
    @patch("modelexpress.gds_loader.is_gds_available", return_value=True)
    def test_pending_cleanup_failure_does_not_skip_remaining_transfers(
        self,
        _mock_available,
        _mock_set_device,
        mock_fstat,
        mock_close,
    ):
        loader, manager = self._loader_with_manager()
        transfers = [MagicMock(label=f"gms:{index}.bin") for index in range(2)]
        original_error = RuntimeError("prepare failed")
        cleanup_error = RuntimeError("release failed")
        manager.prepare_read.side_effect = [*transfers, original_error]
        mock_fstat.return_value.st_size = 1024
        events = []

        def release(transfer):
            events.append(("release", transfer))
            if transfer is transfers[0]:
                raise cleanup_error

        manager.release.side_effect = release
        mock_close.side_effect = lambda fd: events.append(("close", fd))
        sources = [
            self._source(f"a{index}", f"/snapshot/{index}.bin")
            for index in range(3)
        ]
        targets = {
            source.allocation_id: self._target(source.allocation_id)
            for source in sources
        }

        with (
            patch.object(loader, "_open_gds_file", side_effect=[10, 11, 12]),
            patch("modelexpress.gds_loader.logger.warning") as mock_warning,
            pytest.raises(
                RuntimeError,
                match="GMS snapshot GDS restore failed: prepare failed",
            ) as exc_info,
        ):
            self._restore(
                loader,
                sources=sources,
                targets=targets,
                max_inflight_batches=3,
            )

        assert exc_info.value.__cause__ is original_error
        assert events == [
            ("close", 12),
            ("release", transfers[0]),
            ("close", 10),
            ("release", transfers[1]),
            ("close", 11),
        ]
        mock_warning.assert_called_once()

    @patch("modelexpress.gds_loader.os.close")
    def test_wait_failure_releases_transfer_and_closes_fd(self, mock_close):
        loader, manager = self._loader_with_manager()
        transfer = MagicMock(label="gms:shard-0.bin")
        wait_error = RuntimeError("wait failed")
        manager.wait.side_effect = wait_error

        with pytest.raises(RuntimeError, match="wait failed") as exc_info:
            loader._wait_and_release(transfer, 10)

        assert exc_info.value is wait_error
        manager.release.assert_called_once_with(transfer)
        mock_close.assert_called_once_with(10)

    @patch("modelexpress.gds_loader.os.close")
    def test_wait_error_is_preserved_when_release_raises(self, mock_close):
        loader, manager = self._loader_with_manager()
        transfer = MagicMock(label="gms:shard-0.bin")
        wait_error = RuntimeError("wait failed")
        manager.wait.side_effect = wait_error
        manager.release.side_effect = RuntimeError("release failed")

        with (
            patch("modelexpress.gds_loader.logger.warning") as mock_warning,
            pytest.raises(RuntimeError, match="wait failed") as exc_info,
        ):
            loader._wait_and_release(transfer, 10)

        assert exc_info.value is wait_error
        manager.release.assert_called_once_with(transfer)
        mock_close.assert_called_once_with(10)
        mock_warning.assert_called_once()

# ---------------------------------------------------------------------------
# Direct file opening
# ---------------------------------------------------------------------------


class TestOpenGdsFile:
    def test_uses_read_only_and_direct_flags(self):
        from modelexpress.gds_loader import MxGdsLoader

        direct_flag = 0x4000
        with (
            patch("modelexpress.gds_loader.os.O_DIRECT", direct_flag, create=True),
            patch("modelexpress.gds_loader.os.open", return_value=17) as mock_os_open,
        ):
            fd = MxGdsLoader._open_gds_file("/snapshot/shard.bin")

        assert fd == 17
        mock_os_open.assert_called_once_with(
            "/snapshot/shard.bin",
            os.O_RDONLY | direct_flag,
        )

    def test_wraps_open_error_with_path_and_preserves_cause(self):
        from modelexpress.gds_loader import MxGdsLoader

        open_error = OSError("permission denied")
        with (
            patch("modelexpress.gds_loader.os.O_DIRECT", 0x4000, create=True),
            patch("modelexpress.gds_loader.os.open", side_effect=open_error),
            pytest.raises(RuntimeError, match="failed to open") as exc_info,
        ):
            MxGdsLoader._open_gds_file("/snapshot/shard.bin")

        assert "/snapshot/shard.bin" in str(exc_info.value)
        assert exc_info.value.__cause__ is open_error


# ---------------------------------------------------------------------------
# Safetensors header parsing
# ---------------------------------------------------------------------------


class TestParseSafetensorsHeader:
    """Tests for safetensors header parsing."""

    def test_parses_header(self, tmp_path):
        from modelexpress.gds_loader import MxGdsLoader

        header = {
            "weight": {
                "dtype": "F32",
                "shape": [4, 2],
                "data_offsets": [0, 32],
            }
        }
        header_bytes = json.dumps(header).encode()
        file_path = tmp_path / "test.safetensors"
        with open(file_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * 32)

        loader = MxGdsLoader(CudaAcceleratorBackend())
        parsed = loader._parse_safetensors_header(str(file_path))
        assert "weight" in parsed
        assert parsed["weight"]["dtype"] == "F32"
        assert parsed["weight"]["shape"] == [4, 2]
        assert parsed["weight"]["size"] == 32
        assert parsed["weight"]["file_offset"] == 8 + len(header_bytes)


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------


class TestResolveModelPath:
    """Tests for model path resolution."""

    @patch("huggingface_hub.snapshot_download")
    def test_hf_model_calls_snapshot_download(self, mock_download):
        from modelexpress.gds_loader import MxGdsLoader
        mock_download.return_value = "/cache/models/org/model"
        result = MxGdsLoader._resolve_model_path("org/model")
        mock_download.assert_called_once_with("org/model", revision=None)
        assert result == "/cache/models/org/model"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


class TestResolveSafetensorsFiles:
    """Tests for safetensors file discovery."""

    def test_sharded_index(self, tmp_path):
        from modelexpress.gds_loader import MxGdsLoader

        # Create index file
        index = {
            "weight_map": {
                "layer.0.weight": "model-00001.safetensors",
                "layer.1.weight": "model-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        loader = MxGdsLoader(CudaAcceleratorBackend())
        result = loader._resolve_safetensors_files(str(tmp_path))
        assert len(result) == 2

    def test_no_files_raises(self, tmp_path):
        from modelexpress.gds_loader import MxGdsLoader
        loader = MxGdsLoader(CudaAcceleratorBackend())
        with pytest.raises(FileNotFoundError, match=r"No \.safetensors"):
            loader._resolve_safetensors_files(str(tmp_path))


# ---------------------------------------------------------------------------
# GdsStrategy integration
# ---------------------------------------------------------------------------


class TestGdsStrategyIntegration:
    """Tests for GdsStrategy loading behavior."""

    class _FakeAdapter(EngineAdapter):
        def discover_tensors(self, result: LoadResult):
            return {}

        def after_weight_iter_load(self, result: LoadResult):
            return result

        def apply_weight_iter(self, result: LoadResult, weights_iter):
            if result.model is not None:
                result.model.load_weights(weights_iter)
            return result

    def _make_context(self):
        from modelexpress.load_strategy import LoadContext
        return LoadContext(
            model_config=MagicMock(),
            load_config=MagicMock(),
            target_device=torch.device("cpu"),
            global_rank=0,
            worker_rank=0,
            device_id=0,
            identity=MagicMock(),
            mx_client=MagicMock(),
            worker_id="test-worker",
            adapter=self._FakeAdapter(),
        )

    @patch("modelexpress.gds_transfer.is_gds_available", return_value=True)
    @patch("modelexpress.gds_loader.MxGdsLoader")
    @patch("modelexpress.load_strategy.base.publish_metadata_and_ready")
    @patch("modelexpress.load_strategy.base.is_nixl_available", return_value=False)
    def test_gds_success(self, _mock_nixl, _mock_pub, mock_gds_cls, _mock_avail):
        from modelexpress.load_strategy.gds_strategy import GdsStrategy

        mock_gds = MagicMock()
        mock_gds.load_iter.return_value = iter([("w", torch.zeros(1))])
        mock_gds_cls.return_value = mock_gds

        ctx = self._make_context()
        ctx.model_config.model = "test-model"

        strategy = GdsStrategy()
        model = MagicMock()
        result = strategy.load(model, ctx)

        assert isinstance(result, LoadResult)
        assert result.model is model
        mock_gds.load_iter.assert_called_once()
        model.load_weights.assert_called_once()
        mock_gds.shutdown.assert_called_once()

    @patch("modelexpress.gds_transfer.is_gds_available", return_value=True)
    @patch("modelexpress.gds_loader.MxGdsLoader")
    def test_gds_failure_raises_strategy_failed(self, mock_gds_cls, _mock_avail):
        from modelexpress.load_strategy.gds_strategy import GdsStrategy

        mock_gds = MagicMock()
        mock_gds.load_iter.side_effect = RuntimeError("GDS error")
        mock_gds_cls.return_value = mock_gds

        ctx = self._make_context()
        ctx.model_config.model = "test-model"

        strategy = GdsStrategy()
        with pytest.raises(StrategyFailed, match="GDS error") as exc:
            strategy.load(MagicMock(), ctx)

        assert exc.value.mutated is False
        mock_gds.shutdown.assert_called_once()

    @patch("modelexpress.gds_transfer.is_gds_available", return_value=True)
    @patch("modelexpress.gds_loader.MxGdsLoader")
    def test_gds_apply_weight_iter_failure_is_mutated(self, mock_gds_cls, _mock_avail):
        from modelexpress.load_strategy.gds_strategy import GdsStrategy

        mock_gds = MagicMock()
        mock_gds.load_iter.return_value = iter([("w", torch.zeros(1))])
        mock_gds_cls.return_value = mock_gds

        ctx = self._make_context()
        ctx.adapter.apply_weight_iter = MagicMock(side_effect=RuntimeError("partial load"))
        ctx.model_config.model = "test-model"

        strategy = GdsStrategy()
        with pytest.raises(StrategyFailed, match="partial load") as exc:
            strategy.load(MagicMock(), ctx)

        assert exc.value.mutated is True
        mock_gds.shutdown.assert_called_once()

    @patch("modelexpress.gds_transfer.is_gds_available", return_value=False)
    def test_gds_not_available(self, _mock_avail):
        from modelexpress.load_strategy.gds_strategy import GdsStrategy

        ctx = self._make_context()
        strategy = GdsStrategy()
        assert strategy.is_available(ctx) is False
