# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GMS package imports and wiring."""

import pytest


class TestGmsImports:
    """Verify all new GMS modules are importable."""

    def test_import_config(self):
        from modelexpress.gms.config import (
            EngineType,
            MxConfig,
            GmsConfig,
            GmsMode,
            WeightSourceType,
        )

        assert EngineType.VLLM == "vllm"

    def test_import_mx_hooks(self):
        pytest.importorskip("torch")
        from modelexpress.gms.mx_hooks import (
            source_commit_gms,
            source_connect_gms,
            source_finalize,
            source_register_nixl,
            target_allocate,
            target_commit,
            target_receive,
        )

        assert callable(source_connect_gms)
        assert callable(source_register_nixl)
        assert callable(source_commit_gms)
        assert callable(source_finalize)
        assert callable(target_allocate)
        assert callable(target_receive)
        assert callable(target_commit)

    def test_import_vllm_launcher(self):
        pytest.importorskip("torch")
        from modelexpress.gms.launchers.vllm import run

        assert callable(run)

    def test_import_main(self):
        from modelexpress.gms.main import get_launcher, main, parse_args

        assert callable(main)
        assert callable(parse_args)
        assert callable(get_launcher)

    def test_import_weight_sources(self):
        from modelexpress.gms.weight_sources.disk import get_weights_iterator as disk_iter
        from modelexpress.gms.weight_sources.gds import get_weights_iterator as gds_iter
        from modelexpress.gms.weight_sources.s3 import get_weights_iterator as s3_iter

        assert callable(disk_iter)
        assert callable(gds_iter)
        assert callable(s3_iter)


class TestExistingImports:
    """Verify existing package imports still work."""

    def test_import_mx_client(self):
        from modelexpress import MxClient

        assert MxClient is not None

    def test_import_register_loaders(self):
        from modelexpress import register_modelexpress_loaders

        assert callable(register_modelexpress_loaders)

    def test_import_types(self):
        from modelexpress.types import (
            GetMetadataResponse,
            TensorDescriptor,
            WorkerMetadata,
        )

        assert TensorDescriptor is not None
