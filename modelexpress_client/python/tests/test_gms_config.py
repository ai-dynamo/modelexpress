# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GMS configuration models."""

import pickle

import pytest

from modelexpress.gms.config import (
    EngineType,
    MxConfig,
    GmsConfig,
    GmsMode,
    WeightSourceType,
)


class TestEnums:
    """Tests for configuration enums."""

    def test_engine_type_values(self):
        assert EngineType.VLLM == "vllm"
        assert EngineType.SGLANG == "sglang"
        assert EngineType.TRTLLM == "trtllm"

    def test_gms_mode_values(self):
        assert GmsMode.SOURCE == "source"
        assert GmsMode.TARGET == "target"

    def test_weight_source_type_values(self):
        assert WeightSourceType.DISK == "disk"
        assert WeightSourceType.GDS == "gds"
        assert WeightSourceType.S3 == "s3"
        assert WeightSourceType.MODEL_STREAMER == "model-streamer"

    def test_enum_creation_from_string(self):
        assert EngineType("vllm") == EngineType.VLLM
        assert GmsMode("source") == GmsMode.SOURCE
        assert WeightSourceType("gds") == WeightSourceType.GDS


class TestMxConfig:
    """Tests for MxConfig model."""

    def test_defaults(self):
        config = MxConfig()
        assert config.mx_server == "localhost:8001"
        assert config.model_name == ""
        assert config.contiguous_reg is False
        assert config.sync_start is True
        assert config.expected_workers == 1

    def test_custom_values(self):
        config = MxConfig(
            mx_server="mx:9000",
            model_name="llama-70b",
            contiguous_reg=True,
            sync_start=False,
            expected_workers=8,
        )
        assert config.mx_server == "mx:9000"
        assert config.model_name == "llama-70b"
        assert config.contiguous_reg is True
        assert config.sync_start is False
        assert config.expected_workers == 8

    def test_pickle_round_trip(self):
        config = MxConfig(model_name="test", expected_workers=4)
        restored = pickle.loads(pickle.dumps(config))
        assert restored == config


class TestGmsConfig:
    """Tests for GmsConfig model."""

    def test_model_required(self):
        with pytest.raises(Exception):
            GmsConfig()

    def test_defaults(self, default_gms_config):
        config = default_gms_config
        assert config.model == "test-model"
        assert config.engine == EngineType.VLLM
        assert config.mode == GmsMode.SOURCE
        assert config.tp_size == 1
        assert config.ep_size == 1
        assert config.device == 0
        assert config.mx_server == "localhost:8001"
        assert config.model_name is None
        assert config.dtype == "auto"
        assert config.trust_remote_code is False
        assert config.revision is None
        assert config.max_model_len is None
        assert config.weight_source == WeightSourceType.DISK
        assert config.s3_bucket is None
        assert config.s3_prefix is None
        assert config.cache_endpoint is None

    def test_total_workers_single_gpu(self, default_gms_config):
        assert default_gms_config.total_workers == 1

    def test_total_workers_multi_gpu(self, multi_gpu_gms_config):
        assert multi_gpu_gms_config.total_workers == 8  # 4 * 2

    def test_to_mx_config(self, default_gms_config):
        mx = default_gms_config.to_mx_config()
        assert isinstance(mx, MxConfig)
        assert mx.mx_server == "localhost:8001"
        assert mx.model_name == "test-model"
        assert mx.expected_workers == 1

    def test_to_mx_config_with_model_name_override(self):
        config = GmsConfig(
            model="meta-llama/Llama-3.3-70B",
            model_name="llama-70b-custom",
            tp_size=4,
        )
        mx = config.to_mx_config()
        assert mx.model_name == "llama-70b-custom"
        assert mx.expected_workers == 4

    def test_to_mx_config_uses_model_when_no_override(self):
        config = GmsConfig(model="meta-llama/Llama-3.3-70B")
        mx = config.to_mx_config()
        assert mx.model_name == "meta-llama/Llama-3.3-70B"

    def test_pickle_round_trip(self, default_gms_config):
        restored = pickle.loads(pickle.dumps(default_gms_config))
        assert restored.model == default_gms_config.model
        assert restored.total_workers == default_gms_config.total_workers

    def test_all_fields(self):
        config = GmsConfig(
            model="deepseek-v3",
            engine=EngineType.VLLM,
            mode=GmsMode.TARGET,
            tp_size=4,
            ep_size=2,
            device=0,
            mx_server="mx:8001",
            model_name="ds-v3",
            dtype="bfloat16",
            trust_remote_code=True,
            revision="main",
            max_model_len=4096,
            weight_source=WeightSourceType.GDS,
            s3_bucket="bucket",
            s3_prefix="prefix/",
            cache_endpoint="http://cache",
        )
        assert config.total_workers == 8
        assert config.weight_source == WeightSourceType.GDS
