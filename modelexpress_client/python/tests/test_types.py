# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelExpress type definitions."""

from modelexpress.types import TensorDescriptor, WorkerMetadata, GetMetadataResponse


class TestTensorDescriptor:
    """Tests for TensorDescriptor dataclass."""

    def test_creation(self):
        """Test basic tensor descriptor creation."""
        desc = TensorDescriptor(
            name="model.layers.0.self_attn.q_proj.weight",
            addr=0x7F8A00000000,
            size=1024 * 1024 * 1024,
            device_id=0,
            dtype="bfloat16",
        )
        assert desc.name == "model.layers.0.self_attn.q_proj.weight"
        assert desc.size == 1024 * 1024 * 1024
        assert desc.dtype == "bfloat16"

    def test_dtype_required(self):
        """Test that dtype is a required field."""
        import pytest
        with pytest.raises(TypeError):
            TensorDescriptor(
                name="test",
                addr=0,
                size=0,
                device_id=0,
            )

    def test_large_tensor(self):
        """Test with realistic large tensor values."""
        desc = TensorDescriptor(
            name="model.embed_tokens.weight",
            addr=0x7F8A00000000,
            size=32000 * 8192 * 2,
            device_id=7,
            dtype="bfloat16",
        )
        assert desc.size == 524288000


class TestWorkerMetadata:
    """Tests for WorkerMetadata dataclass."""

    def test_creation(self):
        """Test basic worker metadata creation."""
        tensors = [
            TensorDescriptor(
                name=f"layer.{i}.weight",
                addr=0x7F8A00000000 + i * 1024,
                size=1024,
                device_id=0,
                dtype="bfloat16",
            )
            for i in range(3)
        ]
        metadata = WorkerMetadata(
            worker_rank=0,
            nixl_metadata=b"test_metadata",
            tensors=tensors,
        )
        assert metadata.worker_rank == 0
        assert len(metadata.tensors) == 3
        assert metadata.nixl_metadata == b"test_metadata"


class TestGetMetadataResponse:
    """Tests for GetMetadataResponse dataclass."""

    def test_found_response(self):
        """Test response when source is found."""
        workers = [
            WorkerMetadata(
                worker_rank=i,
                nixl_metadata=b"metadata",
                tensors=[],
            )
            for i in range(4)
        ]
        response = GetMetadataResponse(
            found=True,
            workers=workers,
        )
        assert response.found is True
        assert len(response.workers) == 4

    def test_not_found_response(self):
        """Test response when source is not found."""
        response = GetMetadataResponse(
            found=False,
            workers=[],
        )
        assert response.found is False
        assert len(response.workers) == 0
