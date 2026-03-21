# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelExpress type definitions."""

import re

from google.protobuf import __version__ as _pb_version

from modelexpress.types import TensorDescriptor, WorkerMetadata, GetMetadataResponse


class TestProtobufCompatibility:
    """Guard against generated protobuf code drifting from the installed runtime."""

    def test_p2p_pb2_gencode_matches_runtime_major_version(self):
        """Regenerate p2p_pb2.py if this fails (see pyproject.toml [dev] deps)."""
        import modelexpress.p2p_pb2 as pb2

        with open(pb2.__file__) as f:
            src = f.read()
        m = re.search(r"Protobuf Python Version: (\d+)\.", src)
        assert m, "Could not parse gencode version from p2p_pb2.py"
        gencode_major = int(m.group(1))
        runtime_major = int(_pb_version.split(".")[0])
        assert gencode_major == runtime_major, (
            f"p2p_pb2.py was generated with protobuf {gencode_major}.x "
            f"but runtime is {runtime_major}.x - regenerate with: "
            f"python -m grpc_tools.protoc -I../../modelexpress_common/proto "
            f"--python_out=modelexpress --grpc_python_out=modelexpress "
            f"../../modelexpress_common/proto/p2p.proto"
        )


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

    def test_creation_with_endpoint(self):
        """Test worker metadata creation with metadata endpoint."""
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
            metadata_endpoint="10.0.1.5:50051",
            tensors=tensors,
        )
        assert metadata.worker_rank == 0
        assert len(metadata.tensors) == 3
        assert metadata.metadata_endpoint == "10.0.1.5:50051"

    def test_default_fields(self):
        """Test that optional fields default to empty strings."""
        metadata = WorkerMetadata(worker_rank=0, tensors=[])
        assert metadata.metadata_endpoint == ""
        assert metadata.agent_name == ""
        assert metadata.transfer_engine_session_id == ""


class TestGetMetadataResponse:
    """Tests for GetMetadataResponse dataclass."""

    def test_found_response(self):
        """Test response when source is found."""
        workers = [
            WorkerMetadata(
                worker_rank=i,
                metadata_endpoint=f"10.0.1.{i}:50051",
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
