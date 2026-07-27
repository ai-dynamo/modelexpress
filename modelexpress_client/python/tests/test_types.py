# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelExpress type definitions."""

import re

from google.protobuf import __version__ as _pb_version
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from modelexpress import p2p_pb2
from modelexpress.metadata.payload import (
    tensor_source_metadata,
    worker_tensor_descriptors,
)
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

    def test_publish_request_is_readable_by_0_4_schema(self):
        legacy_file = descriptor_pb2.FileDescriptorProto(
            name="modelexpress_0_4_publish_metadata.proto",
            package="model_express.p2p.compat_0_4",
            dependency=[p2p_pb2.DESCRIPTOR.name],
            syntax="proto3",
        )
        request = legacy_file.message_type.add(name="PublishMetadataRequest")
        for name, number, type_name in (
            ("identity", 1, ".model_express.p2p.SourceIdentity"),
            ("worker", 2, ".model_express.p2p.WorkerMetadata"),
        ):
            request.field.add(
                name=name,
                number=number,
                label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
                type=descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
                type_name=type_name,
            )
        request.field.add(
            name="worker_id",
            number=3,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
            type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        )

        pool = descriptor_pool.Default()
        pool.Add(legacy_file)
        legacy_request = message_factory.GetMessageClass(
            pool.FindMessageTypeByName(
                "model_express.p2p.compat_0_4.PublishMetadataRequest"
            )
        )()
        current_request = p2p_pb2.PublishMetadataRequest(
            identity=p2p_pb2.SourceIdentity(model_name="model"),
            worker=p2p_pb2.WorkerMetadata(worker_rank=1),
            worker_id="worker",
            pod_name="source-0",
            pod_uid="pod-uid",
            pod_namespace="default",
        )

        legacy_request.ParseFromString(current_request.SerializeToString())

        assert legacy_request.identity.model_name == "model"
        assert legacy_request.worker.worker_rank == 1
        assert legacy_request.worker_id == "worker"


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

    def test_worker_tensor_descriptors_prefers_tensor_source(self):
        legacy = p2p_pb2.TensorDescriptor(name="legacy", addr=1, size=1)
        current = p2p_pb2.TensorDescriptor(name="current", addr=2, size=2)
        worker = p2p_pb2.WorkerMetadata(
            tensors=[legacy],
            tensor_source=tensor_source_metadata([current]),
        )

        tensors = worker_tensor_descriptors(worker)

        assert len(tensors) == 1
        assert tensors[0].name == "current"

    def test_worker_tensor_descriptors_falls_back_to_legacy_tensors(self):
        legacy = p2p_pb2.TensorDescriptor(name="legacy", addr=1, size=1)
        worker = p2p_pb2.WorkerMetadata(tensors=[legacy])

        tensors = worker_tensor_descriptors(worker)

        assert len(tensors) == 1
        assert tensors[0].name == "legacy"

    def test_worker_tensor_descriptors_returns_empty_for_artifact_source(self):
        legacy = p2p_pb2.TensorDescriptor(name="legacy", addr=1, size=1)
        worker = p2p_pb2.WorkerMetadata(
            tensors=[legacy],
            artifact_source=p2p_pb2.ArtifactSourceMetadata(
                artifact_id="artifact",
                total_size=128,
                file_count=1,
                chunk_count=1,
            ),
        )

        tensors = worker_tensor_descriptors(worker)

        assert len(tensors) == 0


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
