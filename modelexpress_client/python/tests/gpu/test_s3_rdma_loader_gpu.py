# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in NIXL OBJ test against a real S3-over-RDMA endpoint.

Requires a NIXL build with the generic accelerated OBJ engine, a CUDA GPU, the
AWS_* endpoint/credential variables, MX_TEST_S3_RDMA_URI (s3://bucket/prefix of a
safetensors checkpoint) and MX_TEST_S3_RDMA_REFERENCE (a local copy of the same
checkpoint used as the byte-for-byte reference).
"""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.slow
def test_s3_rdma_loader_matches_local_checkpoint():
    uri = os.environ.get("MX_TEST_S3_RDMA_URI")
    reference_dir = os.environ.get("MX_TEST_S3_RDMA_REFERENCE")
    if not uri or not reference_dir:
        pytest.skip("set MX_TEST_S3_RDMA_URI and MX_TEST_S3_RDMA_REFERENCE")
    if not torch.cuda.is_available():
        pytest.skip("a CUDA GPU is required")

    from safetensors import safe_open

    from modelexpress.accelerators import CudaAcceleratorBackend
    from modelexpress.s3_rdma_loader import MxS3RdmaLoader, S3RdmaLoadConfig
    from modelexpress.s3_rdma_transfer import is_s3_rdma_available

    if not is_s3_rdma_available():
        pytest.skip("NIXL OBJ with cuObject is not available")

    reference = {}
    for path in sorted(Path(reference_dir).glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                reference[name] = path

    loader = MxS3RdmaLoader(CudaAcceleratorBackend(), 0, S3RdmaLoadConfig.from_env())
    loader.prepare(uri)
    seen = set()
    try:
        for name, tensor in loader.iter_tensors():
            assert tensor.is_cuda
            with safe_open(str(reference[name]), framework="pt", device="cpu") as handle:
                want = handle.get_tensor(name)
            assert tensor.dtype == want.dtype and tensor.shape == want.shape, name
            got_bytes = tensor.reshape(-1).view(torch.uint8)
            want_bytes = want.reshape(-1).view(torch.uint8).to(tensor.device)
            assert torch.equal(got_bytes, want_bytes), name
            seen.add(name)
    finally:
        loader.release()
    loader.log_timing()
    assert seen == set(reference)
