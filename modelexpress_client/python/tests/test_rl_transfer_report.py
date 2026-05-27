# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from modelexpress import p2p_pb2
from modelexpress.rl_metadata import RlSourceCandidate, RlSourceMetadata, RlSourceRole
from modelexpress.rl_transfer_report import (
    failed_attempt_from_candidate,
    successful_attempt_from_candidate,
)


def _candidate() -> RlSourceCandidate:
    return RlSourceCandidate(
        mx_source_id="source-a",
        worker_id="worker-a",
        model_name="test-model",
        worker_rank=0,
        metadata=RlSourceMetadata(
            model_version=7,
            role=RlSourceRole.TRAINER,
            world_size=1,
        ),
        status=p2p_pb2.SOURCE_STATUS_READY,
        updated_at=1234567890000,
    )


def test_successful_attempt_includes_discovery_health():
    attempt = successful_attempt_from_candidate(
        _candidate(),
        bytes_transferred=1024,
        tensor_count=2,
        duration_seconds=0.01,
    )

    assert attempt.source_status == p2p_pb2.SOURCE_STATUS_READY
    assert attempt.source_updated_at == 1234567890000


def test_failed_attempt_includes_discovery_health():
    attempt = failed_attempt_from_candidate(_candidate(), RuntimeError("boom"))

    assert attempt.error == "boom"
    assert attempt.source_status == p2p_pb2.SOURCE_STATUS_READY
    assert attempt.source_updated_at == 1234567890000
