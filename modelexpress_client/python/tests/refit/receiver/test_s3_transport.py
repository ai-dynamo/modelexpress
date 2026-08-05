# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from modelexpress.refit.transport.s3 import S3CanonicalTransport

from modelexpress.refit.receiver import (
    build_modelexpress_s3_transport,
)


class FakeS3Client:
    pass


def args(**overrides):
    values = {
        "modelexpress_delta_s3_bucket": "weights",
        "modelexpress_delta_s3_prefix": "runs/policy",
        "modelexpress_delta_s3_endpoint": "https://s3.example.test",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_phase2_production_transport_is_s3():
    transport = build_modelexpress_s3_transport(args(), client=FakeS3Client())

    assert isinstance(transport, S3CanonicalTransport)
    assert transport.identity.namespace == "s3://weights/runs/policy"
