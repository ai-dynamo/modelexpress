# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query mx-server for READY sources over gRPC, split by kind.

Two consumers:
  - imported as a library (`list_ready_sources_by_kind`) by the pytest tests
    that already hold a stub (e.g. test_stale_metadata.py);
  - run as a CLI by run-mx-p2p-test's source-readiness gate, which needs a
    backend-agnostic count. The gate used to count `kubectl get modelmetadata`
    CRs, but the Redis backend stores metadata in Redis with no CRs, so that
    count is always 0 there. Querying the server's ListSources instead works
    for both the Kubernetes-CRD and Redis backends.

CLI: prints "<weight_count> <artifact_count>" to stdout. On any failure it
prints "0 0" to stdout (so the gate's poll loop simply retries) and the
exception to stderr (so a real problem — missing stubs, unreachable server,
stuck port-forward — is visible in the CI log instead of looking like
"0 sources forever").

Requires the generated p2p_pb2 / p2p_pb2_grpc stubs (grpc_tools.protoc over
modelexpress_common/proto/p2p.proto) alongside this file.
"""

from __future__ import annotations

import argparse
import socket
import sys

import grpc

import p2p_pb2
import p2p_pb2_grpc
from kube_utils import port_forward


def list_ready_sources_by_kind(
    stub: "p2p_pb2_grpc.P2pServiceStub",
) -> tuple[list, list]:
    """Return READY sources split into (weight_sources, artifact_sources).

    No identity filter on purpose: the server computes `mx_source_id` as a
    SHA256 over all identity fields (model_name, tp/pp/ep size, dtype, revision,
    mx_version, …), so a partial identity from outside the engine can't produce
    a matching hash. `status_filter=READY` scopes to published, healthy sources.
    Artifact transfer publishes additional READY metadata in the same
    namespace, so classify both kinds explicitly.
    """
    response = stub.ListSources(
        p2p_pb2.ListSourcesRequest(status_filter=p2p_pb2.SOURCE_STATUS_READY),
        timeout=30,
    )
    weight_sources = []
    artifact_sources = []
    for source in response.instances:
        metadata = stub.GetMetadata(
            p2p_pb2.GetMetadataRequest(
                mx_source_id=source.mx_source_id,
                worker_id=source.worker_id,
            ),
            timeout=30,
        )
        if not metadata.found:
            continue
        # Classify via the source_payload oneof: artifact_source => artifact;
        # anything else (tensor_source, or an absent oneof on servers predating
        # it — e.g. the 0.4.0 image the mixed-version entry runs against) =>
        # weight. Matches the CR gate's "empty sourceType == weight" handling.
        if metadata.worker.WhichOneof("source_payload") == "artifact_source":
            artifact_sources.append(source)
        else:
            weight_sources.append(source)
    return weight_sources, artifact_sources


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()

    try:
        with port_forward(args.namespace, "svc/mx-server", _free_local_port(), 8000) as port:
            channel = grpc.insecure_channel(f"localhost:{port}")
            stub = p2p_pb2_grpc.P2pServiceStub(channel)
            weights, artifacts = list_ready_sources_by_kind(stub)
    except Exception as exc:
        # stdout stays parseable so the gate retries; the cause goes to stderr.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("0 0")
        return 0

    print(f"{len(weights)} {len(artifacts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
