#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E test: each pod publishes a unique source and discovers the other's."""

import logging
import os
import socket
import sys
import time
import uuid

from modelexpress import p2p_pb2
from modelexpress.peer_direct_client import PeerDirectMetadataClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("peer_test")


def get_my_ip() -> str:
    # POD_IP is injected via downward API
    pod_ip = os.environ.get("POD_IP")
    if pod_ip:
        return pod_ip
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def main():
    role = os.environ.get("PEER_ROLE", "unknown")
    substrate = os.environ.get("MX_PEER_DISCOVERY_SUBSTRATE", "static")
    my_ip = get_my_ip()
    log.info(f"starting role={role} substrate={substrate} ip={my_ip}")

    client = PeerDirectMetadataClient(substrate=substrate, ip=my_ip)

    # Publish a distinct source per role so we can tell them apart.
    identity = p2p_pb2.SourceIdentity(
        mx_version="0.3.0",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=f"test-model-from-{role}",
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype="bfloat16",
    )
    worker = p2p_pb2.WorkerMetadata(
        worker_rank=0,
        metadata_endpoint=f"{my_ip}:9999",
        agent_name=f"agent-{role}",
        status=p2p_pb2.SOURCE_STATUS_READY,
    )
    worker_id = f"{role}-{uuid.uuid4().hex[:8]}"
    my_source_id = client.publish_metadata(identity, worker, worker_id)
    log.info(f"PUBLISHED source_id={my_source_id} worker_id={worker_id}")

    # Discover peers - loop until we find at least one non-self source or time out.
    deadline = time.monotonic() + 60
    seen_other = False
    while time.monotonic() < deadline:
        response = client.list_sources()
        others = [
            inst for inst in response.instances
            if inst.mx_source_id != my_source_id
        ]
        if others:
            log.info(f"DISCOVERED {len(others)} other source(s):")
            for inst in others:
                log.info(
                    f"  mx_source_id={inst.mx_source_id} "
                    f"worker_id={inst.worker_id} "
                    f"model={inst.model_name} "
                    f"rank={inst.worker_rank} "
                    f"status={p2p_pb2.SourceStatus.Name(inst.status)}"
                )
                meta = client.get_metadata(inst.mx_source_id, inst.worker_id)
                if meta.found:
                    log.info(
                        f"  GET_METADATA found "
                        f"endpoint={meta.worker.metadata_endpoint} "
                        f"agent={meta.worker.agent_name} "
                        f"grpc={meta.worker.worker_grpc_endpoint} "
                        f"tensors={len(meta.worker.tensors)}"
                    )
                else:
                    log.warning(f"  GET_METADATA not found for {inst.mx_source_id}")
            seen_other = True
            break
        time.sleep(2)

    if seen_other:
        log.info("TEST_RESULT=PASS")
        rc = 0
    else:
        log.error("TEST_RESULT=FAIL timed out waiting for peer")
        rc = 1

    # Stay alive a bit longer so the other side can also confirm
    time.sleep(20)
    client.close()
    log.info("closed")
    sys.exit(rc)


if __name__ == "__main__":
    main()
