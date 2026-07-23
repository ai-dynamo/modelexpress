# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Miles post-write hook for publishing delta weights through S3.

The POC data plane is S3 only. On a single trainer node the rollout receiver
discovers versions directly from the S3 key convention (weight_v{N:06d}/) and
walks the contiguous chain, so no ModelExpress control-plane record is needed to
tell it which version or lineage to pull. Control-plane publish (list_sources
discovery + lineage attestation) is deferred to the cross-region design (P1).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket

from .s3_delta import s3_client, upload_version_dir

logger = logging.getLogger("modelexpress.rl.delta_publish")

_baseline_failure: Exception | None = None


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _assert_shared_fs(args, path: str) -> None:
    delta_dir = os.path.realpath(args.update_weight_disk_dir)
    version_path = os.path.realpath(path)
    actor_nodes = int(getattr(args, "actor_num_nodes", 1))
    actor_gpus = int(getattr(args, "actor_num_gpus_per_node", 1))
    world_size = int(getattr(args, "world_size", actor_nodes * actor_gpus))
    logger.info(
        "mx_delta_publish: topology host=%s actor_nodes=%d "
        "actor_gpus_per_node=%d world_size=%d delta_dir=%s",
        socket.gethostname(),
        actor_nodes,
        actor_gpus,
        world_size,
        delta_dir,
    )
    assert actor_nodes == 1, (
        "P0 delta publishing requires one trainer node with a filesystem shared "
        "by every trainer rank"
    )
    assert version_path == delta_dir or os.path.dirname(version_path) == delta_dir, (
        "delta version directory must be the shared update_weight_disk_dir or its child"
    )


def _record_baseline(args, version_dir: str) -> None:
    global _baseline_failure
    _baseline_failure = None
    try:
        _assert_shared_fs(args, version_dir)
        logger.info(
            "mx_delta_publish: baseline uses local model_path; no S3 anchor uploaded"
        )
    except Exception as exc:
        _baseline_failure = exc
        logger.exception("mx_delta_publish: baseline lineage setup failed; deferred")


def publish_delta(args, version_dir: str, rollout_engines=None) -> None:
    """Publish a completed Miles delta directory to S3 from rank 0."""
    del rollout_engines
    if _rank() != 0:
        return

    if os.path.realpath(version_dir) == os.path.realpath(args.update_weight_disk_dir):
        _record_baseline(args, version_dir)
        return

    if _baseline_failure is not None:
        raise RuntimeError("baseline lineage setup failed") from _baseline_failure

    _assert_shared_fs(args, version_dir)
    bucket = os.environ["MX_S3_BUCKET"]
    stream_prefix = os.environ["MX_DELTA_STREAM_PREFIX"]

    index_path = os.path.join(version_dir, "model.safetensors.index.json")
    with open(index_path) as file:
        metadata = json.load(file)["metadata"]
    training_step = int(metadata["version"])

    s3 = s3_client()
    uploaded = upload_version_dir(s3, bucket, stream_prefix, version_dir)

    if os.environ.get("MX_DELTA_POC_EVICT_LOCAL") == "1":
        shutil.rmtree(version_dir)

    logger.info(
        "mx_delta_publish: uploaded %d files from weight_v%06d",
        uploaded,
        training_step,
    )
