# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def replay_ops(tensor, op_chain):
    for name, args, kwargs in op_chain:
        tensor = getattr(tensor, name)(*args, **dict(kwargs))
    return tensor


def install_replay(batch, layout, installed=None):
    if installed is None:
        installed = {}
    for _, copy, source in batch.writes:
        if copy.param_name not in installed:
            shape, dtype = layout[copy.param_name]
            installed[copy.param_name] = torch.zeros(shape, dtype=dtype)
        target = installed[copy.param_name]
        target.as_strided(
            copy.dest_shape,
            copy.dest_stride,
            target.storage_offset() + copy.dest_offset,
        ).copy_(replay_ops(source, copy.op_chain))
    return installed


def compile_batches(capture, layout, sources, capacity):
    from modelexpress_rl.inference.trainer_plan import TrainerTransferPlan
    from modelexpress_rl import envs
    from modelexpress_rl.inference.plan import StreamingSettings
    from modelexpress_rl.version import MeshIdentity

    return TrainerTransferPlan.compile(
        mesh_identity=MeshIdentity("mesh", 1),
        capture=capture,
        parameter_layout=layout,
        sources=sources,
        streaming=StreamingSettings(capacity),
        verify_full_sources=envs.MX_RESHARD_PUBLISH_DIGEST,
    ).batches


def consume_stream(prepared, install):
    with prepared.consume(lambda: None):
        prepared.apply(install)


def resolve_sources(manifests):
    from modelexpress_rl.inference.adapter import TrainerSourceShard
    from modelexpress_rl.inference.nixl_staged_transfer import _SourceBinding
    from modelexpress_rl.inference.plan import TrainerSourceSnapshot
    from modelexpress_rl.version import MeshIdentity

    snapshot = TrainerSourceSnapshot(
        MeshIdentity("mesh", 1),
        tuple(
            TrainerSourceShard(str(index), str(index), "digest", "source:19000", blob)
            for index, blob in enumerate(manifests)
        ),
    )
    return _SourceBinding.from_snapshot(snapshot, {}).resolved
