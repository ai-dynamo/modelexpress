# FSDP CPU Staging

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Use this reference when an FSDP trainer needs a stable weight snapshot without
retaining another wire-format copy in GPU memory. This is the **NIXL full-tensor
publication path**, not an option on `RefitClientTrainer`'s NCCL M2N publisher or
S3 delta publication. For the surrounding mesh and generator lifecycle, read
[nixl-reshard-refit.md](nixl-reshard-refit.md).

## Choose Where The Snapshot Lives

| Mode | Use when | Cost and contract |
|---|---|---|
| `IN_PLACE` | Synchronous refit, contiguous sources already in wire dtype, stable addresses | No second snapshot; do not mutate or free source storage until readers finish and the version is retired |
| `COPY_TO_HOST` | Source storage moves, dtype conversion is needed, or a second GPU copy is too expensive | Persistent pinned CPU arenas, registered once; each version copies the current local shards into them |
| `COPY_TO_DEVICE` | A measured latency benefit justifies the extra VRAM | Persistent wire-format device arenas; budget all local shards and replicated tensors |

Host staging is independent of optimizer-state offload, FSDP parameter offload,
and the generator's `staging_device="cpu"`. Leave the training precision and
optimizer policy unchanged when comparing staging modes. A host-staged trainer
can feed a generator whose receive arena remains on CUDA.

For FP32 trainer parameters sent as BF16, `IN_PLACE` is not valid: that requires
a cast. Preserve FP32-sensitive tensors with exact state-dict names in
`wire_dtype_overrides`. Casting an already-rounded BF16 value back to FP32 at the
receiver cannot restore precision.

## Configure The FSDP Publisher

After distributed/CUDA initialization, with `MX_WORKER_HOST` reachable from the
generators and the NIXL prerequisites satisfied:

```python
import torch
from modelexpress_rl import (
    FSDPTrainerContext,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    TrainerStagingMode,
    WeightPayloadFormat,
)

trainer = ModelExpressTrainerClient.initialize(
    ModelExpressTrainerConfig(
        model_name=model_name,
        server_url=mx_server,
        device_id=torch.cuda.current_device(),
        engine_context=FSDPTrainerContext(
            wire_dtype_overrides=fp32_overrides,
        ),
        staging_mode=TrainerStagingMode.COPY_TO_HOST,
        payload_format=WeightPayloadFormat.FULL_TENSOR,
    )
)
bound_state = model.state_dict()
binding = trainer.bind_tensors(bound_state)
```

`fp32_overrides` is a framework-supplied mapping of actual state-dict names to
`torch.float32` (or `{}` when no exception is needed). Other floating tensors
use BF16. Unknown override names and non-floating targets are rejected; the
supported override types are FP16, BF16 and FP32, not packed FP8/FP4.

The equivalent environment default is `MX_TRAINER_STAGING_MODE=COPY_TO_HOST`;
an explicit `ModelExpressTrainerConfig.staging_mode` takes precedence. Choose the
mode before initializing the client; changing an active adapter's staging mode
requires retiring its versions and creating a new client/binding.

The orchestrator collects **every** worker's binding, creates a trainer mesh,
and creates the new `FULL_TENSOR` version against that mesh. Each trainer then
calls `trainer.publish_version(version=version.ref)`. Wait for the global
version to become `READY`; one rank's return does not establish global readiness.

### Publish Current Storage, Not A Stale State Dict

The public client binds the mapping once and retains it. The FSDP adapter
re-reads that mapping on each publish, but it does not call `model.state_dict()`
for you. If the framework replaces tensor objects or rematerializes storage,
refresh the **same bound mapping** before publishing, after the previous version
has been retired:

```python
current_state = model.state_dict()
bound_state.clear()
bound_state.update(current_state)
trainer.publish_version(version=version.ref)
```

Do not call `bind_tensors` twice on the same client. Tensor names, source dtypes,
global/local shapes and shard offsets must still match the bound layout. A real
layout or precision-policy change requires retirement and a new client/mesh
binding; CPU staging is not a way to bypass coverage validation.

### Respect The Snapshot Lifetime

- Order optimizer writes before the snapshot copy. If training uses another
  CUDA stream, establish that dependency before publication; the staging fence
  only covers the copy work recorded on the publishing stream.
- The adapter copies into persistent pinned arenas and records CUDA completion
  events. Publication waits on `publish_ready` before advertising the bytes.
  Do not expose a host buffer while its asynchronous copy is incomplete.
- Keep source tensors stable through snapshot completion. After that, training
  can change its original storage, but the published CPU snapshot must stay
  immutable until retirement.
- The adapter reuses its arenas across versions; it does **not** allocate a
  separate snapshot for every outstanding version. Before staging the next
  version, let consumers finish and release leases, delete the old version via
  the control client, and call `trainer.release_version(version=old.ref)` on
  every publisher. Keep the mesh for subsequent versions; delete it at teardown.
  Do not assume a server lease prevents a local overwrite of reused arenas.

## Verify Memory And Transport Independently

For each rank, estimate persistent host snapshot bytes as
`sum(local_shard.numel() * wire_dtype.itemsize)`. Include replicated tensors and
sum all processes on a node. Budget pinned RAM alongside optimizer offload,
checkpoint loading and other host allocations; verify host/cgroup limits and
locked-memory limits when registration fails. CUDA-enabled PyTorch is required
for this implementation's pinned allocations.

The snapshot is registered as NIXL `DRAM`; host-source manifests use
`mx.reshard.shard_table.v2`. Both publisher and receiver need host-aware clients.
Do not relabel host addresses as VRAM to satisfy an old receiver. The normal
receiver handles per-source memory kinds; custom `NixlReshardTransport` wiring
must pass the source memory map and an explicit local destination memory kind.
Source DRAM does not require destination DRAM.

Host staging removes the persistent wire-format GPU snapshot, not all temporary
CUDA workspace or the generator's buffers. Measure peak RAM and VRAM separately;
timing logs cannot establish memory savings. For slow DRAM-to-GPU reads, also
check CPU/NUMA placement and NIC assignment using [rdma-rails.md](rdma-rails.md).

## Acceptance And Measurement

Qualify multiple changing versions, including a case where source storage is
replaced. Check every installed parameter against the expected wire-cast value,
FP32 exceptions using values distinguishable from BF16 rounding, and generation
separately. A successful READ or matching tokens alone is insufficient. Confirm
host buffers are pinned, registered addresses are reused, and publication does
not precede copy completion. Exercise the receiver's failure/retirement path
before allowing snapshot reuse.

Measure source preparation (copy enqueue **and completion**), publication,
receiver READ, installation, and pause/update/resume as separate boundaries.
Compare modes with the same model, precision, topology, NIC assignment and
workload; distinguish cold registration from warm updates and repeat independent
launches. Keep expensive hashes/native-load comparisons out of clean timings
while retaining a matching correctness run. Do not add nested timings or subtract
independent medians to infer overhead.

The earlier large-model host-staging experiments established repeated-update
correctness with CUDA receive staging. Their unpaired diagnostic launches did
not establish a universal latency benefit or exact peak-memory savings. Choose
host staging for its snapshot/memory contract, then measure the latency tradeoff.

## Source And Validation Pointers

- `docs/rl/CONFIGURATION.md`: FSDP precision and trainer staging.
- `modelexpress_client/python/modelexpress_rl/train/runtime.py`: bound mapping and publication.
- `modelexpress_client/python/modelexpress_rl/train/engines/fsdp/adapter.py`: persistent arenas, layout checks and CUDA fence.
- `modelexpress_client/python/modelexpress_rl/train/engines/fsdp/publisher.py`: DRAM manifests and wire dtype.
- `modelexpress_client/python/modelexpress_rl/train/methods/full_tensor.py`: publication and retirement.
- `modelexpress_client/python/tests/test_refit_fsdp_adapter.py`, `test_refit_fsdp_publisher.py`, and `test_refit_dram_read.py`: snapshot, precision and memory-kind checks. CPU-only tests do not qualify actual pinned DMA or RDMA.
