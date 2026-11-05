# NIXL Reshard Refit

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Public API: the `modelexpress_rl` top-level namespace. No-gather resharding
lives in `modelexpress_rl.refit.reshard`. Authoritative sources:
`docs/rl/README.md`, `docs/rl/ARCHITECTURE.md`, `docs/rl/CONFIGURATION.md`.
Reference integration: `examples/rl/dynamo_vllm_reshard_refit/`.

## Model

- The trainer publishes an immutable `WeightVersion`. Weight bytes and full
  tensor manifests stay on trainer workers; MX (`RefitService`, Redis-only)
  stores registrations, versions, and compact shard publications.
- A `TrainerMesh` declares the complete trainer worker set. A worker-sharded
  version becomes `READY` when every logical shard has a current, live
  publication.
- Generators resolve the manifests, intersect their own loader's slices with
  trainer shards, and read one-sided over NIXL without gathering a full tensor.
- Version leases protect sources while a generator installs a version.

## Trainer Side

```python
from modelexpress_rl import (
    FSDPTrainerContext,
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    WeightPayloadFormat,
    WeightVersionState,
)

trainer = ModelExpressTrainerClient.initialize(
    ModelExpressTrainerConfig(
        engine_context=FSDPTrainerContext(),
        device_id=0,
        model_name=model_name,
        server_url=mx_server,
    )
)
control = ModelExpressControlClient.connect(server_url=mx_server)

binding = trainer.bind_tensors(model.state_dict())
mesh = control.create_trainer_mesh(
    model_name=model_name,
    idempotency_key=mesh_key,
    workers={trainer.worker_id: binding},     # every trainer worker's binding
)
version = control.create_weight_version(
    model_name=model_name,
    idempotency_key=version_key,
    payload_format=WeightPayloadFormat.FULL_TENSOR,
    trainer_mesh_id=mesh.mesh_id,
)
trainer.publish_version(version=version.ref)
assert control.get_weight_version(version.version_id).state is WeightVersionState.READY
```

Engine contexts:

- `FSDPTrainerContext(wire_dtype_overrides=...)`: `bind_tensors` takes the FSDP
  `state_dict()` (`dict[str, Tensor]`). Tensors not overridden go on the wire as BF16.
- `MegatronTrainerContext()`: `bind_tensors` takes a list of `MegatronTensorSpec`;
  build it from Megatron-Bridge conversion tasks with
  `modelexpress_rl.train.engines.megatron.build_megatron_tensor_specs`.

Set `ModelExpressTrainerConfig.staging_mode` explicitly
(`TrainerStagingMode`): prefer `IN_PLACE` when the source bytes stay stable and
match the wire dtype until retirement; otherwise `COPY_TO_HOST`.
`COPY_TO_DEVICE` costs substantial VRAM and should be a measured exception.

Retire in this order: `control.delete_weight_version(...)`,
`trainer.release_version(version=...)`, `control.delete_trainer_mesh(...)`,
then `close()` both clients.

## Generator Side: vLLM Plugin

The distribution registers `modelexpress_rl` under `vllm.general_plugins`. It
exposes a `modelexpress` native weight-transfer backend, selected with the
serve flag `--weight-transfer-config '{"backend":"modelexpress"}'`
(`examples/rl/dynamo_vllm_reshard_refit/dgd.yaml` shows the full command line,
together with `--load-format modelexpress` for the initial load).

- If plugin loading is filtered, `VLLM_PLUGINS` must include `modelexpress_rl`
  (not only `modelexpress`). For manual registration call
  `modelexpress_rl.inference.engines.vllm.registration.register()`.
- `MX_MODEL_NAME_OVERRIDE` sets the MX model identity when it differs from
  vLLM's model path; use the same name for trainers and versions.

Per refit, on every generator worker, with generation paused:

1. `init_weight_transfer_engine` with `{"init_info": {...}}` (empty selects the NIXL path).
2. `start_weight_update`.
3. `update_weights` with `{"update_info": {"version_id": "<WeightVersion id>"}}`.
4. `finish_weight_update`.

Then read back the installed version on every worker before resuming
generation (`examples/rl/dynamo_vllm_reshard_refit/rl_coordinator.py` does this
through Dynamo's `get_weight_version` control route). Each session accepts one
version; out-of-order or repeated calls raise.

## Generator Side: Direct Client

For framework integrations that drive the engine themselves:

```python
generator = ModelExpressGeneratorClient.initialize(
    ModelExpressGeneratorConfig(
        engine_context=engine_context,        # e.g. VllmGeneratorContext
        model_name=model_name,
        source_order=(WeightSource.TRAINER,),
        staging_buffer_bytes=4 * 1024**3,     # enables bounded staging
        staging_buffers_count=2,
        staging_device="cuda",                # or "cpu" for pinned host arenas
    )
)
staged = generator.stage_weight(version=WeightVersionRef(version_uid))
try:
    metrics = generator.apply_weight(staged)
finally:
    staged.release()
```

Pause generation on every replica before `apply_weight` and resume only after
every replica succeeds. Any failure after installation starts: keep the
deployment paused and restart the engines; a partially installed model is not
the old version.

## Cold Start At An Exact Version

| Variable | Use |
|---|---|
| `MX_LOAD_STRATEGY_CHAIN=RL` | Engine-neutral RL initial-load policy |
| `MX_REFIT_DESIRED_VERSION_UID` | Exact version required; startup fails rather than falling back |
| `MX_GENERATOR_SOURCE_ORDER` | Default `GENERATOR,OBJECT_STORAGE` for cold start; `TRAINER` is rejected there |
| `MX_REFIT_CHECKPOINT_DIR` | Host-local cache for S3 replay; the S3 strategy skips when unset |
| `MX_REFIT_METADATA_PORT` | Base NIXL metadata port for generator refit (default `7555`, plus local device id) |

## Settings Worth Knowing

| Variable | Default | Why |
|---|---|---|
| `MX_P2P_METADATA` | `1` | `0` makes active refit of a runtime-publishing generator fail before mutation |
| `MX_RESHARD_MAX_SEGMENTS_PER_COPY` | `64` | Descriptor cap before a dim-0 source is pulled once and sliced locally |
| `MX_RESHARD_MIN_GBPS` | `0` | Per-rank throughput floor; emits a `refit-slow-throughput-v1` warning record |
| `MX_RESHARD_PUBLISH_DIGEST` | `0` | Trainer publishes per-shard digests so receivers can confirm installed bytes |
| `MX_REFIT_STAGE_RECORD` | `1` | One `refit-stage-v2` JSON timing record per refit |

## Leases

- Generator installs hold a version lease (`RegisterVersionLease` /
  `DeleteVersionLease`). Releasing a published version retries only the
  active-reader rejection until its deadline.
- Any trainer mesh update that changes its generation is rejected while a
  linked version has an active lease. Drain readers first.
- Generator-to-generator reads hold a donor-timed lease; an abandoned lease
  (target `SIGKILL`, OOM, node loss) expires after its bounded lifetime.
- Upgrades of the split worker metadata protocol need the server, trainers,
  and generators upgraded together; retire active refit rounds and their
  leases before stopping workers.
