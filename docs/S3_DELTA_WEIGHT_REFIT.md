# S3 Delta Weight Refit

This guide shows how to publish XOR-delta weight updates to S3 and install them
in a running vLLM or SGLang model with ModelExpress. The trainer and generator
start from the same local checkpoint; integrations may periodically publish full
HF checkpoints to reset that base. ModelExpress coordinates each version's
lineage and readiness.

Complete the [shared setup](#shared-setup), choose a backend, and use the
[shared publication workflow](#shared-weight-publication) for each update.

For a runnable Vime TP2 trainer, Dynamo TP1 rollout worker, and SeaweedFS setup, see
[`examples/rl/vime_dynamo_delta_refit`](../examples/rl/vime_dynamo_delta_refit/README.md).

| Backend | Receiver configuration | Apply a READY version |
|---|---|---|
| [vLLM](#vllm-backend) | `POST /init_weight_transfer_engine` | `start_weight_update` → `update_weights` → `finish_weight_update` |
| [SGLang](#sglang-backend) | `--modelexpress-config` | `POST /update_weights_from_modelexpress` |

The [checkpoint cache](#shared-checkpoint-cache) and
[S3 artifact contract](#s3-artifact-contract) are shared by both backends.

## Shared setup

### Components

| Component | Responsibility |
|---|---|
| `ModelExpressControlClient` | Create and transition immutable weight-version records in the ModelExpress catalog. |
| `ModelExpressTrainerClient` | Capture the seed-checkpoint base and publish either XOR deltas or full HF checkpoint batches to S3. |
| `ModelExpressGeneratorClient` | Validate READY versions, apply the requested S3 payload to its refit checkpoint, and reload the live model. |

### Requirements

- A Redis-backed ModelExpress Refit service.
- A ModelExpress build containing the canonical S3 delta integration and the
  selected inference backend.
- The `modelexpress` Python package installed in both the trainer and inference
  environments.
- Trainer ranks with S3 read/write access and inference hosts with read access.
- The corresponding base checkpoint in safetensors format on every trainer and
  inference host.

Minimal ModelExpress server configuration:

```bash
export MX_METADATA_BACKEND=redis
export REDIS_URL=redis://redis:6379
```

### Common environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MX_SERVER_ADDRESS` | `localhost:8001` | ModelExpress server address. |
| `MX_AUTH_TOKEN_PATH` | unset | Optional ModelExpress bearer-token file. |
| `MX_AUTH_TOKEN_TTL_SECONDS` | `60` | Token-file reread interval in seconds. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | unset | S3 credentials when an IAM role or workload identity is unavailable. |
| `AWS_SESSION_TOKEN` | unset | Session token when using temporary AWS credentials. |
| `AWS_DEFAULT_REGION` | unset | S3 region when `region_name` is not supplied in client configuration. |
| `MX_GENERATOR_SOURCE_ORDER` | Backend default | Set to `OBJECT_STORAGE` for S3-only refit. See the vLLM section for generator-peer fallback. |
| `MX_REFIT_DELTA_BUCKET_BYTES` | `536870912` (512 MiB) | Optional tensor-bucket size override for framework integrations. |
| `MX_REFIT_DELTA_WORKERS` | `min(32, CPU count)` | CPU workers used to compute and apply XOR deltas. |
| `MX_REFIT_CHECKSUM_FORMAT` | `adler32` | Checksum algorithm written by canonical S3 trainers. |
| `MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES` | `4294967296` (4 GiB) | Maximum tensor bytes grouped into one full-checkpoint safetensors object. |
| `MX_S3_UPLOAD_WORKERS` | `8` | Maximum concurrent multipart uploads per trainer rank. |
| `MX_S3_DOWNLOAD_WORKERS` | `16` | Generator download concurrency. |
| `MX_S3_MAX_POOL_CONNECTIONS` | `32` | Botocore HTTP connection-pool size. |
| `MX_S3_MAX_ATTEMPTS` | `5` | Total S3 request attempts. |

Optional S3 tuning variables and defaults are documented in
[`modelexpress_client/python/README.md`](../modelexpress_client/python/README.md#canonical-s3-transfer-tuning).

### Create the base version

Create a READY initial WeightVersion before initializing trainer or generator
clients. The current API requires a syntactically valid object-storage URI, but
this local seed-checkpoint flow does not upload or read an object at that URI.

```python
from modelexpress_rl import (
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    ObjectStorageConfig,
    ObjectStorageSource,
    ObjectStorageType,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionState,
)

S3_URI_PREFIX = "s3://my-bucket/weights"

with ModelExpressControlClient.connect(
    server_url="modelexpress:8001",
) as control:
    base = control.create_weight_version(
        uid="v0",
        model_name="Qwen/Qwen3-30B-A3B",
        idempotency_key="initialize-v0",
        payload_format=WeightPayloadFormat.FULL_TENSOR,
        object_storage=ObjectStorageSource(  # Catalog-only; no object is uploaded.
            storage_type=ObjectStorageType.S3,
            uri=f"{S3_URI_PREFIX}/v0/model.safetensors.index.json",
        ),
        state=WeightVersionState.READY,
    )
```

Trainer and generator initialization use `base.version_id` as their initial base
ID. They read the real weights from `seed_checkpoint_path`; the base version URI
is not downloaded.

See [`refit.proto`](../modelexpress_common/proto/refit.proto) for the complete
control-plane request and response schemas.

### Initialize the trainer client

Initialize one trainer client on each rank participating in MX publication and
keep it for the full training run. `hf_tensor_buckets()` below represents a
framework-owned function that returns a fresh iterable of Hugging Face tensor
buckets.

```python
import torch.distributed as dist

refit_process_group = ...  # Existing Gloo group of publishing ranks.

trainer = ModelExpressTrainerClient.initialize(
    ModelExpressTrainerConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
        payload_format=WeightPayloadFormat.XOR_DELTA,
        server_url="modelexpress:8001",
        process_group=refit_process_group,
        object_storage=ObjectStorageConfig(
            storage_type=ObjectStorageType.S3,
            uri_prefix=S3_URI_PREFIX,
            initial_base_version_id="v0",
            seed_checkpoint_path="/models/Qwen3-30B-A3B",
            region_name="us-west-2",
        ),
    )
)

# Call once before the first optimizer update.
trainer.prepare_delta_base(hf_tensor_iter=hf_tensor_buckets())
```

The bucket names must match tensors in the seed checkpoint. Keep the trainer
client alive because each successful publication advances its retained base
from `v0` to `v1`, then `v2`, and so on.

Use a dedicated Gloo process group containing the ranks that call MX staging
and publication. ModelExpress uses it for CPU object collectives that coordinate
shard and index publication. Other ranks may still participate in the framework's
weight gathers; Miles, for example, creates MX clients only on sender ranks.

## vLLM backend

This example uses vLLM 0.27.1 and the ModelExpress weight-transfer plugin.
For a runnable Vime TP2 trainer, Dynamo TP1 rollout worker, and MinIO setup, see
[`examples/rl/vime_dynamo_delta_refit`](../examples/rl/vime_dynamo_delta_refit/README.md).

Backend-specific environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MX_MODEL_NAME_OVERRIDE` | unset | Logical MX model name used by vLLM at startup and during refit. Use the same name for trainer clients and published WeightVersions. Without it, vLLM's configured model path or ID is used. |
| `VLLM_SERVER_DEV_MODE` | unset | Set to `1` to enable vLLM's weight-update HTTP routes. Network-isolate these routes. |
| `VLLM_PLUGINS` | unset | Set to `modelexpress` to load the ModelExpress vLLM plugin. |

### Start vLLM

Install the `modelexpress` Python package in the same environment used to run
`vllm serve`. The package provides the vLLM plugin entry point:

```toml
[project.entry-points."vllm.general_plugins"]
modelexpress = "modelexpress:register_modelexpress"
```

`VLLM_PLUGINS=modelexpress` selects that installed plugin. Then start vLLM:

```bash
export VLLM_SERVER_DEV_MODE=1
export VLLM_PLUGINS=modelexpress
export MX_SERVER_ADDRESS=modelexpress:8001
export MX_MODEL_NAME_OVERRIDE=Qwen/Qwen3-30B-A3B

vllm serve /models/Qwen3-30B-A3B \
  --tensor-parallel-size 4 \
  --weight-transfer-config '{"backend":"modelexpress"}'
```

Set `MX_MODEL_NAME_OVERRIDE` before starting vLLM so cold-start version validation,
generator registration, peer identity, and the checkpoint cache use the same
logical name.
For an S3 launch such as `--model s3://bucket/model`, vLLM can rewrite its internal
model name to a local streamer cache path.
`MX_MODEL_NAME_OVERRIDE=s3://bucket/model` keeps the original URI as the MX name;
a name such as `customer-bot` works too. Publish every WeightVersion with that
exact name and configure trainer clients to match.
The adapter resolves the override once on a model-configuration copy used only
for MX identity construction. The original configuration continues to provide
vLLM's model-loading path. Without the override, the existing behavior, including
use of the rewritten path, is unchanged.

An explicit `init_info.model_name` still overrides the transfer client's default
after startup. Keep it consistent with `MX_MODEL_NAME_OVERRIDE`; it cannot change
an identity already used during cold start.

### Initialize the weight-transfer engine

After vLLM is ready, initialize each server once:

```python
import requests

VLLM_URL = "http://vllm-generator:8000"

response = requests.post(
    f"{VLLM_URL}/init_weight_transfer_engine",
    json={
        "init_info": {
            "model_name": "Qwen/Qwen3-30B-A3B",
            "server_url": "modelexpress:8001",
            "object_storage_type": "S3",
            "initial_base_version_id": "v0",
            "seed_checkpoint_path": "/models/Qwen3-30B-A3B",
            "refit_checkpoint_dir": "/var/cache/modelexpress",
            "refit_checkpoint_max_size_gb": 500,
            "object_storage_region_name": "us-west-2",
            "max_transfer_attempts": 3,
            "max_replay_chain_length": 64,
            "rpc_timeout_seconds": 30,
        }
    },
    timeout=900,
)
response.raise_for_status()
```

`POST /init_weight_transfer_engine` fans out to every vLLM worker. Each worker:

1. initializes the seed lineage and host-local checkpoint cache;
2. fetches `initial_base_version_id` from the ModelExpress server;
3. verifies that the base is READY and has the configured model name; and
4. registers itself as a ModelExpress generator worker.

Initialization fails before serving updates if any worker cannot read the
seed checkpoint, write the cache, or validate the base version.

See [seed checkpoint requirements](#seed_checkpoint_path) and
[cache configuration](#refit_checkpoint_dir) for the shared storage settings.

### Reuse a cached seed

If desired-version cold start already cached the full checkpoint for
`initial_base_version_id`, omit `seed_checkpoint_path` from `init_info` or set it
to `null`. With the Python `ObjectStorageGeneratorConfig`, pass
`seed_checkpoint_path=None`. ModelExpress resolves the cached full root using
the model name, `refit_checkpoint_dir`, and `initial_base_version_id`; callers do
not need to reproduce the cache path or its URL encoding.

This mode reuses the cached checkpoint without downloading or copying a seed.
Initialization fails if the required root is missing or cannot be restored.
Passing the cached full-root path explicitly retains the same reuse behavior.
Passing an external checkpoint path retains the existing seed-import behavior.

With a cached seed, later full updates carry non-weight files forward from the
current prepared checkpoint. Its directory is protected from eviction until the
copy finishes, so eviction of the original seed does not drop these files.
Explicit external seed paths remain the source of their non-weight files.

### Apply a published version

After [publishing `v1`](#publish-v1) and marking it READY, run the update while
generation is paused. `mode=abort` clears active requests and vLLM caches before
the weight update.

```python
import requests

VLLM_URL = "http://vllm-generator:8000"


def post(path, *, body=None, params=None):
    response = requests.post(
        f"{VLLM_URL}/{path}",
        json=body or {},
        params=params,
        timeout=900,
    )
    response.raise_for_status()
    return response.json()


post("pause", body={}, params={"mode": "abort"})
post("start_weight_update", body={})
post("update_weights", body={"update_info": {"version_id": "v1"}})
post("finish_weight_update", body={"weight_version": "v1"})
post("resume", body={})
```

If an update request fails, keep generation paused while recovering the engine.
The sequence resumes generation only after the update succeeds.

### Source selection and peer fallback

By default, with full-tensor engine support, active refit uses this order:

1. load the exact requested version from a same-rank generator peer;
2. if no peer can prepare it, reconstruct the complete S3 lineage from its full
   checkpoint root through the target deltas and install that checkpoint.

Set `MX_GENERATOR_SOURCE_ORDER=OBJECT_STORAGE` to use S3 directly.

Post-load generator P2P requires registered runtime tensors and an initialized
loader-owned NIXL manager. Quantized models and FP8 KV caches can select this
path in either eager or graph mode. The peer transfer copies the registered
runtime representation without rerunning post-load processing. Eager
installation then refreshes q/k/v host
scale mirrors and invalidates FlashInfer launch-scale caches for recomputation
on the next forward. Graph-mode refits retain direct-copy behavior without
this refresh; captured scalar updates are not handled here. This does not clear
stored KV entries; retaining quantized KV entries across a change to their
K/V scales remains unsupported.

A successful peer install does not trigger checkpoint reconstruction. If a
later active refit cannot use a same-rank generator peer, that foreground refit
resolves the immutable full root and delta lineage before installation. The
receiver validates its local checkpoint under the cache lock and, when it is a
source-verified ancestor of the target, downloads and applies only the missing
revisions. The local checkpoint can lag GPU weights after P2P updates, so its
version determines the replay suffix. When no matching, source-verified local
checkpoint exists, the receiver reconstructs from the full root.

## SGLang backend

Use an SGLang build with the `/update_weights_from_modelexpress` endpoint and an
MX build exposing `get_modelexpress_generator()` in
`modelexpress_rl.inference.engines.sglang`. The ModelExpress package must be
installed in each SGLang worker's environment.

### Start SGLang

Pass the receiver settings through `--modelexpress-config`. The model name and
initial base ID must match the trainer and catalog records from the shared setup.

```bash
export MX_GENERATOR_SOURCE_ORDER=OBJECT_STORAGE

python -m sglang.launch_server \
  --model-path /models/Qwen3-30B-A3B \
  --host 0.0.0.0 --port 30000 --tp-size 4 \
  --modelexpress-config '{
    "model_name": "Qwen/Qwen3-30B-A3B",
    "server_url": "modelexpress:8001",
    "initial_base_version_id": "v0",
    "seed_checkpoint_path": "/models/Qwen3-30B-A3B",
    "refit_checkpoint_dir": "/var/cache/modelexpress",
    "refit_checkpoint_max_size_gb": 500,
    "object_storage_region_name": "us-west-2",
    "max_transfer_attempts": 3,
    "max_replay_chain_length": 64,
    "rpc_timeout_seconds": 30
  }'
```

The AWS example uses the default endpoint. For MinIO or another S3-compatible
endpoint, add `object_storage_endpoint_url` to the JSON configuration:

```json
{
  "object_storage_endpoint_url": "http://minio:9000",
  "object_storage_region_name": "us-east-1"
}
```

Both backends use these field names. MX maps them to `endpoint_url` and
`region_name` in `ObjectStorageGeneratorConfig`.

### Receiver initialization and ownership

Each TP worker creates its MX client on the first refit request and reuses it
for later versions. There is no separate initialization HTTP request. The
SGLang handler obtains the client through this public MX API:

```python
from modelexpress_rl.inference.engines.sglang import get_modelexpress_generator

generator = get_modelexpress_generator(model_runner)
```

MX parses the configuration, resolves the seed, validates the READY base version,
and caches the client on `model_runner.modelexpress_generator`. SGLang closes
that client during scheduler shutdown. If `seed_checkpoint_path` is omitted or
`null`, the factory resolves the launch checkpoint through the model runner's
loader. Use an explicit local seed path to select a different checkpoint.

The shared [seed requirements](#seed_checkpoint_path) and
[cache settings](#refit_checkpoint_dir) apply. Speculative draft-model refits and
incompatible shared or derived weight-cache modes are rejected by SGLang.

### Apply a published version

After [publishing `v1`](#publish-v1) and marking it READY, pause generation,
flush serving caches, refit, and resume:

```python
import requests

SGLANG_URL = "http://sglang-generator:30000"

for path, body in (
    ("pause_generation", {"mode": "abort"}),
    ("flush_cache", {}),
    ("update_weights_from_modelexpress", {"weight_version": "v1", "flush_cache": False}),
    ("continue_generation", {}),
):
    response = requests.post(f"{SGLANG_URL}/{path}", json=body, timeout=900)
    response.raise_for_status()
```

The explicit flush runs while generation is paused; `flush_cache=False` avoids
a second flush after installation. The handler stages and applies the version,
releases the staged handle, and combines TP results before reporting success.
A failed refit returns an error, so the sequence above leaves generation paused.

### Miles integration

With Miles-managed SGLang, select `--update-weight-transfer-mode modelexpress`
and pass `--modelexpress-config` to Miles. Include the receiver settings above,
the trainer's `object_storage_uri_prefix`, and an optional `full_hf_checkpoint_interval`.
Miles launches SGLang and owns the shared base initialization and publication
steps described in this guide.

Miles sends each real tensor bucket through `stage_shard(tensors=bucket)` and
publishes once all buckets have been submitted. Only sender ranks join the MX
publication group; other ranks still participate in framework weight gathering.
Miles pauses and flushes the engines before refit, then restores its numeric
weight-version label through `/update_weight_version` before resuming generation.
MX retains its opaque version ID for checkpoint lineage.

## Shared weight publication

Both backends consume the same immutable WeightVersions and S3 artifacts. The
trainer chooses whether each version is a delta or a full checkpoint; the
backend's apply request carries the target version ID.

### Publish `v1`

Create `v1` as STAGING, upload the delta shards and index, and then mark it
READY. The S3 URI is the exact index URI, not a directory prefix.

```python
import torch.distributed as dist

from modelexpress_rl import (
    ModelExpressControlClient,
    ObjectStorageSource,
    ObjectStorageType,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionState,
)

# The coordinator creates the target version.
if dist.get_rank(group=refit_process_group) == 0:
    with ModelExpressControlClient.connect(
        server_url="modelexpress:8001",
    ) as control:
        control.create_weight_version(
            uid="v1",
            model_name="Qwen/Qwen3-30B-A3B",
            idempotency_key="publish-v1",
            payload_format=WeightPayloadFormat.XOR_DELTA,
            base_version_id="v0",
            object_storage=ObjectStorageSource(
                storage_type=ObjectStorageType.S3,
                uri=f"{S3_URI_PREFIX}/v1/model.safetensors.index.json",
            ),
            state=WeightVersionState.STAGING,
        )
dist.barrier(group=refit_process_group)

# Every rank in refit_process_group computes and publishes its contribution.
# The global index is written after all rank-local delta shards are durable.
staged = trainer.stage_shard(
    version=WeightVersionRef("v1"),
    hf_tensor_iter=hf_tensor_buckets(),
)
staged.publish()
dist.barrier(group=refit_process_group)

# The coordinator exposes the completed version to generators.
if dist.get_rank(group=refit_process_group) == 0:
    with ModelExpressControlClient.connect(
        server_url="modelexpress:8001",
    ) as control:
        control.update_weight_version_state(
            "v1",
            WeightVersionState.READY,
        )
```

### Submit one bucket at a time

Frameworks with a bucket callback, such as Miles, can replace the iterator-based
stage/publish pair above with:

```python
for bucket in hf_tensor_buckets():
    staged = trainer.stage_shard(
        version=WeightVersionRef("v1"),
        tensors=bucket,
    )
staged.publish()
```

Each participating publisher submits at least one nonempty bucket. All calls for
the same version share one staged payload; `publish()` waits for pending bucket
processing before uploading. Submitted tensors must remain stable until
publication finishes. The iterator form completes staging before returning.

The next delta must be `v2` with `base_version_id="v1"`. An integration may
instead create a `FULL_HF_CHECKPOINT` version without `base_version_id`; that
version becomes the exact base for the following delta. Full checkpoint batches
may declare `checksum_format="adler32"` in the index and carry per-tensor
checksums under tensor-name keys in shard metadata. They are retained as an
immutable full artifact. XOR deltas require the complete index metadata contract
above and are replayed in order in the canonical lineage; each preparation
applies only the incoming delta to its exact active base.

## Shared checkpoint cache

### `seed_checkpoint_path`

When supplied, this must be a complete local safetensors checkpoint for
`initial_base_version_id`, readable by every inference engine worker. It may be
either:

- one unsharded `.safetensors` file containing the full model; or
- a directory containing all `.safetensors` shards. If
  `model.safetensors.index.json` is present, every shard referenced by its
  `weight_map` must also be present.

For typical sharded models such as Qwen3-30B-A3B, use the full Hugging Face
snapshot directory.

The backend sections describe how an omitted seed is resolved: vLLM can
[reuse a cached full root](#reuse-a-cached-seed), while SGLang's factory resolves
the launch checkpoint through its model runner.

### `refit_checkpoint_dir`

This is the root of ModelExpress's host-local immutable checkpoint cache. During
initialization, ModelExpress creates a model-specific subdirectory containing
full checkpoints, delta payloads, resolved chains, derived materializations,
and activation state.

`full/<version>/` and `deltas/<version>/` contain canonical immutable artifacts.
`chains/<version>.json` resolves a version to one full checkpoint plus its
ordered deltas. The first delta after a full checkpoint copies that immutable
full checkpoint into `materialized/<version>/`. Later sequential deltas rename
the active derived checkpoint and apply only the incoming delta in place, so
they do not copy the full model. Current vLLM and SGLang installers consume that
ordinary checkpoint directory. Materializations are derived and can be rebuilt
from the canonical lineage. If an in-place delta fails, the running engine keeps
its previous weights, the cache remains `UPDATING`, and initialization rebuilds
the checkpoint before accepting another update.

`state.json` records whether preparation is `READY` or `UPDATING` and protects
against interrupted writes. `active.json` changes only after engine installation
succeeds, so a failed download, reconstruction, or install retains the previous
active engine version. The cache lock coordinates artifact mutations. The
installation lock is held shared by concurrent co-located installers and
exclusively by preparation, preventing another preparation from entering before
activation.

Paths under `<refit_checkpoint_dir>/<URL-quoted-MX-model-name>/`:

| Path | Contents or role |
|---|---|
| `.lock` | Coordinates cache metadata and artifact mutations. |
| `.prepare.lock` | Serializes checkpoint preparation requests. |
| `.install.lock` | Shared during installation and exclusive during reconstruction. |
| `active.json`, `state.json` | Active version and preparation state. |
| `full/<version>/` | Immutable full checkpoint: safetensors, optional HF index, and files such as `config.json`. |
| `deltas/<version>/` | Delta index and compressed safetensors shards. |
| `chains/<version>.json` | Full root and ordered deltas for the version. |
| `materialized/<version>/` | Derived checkpoint consumed by the engine's native loader. |

The generator may request a target several revisions ahead of its active
version. ModelExpress first resolves the complete READY chain, rejecting cycles,
missing or incompatible revisions, and chains longer than
`max_replay_chain_length` (64 by default). It then prepares the ordered chain as
one immutable target checkpoint and installs only that final target. If engine
installation starts and fails, the checkpoint remains `READY`, `active.json`
continues to identify the last successfully installed version, and the local
engine is marked uncertain. The next request may reinstall that active version
or install any target reconstructed from it; either successful installation
clears the uncertain state.

All ranks sharing one host filesystem can share the same cache. Each host without
a shared filesystem needs its own cache.

A Kubernetes volume layout for either backend:

```yaml
spec:
  containers:
    - name: generator
      image: your-generator-image
      env:
        - name: HF_HOME
          value: /root/.cache/huggingface
      volumeMounts:
        # Immutable seed checkpoint in the default HF cache.
        - name: hf-cache
          mountPath: /root/.cache/huggingface
          readOnly: true

        # Host-local immutable artifacts and derived materializations.
        - name: mx-refit-checkpoint
          mountPath: /var/cache/modelexpress

  volumes:
    - name: hf-cache
      persistentVolumeClaim:
        claimName: huggingface-cache

    # To retain the prepared checkpoint across Pod recreation on the same node.
    # emptyDir is also a valid choice.
    - name: mx-refit-checkpoint
      hostPath:
        path: /var/lib/modelexpress/refit
        type: DirectoryOrCreate
```

The corresponding generator configuration would use:

```json
{
  "seed_checkpoint_path": "/root/.cache/huggingface/hub/models--ORG--MODEL/snapshots/SNAPSHOT_ID",
  "refit_checkpoint_dir": "/var/cache/modelexpress",
  "refit_checkpoint_max_size_gb": 500
}
```

`refit_checkpoint_max_size_gb` is a positive per-model quota in decimal
gigabytes (`1 GB = 1,000,000,000 bytes`) for payload files under `full/`,
`deltas/`, and `materialized/`. The SDK and vLLM backend default to 2000 GB;
the SGLang factory defaults to 500 GB. Set it explicitly to use the same quota
across backends, or set it to `null` to disable the configured quota.
At initialization, ModelExpress caps the quota at the existing model cache size
plus available filesystem space. A short INFO log reports the cap and free space
when this reduces the configured quota or replaces `null` with a disk-based
limit. This safety cap also applies when the configured quota is disabled.
ModelExpress rechecks free space before known writes and copies as other disk
usage changes. It evicts stale derived materializations before stale canonical
artifacts, but never evicts the active lineage or the checkpoint being prepared
or installed. Capacity must therefore cover the active checkpoint plus the
rollback-safe working set for one update. A capacity rejection preserves the
active checkpoint as READY so a later update can retry. On initialization, the
configured seed is restored as the initial full artifact and becomes the active
version.

## S3 artifact contract

The weight version's `object_storage.uri` points to a global JSON index. Shard
filenames in `weight_map` are resolved relative to that index.

### `XOR_DELTA`

```json
{
  "metadata": {
    "version": "v1",
    "base_version": "v0",
    "delta_encoding": "xor",
    "compression_format": "zstd",
    "checksum_format": "adler32"
  },
  "weight_map": {
    "model.layers.0.example.weight": "model-00000-of-00004.safetensors"
  }
}
```

The generator requires `weight_map`, `delta_encoding="xor"`,
`checksum_format="adler32"`, and a supported `compression_format` to select the
decompressor. `metadata.version` and `metadata.base_version` are optional
descriptive fields; the receiver does not compare them with MX IDs. This allows
existing S3 artifacts to be registered under different IDs without rewriting
their indexes.

Reconstruction and cache bookkeeping use the registered MX IDs and base
relationships. The caller must register the correct S3 artifact URI and exact
base checkpoint. MX's exact-base checks, tensor validation, and checksums remain
enabled.

Each delta shard contains compressed `U8` XOR bytes. Its safetensors
`__metadata__` must contain the Adler-32 checksum of every reconstructed full
tensor:

```json
{
  "__metadata__": {
    "model.layers.0.example.weight": "12ab34cd"
  },
  "model.layers.0.example.weight": {
    "dtype": "U8",
    "shape": [1234],
    "data_offsets": [0, 1234]
  }
}
```

### `FULL_HF_CHECKPOINT`

Full checkpoints use a standard Hugging Face safetensors index:

```json
{
  "metadata": {
    "total_size": 8,
    "checksum_format": "adler32"
  },
  "weight_map": {
    "model.layers.0.example.weight": "model-00001-of-00004.safetensors"
  }
}
```

The generator requires a non-empty `weight_map` covering exactly the local
checkpoint tensors. The index `metadata` field is optional. When it contains
`checksum_format`, the only supported value is `adler32`.

Each referenced shard contains native HF tensors. Safetensors `__metadata__`
may contain arbitrary string-to-string entries. When the index declares
`checksum_format="adler32"`, it must also contain a checksum keyed by tensor
name for every referenced tensor in the shard:

```json
{
  "__metadata__": {
    "format": "pt",
    "model.layers.0.example.weight": "12ab34cd"
  },
  "model.layers.0.example.weight": {
    "dtype": "F32",
    "shape": [2],
    "data_offsets": [0, 8]
  }
}
```

When the index omits `checksum_format`, checksum verification is skipped and
shard metadata is not interpreted as checksums. Tensor names, dtypes, shapes,
and byte sizes are always checked before the immutable full artifact is
promoted.
