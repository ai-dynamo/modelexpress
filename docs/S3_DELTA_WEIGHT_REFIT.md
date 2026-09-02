# S3 Delta Weight Refit

This guide shows how to publish XOR-delta weight updates to S3 and install them
in a running vLLM model with ModelExpress. The trainer and generator start from
the same local checkpoint; integrations may use a framework-selected cadence of
full HF checkpoints to reset that base. ModelExpress coordinates each version's
lineage and readiness.

## Components

| Component | Responsibility |
|---|---|
| `ModelExpressControlClient` | Create and transition immutable weight-version records in the ModelExpress catalog. |
| `ModelExpressTrainerClient` | Capture the seed-checkpoint base and publish either XOR deltas or full HF checkpoint batches to S3. |
| `ModelExpressGeneratorClient` | Validate READY versions, apply the requested S3 payload to its refit checkpoint, and reload the live model. |

## Requirements

- A Redis-backed ModelExpress Refit service.
- A ModelExpress build containing the canonical S3 delta and vLLM integrations.
- The `modelexpress` Python package installed in both the trainer environment
  and the environment used to launch vLLM.
- vLLM 0.27.1.
- Trainer ranks with S3 read/write access and vLLM hosts with read access.
- The corresponding base checkpoint in safetensors format on every trainer and
  vLLM host.

Minimal ModelExpress server configuration:

```bash
export MX_METADATA_BACKEND=redis
export REDIS_URL=redis://redis:6379
```

Environment variables used by the clients and vLLM engine:

| Variable | Default | Purpose |
|---|---|---|
| `MX_SERVER_ADDRESS` | `localhost:8001` | ModelExpress server address. |
| `MX_AUTH_TOKEN_PATH` | unset | Optional ModelExpress bearer-token file. |
| `MX_AUTH_TOKEN_TTL_SECONDS` | `60` | Token-file reread interval in seconds. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | unset | S3 credentials when an IAM role or workload identity is unavailable. |
| `AWS_SESSION_TOKEN` | unset | Session token when using temporary AWS credentials. |
| `AWS_DEFAULT_REGION` | unset | S3 region when `region_name` is not supplied in client configuration. |
| `MX_REFIT_DELTA_BUCKET_BYTES` | `536870912` (512 MiB) | Optional tensor-bucket size override for framework integrations. |
| `MX_REFIT_DELTA_WORKERS` | `min(32, CPU count)` | CPU workers used to compute and apply XOR deltas. |
| `MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES` | `4294967296` (4 GiB) | Maximum tensor bytes grouped into one full-checkpoint safetensors object. |
| `MX_S3_UPLOAD_WORKERS` | `8` | Maximum concurrent multipart uploads per trainer rank. |
| `MX_S3_DOWNLOAD_WORKERS` | `16` | Generator download concurrency. |
| `MX_S3_MAX_POOL_CONNECTIONS` | `32` | Botocore HTTP connection-pool size. |
| `MX_S3_MAX_ATTEMPTS` | `5` | Total S3 request attempts. |
| `VLLM_SERVER_DEV_MODE` | unset | Set to `1` to enable vLLM's weight-update HTTP routes. Network-isolate these routes. |
| `VLLM_PLUGINS` | unset | Set to `modelexpress` to load the ModelExpress vLLM plugin. |

Optional S3 tuning variables and defaults are documented in
[`modelexpress_client/python/README.md`](../modelexpress_client/python/README.md#canonical-s3-transfer-tuning).

## Initialization

### 1. Create the base version

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

### 2. Initialize the trainer client

Initialize one trainer client on every distributed trainer rank and keep it for
the full training run. `hf_tensor_buckets()` below represents a framework-owned
function that returns a fresh iterable of Hugging Face tensor buckets.

```python
import torch.distributed as dist

trainer = ModelExpressTrainerClient.initialize(
    ModelExpressTrainerConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
        payload_format=WeightPayloadFormat.XOR_DELTA,
        server_url="modelexpress:8001",
        process_group=...,  # Gloo process group for control plane
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

Use a dedicated Gloo process group containing every participating trainer rank.
ModelExpress uses it for CPU object collectives that coordinate shard and index
publication; do not reuse the model-training NCCL group for this example.

### 3. Initialize the generator

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

vllm serve /models/Qwen3-30B-A3B \
  --tensor-parallel-size 4 \
  --weight-transfer-config '{"backend":"modelexpress"}'
```

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
            "object_storage_region_name": "us-west-2",
            "max_transfer_attempts": 3,
            "rpc_timeout_seconds": 30,
        }
    },
    timeout=900,
)
response.raise_for_status()
```

#### `seed_checkpoint_path`

This must be a complete local safetensors checkpoint for
`initial_base_version_id`, readable by every inference engine worker. It may be
either:

- one unsharded `.safetensors` file containing the full model; or
- a directory containing all `.safetensors` shards. If
  `model.safetensors.index.json` is present, every shard referenced by its
  `weight_map` must also be present.

For typical sharded models such as Qwen3-30B-A3B, use the full Hugging Face
snapshot directory.

#### `refit_checkpoint_dir`

This is the root of ModelExpress's mutable working checkpoint. During
initialization, ModelExpress creates a model-specific subdirectory containing
`checkpoint/`, `state.json`, and `.lock`.

The `checkpoint/` directory starts as a full copy of `seed_checkpoint_path` and
is updated in place as sequential deltas are applied. `state.json` records its
current version and whether the checkpoint is `READY` or `UPDATING`. The file
lock prevents co-located ranks from seeding or updating the same checkpoint
concurrently. With an HF snapshot seed checkpoint, the resulting layout is:

Before seeding or mutating checkpoint bytes, ModelExpress writes `UPDATING` with
the last READY version. It writes `READY` for the target only after all writes
and verification succeed. Preparation and installation reject `UPDATING`; a
fresh initialization reseeds it from `seed_checkpoint_path`.

```text
<refit_checkpoint_dir>/<URL-quoted-vLLM-model-path-or-ID>/
  .lock
  state.json
  checkpoint/
    *.safetensors
    config.json
    ...
```

All ranks sharing one host filesystem can share the same cache. Each host without
a shared filesystem needs its own cache.

A Kubernetes example:

```yaml
spec:
  containers:
    - name: vllm
      image: your-vllm-image
      env:
        - name: HF_HOME
          value: /root/.cache/huggingface
      volumeMounts:
        # Immutable seed checkpoint in the default HF cache.
        - name: hf-cache
          mountPath: /root/.cache/huggingface
          readOnly: true

        # Mutable, host-local refit checkpoint directory.
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
  "refit_checkpoint_dir": "/var/cache/modelexpress"
}
```

Ensure the host directory has space for at least one complete checkpoint copy.
ModelExpress creates the model-specific subdirectory automatically. On
initialization, it reuses the cache only when its recorded version is
`initial_base_version_id`, its state is `READY`, and safetensors files are
present. A later-version or interrupted `UPDATING` cache is reseeded from
`seed_checkpoint_path`.

#### Initialization behavior

`POST /init_weight_transfer_engine` fans out to every vLLM worker. Each worker:

1. initializes or reuses its refit checkpoint;
2. fetches `initial_base_version_id` from the ModelExpress server;
3. verifies that the base is READY and has the configured model name; and
4. registers itself as a ModelExpress generator worker.

Initialization fails before serving updates if any worker cannot read the
seed checkpoint, write the cache, or validate the base version.

## Weight Update

### 1. Publish `v1`

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

# Every trainer rank computes and publishes its contribution. This writes the
# global index after all rank-local delta shards are durable.
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

The index has this shape; shard names are resolved relative to its parent URI:

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

### 2. Apply `v1` in vLLM

Run the update while generation is paused. `mode=abort` clears active requests
and vLLM caches before the weight update.

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
try:
    post("start_weight_update", body={})
    post("update_weights", body={"update_info": {"version_id": "v1"}})
    post("finish_weight_update", body={"weight_version": "v1"})
finally:
    post("resume", body={})
```

The next delta must be `v2` with `base_version_id="v1"`. An integration may
instead create a `FULL_HF_CHECKPOINT` version without `base_version_id`; that
version becomes the exact base for the following delta. Full checkpoint batches
carry per-tensor Adler-32 checksums and are copied directly into the existing
refit checkpoint as they download. XOR deltas require
`compression_format="zstd"`.
