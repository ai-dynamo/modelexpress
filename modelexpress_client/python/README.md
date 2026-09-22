# ModelExpress Python client

ModelExpress helps inference workers load model weights and RL rollout workers install new trainer versions. This package supplies the runtime integrations, Python clients, and weight-transfer code; your inference or training runtime owns GPU execution.

- **Inference:** start with the [vLLM Kubernetes quickstart](../../docs/integrations/runtimes/vllm.md), then use [SGLang](../../docs/integrations/runtimes/sglang.md), [TensorRT-LLM](../../docs/integrations/runtimes/tensorrt-llm.md), or [Dynamo](../../docs/integrations/orchestrators/dynamo.md) if that is your serving stack.
- **RL:** start with [RL weight updates](../../docs/guides/rl.md) for trainer-to-rollout refit and version handling.
- **Direct storage loading:** use the [ModelStreamer examples](../../examples/model_streamer_k8s/README.md) to load safetensors without an MX server.

## Installation

Install into your runtime environment using Python 3.10 or newer. From the ModelExpress repository root:

```bash
pip install ./modelexpress_client/python

# For client development and tests
pip install -e './modelexpress_client/python[dev]'

# Only when regenerating protobuf APIs
pip install -e './modelexpress_client/python[codegen]'
```

The package does not install vLLM, SGLang, TensorRT-LLM, or NIXL. Use a [supported runtime image](../../docs/COMPATIBILITY.md) with a compatible CUDA/NIXL stack. Prefer the NIXL package supplied by that image; for a bare environment, install `nixl-cu12` or `nixl-cu13` to match the runtime's CUDA stack.

P2P requires compatible GPU workers, a working NIXL transport, and source discovery through a ModelExpress server or the `k8s-service` backend. Direct ModelStreamer storage loading does not require NIXL, RDMA, or a server. Ordinary Python control-plane API calls do not require a GPU.

## How inference loading works

1. A source worker loads a checkpoint through an eligible storage loader and publishes its availability.
2. A compatible target discovers the source and copies its ready GPU tensors through NIXL. For vLLM, these are **post-processed** tensors, including the layouts produced by quantization processing; the target reconstructs the corresponding layout before receiving them.
3. The runtime finishes startup and serves requests. Verify the target's transfer completion log and an inference response; automatic fallback can make a healthy worker look like a successful P2P run.

The central server carries P2P discovery metadata, while weight bytes move between workers. Sources must stay available during transfer. Targets still need configuration and tokenizer files. Optional `MX_ARTIFACT_TRANSFER=1` also reuses compatible vLLM JIT caches from a healthy source. See [Choose a path](../../docs/guides/choose-a-path.md) for storage and offline deployment choices.

## Programmatic usage

### MxClient

`MxClient` is a lightweight gRPC client for communicating with the ModelExpress server. This example lists ready sources; runtime integrations perform compatibility checks and transfers:

```python
from modelexpress import MxClient, p2p_pb2

client = MxClient(server_url="modelexpress-server:8001")
try:
    response = client.list_sources(status_filter=p2p_pb2.SOURCE_STATUS_READY)
    for source in response.instances:
        print(source.model_name, source.worker_rank, source.worker_id)
finally:
    client.close()
```

### Registering Loaders Manually

Manual registration is only needed for integrations that construct vLLM loaders outside vLLM 0.23.0's native load-format path.

```python
from modelexpress import register_modelexpress_loaders

register_modelexpress_loaders()
# Now vLLM recognizes --load-format modelexpress and mx
```

### RL trainer publication

Start with [RL weight updates](../../docs/guides/rl.md). The [RL integration reference](../../docs/RL_REFIT.md#trainer-publication) contains the trainer API, lifecycle requirements, and adapter contracts.

## Environment variables

For loading policy, defaults, and deployment settings, use the [configuration reference](../../docs/CONFIGURATION.md). RL-specific settings are in the [refit reference](../../docs/RL_REFIT.md#client-settings). `MX_LOAD_STRATEGY_CHAIN=INFERENCE` is the default; `RL` selects the separate RL startup policy and does not by itself perform an update on a running worker.

<details>
<summary>Advanced client and transport settings</summary>

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_SERVER_ADDRESS` | `localhost:8001` | ModelExpress gRPC server address (recommended) |
| `MODEL_EXPRESS_URL` | `localhost:8001` | Legacy server address; takes precedence over `MX_SERVER_ADDRESS` when both are set. Prefer `MX_SERVER_ADDRESS`; if an older integration requires both, set them to the same endpoint. |
| `MX_DISABLE_PATCHES` | `0` | Emergency escape hatch that skips all runtime compatibility patches. Set to `1`, `true`, `yes`, or `on` if a patch is incompatible with the installed engine. |
| `MX_POOL_REG` | `0` | Allocation-level NIXL registration (registers cudaMalloc blocks instead of individual tensors) |
| `MX_P2P_METADATA` | `1` | Serve tensor and artifact manifests directly from source workers; set to `0` to route full tensor metadata through the central server |
| `MX_LOAD_STRATEGY_CHAIN` | `INFERENCE` | Select the engine-neutral initial-load policy. With `RL`, a configured desired UID permits only desired-version P2P and S3 replay; without one, loading falls back through `MX_MODEL_URI` and then the engine default. vLLM speculative draft models are rejected in `RL` mode. |
| `MX_ARTIFACT_TRANSFER` | `0` | Transfer compatible vLLM TorchInductor, Triton, DeepGEMM, TileLang, CuTe DSL, and FlashInfer JIT caches, including persistent autotune files when supported by vLLM |
| `MX_ARTIFACT_BUNDLE_ROOT` | `$TMPDIR/modelexpress-artifacts` | Staging root for tarred cache artifact bundles |
| `MX_ARTIFACT_COMPILE_CONFIG_DIGEST` | empty | Optional compile-configuration compatibility digest for cache discovery |
| `MX_ARTIFACT_READY_URL` | Framework default | Readiness endpoint checked before a source publishes weights or JIT cache artifacts (`http://127.0.0.1:8000/health` for vLLM; `http://127.0.0.1:30000/health` for SGLang). On the non-head nodes of a multi-node engine, a loopback host is rewritten onto the head's address (the engine's own distributed-init address, else `LWS_LEADER_ADDRESS`), preserving the configured port and path. A non-loopback host is used verbatim |
| `MX_ARTIFACT_READY_TIMEOUT_SECS` | `1800` | Maximum time to wait for readiness and successful artifact publication |
| `MX_HEARTBEAT_INTERVAL_SECS` | `30` | Seconds between READY status heartbeats for published sources, including reshard rendezvous sources; keep below the server heartbeat timeout |

### Canonical S3 Transfer Tuning

See [S3 transfer tuning](../../docs/S3_DELTA_WEIGHT_REFIT.md#transfer-tuning) for upload, download, concurrency, and memory settings.

### UCX/NIXL Tuning

| Variable | Recommended | Description |
|----------|-------------|-------------|
| `UCX_RNDV_SCHEME` | `get_zcopy` | Zero-copy RDMA reads |
| `UCX_RNDV_THRESH` | `0` | Force rendezvous for all transfers |
| `NIXL_LOG_LEVEL` | `INFO` | NIXL logging level |

</details>

## Package Structure

| Module | Description |
|--------|-------------|
| `modelexpress.client` | `MxClient` -- gRPC client for the ModelExpress server |
| `modelexpress.metadata` | Metadata clients, source identity, publishing, and worker manifest serving |
| [`modelexpress.refit`](modelexpress/refit/README.md) | Shared geometry, transfer planning, transport, and timing primitives |
| `modelexpress.engines.vllm.loader` | `MxModelLoader` -- vLLM integration |
| `modelexpress.refit.reshard` | Engine-agnostic loader-geometry capture and bounded no-gather transfer planning |
| `modelexpress.engines.sglang.loader` | `MxModelLoader` -- SGLang `remote_instance` integration |
| `modelexpress.engines.trtllm.loader` | `MxModelLoader` -- TensorRT-LLM shared-strategy integration |
| `modelexpress.vllm_loader` | Compatibility shim for the vLLM loader |
| `modelexpress.nixl_transfer` | `NixlTransferManager` -- NIXL agent lifecycle and RDMA transfers |
| `modelexpress.types` | `TensorDescriptor`, `WorkerMetadata` -- core data types |
| `modelexpress.vllm_worker` | Compatibility worker extension for older manual-registration workflows |

## License

Apache-2.0
