# ModelExpress P2P Transfer - Engineering Context

This document provides deep technical context for engineers continuing development on the ModelExpress P2P GPU weight transfer system. Use this as a reference when resuming work with Cursor or any AI-assisted coding session.

## Project Overview

ModelExpress enables high-performance GPU-to-GPU model weight transfers between vLLM instances using NVIDIA NIXL over RDMA/InfiniBand. The primary use case is rapid model loading: instead of each vLLM instance downloading weights from storage, one "source" instance loads the model and transfers weights directly to "target" instances via GPU memory.

### Key Value Proposition

- **Speed**: Transfer 140GB model weights in ~4-5 seconds vs. minutes from network storage
- **Efficiency**: GPU-to-GPU transfers bypass CPU entirely (zero-copy)
- **Scalability**: Coordinate transfers across multiple vLLM instances in a cluster

## Architecture Overview

```
   Node A (first instance)                    Node B (second instance)
   +-------------------------+                +-------------------------+
   | vLLM + MxWorkerExtension|                | vLLM + MxWorkerExtension|
   | (NIXL agent + tensors)  | <== RDMA ===> | (NIXL agent + tensors)  |
   +----------+--------------+                +----------+--------------+
              | ZMQ (IPC)                                | ZMQ (IPC)
              v                                          v
   +-------------------------+                +-------------------------+
   | ModelExpress Client     |                | ModelExpress Client     |
   | - Query server: no src  |                | - Query server: found A |
   | - Publish metadata      |                | - Tell worker: receive  |
   |                         |                | - Publish metadata      |
   +-------------------------+                +-------------------------+
              ^                                          ^
              |              ModelExpress Server         |
              +--------->  (model_name -> metadata)  <---+
```

### Key Design Principles

1. **Symmetric Clients**: Every client can be either source or target. Role is determined dynamically based on whether metadata for the model already exists.

2. **Model-Name Keyed Storage**: The server stores metadata keyed by model name. First instance to publish becomes the source; subsequent instances query and receive.

3. **NIXL in vLLM Workers**: NIXL agents live in vLLM worker processes (not the client) because GPU memory must be registered by the owning process for GPUDirect RDMA. CUDA IPC-mapped memory cannot be registered with `ibv_reg_mr`.

4. **Tensor Parallelism (TP) Support**: Each GPU worker gets its own ZMQ socket and NIXL agent. Transfers are rank-matched (source rank 0 -> target rank 0, etc.).

5. **Collective RPC Integration**: The client triggers weight server startup via vLLM's `collective_rpc` endpoint before connecting via ZMQ.

## Tensor Parallelism (TP > 1)

The architecture fully supports tensor parallelism. When running with TP > 1:

### How TP Works

1. **vLLM Extension**: When `start_weight_server()` is called, each worker process:
   - Generates a unique ZMQ address by appending its rank
   - Creates a NIXL agent and registers GPU tensors for GPUDirect RDMA
   - Starts ZMQ server to serve metadata and handle transfer commands
   - Addresses: Base `ipc:///tmp/vllm.sock` becomes `ipc:///tmp/vllm-0.sock`, `ipc:///tmp/vllm-1.sock`, etc.

2. **Client**: Connects to ALL worker ZMQ sockets, fetches NIXL metadata, and orchestrates transfers.

3. **Transfers**: Matched by worker rank (source rank 0 -> target rank 0). Each worker executes its own RDMA transfer.

### TP=4 Architecture

```
Source Node                                      Target Node
+------------------+                             +------------------+
| Worker 0 (NIXL)  |<=========== RDMA ==========>| Worker 0 (NIXL)  |
| Worker 1 (NIXL)  |<=========== RDMA ==========>| Worker 1 (NIXL)  |
| Worker 2 (NIXL)  |<=========== RDMA ==========>| Worker 2 (NIXL)  |
| Worker 3 (NIXL)  |<=========== RDMA ==========>| Worker 3 (NIXL)  |
+------------------+                             +------------------+
        |                                                |
        | ZMQ                                       ZMQ |
        v                                                v
+------------------+                             +------------------+
|     Client       |                             |     Client       |
| (orchestration)  |                             | (orchestration)  |
+------------------+                             +------------------+
        |                                                |
        v                                                v
+--------------------------------------------------------------+
|                    ModelExpress Server                       |
|                  (model metadata by name)                    |
+--------------------------------------------------------------+
```

### TP Usage Examples

**vLLM with TP=4** (workers auto-create unique ZMQ addresses when `start_weight_server` is called):
```bash
MX_ZMQ_ADDRESS=ipc:///tmp/vllm.sock \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B \
    --tensor-parallel-size 4 \
    --worker-extension-cls modelexpress.vllm_extension.MxWorkerExtension
```

**Client with TP=4** (auto-generates addresses from base):
```bash
python -m modelexpress.client \
    --model-name meta-llama/Llama-3.1-70B \
    --zmq-base ipc:///tmp/vllm.sock \
    --tp-size 4 \
    --engine-address http://localhost:8000 \
    --server-address modelexpress:8001
```

**Client with explicit addresses**:
```bash
python -m modelexpress.client \
    --model-name meta-llama/Llama-3.1-70B \
    --zmq-addresses \
        ipc:///tmp/vllm-0.sock \
        ipc:///tmp/vllm-1.sock \
        ipc:///tmp/vllm-2.sock \
        ipc:///tmp/vllm-3.sock \
    --device-ids 0 1 2 3
```

## Repository Structure

```
modelexpress/                     # Repository root
├── modelexpress_server/          # Rust gRPC server
│   └── src/
│       ├── main.rs               # Server entrypoint
│       ├── p2p_service.rs        # P2P gRPC service (PublishMetadata, GetMetadata)
│       ├── state.rs              # Redis-backed model metadata storage
│       ├── services.rs           # Health/API/Model services
│       ├── database.rs           # SQLite database for model cache
│       ├── cache.rs              # Cache eviction service
│       ├── config.rs             # Server configuration
│       └── bin/
│           └── config_gen.rs     # Configuration file generator
├── modelexpress_client/
│   ├── src/                      # Rust client library + CLI
│   │   ├── lib.rs
│   │   └── bin/
│   │       ├── cli.rs            # CLI tool (modelexpress-cli)
│   │       ├── test_client.rs    # Test client
│   │       └── fallback_test.rs  # Fallback testing
│   └── python/
│       └── modelexpress/
│           ├── client.py         # Main P2P client process (NIXL transfers)
│           ├── vllm_extension.py # vLLM worker extension (ZMQ weight server)
│           ├── nixl_transfer.py  # NIXL transfer manager
│           ├── types.py          # Data types (TensorDescriptor, WorkerMetadata)
│           ├── p2p_pb2.py        # Generated protobuf
│           └── p2p_pb2_grpc.py   # Generated gRPC stubs
├── modelexpress_common/
│   ├── proto/
│   │   ├── p2p.proto             # P2P metadata service definition
│   │   ├── model.proto           # Model download service
│   │   ├── health.proto          # Health check service
│   │   └── api.proto             # General API service
│   └── src/
│       ├── lib.rs
│       ├── models.rs
│       ├── config.rs
│       ├── download.rs
│       └── providers/            # Model providers (HuggingFace)
├── examples/
│   ├── aggregated_k8s/           # Dynamo integration example
│   └── p2p_transfer_k8s/         # P2P GPU transfer Kubernetes manifests
│       ├── redis.yaml
│       ├── modelexpress-server.yaml
│       ├── vllm-source.yaml
│       ├── vllm-target.yaml
│       └── Dockerfile.client
├── helm/                         # Helm chart for Kubernetes deployment
├── workspace-tests/              # Integration tests and benchmarks
│   ├── tests/
│   │   └── integration_tests.rs
│   └── benches/
│       └── performance.rs
├── docs/
│   ├── CLI.md                    # CLI documentation
│   ├── CONTEXT.md                # This file
│   └── QUICK_START.md
├── docker-compose.yml
├── Dockerfile
├── k8s-deployment.yaml
└── run_integration_tests.sh
```

## Core Components

### 1. ModelExpress Server (Rust)

**Location**: `modelexpress_server/`

The server provides multiple gRPC services and stores model metadata via Redis for P2P coordination. It does NOT participate in data transfer - only coordination.

**gRPC Services**:

| Service | Description |
|---------|-------------|
| `P2pService` | P2P metadata storage and retrieval |
| `ModelService` | Model download and file streaming |
| `HealthService` | Health check endpoints |
| `ApiService` | General API endpoints |

**P2P gRPC APIs** (defined in `p2p.proto`):

| RPC | Description |
|-----|-------------|
| `PublishMetadata` | Client publishes NIXL metadata + tensor descriptors for a model |
| `GetMetadata` | Client queries for existing source with same model name |

**State Storage**: Redis
- Model metadata: `mx:model:{model_name}` (JSON with workers, tensors)
- Model set: `mx:models` (set of all registered model names)

**Key Data Structures**:

```rust
// Per-worker metadata stored in Redis
struct WorkerRecord {
    worker_rank: u32,
    nixl_metadata: Vec<u8>,  // Serialized NIXL agent metadata
    tensors: Vec<TensorRecord>,
}

struct TensorRecord {
    name: String,
    addr: u64,      // GPU memory address
    size: u64,      // Size in bytes
    device_id: u32,
    dtype: String,
}
```

### 2. ModelExpress Client (Python)

**Location**: `modelexpress_client/python/modelexpress/client.py`

The client orchestrates transfers between vLLM workers. It does NOT create NIXL agents itself (those live in vLLM workers). The client handles:

1. **Trigger weight server via collective_rpc** to start ZMQ servers and NIXL agents on vLLM workers
2. **Connect to vLLM workers via ZMQ** to get NIXL metadata and tensor descriptors
3. **Query ModelExpress server** for existing sources with the same model
4. **Instruct workers to receive** via ZMQ if a source exists (workers execute RDMA)
5. **Publish metadata** to become a source for future instances

**Key Classes**:
- `ModelExpressClient`: Orchestrates transfers via ZMQ commands to workers

**Client Flow**:
```python
def run(self, zmq_addresses, zmq_base):
    # 1. Connect to vLLM workers and fetch NIXL metadata
    self.connect_to_workers(zmq_addresses, zmq_base)
    self.fetch_all_worker_metadata()

    # 2. Query for existing source
    response = self._get_metadata(self.model_name)

    if response.found:
        # 3a. Instruct workers to receive from source
        self.receive_from_source(response.workers)

    # 4. Publish metadata (become a source)
    self.publish_metadata()
```

**Usage**:
```bash
python -m modelexpress.client \
    --model-name meta-llama/Llama-3.1-70B \
    --zmq-base ipc:///tmp/vllm.sock \
    --tp-size 4 \
    --engine-address http://localhost:8000 \
    --server-address modelexpress:8001
```

### 3. vLLM Worker Extension (Python)

**Location**: `modelexpress_client/python/modelexpress/vllm_extension.py`

The extension is loaded by vLLM and handles NIXL registration and transfers. NIXL agents live here (not in the client) because GPU memory must be registered by the owning process for GPUDirect RDMA.

**Key Features**:
- Creates NIXL agent and registers GPU tensors for GPUDirect RDMA
- Starts ZMQ server when `start_weight_server()` is called via collective_rpc
- Auto-generates rank-specific ZMQ addresses for TP > 1
- Executes RDMA transfers on `receive_from` command from client

**Loading in vLLM**:
```bash
MX_ZMQ_ADDRESS=ipc:///tmp/vllm.sock \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B \
    --tensor-parallel-size 4 \
    --worker-extension-cls modelexpress.vllm_extension.MxWorkerExtension
```

**ZMQ Protocol**:

| Request | Response |
|---------|----------|
| `{"cmd": "get_metadata"}` | `{worker_rank, nixl_metadata, tensors}` |
| `{"cmd": "receive_from", nixl_metadata, tensors, ...}` | `{success, bytes_transferred, tensors_transferred}` |
| `{"cmd": "done"}` | "ok" and shutdown signal |

### 4. NIXL Transfer Manager (Python)

**Location**: `modelexpress_client/python/modelexpress/nixl_transfer.py`

Manages a single NIXL agent and RDMA transfers for one GPU. Each vLLM worker creates its own instance.

**Key Concepts**:
- **Single-Worker Design**: Each instance manages one NIXL agent for one GPU
- **Metadata Exchange**: NIXL metadata is serialized and exchanged via ModelExpress server
- **Memory Registration**: GPU memory must be registered by the owning process (vLLM worker)
- **Prepped Transfers**: For batched efficiency, use `prep_xfer_dlist` and `make_prepped_xfer`

**Usage Pattern** (in vLLM extension):
```python
# Initialize and register tensors
manager = NixlTransferManager(agent_name="vllm-abc123-0", device_id=0)
manager.initialize()
manager.register_tensors(tensors)

# Receive from remote source
manager.receive_from_source(
    source_metadata=remote_nixl_metadata,
    source_tensors=remote_tensor_descriptors,
)
```

### 5. NIXL (NVIDIA Interconnect eXchange Library)

NIXL enables zero-copy GPU-to-GPU transfers over various backends (UCX/RDMA, IPC, GDS).

The current implementation uses:
- `nixl._api.nixl_agent` - Agent class
- `nixl._api.nixl_agent_config` - Configuration with UCX backend
- `agent.register_memory()` - Register GPU tensors
- `agent.get_agent_metadata()` - Get serialized agent metadata
- `agent.add_remote_agent()` - Add remote agent from metadata
- `agent.prep_xfer_dlist()` - Prepare transfer descriptor list
- `agent.make_prepped_xfer()` - Create prepared transfer handle
- `agent.transfer()` - Execute transfer
- `agent.check_xfer_state()` - Check transfer status

## Kubernetes Deployment

### Pod Configuration

**Source Instance** (first to start):
- Loads real weights from HuggingFace
- Client publishes metadata to become source

**Target Instance** (starts later):
- Starts with `--load-format dummy` (random weights)
- Client queries server, finds source, receives weights
- Also publishes metadata (becomes additional source)

### Environment Variables

```yaml
env:
  # For vLLM extension
  - name: MX_ZMQ_ADDRESS
    value: "ipc:///tmp/mx/vllm.sock"     # Base ZMQ address (rank appended)

  # For client
  - name: MX_SERVER
    value: "modelexpress-server:8001"    # gRPC server address
  - name: MX_TP_SIZE
    value: "4"                            # Tensor parallel size
  - name: MX_ENGINE_ADDRESS
    value: "http://localhost:8000"        # vLLM engine for collective_rpc

  # NIXL/UCX configuration
  - name: UCX_TLS
    value: "rc_x,rc,dc_x,dc,cuda_copy"    # RDMA transports
  - name: UCX_RNDV_SCHEME
    value: "get_zcopy"                     # Zero-copy rendezvous

  # ModelExpress Server
  - name: REDIS_URL
    value: "redis://modelexpress-redis:6379"
```

**ZMQ Address Pattern (TP Support)**:
- When `MX_ZMQ_ADDRESS=ipc:///tmp/mx/vllm.sock` is set:
  - Worker rank 0 binds to `ipc:///tmp/mx/vllm-0.sock`
  - Worker rank 1 binds to `ipc:///tmp/mx/vllm-1.sock`
  - etc.
- The client uses `--zmq-base` + `--tp-size` to auto-generate matching addresses

### Deployment Steps

1. **Deploy Redis** (state storage)
2. **Deploy ModelExpress Server**
3. **Deploy Source vLLM Instance** with extension + client
4. **Deploy Target vLLM Instance** with extension + client

The target client will automatically:
- Start weight server via collective_rpc
- Query server for the model
- Find source metadata
- Initiate RDMA transfer
- Receive weights
- Publish its own metadata

## Proto Definitions

### P2P Service (p2p.proto)

```protobuf
service P2pService {
  rpc PublishMetadata(PublishMetadataRequest) returns (PublishMetadataResponse);
  rpc GetMetadata(GetMetadataRequest) returns (GetMetadataResponse);
}

message TensorDescriptor {
  string name = 1;
  uint64 addr = 2;
  uint64 size = 3;
  uint32 device_id = 4;
  string dtype = 5;
}

message WorkerMetadata {
  uint32 worker_rank = 1;
  bytes nixl_metadata = 2;
  repeated TensorDescriptor tensors = 3;
}

message PublishMetadataRequest {
  string model_name = 1;
  repeated WorkerMetadata workers = 2;
}

message GetMetadataRequest {
  string model_name = 1;
}

message GetMetadataResponse {
  bool found = 1;
  repeated WorkerMetadata workers = 2;
}
```

### Model Service (model.proto)

```protobuf
service ModelService {
  rpc EnsureModelDownloaded(ModelDownloadRequest) returns (stream ModelStatusUpdate);
  rpc StreamModelFiles(ModelFilesRequest) returns (stream FileChunk);
  rpc ListModelFiles(ModelFilesRequest) returns (ModelFileList);
}
```

### Health Service (health.proto)

```protobuf
service HealthService {
  rpc GetHealth(HealthRequest) returns (HealthResponse);
}
```

## Debugging Commands

```bash
# Check pod status
kubectl get pods -o wide

# Stream client logs
kubectl logs -f deployment/mx-source | grep -E "Client|NIXL|transfer"

# Check vLLM extension logs
kubectl logs -f deployment/mx-source | grep "\[MX\]"

# Check Redis state
kubectl exec deployment/redis -- redis-cli KEYS "mx:*"
kubectl exec deployment/redis -- redis-cli GET "mx:model:meta-llama/Llama-3.1-70B"

# Check ZMQ sockets are created
kubectl exec -it deployment/mx-source -- ls -la /tmp/mx/

# Verify InfiniBand is working
kubectl exec -it deployment/mx-source -- ibstat

# Check UCX configuration
kubectl exec -it deployment/mx-source -- ucx_info -d
```

## Testing Transfer Success

```bash
# Both instances should produce identical outputs
SOURCE_RESP=$(kubectl exec deployment/mx-source -- curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-70b", "prompt": "The capital of France is", "max_tokens": 5, "temperature": 0}')

TARGET_RESP=$(kubectl exec deployment/mx-target -- curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-70b", "prompt": "The capital of France is", "max_tokens": 5, "temperature": 0}')

echo "Source: $SOURCE_RESP"
echo "Target: $TARGET_RESP"
# Both should respond with "Paris" or similar
```

## Known Issues and Solutions

### Issue 1: Client can't connect to ZMQ socket

**Symptom**: `ZMQ connection refused` or timeout
**Root Cause**: vLLM extension hasn't started weight server yet
**Solution**: The client now calls collective_rpc to trigger `start_weight_server()` before connecting

### Issue 2: NIXL transfer fails with "not found"

**Symptom**: Target fails during `receive_from_source`
**Root Cause**: Source NIXL metadata was not captured correctly
**Solution**: Ensure `get_agent_metadata()` is called after `register_memory()`

### Issue 3: Model name mismatch

**Symptom**: Target doesn't find source even though source is running
**Root Cause**: Model names don't match exactly (e.g., trailing slashes)
**Solution**: Use consistent model naming across all instances

### Issue 4: NIXL_ERR_REMOTE_DISCONNECT

**Symptom**: `nixlRemoteDisconnectError` during transfer
**Root Cause**: Source instance was terminated during transfer
**Solution**: Ensure source remains running until all transfers complete

### Issue 5: GPUDirect RDMA fails with "Bad address" (SOLVED)

**Symptom**: `ibv_reg_mr failed: Bad address` when registering GPU memory
**Root Cause**: CUDA IPC-mapped memory cannot be registered with InfiniBand for GPUDirect RDMA. When the client used IPC handles to map vLLM worker's GPU memory, that mapped memory couldn't be registered with `ibv_reg_mr` because nvidia-peermem only works with memory allocated by the calling process.
**Solution**: NIXL agents now live in vLLM workers (which own the GPU memory), not in the client. The client only orchestrates transfers via ZMQ commands.

## Development Workflow

### Making Changes to Client

1. Edit `modelexpress_client/python/modelexpress/client.py`
2. Rebuild Docker image with Python package
3. Restart containers

### Making Changes to vLLM Extension

1. Edit `modelexpress_client/python/modelexpress/vllm_extension.py`
2. Rebuild Docker image
3. Restart pods

### Making Changes to Server

1. Edit Rust code in `modelexpress_server/src/`
2. Run `cargo clippy` and `cargo test`
3. Build and push Docker image
4. Restart server deployment

### Regenerating Proto Bindings

```bash
# Python
cd modelexpress_client/python
bash generate_proto.sh

# Rust (automatic via build.rs)
cargo build
```

### Running Tests

```bash
# Run all Rust tests
cargo test

# Run integration tests
./run_integration_tests.sh

# Run clippy for linting
cargo clippy
```

## Configuration Examples

### Single Node Testing (IPC transfers)

```bash
# Terminal 1: Start vLLM with extension
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --worker-extension-cls modelexpress.vllm_extension.MxWorkerExtension

# Terminal 2: Start client
python -m modelexpress.client \
    --model-name facebook/opt-125m \
    --zmq-base ipc:///tmp/vllm.sock \
    --tp-size 1 \
    --engine-address http://localhost:8000 \
    --server-address localhost:8001
```

### Multi-Node with TP4

```yaml
# vLLM + Client in same container
env:
  - name: MX_ZMQ_ADDRESS
    value: "ipc:///tmp/mx/vllm.sock"
  - name: MODEL_NAME
    value: "meta-llama/Llama-3.3-70B-Instruct"

command: ["bash", "-c"]
args:
  - |
    # Start vLLM in background
    python3 -m vllm.entrypoints.openai.api_server \
      --model ${MODEL_NAME} \
      --tensor-parallel-size 4 \
      --worker-extension-cls modelexpress.vllm_extension.MxWorkerExtension &
    
    # Wait for vLLM to be ready
    while ! curl -s http://localhost:8000/health; do sleep 5; done
    
    # Start ModelExpress client
    python3 -m modelexpress.client \
      --model-name ${MODEL_NAME} \
      --zmq-base ipc:///tmp/mx/vllm.sock \
      --tp-size 4 \
      --engine-address http://localhost:8000 \
      --server-address modelexpress-server:8001 &
    
    wait
```

## Performance Expectations

| Model | Size | TP | Transfer Time | Throughput |
|-------|------|-----|---------------|------------|
| Llama 3.1 70B | ~140GB | 4 | ~4-5s | ~300 Gbps |
| Llama 3.1 405B | ~750GB | 8 | ~25s | ~300 Gbps |

## Next Steps / Future Work

1. **Auto-Start Client**: Integrate client into vLLM container as init container
2. **Multi-Source Fanout**: Load balancing across multiple sources
3. **Incremental Updates**: Transfer only changed parameters during fine-tuning
4. **Health Checks**: Automatic recovery if transfer fails
5. **Metrics/Observability**: Prometheus metrics for transfer throughput

## Related Repositories

- **NIXL**: NVIDIA's low-level transfer library (https://github.com/ai-dynamo/nixl)
- **vLLM**: The LLM serving engine we extend

## Contact

For questions about this codebase, reach out to the Dynamo team.
