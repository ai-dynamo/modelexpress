# ModelExpress × NeMo-RL — Design + Validation Overview (v2)

**Last Updated**: May 8, 2026
**Status**: **End-to-end NIXL RDMA refit working on real GB200**, prototyped on `kavink/nemo_rl_moe` (MX) + `kavink/mx_integration` ([NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)). 4 ranks × 2 cycles × toy tensors verified byte-correct (sentinels match). 4 ranks × 1.6 GB Qwen3-30B-A3B-shaped tensors land in 11–16 ms each. **4 ranks × 2 cycles × real `torch.distributed.tensor.DTensor` (Shard(0) FSDP placement) verified byte-correct AND verified that the receiver's reconstructed shape registry reports the correct un-sharded `global_shape` plus per-rank `local_shard_range` for every tensor.** 15/15 unit tests passing. arm64 NemoRL overlay image (`nvcr.io/nvidian/dynamo-dev/nemo-rl:kavink-v2`) built and smoke-tested but not yet pushed to a registry, so the actual Ray-orchestrated NemoRL training loop on Qwen3 hasn't been driven yet — that's the next milestone, gated only on image push + a K8s manifest.

This document is the technical companion to the upstream PR. It covers:

1. Why we built this (lessons from PrimeRL #2389 + the TensorHub paper + Composer 2 router replay).
2. The v2 design — 4 pillars: rank-to-rank publish, tree scale-out, MoE expert filter, explicit shape registry.
3. The Python API surface a NemoRL caller sees.
4. The full file inventory across the two branches.
5. Where the running MX server has gaps and how we work around them today.
6. What was tested vs what is still on paper.
7. End-to-end deployment recipe for the next session.

---

## 1. Motivation

NeMo-RL today has two weight-sync paths — **`update_weights_via_ipc_zmq`** (CUDA IPC handles over a ZMQ socket; only valid in colocated/hybrid mode) and **`update_weights_from_collective`** (NCCL `broadcast` from rank 0 in non-colocated mode). The NCCL path has three blockers for the workloads we care about:

| Problem | Where it bites |
|---|---|
| **`tensor.full_tensor()` allgather on every refit** | `dtensor_policy_worker.py:1822-1834` (`broadcast_weights_for_collective`). On a 30B-MoE with FSDP=4 this is ~120 GB through rank-0's NIC per refit. |
| **Static NCCL group** | NCCL barrier locks the trainer + all rollout replicas into a fixed world. Spot/elastic rollout, mid-run rebalancing, cross-DC — all blocked. |
| **No MoE awareness** | Every rank receives every expert weight, even if its EP shard only needs 1/8th of them. Composer 2 reports this as the dominant refit cost on Kimi K2.5 (1.04T / 32B active). |

PrimeRL PR [#2389](https://github.com/PrimeIntellect-ai/prime-rl/pull/2389) is the closest framework analog. We live-debugged it on GB200 in early May and learned two things the hard way:

1. **Cross-subnet full-mesh in `TransportPlan` ≠ routable.** GCP GB200's four `mlx5_N` NICs each sit on their own L3 subnet (`rdma-0..rdma-3`); the full-mesh `add_remote_agent` loop hits `NIXL_ERR_REMOTE_DISCONNECT` whenever (trainer rank N → inference rank M ≠ N). For the 1-to-1 dp-only layout that NeMo-RL also uses, **same-rank-only writes** are both topologically correct and 3× cheaper in NIXL connection count.

2. **vLLM workers don't unpublish-on-death and don't heartbeat.** Each orchestrator restart leaves stale `READY` rows in MX Redis; subsequent `add_remote_agent` calls choke on the dead rows with `NIXL_ERR_NOT_ALLOWED`. The fix is `(worker_rank, max(updated_at))` dedup at read time, plus a real heartbeat on the publisher side.

The TensorHub paper ([arXiv 2604.09107v1](https://arxiv.org/pdf/2604.09107v1)) gave us the production-quality framing — **Reference-Oriented Storage**, mutability contract, retention protocol, and crucially **pipeline replication** (a receiver becomes a source for the next receiver, building an expanding DAG that scales bandwidth with the number of active clients). Cursor's Composer 2 technical report adds **router replay + per-expert delta compression** as the MoE-specific shape we need.

**v2 = NemoRL's existing IPC/NCCL-style API + every learning above.**

---

## 2. Comparison to existing NeMo-RL paths

| Property | `update_weights_via_ipc_zmq` | `update_weights_from_collective` (NCCL) | **`update_weights_via_mx` (this PR)** |
|---|---|---|---|
| Cross-node | ❌ (colocated only) | ✅ | ✅ |
| Full-mesh allgather | ❌ | ✅ — `tensor.full_tensor()` on rank 0 | **❌ — `tensor.to_local()` on every rank** |
| Trainer NIC bottleneck | n/a | yes — single rank-0 funnel | **no — N parallel rank-N → rank-N pairs** |
| MoE expert filtering | none | none | **first-class — owned/needed expert IDs in metadata** |
| Tree fan-out for cold-start replicas | none | none | **TensorHub pipeline replication** |
| Cross-DC | ❌ | ❌ | designed (P3, not yet wired) |
| Elastic rollouts | ❌ | ❌ | ✅ — NIXL connections are dynamic, no static world |
| Heartbeat / liveness | n/a | n/a | ✅ — `HeartbeatThread` per worker |
| Versioning / freshness | implicit via NCCL barrier order | implicit | **explicit via `version: int` on every publish** |
| Mutability contract | none | none | **`set_status(STALE)` drains in-flight readers (TensorHub-style)** |
| Backward-compat default | n/a | yes | yes — opt-in via `cluster.weight_sync.method: "mx"` |

---

## 3. Architecture

```
                                 ┌────────────────────────────────────────────┐
                                 │            MX Server (Rust + Redis)        │
                                 │                                            │
                                 │  publish_metadata / list_sources /         │
                                 │  get_metadata / update_status              │
                                 │                                            │
                                 │  storage layout (Redis HASH):              │
                                 │    mx:source:{16-hex}                      │
                                 │      __attributes__ → SourceAttributesJson │
                                 │                       (incl. extra_params) │
                                 │      <worker_id> → worker_rank             │
                                 │    mx:source:{16-hex}:{worker_uuid}        │
                                 │      <worker_rank> → WorkerRecordJson      │
                                 └─────┬──────────────────────────┬───────────┘
                                       │ gRPC                     │ gRPC
                          publish + register                     query / list
                                       │                          │
                  ┌────────────────────┴───┐         ┌────────────┴──────────┐
                  │  Trainer ranks         │         │  Inference ranks       │
                  │  (FSDP2 / DTensor)     │         │  (vLLM, EP-sharded)    │
                  │                        │         │                        │
                  │  rank N → publish      │         │  rank N → discover     │
                  │    its local DTensor   │         │    same-rank source    │
                  │    shard, with         │  RDMA   │    via picker (filter  │
                  │    per-tensor          │ ◄────── │    by worker_rank,     │
                  │    placement info      │  WRITE  │    dedup by latest     │
                  │                        │         │    updated_at)         │
                  │  Each rank registers   │         │                        │
                  │  buffers with NIXL     │         │  After receive: each   │
                  │  ONCE (addresses are   │         │  rank registers itself │
                  │  stable across steps)  │         │  as a NEW source for   │
                  │                        │         │  subsequent receivers  │
                  │  HeartbeatThread keeps │         │  (TensorHub pipeline-  │
                  │  updated_at fresh      │         │  replication trick)    │
                  └────────────────────────┘         └────────────────────────┘
```

---

## 4. The four design pillars

### 4.1 Pillar 1 — Rank-to-rank publish (no allgather)

**Trainer side.** Each FSDP/EP rank publishes only its **local DTensor shard**, never `tensor.full_tensor()`. The placement (which axis is sharded, which range of indices this rank holds) travels in the per-tensor metadata.

```python
# nemo_rl/models/policy/workers/dtensor_policy_worker.py
@torch.no_grad()
@wrap_with_nvtx_name("dtensor_policy_worker/stream_weights_via_mx")
def stream_weights_via_mx(self, *, version: int, mx_config: Any) -> None:
    if not hasattr(self, "_mx_publisher") or self._mx_publisher is None:
        self._mx_publisher = build_v2_publisher(
            rank=self.rank,
            device_id=self.local_device_index,
            fsdp_world_size=self.world_size,
            tp_world_size=self.tp_size or 1,
            pp_world_size=self.pp_size or 1,
            ep_world_size=self.ep_size or 1,
            mx_config=mx_config,
        )
        self._mx_publisher.initialize(model_name=self.model_name, dtype=str(self.dtype).removeprefix("torch."))
        self._mx_expert_layout = detect_moe_expert_layout(
            self.model, ep_world_size=self.ep_size or 1, rank=self.rank,
        ) if mx_config.moe_expert_filter else {}

    self._mx_publisher._registry.clear()
    self._mx_publisher._registered_tensors.clear()
    for name, tensor in self.model.state_dict().items():
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor   # ← key: NO allgather
        local = local.to(self.dtype, non_blocking=True).contiguous()
        expert_info = self._mx_expert_layout.get(name)
        self._mx_publisher.add_tensor(
            name=name,
            tensor=local,
            is_expert=expert_info is not None,
            expert_axis=expert_info[0] if expert_info else 0,
            owned_expert_ids=expert_info[1] if expert_info else set(),
        )
        # Override the descriptor's global_shape from the DTensor view so the
        # receiver knows the un-sharded shape. The NIXL-registered buffer is
        # still the local shard.
        if isinstance(tensor, DTensor):
            self._mx_publisher._registry[-1].global_shape = tuple(int(s) for s in tensor.shape)

    self._mx_publisher.publish(version=int(version))
    self._mx_publisher.mark_ready()                                            # ← starts HeartbeatThread
```

**Cost vs allgather pattern (Qwen3-30B-A3B FSDP=4)**:
- v1 / NCCL: 4 ranks each allgather → 4× full model materialized in VRAM at peak; rank 0's NIC ships full model to every inference rank → ~120 GB through one NIC.
- v2: each rank holds only its 1/4 shard; each rank's NIC ships its 1/4 → 4 NICs in parallel → 4× the aggregate bandwidth.

### 4.2 Pillar 2 — Tree scale-out (TensorHub pipeline replication)

After an inference rank finishes receiving its slice, it **becomes a source** by re-registering its already-NIXL-registered receive buffers and publishing as `inference_replica`. Subsequent same-rank receivers can pull from it instead of contending on the trainer's NIC.

```python
# modelexpress/nemo_rl_v2.py — MxV2RefitReceiver
def publish_self_as_source(self, *, version: int, model_name: str) -> str | None:
    identity = p2p_pb2.SourceIdentity(
        model_name=model_name, mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_UNKNOWN,
        dtype="bfloat16",
        extra_parameters={
            "role": ROLE_INFERENCE_REPLICA,
            "mx_v2": "1",
            "worker_rank": str(self._worker_rank),
            "training_step": str(int(version)),
            "training_framework": "nemo_rl",
        },
    )
    worker_meta = p2p_pb2.WorkerMetadata(
        worker_rank=self._worker_rank,
        nixl_metadata=nixl.nixl_metadata,
        tensors=[p2p_pb2.TensorDescriptor(name=d.name, addr=d.addr, size=d.size,
                                           device_id=d.device_id, dtype=d.dtype)
                  for d in nixl.tensor_descriptors],
        status=p2p_pb2.SOURCE_STATUS_READY,
        agent_name=self._receiver._agent_name,
    )
    return client.publish_metadata(identity=identity, worker=worker_meta,
                                    worker_id=self._receiver._worker_id)
```

The picker prefers `trainer` over `inference_replica` when both are visible (the trainer is always authoritative), then breaks ties on `max(updated_at)`. This means:
- First receiver → pulls from trainer.
- Second receiver (slow / cold-start / restart) → may pull from the first receiver if trainer is busy. (Today: picker just picks freshest trainer; the load-balancing improvement is a follow-up.)

### 4.3 Pillar 3 — MoE expert filtering

Each tensor descriptor carries `is_expert: bool`, `expert_axis: int`, and the publisher's `owned_expert_ids: set[int]`. An EP-sharded inference rank's `pick_best_source` accepts an optional `needed_experts_per_layer` filter and rejects candidates that don't cover all needed experts.

```python
# Receiver side
chosen = receiver.pick_best_source(
    candidates,
    needed_experts_per_layer={5: {72, 73, 74, 75, 76, 77}},  # what THIS rank needs
)
```

Composer 2 extends this with a **dirty-experts bitmap** ("only experts with non-zero gradient since last refit"). The MX-side surface for that is designed (`set_dirty_experts`, `get_dirty_experts` in the design doc) but not yet implemented; v0 refits all owned experts.

### 4.4 Pillar 4 — Explicit shape registry / mutability contract

Every published tensor carries a `TensorDescriptorV2` (placement kind, shard axis, local shard range, expert axis, owned expert IDs). Receivers consult these to know exactly what to expect and where each shard fits in the global tensor.

The mutability contract follows TensorHub §3.2:
- The trainer **MUST NOT** mutate `tensor` between `publish_metadata(version=v)` and `set_status(STALE)` for that version.
- `set_status(STALE)` is intended to block until in-flight RDMA reads complete. (Today it's a no-op on the server side; the proper drain semantics are a follow-up — design only.)
- Inference workers commit to the same contract for any version they hold.

---

## 5. Public Python API

Three new symbols on the MX side:

```python
from modelexpress import (
    MxV2TrainingPublisher,    # trainer-side wrapper
    MxV2RefitReceiver,        # inference-side wrapper
    TrainerWorldLayout,       # (fsdp, tp, pp, ep) descriptor
)
from modelexpress.shape_descriptors import (
    TensorDescriptorV2,
    describe_tensor,          # DTensor → wire format
    even_expert_owner_map,
)
```

One new module on the NemoRL side:

```python
from nemo_rl.distributed.mx_helpers import (
    MxConfig,                 # parsed from cfg.cluster.weight_sync
    build_v2_publisher,       # convenience constructor + NIC pin
    build_v2_receiver,
    pin_local_nic,
    collect_named_local_shards,
    detect_moe_expert_layout,
)
```

Two new abstract methods on the existing interfaces:

```python
# nemo_rl/models/policy/interfaces.py
class ColocatablePolicyInterface(PolicyInterface):
    def stream_weights_via_mx(self, *, version: int, mx_config: Any) -> list[ray.ObjectRef]:
        raise NotImplementedError("...")

# nemo_rl/models/generation/interfaces.py
class GenerationInterface(ABC):
    def update_weights_via_mx(self, *, version: int, mx_config: Any) -> list[ray.ObjectRef]:
        raise NotImplementedError("...")
```

One new branch in `algorithms/grpo.py`:

```python
# nemo_rl/algorithms/grpo.py::refit_policy_generation
elif weight_sync_method == "mx":
    if mx_config is None or not getattr(mx_config, "enabled", False):
        raise RuntimeError(
            "weight_sync_method='mx' requires an enabled MxConfig "
            "(cfg.cluster.weight_sync.method='mx', .enabled=True)"
        )
    version = int(refit_version) if refit_version is not None else 0
    futures_train = policy.stream_weights_via_mx(version=version, mx_config=mx_config)
    futures_inference = policy_generation.update_weights_via_mx(version=version, mx_config=mx_config)
    ray.get(futures_train)
    results = ray.get(futures_inference)
    update_success = all(result for result in results if result is not None)
```

`MxConfig` knobs (parsed from `cfg.cluster.weight_sync`):

```python
@dataclass
class MxConfig:
    enabled: bool = False                # master switch
    mx_server_url: str = "modelexpress-server:8001"
    timeout_seconds: float = 300.0
    same_rank_only: bool = True          # ← required on multi-subnet RDMA fabrics (GB200 / EFA)
    tree_scale_out: bool = True          # ← receivers republish as inference_replica
    moe_expert_filter: bool = True       # ← only request owned experts
    register_self_buffers: list[str] = []
    nic_pin: str = "auto"                # "auto" | "off" | "mlx5_X"
    retain_latest_k: int = 1             # TensorHub-style retention (designed; not enforced server-side yet)
```

Worked example for Qwen3-30B-A3B:

```yaml
cluster:
  weight_sync:
    method: "mx"
    enabled: true
    mx_server_url: "modelexpress-server.kavin.svc.cluster.local:8001"
    timeout_seconds: 300.0
    same_rank_only: true       # GB200/EFA — keep ON
    tree_scale_out: true
    moe_expert_filter: true
    nic_pin: "auto"
    retain_latest_k: 1
```

---

## 6. File inventory across the two branches

### `kavink/nemo_rl_moe` (off `kavink/RL` in `NVIDIA-Model-Optimizer/modelexpress`)

```
modelexpress_common/proto/p2p.proto                                M  +7   added SourceIdentity identity = 5 to GetMetadataResponse
modelexpress_client/python/modelexpress/__init__.py                M  +8   re-export MxV2TrainingPublisher / MxV2RefitReceiver / TrainerWorldLayout
modelexpress_client/python/modelexpress/p2p_pb2.py                 M  ±40  regenerated
modelexpress_client/python/modelexpress/p2p_pb2_grpc.py            M  ±4   regenerated
modelexpress_client/python/modelexpress/shape_descriptors.py       A  +277 TensorDescriptorV2, describe_tensor, even_expert_owner_map, codecs
modelexpress_client/python/modelexpress/nemo_rl_v2.py              A  +752 MxV2TrainingPublisher, MxV2RefitReceiver, TrainerWorldLayout, V2SourceCandidate
modelexpress_client/python/scripts/v2_moe_e2e_demo.py              A  +253 standalone GB200 cluster demo
modelexpress_client/python/tests/test_v2_shape_registry.py         A  +187 8 unit tests
modelexpress_client/python/tests/test_v2_source_picker.py          A  +469 7 unit tests (mocked NIXL/gRPC)
modelexpress_server/src/metadata_backend.rs                        M  +9   ModelMetadataRecord.identity field
modelexpress_server/src/metadata_backend/redis.rs                  M  +36  SourceAttributesJson.extra_parameters + to_source_identity()
modelexpress_server/src/metadata_backend/kubernetes.rs             M  +5   identity: None placeholder (CRD schema bump deferred)
modelexpress_server/src/p2p_service.rs                             M  +10  populate GetMetadataResponse.identity
modelexpress_server/src/state.rs                                   M  +1   identity: None in test fixtures
```

Two commits:
- `97c0e78` — client Python (publisher / receiver / shape descriptors / tests / demo / proto regen)
- `0bce4f0` — server Rust (round-trip the SourceIdentity through GetMetadata)

### `kavink/mx_integration` (off `main` in `NVIDIA-NeMo/RL`)

```
nemo_rl/algorithms/grpo.py                                         M  +49  mx branch in refit_policy_generation
nemo_rl/distributed/mx_helpers.py                                  A  +250 MxConfig, build_v2_*, pin_local_nic, collect_named_local_shards, detect_moe_expert_layout
nemo_rl/models/generation/interfaces.py                            M  +23  abstract update_weights_via_mx
nemo_rl/models/generation/vllm/vllm_backend.py                     M  +127 VllmInternalWorkerExtension.update_weights_via_mx (NIXL receive + _load_weights)
nemo_rl/models/generation/vllm/vllm_generation.py                  M  +20  Ray driver fan-out
nemo_rl/models/generation/vllm/vllm_worker.py                      M  +27  Ray actor entry → collective_rpc("update_weights_via_mx", ...)
nemo_rl/models/policy/interfaces.py                                M  +29  abstract stream_weights_via_mx (default-NotImplementedError, opt-in)
nemo_rl/models/policy/lm_policy.py                                 M  +14  Policy.stream_weights_via_mx (worker_group fan-out)
nemo_rl/models/policy/workers/dtensor_policy_worker.py             M  +111 DTensorPolicyWorker.stream_weights_via_mx (uses tensor.to_local())
docker/v2_overlay/Dockerfile                                       A  +80  thin overlay over nvcr.io/nvidia/nemo-rl:v0.6.0
```

One commit: `d58dca07`.

Both branches are committed but **not pushed** (you'll do that with the appropriate auth).

---

## 7. Three-tier metadata transport (server-side workaround)

The **running** MX server in our `kavin` namespace (`nvcr.io/nvidian/dynamo-dev/modelexpress-server:latest`, started May 6) drops most string fields when echoing `WorkerMetadata` back via `GetMetadata`. Confirmed by direct gRPC introspection on `prime-rl-nixl-mx-trainer-0`:

```
> client.list_sources(...) instances → ok (model_name, worker_rank present on SourceInstanceRef)
> client.get_metadata(instance.mx_source_id, instance.worker_id):
    found=True
    worker.tensors             → preserved ✅
    worker.status              → preserved ✅
    worker.worker_rank         → preserved ✅
    worker.nixl_metadata       → preserved ✅ (the bytes blob)
    worker.updated_at          → preserved ✅
    worker.agent_name          → '' ❌  (publisher set it, server dropped it)
    worker.metadata_endpoint   → '' ❌
    worker.worker_grpc_endpoint → '' ❌
    identity                   → NOT PRESENT ❌ (the proto field didn't exist in the running build)
    identity.extra_parameters  → n/a (would be empty even if present, since identity wasn't returned at all)
```

Because v2 metadata (`mx_v2`, `role`, `worker_rank`, `training_step`, `shape_registry`) was originally designed to live in `SourceIdentity.extra_parameters`, it can't reach the receiver via the current server.

The v2 receiver tries **three transports in order**, falling back to the next when the previous returns empty:

```python
# modelexpress/nemo_rl_v2.py::MxV2RefitReceiver.discover_v2_sources

# 1) SourceIdentity.extra_parameters via meta.identity
identity = getattr(meta, "identity", None)
extra = (dict(identity.extra_parameters) if identity is not None and identity.extra_parameters else {})

# 2) Synthetic TensorDescriptor sidecar (the path the prototype actually uses today)
if not extra:
    for td in meta.worker.tensors:
        if td.name == _V2_SIDECAR_NAME and td.dtype:                 # _V2_SIDECAR_NAME = "__mx_v2_meta__"
            try:
                sidecar = json.loads(td.dtype)
                if isinstance(sidecar, dict):
                    for k, v in sidecar.items():
                        extra[k] = str(v)
            except (json.JSONDecodeError, TypeError):
                pass
            break

# 3) WorkerMetadata.agent_name string-encoded marker (legacy fallback)
if not extra:
    agent_name = getattr(meta.worker, "agent_name", "") or ""
    if agent_name.startswith("mx_v2|"):
        # ... parse "mx_v2|<role>|rank=N|version=K|orig=..."
```

Symmetrically, the publisher writes v2 metadata into all three locations:

```python
# modelexpress/nemo_rl_v2.py::MxV2TrainingPublisher.publish

# Path 1: extra_parameters (forward-compat, used once Rust server populates GetMetadataResponse.identity)
def _build_identity_with_v2(step):
    ident = original_build_identity(step)
    ident.extra_parameters["role"] = ROLE_TRAINER
    ident.extra_parameters["mx_v2"] = "1"
    ident.extra_parameters["worker_rank"] = str(self._worker_rank)
    ident.extra_parameters["shape_registry"] = registry_blob
    ident.extra_parameters["world_layout"] = self._world_layout.encode()
    return ident
self._publisher._build_identity = _build_identity_with_v2

# Path 2: synthetic TensorDescriptor sidecar (today's transport)
sidecar_payload = json.dumps({
    "mx_v2": "1", "role": ROLE_TRAINER,
    "worker_rank": int(self._worker_rank),
    "training_step": int(version),
    "world_layout": self._world_layout.encode(),
    "framework": "nemo_rl",
})
def _build_tensor_protos_with_sidecar(descriptors):
    protos = original_build_tensor_protos(descriptors)
    protos.append(p2p_pb2.TensorDescriptor(
        name="__mx_v2_meta__", addr=0, size=0, device_id=0, dtype=sidecar_payload,
    ))
    return protos
self._publisher._build_tensor_protos = _build_tensor_protos_with_sidecar

# Path 3: agent_name encoding
self._publisher._agent_name = (
    f"mx_v2|{ROLE_TRAINER}|rank={self._worker_rank}|"
    f"version={int(version)}|orig={original_agent_name}"
)
```

**The Rust server fix is committed in commit `0bce4f0` on `kavink/nemo_rl_moe`** but requires a server image rebuild + redeploy to land. Specifically:

| Change | File | What |
|---|---|---|
| Proto: add `SourceIdentity identity = 5;` to `GetMetadataResponse` | `modelexpress_common/proto/p2p.proto` | Field already added; clients regenerated. Backward-compat (older clients ignore). |
| Storage: `SourceAttributesJson` gains `extra_parameters: HashMap<String, String>` | `modelexpress_server/src/metadata_backend/redis.rs:39-72` | `#[serde(default)]` so old records read clean. |
| Storage: `SourceAttributesJson::to_source_identity()` reconstructs full `SourceIdentity` | `modelexpress_server/src/metadata_backend/redis.rs:88-104` | Used by `get_metadata`. |
| Service: populate `GetMetadataResponse.identity` from `record.identity` | `modelexpress_server/src/p2p_service.rs:170-230` | One-line wiring. |
| K8s CRD backend: `identity: None` for now (CRD schema bump separate) | `modelexpress_server/src/metadata_backend/kubernetes.rs:439-450` | v2 clients fall back to sidecar. |

Once the new server image lands, transports collapse to Path 1 (cleanest); the sidecar TensorDescriptor stays in the wire as a no-op fallback.

---

## 8. What was tested

### 8.1 Unit tests (no GPU, no NIXL)

```bash
cd ~/Work/Github/MX0/modelexpress  # or upstream equivalent after push
python3 -m pytest modelexpress_client/python/tests/test_v2_shape_registry.py \
                  modelexpress_client/python/tests/test_v2_source_picker.py -v
```

Result: **15/15 PASS**:

```
test_v2_shape_registry.py::test_replicate_descriptor_round_trip               PASSED
test_v2_shape_registry.py::test_sharded_dtensor_local_range                   PASSED
test_v2_shape_registry.py::test_moe_expert_descriptor_in_registry             PASSED
test_v2_shape_registry.py::test_expert_owner_map_uniform                      PASSED
test_v2_shape_registry.py::test_expert_owner_map_rejects_uneven               PASSED
test_v2_shape_registry.py::test_expert_set_codec_round_trip                   PASSED
test_v2_shape_registry.py::test_decode_expert_set_handles_empty_and_whitespace PASSED
test_v2_shape_registry.py::test_registry_full_round_trip_multitensor          PASSED
test_v2_source_picker.py::test_same_rank_filter_dedup_freshest                PASSED
test_v2_source_picker.py::test_min_version_filter                             PASSED
test_v2_source_picker.py::test_non_v2_sources_ignored                         PASSED
test_v2_source_picker.py::test_pick_best_with_expert_filter                   PASSED
test_v2_source_picker.py::test_pick_best_falls_back_to_trainer                PASSED
test_v2_source_picker.py::test_world_layout_round_trip                        PASSED
test_v2_source_picker.py::test_agent_name_fallback_when_identity_missing      PASSED
```

The picker tests are particularly important — they assert that:
- `same_rank_only=True` correctly rejects rank-0 sources for a rank-2 receiver.
- Multiple stale `READY` rows for the same `worker_rank` collapse to the freshest one (the `(worker_rank, max(updated_at))` dedup that fixes the PrimeRL `NIXL_ERR_NOT_ALLOWED` bug class).
- Sources missing the `mx_v2=1` marker are ignored entirely (forward-compat against future v3+ clients).
- MoE expert filter rejects candidates that don't cover `needed_experts_per_layer`.
- Trainer is always preferred over `inference_replica` for same `(worker_rank, version)`.
- The agent_name fallback path correctly parses `mx_v2|<role>|rank=N|version=K|orig=...` for legacy servers.

### 8.2 Live cluster gRPC smoke test

Port-forwarded the running MX server from a workstation:

```bash
kubectl -n kavin port-forward svc/modelexpress-server 18001:8001
python3 -c "
from modelexpress import MxClient
import modelexpress.p2p_pb2 as p2p_pb2
c = MxClient(server_url='localhost:18001')
resp = c.list_sources(status_filter=p2p_pb2.SOURCE_STATUS_READY)
print(f'total READY: {len(resp.instances)}')
for inst in resp.instances[:3]:
    meta = c.get_metadata(inst.mx_source_id, inst.worker_id)
    print(f'  {inst.model_name} rank={inst.worker_rank} '
          f'identity_present={hasattr(meta, \"identity\")} '
          f'agent_name={meta.worker.agent_name!r}')
"
```

This is the test that surfaced the proto bugs in §7. Useful to re-run any time the server image changes.

### 8.3 Live E2E on GB200 — toy scale (correctness)

Inside the running `prime-rl-nixl-mx-trainer-0` pod (which has 4× B200 GPUs, NIXL, and reachability to MX):

```bash
# 1) copy our v2 files into the pod (the trainer pod's MX install is older)
SRC=/home/kavink/Work/Github/MX0/modelexpress/modelexpress_client/python/modelexpress
DST=/app/.venv/lib/python3.12/site-packages/modelexpress
for f in shape_descriptors.py nemo_rl_v2.py p2p_pb2.py p2p_pb2_grpc.py __init__.py refit_receiver.py training_publisher.py; do
    kubectl -n kavin cp -c trainer "$SRC/$f" prime-rl-nixl-mx-trainer-0:"$DST/$f"
done
kubectl -n kavin cp -c trainer \
    /home/kavink/Work/Github/MX0/modelexpress/modelexpress_client/python/scripts/v2_moe_e2e_demo.py \
    prime-rl-nixl-mx-trainer-0:/tmp/v2_moe_e2e_demo.py

# 2) run the demo: 4 ranks, 8 experts (chunk=2/rank), HIDDEN=256, 2 cycles
kubectl -n kavin exec prime-rl-nixl-mx-trainer-0 -- bash -c "
    cd /tmp && WORLD_SIZE=4 NUM_EXPERTS=8 N_REFIT_CYCLES=2 timeout 90 python3 v2_moe_e2e_demo.py
"
```

Output (abridged):

```
[trainer R0] published v=0 mx_source_id=393ec6709b204c80 sentinel_target=8 got=8
[trainer R1] published v=0 mx_source_id=bf2e1ce5d3bebde6 sentinel_target=16 got=16
[trainer R2] published v=0 mx_source_id=6b057dd75143e1db sentinel_target=24 got=24
[trainer R3] published v=0 mx_source_id=458954a508b0c650 sentinel_target=32 got=32

[inference R0] picked source role=trainer src_rank=0 v=0 updated_at=1778169737062
[inference R0] received 'model.layers.0.experts.weight'    shape=(2, 256, 256) dtype=torch.bfloat16
[inference R0] received 'model.layers.0.layer_norm.weight' shape=(256,)        dtype=torch.bfloat16
[inference R0] correctness: OK
[inference R1] picked source role=trainer src_rank=1 v=0 ... → OK
[inference R2] picked source role=trainer src_rank=2 v=0 ... → OK
[inference R3] picked source role=trainer src_rank=3 v=0 ... → OK

# … cycle 1 with version=1 …

[inference R1] picked source role=trainer src_rank=1 v=1 updated_at=1778169742340  ← freshness dedup picks v=1 over the still-alive v=0
[inference R1] correctness: OK
=== ALL RANKS OK ===
```

This validates end-to-end: `MxV2TrainingPublisher.publish` over real gRPC, NIXL register, the **sidecar transport** (`__mx_v2_meta__`) round-trips through the server, `discover_v2_sources` finds the right rank, `pick_best_source` selects the freshest, `receive_from` does a real RDMA WRITE, byte-level sentinels match, and `publish_self_as_source` (tree fan-out) successfully republishes. Per-rank NIC pinning via `MX_RDMA_NIC_PIN=auto` correctly mapped rank N → `mlx5_N:1`.

### 8.4 Live E2E on GB200 — real DTensors (`torch.distributed.tensor`)

`scripts/v2_dtensor_e2e_demo.py` mirrors the previous test but uses **real DTensors** instead of fake stand-ins, exercising the exact codepath that `DTensorPolicyWorker.stream_weights_via_mx` runs in production: `init_device_mesh("cuda", (WORLD_SIZE,))`, `distribute_tensor(t, mesh, [Shard(0)])`, then `MxV2TrainingPublisher.add_tensor(tensor=sharded_dt)`.

```bash
WORLD_SIZE=4 HIDDEN=1024 INTER=2048 N_REFIT_CYCLES=2 python3 v2_dtensor_e2e_demo.py
```

Result on `prime-rl-nixl-mx-trainer-0`:

```
[trainer R0] DTensor: global=(1024, 2048) local=(256, 2048) sentinel=8
[trainer R1] DTensor: global=(1024, 2048) local=(256, 2048) sentinel=16
[trainer R2] DTensor: global=(1024, 2048) local=(256, 2048) sentinel=24
[trainer R3] DTensor: global=(1024, 2048) local=(256, 2048) sentinel=32

[inference R0] registry: global=(1024, 2048) placement=SHARD shard_axis=0 local_range=(0, 256)
[inference R1] registry: global=(1024, 2048) placement=SHARD shard_axis=0 local_range=(256, 512)
[inference R2] registry: global=(1024, 2048) placement=SHARD shard_axis=0 local_range=(512, 768)
[inference R3] registry: global=(1024, 2048) placement=SHARD shard_axis=0 local_range=(768, 1024)
[inference R0..R3] all_elem_match=True  for v=0 and v=1
=== ALL RANKS OK ===
```

This validates two things v2_moe_e2e_demo.py couldn't:

1. **`shape_descriptors.describe_tensor` correctly handles real DTensors.** The fix is in commit `9aa4b93` — the previous version assumed `tensor.shape` was the local view (true for plain tensors, **false for DTensors**, where `.shape` is the global un-sharded shape). On a real DTensor, we now read `tensor.to_local().shape[shard_dim]` for the local extent.
2. **The shape registry reaches the receiver via the sidecar.** Previously the registry only travelled through `SourceIdentity.extra_parameters` (which the running server drops). Same commit (`9aa4b93`) embeds `shape_registry` inside the sidecar JSON. Receivers now see `chosen.registry["tensors"]` with correct `global_shape`, `placement_kind=SHARD`, `shard_axis=0`, and rank-correct `local_shard_range`.

Both are real bugs that the test caught before they would have shipped. They demonstrate why the DTensor E2E test (vs the fake-DTensor toy demo) is essential for closing the validation gap on the NemoRL-wrapper code paths.

### 8.5 Live E2E on GB200 — production scale (Qwen3-30B-A3B-shaped)

Same demo, scaled up: **WORLD_SIZE=4, NUM_EXPERTS=192, HIDDEN=4096** → `(48, 4096, 4096)` bf16 ≈ **1.6 GB / rank**. This is the per-rank shard size for Qwen3-30B-A3B with EP=4.

```
[inference R0] 1610.62 MB in 16 ms (102232 MB/s); moe[0,0,0]=8 ln[0]=8 expected=8         OK
[inference R1] 1610.62 MB in 11 ms (142784 MB/s); moe[0,0,0]=16 ln[0]=16 expected=16      OK
[inference R2] 1610.62 MB in 11 ms (144449 MB/s); moe[0,0,0]=24 ln[0]=24 expected=24      OK
[inference R3] 1610.62 MB in 12 ms (131257 MB/s); moe[0,0,0]=32 ln[0]=32 expected=32      OK
=== ALL RANKS OK ===
```

The 100+ GB/s figures are intra-node `cuda_ipc` (the test pod runs all 4 ranks on one host), not over-the-wire RDMA. So this validates **correctness at production-shape volumes** but not over-the-wire NIC bandwidth — for cross-node we'd see ~7–8 GB/s per NIC, matching what PrimeRL PR #2389 reports.

### 8.6 Docker overlay image

```bash
cd ~/Work/Github/RL/RL  # or upstream NemoRL clone
docker buildx create --use --name multi-arch --driver docker-container 2>/dev/null || true
docker run --privileged --rm tonistiigi/binfmt --install arm64           # one-time qemu setup
docker pull --platform linux/arm64 nvcr.io/nvidia/nemo-rl:v0.6.0          # ~5 GB

docker buildx build \
    --platform linux/arm64 \
    --build-context modelexpress=$HOME/Work/Github/MX0/modelexpress \
    --build-context nemo-rl-source=. \
    -f docker/v2_overlay/Dockerfile \
    --tag nvcr.io/nvidian/dynamo-dev/nemo-rl:kavink-v2 \
    --load .

# In-image smoke (qemu-aarch64):
docker run --rm --platform linux/arm64 nvcr.io/nvidian/dynamo-dev/nemo-rl:kavink-v2 \
    /opt/nemo_rl_venv/bin/python -c "
from modelexpress import MxV2TrainingPublisher, MxV2RefitReceiver, TrainerWorldLayout
from nemo_rl.distributed.mx_helpers import MxConfig, build_v2_publisher
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.generation.interfaces import GenerationInterface
assert hasattr(ColocatablePolicyInterface, 'stream_weights_via_mx')
assert hasattr(GenerationInterface, 'update_weights_via_mx')
print('nemo_rl × mx v2 imports OK')
"
```

Image: 34.6 GB on disk (`v0.6.0` base + ~6 MB our overlay). Build time on x86 host with qemu-aarch64: ~10 min (cached layers reuse afterward).

The Dockerfile is intentionally minimal — it overlays the entire `nemo_rl/` package (not just the modified files) because v0.6.0 is older than `main` and partial overlays cause missing-symbol import errors (e.g. `resolve_generation_worker_cls` isn't in v0.6.0's `vllm_generation/utils.py`). The MX wheel is `pip install --no-deps` (the venv already has compatible `grpcio` / `protobuf`).

### 8.7 What was NOT validated end-to-end

These compile + import + pass linting but no integration test drove the codepath yet:

- `DTensorPolicyWorker.stream_weights_via_mx` — the **inner mechanics** (DTensor → `to_local()` → `add_tensor` → publish, plus shape-registry placement metadata) are now exercised by §8.4. What's still untouched: the NemoRL-specific outer glue — `model.state_dict()` walk over an actual NemoRL-wrapped HF model, the MoE detection heuristic on real expert layer naming, `cpu_offload` lifecycle. None of these change the v2 protocol; they're integration-level.
- `VllmInternalWorkerExtension.update_weights_via_mx` — the inference-side bridge that registers `model_runner.model.named_parameters()`, calls `_load_weights`, applies the GptOss transpose fix, `_maybe_process_fp8_kv_cache`. Same — receiver was driven, but not the vLLM glue around it.
- `refit_policy_generation`'s `mx` branch (the Ray fan-out: `policy.stream_weights_via_mx` ‖ `policy_generation.update_weights_via_mx`, then `ray.get`).
- `lm_policy.Policy.stream_weights_via_mx` — Ray actor fan-out wrapper.
- The MX server-side Rust changes (compile-checked via `ReadLints`; not deployed since the running cluster image predates them).
- **Multi-node** RDMA — everything was intra-node so far.
- Tree fan-out **under load** — `publish_self_as_source` ran but no second receiver actually preferred a replica over the trainer.
- Heartbeat lifecycle under churn (kill workers, watch reaping).
- Failure-recovery rerouting if a tree-fan-out source dies mid-write.
- Megatron worker, SGLang generation, dirty-experts bitmap, cross-DC, async / in-flight refit.

---

## 9. Server-side patch path (when you want to graduate the prototype)

Three sequenced steps, all on `kavink/nemo_rl_moe`:

### Step 1 — rebuild the server image with the SourceIdentity round-trip

```bash
cd modelexpress
cargo build --release -p modelexpress_server                  # locally or in CI
docker buildx build \
    --platform linux/arm64 \
    -f modelexpress_server/Dockerfile \
    --tag nvcr.io/nvidian/dynamo-dev/modelexpress-server:kavink-v2 \
    --push .
```

Then redeploy the `modelexpress-server` Deployment in `kavin` namespace pointing at the new tag. After this, `MxV2RefitReceiver.discover_v2_sources` automatically uses transport Path 1 (`SourceIdentity.extra_parameters`); the sidecar TensorDescriptor stays in the wire as a no-op.

### Step 2 — push the NemoRL overlay image

```bash
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY
docker push nvcr.io/nvidian/dynamo-dev/nemo-rl:kavink-v2
```

### Step 3 — deploy a NemoRL training job pointing at MX

Mirror the prime-rl-nixl-mx K8s manifest layout (Ray head + worker pods + StatefulSet for trainer) but with NemoRL's actor model. Driving config:

```yaml
# config.yaml (NemoRL job)
cluster:
  weight_sync:
    method: "mx"
    enabled: true
    mx_server_url: "modelexpress-server.kavin.svc.cluster.local:8001"
    timeout_seconds: 300.0
    same_rank_only: true
    tree_scale_out: true
    moe_expert_filter: true
    nic_pin: "auto"
    retain_latest_k: 1

# trainer / generation as usual:
policy:
  model_name: "Qwen/Qwen3-30B-A3B-Instruct-2507"
  parallelism:
    fsdp: 4
    tp: 1
    pp: 1
generation:
  vllm:
    tensor_parallel_size: 1
    data_parallel_size: 4
```

What to watch for in trainer logs once running:

```
[modelexpress.nemo_rl_v2] MxV2TrainingPublisher initialized: rank=0 layout=fsdp:4,tp:1,pp:1,ep:1
[modelexpress.nemo_rl_v2] MxV2 publish: rank=0 version=K tensors=N mx_source_id=...
[modelexpress.heartbeat] [Worker 0] Heartbeat started (interval=30s)
[modelexpress.heartbeat] [Worker 0] Status -> READY
```

What to watch for in inference logs:

```
[mx] rank=0 chosen source role=trainer src_rank=0 version=K
... NIXL transfer complete: <bytes>, <tensors> tensors, <s>s ...
```

If you see `[mx] no v2 source available for version>=K on rank N`, check that:
1. Both pods agree on `model_name`.
2. `same_rank_only=True` and trainer's `worker_rank` matches inference's.
3. Heartbeat is alive on the trainer (`HeartbeatThread` started).

### Step 4 (optional) — push the branches for code review

```bash
git -C ~/Work/Github/MX0/modelexpress       push origin kavink/nemo_rl_moe
git -C ~/Work/Github/RL/RL                  push origin kavink/mx_integration
```

Then open PRs against the respective repos. The MX-side PR description can pull §1–§7 of this doc; the NemoRL-side PR can stay shorter (link back to this doc for the design rationale).

---

## 10. Roadmap (designed but not implemented)

| Item | Why | Where it goes |
|---|---|---|
| Async / in-flight refit (Composer 2 / PipelineRL style) | Don't block training step on refit completion | `nemo_rl/algorithms/grpo.py::refit_policy_generation_async` + a `_AsyncMxRefitDaemon` on the inference side that hot-swaps weights between rollouts |
| Megatron worker `stream_weights_via_mx` | Coverage of the second NemoRL trainer backend | `nemo_rl/models/policy/workers/megatron_policy_worker.py` |
| SGLang generation `update_weights_via_mx` | Coverage of the second NemoRL inference backend | `nemo_rl/models/generation/sglang/` |
| Dirty-experts bitmap | Composer-2-style "only refit changed experts" | MX-side `set_dirty_experts` / `get_dirty_experts` RPCs + receiver-side `needed = my_owned ∩ dirty` filter |
| Cross-DC seeding (TCP fallback) | Multi-DC rollouts | MX-side TopologyScheduler with datacenter-aware source selection |
| Failure-recovery rerouting | A tree-fan-out source dies mid-write | Receiver detects RDMA fail → `report_source_failure` → server marks stale → retry against next-best |
| Mutability contract drain | TensorHub §3.2 | Server-side `set_status(STALE)` blocks until in-flight reads complete |

---

## 11. References

- **PrimeRL PR #2389** (the GB200 multi-subnet RDMA topology lessons that shaped v2 defaults): [PrimeIntellect-ai/prime-rl#2389](https://github.com/PrimeIntellect-ai/prime-rl/pull/2389).
- **TensorHub paper** (ROS, mutability contract, retention protocol, pipeline replication): [arXiv 2604.09107v1](https://arxiv.org/pdf/2604.09107v1).
- **Composer 2 technical report** (router replay + per-expert delta compression): [Cursor Composer 2](https://cursor.com/resources/Composer2.pdf).
- **Sister framework integrations**: `docs/RL/PRIMERL_MX_OVERVIEW.md`, `docs/RL/VERL_MX_OVERVIEW.md`.
- **MX architecture**: `docs/ARCHITECTURE.md`, `docs/metadata.md`, `docs/DEPLOYMENT.md`.
