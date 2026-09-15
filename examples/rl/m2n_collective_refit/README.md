# End-to-end NCCL M2N collective refit

A real trainer refits a real inference engine over the collective path: a Hugging
Face checkpoint sharded by `torch.distributed.fsdp.fully_shard` publishes its
weights into a live vLLM engine, through the ModelExpress control plane and
`nccl.m2n.reshard`.

The distinction from `modelexpress_client/python/tests/gpu/bench_collective_refit.py`
is the engine boundary. That harness stubs it and measures the transport. This one
does not stub anything: vLLM holds the destination storage, and the weights are a
checkpoint rather than tensors whose shapes the test chose.

## Correctness is generation

A refit that moves nothing is very fast, and it compares equal against a model that
still holds the right weights. So the run does not compare tensors. It asks vLLM to
answer a fixed prompt set greedily, overwrites every live weight with noise, checks
that the answers changed, refits, and requires the answers back token for token.

```
"The capital of France is"  ->  " Paris."
                 corrupted  ->  " shore Kel HOUSE HOUSE"
                  refitted  ->  " Paris."
```

## Where each framework fact lives

The plan is derived from the checkpoint's own safetensors header, which is the only
artifact both sides share. Everything framework-specific sits behind the two SPI
objects:

- `mx_m2n_e2e/trainer.py` - `FsdpPublisher` hands the wire op the FSDP2 local shard
  directly, so there is no staging copy on the trainer side.
- `mx_m2n_e2e/vllm_loader.py` - `VllmLoader` receives whole canonical tensors and
  hands each layer group to vLLM's own `model.load_weights()`, which already knows
  how to fold `q_proj`/`k_proj`/`v_proj` into `qkv_proj` and split it across tensor
  parallel ranks. Re-deriving that layout inside ModelExpress would duplicate the
  engine's loader and rot against it.

`MxRefitWorker` is installed as vLLM's `worker_extension_cls`, so the generator
client runs inside the vLLM workers. That is the only place it can run: the
destination storage is the engine's and exists nowhere else.

## Requirements

- One node with at least `trainers + generators` GPUs. Cross-node reshard is not
  supported by the NCCL build this path needs.
- A reachable ModelExpress server with the collective service.
- `pip install "nccl-extensions[cu12]"` or `[cu13]` to match the CUDA major, then
  the ModelExpress Python client. Reshard needs NCCL 2.30.7 or newer **loaded**;
  `torch.cuda.nccl.version()` reports what torch was compiled against, not what is
  mapped into the process, so read `ctypes.CDLL("libnccl.so.2").ncclGetVersion()`.

## Run

```bash
export MODEL=/models/Qwen3-4B
export MX_ENDPOINT=modelexpress-server.<namespace>.svc.cluster.local:8001
./run_e2e.sh
```

`T` and `G` set the trainer and generator counts, `ROUNDS` the number of refits.
Every parameter's first dimension must divide `T`: the plan rejects a shard that
does not divide evenly, and an FSDP pad would otherwise land wrong bytes with every
rank issuing its agreed op and nothing erroring.

A successful run ends in `E2E PASS` and writes per-rank JSON to `$OUT`.
