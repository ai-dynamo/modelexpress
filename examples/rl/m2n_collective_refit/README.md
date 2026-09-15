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

## Two trainer backends

The Publisher is where a framework's storage layout is encoded, so one backend
cannot show the boundary is general. `TRAINER=fsdp2` (default) shards the
checkpoint with `torch.distributed.fsdp.fully_shard` and hands the wire op its
local shard directly. `TRAINER=deepspeed` runs ZeRO-3, which partitions each
parameter's *flattened* storage instead; that is not a dim-0 shard and cannot be
declared as one, so that Publisher stages, gathering one parameter at a time in
`start_new_round`. ZeRO-3 additionally reads `LOCAL_RANK` and `WORLD_SIZE` from
the environment, which the launcher sets.

## Two destination layouts

`DST=replicate` (default) hands every generator rank the whole tensor and lets
the engine's own loader split it. Correct for any architecture, and it puts the
engine's world size in wire bytes because each rank discards most of what it
receives.

`DST=sharded` declares each parameter with the placement the engine already
holds it in, so a rank receives exactly the slice it keeps and the Loader hands
the wire op a view into the live fused parameter. It carries the bytes once. The
arithmetic refuses rather than guesses: a fused parameter whose declared
constituents do not account for its local extent falls back to the replicated
path. Note that it writes into live weights, so a failed refit leaves the model
inconsistent, where the replicated path is atomic per layer group.

## Verification

`DIFF=1` adds `--diff-checkpoint`, which compares every live parameter against
what the engine's own loader produces from the checkpoint. **Use it.** It is
exact, it names the parameter when something is wrong, and when it runs it
becomes the acceptance gate.

Without it the only test is the token comparison, and that is not a sound
equality test on a larger model: greedy decoding is not bitwise reproducible
across cache states, so a near-tied logit diverges with every weight byte
correct and the run reports `E2E FAIL`. With `DIFF=1` that case is reported as
what it is.

The diff carries its own positive control and the run fails if the control does
not fire: it runs first against the deliberately corrupted model, where every
parameter must disagree. A diff that silently matched nothing would otherwise
report zero differences after the refit and read as the strongest possible pass.

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

`T` and `G` set the trainer and generator counts, `ROUNDS` the number of refits,
`TRAINER` the trainer backend (`fsdp2` or `deepspeed`), `DST` the destination
layout (`replicate` or `sharded`), and `DIFF=1` the exact verification above.
Every parameter's first dimension must divide `T`: the plan rejects a shard that
does not divide evenly, and an FSDP pad would otherwise land wrong bytes with every
rank issuing its agreed op and nothing erroring.

A successful run ends in `E2E PASS` and writes per-rank JSON to `$OUT`.
