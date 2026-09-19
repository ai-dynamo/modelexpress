# End-to-end NCCL M2N collective refit

A Hugging Face checkpoint, sharded across trainer ranks by a real framework,
refits a live vLLM engine over the collective path: the ModelExpress control
plane for rendezvous and admission, `nccl.m2n.reshard` for the weights.

`modelexpress_client/python/tests/gpu/bench_collective_refit.py` measures the
same transport with a stub at the engine boundary. This example has no stub. The
weights come from a checkpoint and the destination is vLLM's own storage.

## Correctness is generation

vLLM answers four fixed prompts greedily. Every live weight is then overwritten
with noise, which must change the answers. The trainer refits. The answers must
come back.

```
"The capital of France is"  ->  " Paris."
                 corrupted  ->  " shore Kel HOUSE HOUSE"
                  refitted  ->  " Paris."
```

Pass `DIFF=1`. It compares every live parameter against what vLLM's own loader
produces from the checkpoint. That comparison is the acceptance test. Why
it matters:

- Token equality is not sound on a larger model. Greedy decoding is not bitwise
  reproducible across cache states, so Qwen3-8B diverges at token 12 into two
  equally coherent continuations with all 291 parameters byte-exact. Without
  `DIFF=1` that run reports `E2E FAIL`.
- The comparison ships with a positive control and the run aborts if the control
  does not fire. It runs first against the corrupted model, where all 291
  parameters must disagree. A comparison that matched no parameters would
  report zero differences after the refit and read as a pass.

## Run

```bash
export MODEL=/models/Qwen3-4B
export MX_ENDPOINT=modelexpress-server.<namespace>.svc.cluster.local:8001
DIFF=1 ./run_e2e.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `T` | 4 | trainer ranks, one GPU each |
| `G` | 4 | generator ranks, one GPU each, vLLM tensor-parallel size |
| `ROUNDS` | 5 | refits after the first |
| `TRAINER` | `fsdp2` | `fsdp2` or `deepspeed` |
| `DST` | `replicate` | `replicate` or `sharded` |
| `DIFF` | unset | `1` turns on the per-parameter comparison |

The run needs `T + G` GPUs on one node, a reachable ModelExpress server, and
NCCL 2.30.7 or newer mapped into the process. `torch.cuda.nccl.version()`
answers a different question: it reports the version torch was compiled
against. Read `ctypes.CDLL("libnccl.so.2").ncclGetVersion()` instead. Install
`nccl-extensions[cu12]` or `[cu13]` to match the CUDA major, then the
ModelExpress Python client.

## Two trainer backends

A Publisher encodes one framework's storage layout. A second one is here so the engine
boundary gets tested on more than one layout.

`fsdp2` shards with `torch.distributed.fsdp.fully_shard` and hands the wire op
its DTensor local shard with no staging copy.

`deepspeed` runs ZeRO-3, which partitions each parameter's flattened storage.
That is not a dim-0 shard and cannot be declared as one, so `start_new_round`
gathers one parameter at a time into a proxy buffer. Cost measured at 1.14 ms
per parameter against 1.02 for FSDP2. ZeRO-3 also reads `LOCAL_RANK` and
`WORLD_SIZE` from the environment; `run_e2e.sh` sets both.

## Two destination layouts

`replicate` gives every generator rank the whole tensor and lets vLLM's
`load_weights` split it. Correct for any architecture. Each rank discards most
of what it receives. Wire volume is `G` times the canonical bytes.

`sharded` declares each parameter with the placement vLLM already holds it in.
A rank receives the slice it keeps, and the Loader hands the wire op a view into
the live fused parameter: `q_proj`, `k_proj` and `v_proj` at their offsets
inside `qkv_proj`, `gate_proj` and `up_proj` inside `gate_up_proj`, row-parallel
projections and norms direct. No scratch buffer, and `install` does no work. On Qwen3-8B this is 335 ms against 415 ms for `replicate`, at one quarter of
the wire bytes.

A fused parameter whose declared parts do not add up to its local extent falls
back to `replicate`. Replicated key/value heads and a padded vocabulary are the
common causes. The offset arithmetic never guesses.

## Where each framework fact lives

The plan comes from the checkpoint's safetensors header, which is the only
artifact both sides share. Everything framework-specific sits behind the SPI
objects:

- `mx_m2n_e2e/trainer.py` - `FsdpPublisher` and `DeepSpeedPublisher`.
- `mx_m2n_e2e/vllm_loader.py` - `VllmLoader`, plus `MxRefitWorker`, installed as
  vLLM's `worker_extension_cls`. The generator client runs inside the vLLM
  worker processes because the destination storage belongs to the engine.

## Caveats

Read these before quoting any number out of this directory.

**This is a reference implementation under `examples/`, not a shipped
integration.** `docs/NCCL_M2N_REFIT.md` §11 lists the vLLM loader and the
Megatron publisher as the two pieces held out of the client PR. This covers the
first. The Megatron publisher does not exist.

**This example puts every parameter on the reshard path. The shipped default is
FFN only.**
`plan.py`'s `is_bulk_param` whitelists FFN projections only, inherited from
NeMo-RL because FFN is 97% of the bytes on large MoE models; everything else is
meant to ride the packed misc broadcast. This example overrides that and leaves
`misc` empty. `is_bulk_param`'s docstring allows it. So §12 lists wider coverage
as an open question and these runs are a data point for it, and **no timing here
describes the FFN-only split.** §12 puts the bulk
fraction at 67% on dense models against 97% on MoE, so those are far apart.

**Single node only.** A cross-node reshard has failed inside NCCL's transport
layer, below any ModelExpress code, with one trainer and one generator. That is
not a scale effect. `run_e2e.sh` assumes one node anyway:
it slices `CUDA_VISIBLE_DEVICES` and sets `MASTER_ADDR=127.0.0.1`.

**One source partition.** `build_plan` hardcodes `source_partition_count=1`, so
this example cannot express pipeline stages. The protocol supports more. This example does not
exercise it against an engine.

**Every parameter's first dimension must divide `T`.** The plan rejects a shard
that does not divide evenly. An FSDP pad puts wrong bytes on the wire with every
rank issuing its agreed op and no error raised. Qwen3 sizes divide by 4 and 8; other
geometries need checking. `build_plan` names what it refuses.

**A tied output head is in the checkpoint and in neither framework's parameter
list.** `config.json`'s `tie_word_embeddings` drops it, including the nested
`text_config` form. The checkpoint is a superset of what either side holds.

**`DST=sharded` writes into live weights,** so a failed refit leaves the model
inconsistent. `replicate` is atomic per layer group.

**`pip install deepspeed` downgrades `nvidia-nccl-cu13` from 2.30.7 to 2.29.7,**
which removes the version the reshard requires. Re-read `ncclGetVersion()` after
any install into the image.

**All measurements quoted above are 4 trainers and 4 generators on one node of
8x B200,** NVLink, torch 2.13.0+cu130, vLLM 0.26.1. None of them is an H100
number and none is cross-node.
