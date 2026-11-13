# NCCL M2N Collective Refit

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Package `modelexpress_rl.collective`. Authoritative source:
`docs/NCCL_M2N_REFIT.md` (Part 1 is the stable client API; Part 2 is internal).
Reference integration: `examples/rl/m2n_collective_refit/`.

## Roles

| Component | Owns |
|---|---|
| RL framework / orchestrator | Picks the version and the refitting generators; calls both sides with the same `version` and `operation_id` |
| MX control plane (`RefitCollectiveService`) | Group identity, admission, rank assignment, NCCL unique id, worker-generation fencing, `FORMING -> READY` |
| `RefitClientTrainer` / `RefitClientGenerator` | Round lifecycle, layer groups, local plan |
| `Publisher` / `Loader` (you write these) | Map each HF parameter name to local storage; stage or install it |

Weight bytes never reach MX. MX stores only the plan digest.

## Wiring One Worker

```python
import grpc
from modelexpress_rl.collective import CollectiveRendezvous, RefitClientTrainer

channel = grpc.insecure_channel(mx_endpoint)
client = RefitClientTrainer(
    rendezvous=CollectiveRendezvous(channel),
    model_name=run_name,
    trainer_slots=[f"t{i}" for i in range(num_trainers)],
    generator_slots=[f"g{i}" for i in range(num_generators)],
    source_partition_count=1,          # reshard lanes, typically trainer PP size
    slot_id=f"t{rank}",
    worker_id=f"t{rank}-{run_id}",     # new id after every process restart
    index_in_role=rank,
    device=device,
    streams=[torch.cuda.Stream(device=device)],
)
client.setup_layer_groups(groupings)   # optional; identical everywhere
client.initialize(publisher, source_partition=0)
membership = client.compute_plan()
```

The generator is the same with `RefitClientGenerator` and
`initialize(loader)`. In vLLM, run the generator client inside the worker
processes through `worker_extension_cls` (the example uses
`mx_m2n_e2e.vllm_loader.MxRefitWorker`), because the destination storage
belongs to the engine.

Rules that must hold on every participant:

- `trainer_slots`, `generator_slots`, `model_name`, layer groupings,
  `receiver_protocol`, and `m2n_abi_version` are identical.
- `source_partition_count` divides `len(trainer_slots)`; lane `i` gets the
  `i`-th contiguous block of trainer slots.
- `MX_NCCL_REFIT_BLOCKING_DATA_PLANE` and `MX_NCCL_REFIT_GROUPED_SUBMISSION`
  are read once at construction and hashed into the plan digest; a cohort that
  disagrees stays `FORMING`.

## Call Sequence

1. Once per worker: `initialize`, optionally `setup_layer_groups` first.
2. Every worker: `compute_plan`. All must return before any `start_weight_update`.
3. Per refit: `create_operation(version, key_prefix=...)` on each participant
   (idempotent per `(key_prefix, group, epoch, version)`), or once by a
   coordinator that hands out the id. `key_prefix` must be non-empty.
4. Every participant: `start_weight_update(version, operation_id)`.
5. Per layer group, same order everywhere: trainer `publish_weights(version, layer_group_id)`,
   generator `update_weights(version, layer_group_id)`.
6. Every participant: `finish_weight_update(version)`. Next refit: step 3.
   After an epoch change: step 2.
7. Terminal: `cleanup()`, then `CollectiveRendezvous.close()`.

If `finish_weight_update` cannot deliver its success report (retried on
UNAVAILABLE and DEADLINE_EXCEEDED, 3 attempts), call `retry_report()`; the next
`start_weight_update` is refused until it succeeds.

## Engine SPI

```python
class Publisher(Protocol):
    def capture(self) -> ReshardPlan: ...
    def parameter_names(self) -> list[str]: ...
    def local_params(self) -> dict[str, LocalParamSpec]: ...
    def start_new_round(self, version: str) -> None: ...
    def cleanup(self) -> None: ...

class Loader(Protocol):
    # Publisher methods, plus:
    def install(self, layer_group_id: int) -> None: ...
    def finish(self) -> None: ...
```

- Both sides' `capture()` must produce the same plan.
- `LocalParamSpec(base=..., pre=..., post=...)`: `base` is the live tensor;
  `pre` returns a `RefitCtx` whose `buf` the wire op uses; `post` commits it.
  `pre`, the wire op and `post` run on one CUDA stream.
- Default bulk classification `is_bulk_param` covers FFN projections only; the
  rest rides the packed misc broadcast. The example overrides this and puts
  every parameter on the reshard path.

## Exceptions

| Exception | Raised from | Caller action |
|---|---|---|
| `GroupNotReadyError` (`.missing`) | `compute_plan` | Fix or restart the missing slots; retry `compute_plan` |
| `EpochChangedError` (`.disagreeing`) | `compute_plan`, `start_weight_update` | Re-run `compute_plan`; fix the plan first if `.disagreeing` is set |
| `TransferBindingError` | `start_weight_update` | Create a new operation at the current epoch |
| `RendezvousError` | any call | Re-run `compute_plan` |
| `PlanCoverageError` | `initialize`, `compute_plan` | Fix the `Publisher`/`Loader` so the plan covers the parameter set |
| `NcclUnavailableError` | `compute_plan` | Install `nccl4py` and `nccl-extensions`; libnccl >= 2.30.7 |
| `CommunicatorWedgedError` | round calls, `cleanup` | Terminate the process with a non-zero `os._exit` |

The first three subclass `RendezvousError`. Any other exception from a round
call is a transport or timeout failure: the round is already aborted and
reported FAILED, so re-run `compute_plan`. A failure while `compute_plan`
bootstraps communicators is fatal for the worker: restart the process.

## Configuration

All values are validated when read; a non-numeric, zero, or negative value
raises (the two window sizes accept `0` to disable the bound).

| Variable | Default | Purpose |
|---|---|---|
| `MX_NCCL_REFIT_GROUP_TIMEOUT_S` | `600` | Deadline for `FORMING -> READY` (`run_e2e.sh` sets `900`) |
| `MX_NCCL_REFIT_POLL_INTERVAL_S` | `0.25` | `GetCollectiveGroup` poll backoff floor |
| `MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S` | `300` | One lane's communicator init; `NCCL_COMM_BLOCKING` other than `0` is rejected |
| `MX_NCCL_REFIT_TRANSFER_TIMEOUT_S` | `600` | Transfer deadline, checked between submissions only |
| `MX_NCCL_REFIT_ABORT_TIMEOUT_S` | `10` | Shared budget to abort all lanes before `CommunicatorWedgedError` |
| `MX_NCCL_REFIT_REGISTRATION_TTL_S` | `3 x MX_HEARTBEAT_INTERVAL_SECS` (`90`) | Worker liveness lease |
| `MX_NCCL_REFIT_PIPELINE_DEPTH` | `2` | Layer groups the generator keeps in flight |
| `MX_NCCL_REFIT_BLOCKING_DATA_PLANE` | `1` | Data traffic on a blocking communicator split from the non-blocking one |
| `MX_NCCL_REFIT_GROUPED_SUBMISSION` | `0` | One `nccl.m2n.group()` per layer group; needs GIN |
| `MX_NCCL_REFIT_RECEIVE_WINDOW_BYTES` | `1073741824` | Generator `pre` scratch bound per layer group |
| `MX_NCCL_REFIT_SEND_WINDOW_BYTES` | `1073741824` | Trainer `pre` scratch bound per layer group |
| `MX_NCCL_REFIT_NUM_STREAMS` | `2` | Stream pool for integrations that build their own `streams`; not read by the refit clients |

## Runner Contract

- Catch `CommunicatorWedgedError` at top level and exit with a non-zero
  `os._exit`. The round is already reported FAILED, so peers re-form at a new
  epoch without this worker.
- A reshard or grouped submission can block the host inside native code, where
  `MX_NCCL_REFIT_TRANSFER_TIMEOUT_S` cannot fire. Bound every trainer and
  generator process from outside. `run_e2e.sh` does this with `PROC_TIMEOUT_S`
  (SIGTERM on expiry, SIGKILL 30 s later).
- A restarted process uses a new `worker_id`; it is admitted as a new
  generation and the epoch moves.

## Reference Example Knobs

`examples/rl/m2n_collective_refit/run_e2e.sh` requires `MODEL` and
`MX_ENDPOINT`, and needs `T + G` GPUs on one node.

| Variable | Default | Meaning |
|---|---|---|
| `T` / `G` | `4` / `4` | Trainer ranks / generator ranks (vLLM tensor-parallel size) |
| `ROUNDS` | `5` | Refits after the first |
| `TRAINER` | `fsdp2` | `fsdp2` or `deepspeed` |
| `DST` | `replicate` | `replicate` or `sharded` (writes into live weights; not atomic) |
| `DIFF` | unset | `1` turns on the per-parameter comparison against vLLM's own load |
| `PROC_TIMEOUT_S` | `0` | Per-process bound in seconds |
| `RUN`, `OUT`, `MASTER_PORT` | `e2e-<epoch>`, `/work/out/$RUN`, `29555` | Run id, output dir, trainer rendezvous port |

Outputs under `$OUT`: `generator.log`, `trainer<rank>.log`, one `.rc` per
process, and `generator.json`. Known caveats from the example README:

- Every parameter's first dimension must divide `T`; `build_plan` names what it refuses.
- `pip install deepspeed` can downgrade `nvidia-nccl-cu13` below 2.30.7;
  re-check `ctypes.CDLL("libnccl.so.2").ncclGetVersion()` after installs.
- `torch.cuda.nccl.version()` reports the compile-time version, not the mapped library.
