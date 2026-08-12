# Live weight refit for Miles

This describes how ModelExpress refits the weights of a running SGLang engine
during Miles training, why the refit is bracketed rather than a single call, and
what has and has not been established about it.

It spans three repositories. This one carries the publisher, the receiver and the
Miles adapter; the other two carry the halves that call into them:

| Repository | Branch | Contains |
|---|---|---|
| `ai-dynamo/modelexpress` | `kavink/miles-sglang-refit` | publisher, reshard receiver, Miles adapter |
| [`sgl-project/sglang`](https://github.com/KavinKrishnan/sglang/tree/kavink/modelexpress-live-refit) | `kavink/modelexpress-live-refit` (fork) | serving-side refit lifecycle and endpoints |
| [`radixark/miles`](https://github.com/KavinKrishnan/miles/tree/kavink/mx-live-refit) | `kavink/mx-live-refit` (fork) | `modelexpress` weight-transfer mode |

All three are needed together. None is useful on its own.

## What this adds

ModelExpress already integrated with SGLang, but only for cold-start loading: an
engine could be brought up from a ModelExpress source, and that was the end of
the relationship. Miles needs the opposite shape. The engine is already serving,
the trainer produces new weights every step, and the weights have to land in the
running engine without a restart.

## Terminology

Worth pinning down, because several things in this area share names.

**Refit** is replacing the weights of a running inference engine with newer ones
from the trainer. It is not a reload; the process, the CUDA graphs and the KV
cache allocation all survive it.

**Publisher** is the trainer-side role. It exposes the weights of one training
rank so receivers can read them.

**Receiver** is the inference-side role. It pulls the slices its own rank needs.

**Resharding** is translating between the trainer's parallelism layout and the
engine's. The two rarely agree: a trainer at EP=2 feeding an engine at EP=1 has
to have its expert dimension rearranged before the bytes mean anything.

**Cohort** is the set of workers a transfer plan was built against, plus a
generation number. A plan is only valid for its cohort.

**Version** is the trainer step a set of weights belongs to. Receivers commit a
version, and a version is what generation runs against.

Two naming collisions are worth calling out, because they have caused real
confusion. "M2N" inside ModelExpress refers to NIXL RDMA transfers, whereas
"NCCL M2N" is a different thing entirely -- the NCCL collective reshard
primitive (`ncclReshardWithWindow`) -- and Miles' "broadcast" mode is neither,
being a plain NCCL broadcast. This work uses the NIXL path.

## Why the refit is bracketed

The obvious API is one call: send the weights. That is not what this implements.
A refit is three phases:

1. `begin_weight_update` pauses generation and hands the ranks to the publisher.
2. The transfer runs. Each receiver pulls the slices its rank needs.
3. `end_weight_update` runs `post_load_weights` and quant finalization, then
   resumes generation.

The split exists so the commit point is explicit. The version is committed only
once every rank has installed the weights, which is what lets a failed refit
leave the previously committed version intact and still servable. With a single
call there is no moment that means "all ranks have it", and a partial failure
leaves an engine serving weights that are half of one version and half of
another -- which does not announce itself, because the engine still answers.

## Pull-mode resharding

The receiver pulls rather than the publisher pushing. This is what lets the two
sides disagree about parallelism: a receiver that knows its own layout can work
out which byte ranges of which publisher it needs and read exactly those,
without the publisher knowing anything about the engine's geometry.

Note what this does and does not save. It removes the layout mismatch, and where
the engine is sharded it reduces the bytes any single rank pulls. It does not
reduce total bytes when every engine needs the whole model -- and co-located
engines each pull their own copy, so the bytes crossing the wire scale with the
number of engines on the node, not with the model.

## Cohorts and replan

A transfer plan is built against a specific set of workers. If a rollout engine
is replaced, every plan referring to it is stale, and reusing it means reading
from a worker that is gone or, worse, from one that has been recycled into a
different role.

The adapter therefore keys the publisher session by cohort and rebuilds it when
membership changes, rather than rejecting the reconfiguration. Teardown is
factored out of `close()` for a related reason: the outgoing receiver's device
buffers have to be released before the incoming one allocates its own, or the
two buffer sets are live simultaneously. On a 30B MoE fleet that overlap was
enough to exhaust device memory during replan.

## Usage

On the Miles side, select the mode and point it at a publisher factory:

```
--update-weight-transfer-mode modelexpress
--modelexpress-publisher-adapter <import.path.to.factory>
```

The factory takes no arguments and returns a publisher. Keeping it behind an
import path means the transport is not wired into the training loop.

Validate with `--check-weight-update-equal`, which compares post-refit weights
against a snapshot. This is worth insisting on: the failure mode of a
misconfigured RDMA fabric is not an error but silently wrong tensors, and
generation continues happily on them.

## Operational notes

**Pin the NIC.** On fabrics where each NIC sits on its own subnet, two ranks
only reach each other if they chose the same one. Both this and other transports
pick NICs by PCIe proximity and neither checks reachability, so ranks on
different hosts can select mutually unreachable NICs. The symptom is not a
connection error; it is corrupt tensors. Set `MX_RDMA_NIC_PIN` to an explicit
device rather than relying on `auto` in that topology.

**`expandable_segments` and RDMA registration.** Running with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` interferes with NIXL's memory
registration. Leave it off for the refit path.

## Status

Correctness is established. On a 30B MoE model across two hosts, refit reaches
full parameter coverage, passes weight equality, survives rollout-engine
replacement with an automatic replan, and holds the previously committed version
when failures are injected before install.

Performance is characterised but not optimised, and the headline is that the
bottleneck is not where it was assumed to be. Control discovery, not wire
transfer, dominates a warm refit. Early figures suggesting otherwise were
measured with trainer and engine on the same host, where the transfer never
touched the network at all; once genuinely cross-host, wire time is a large
minority of the refit and runs near a single NIC's line rate. Two consequences:
same-host numbers should not be quoted as transport measurements, and the
optimisation target is discovery rather than bandwidth.

## Code

| Path | Role |
|---|---|
| `modelexpress_client/python/modelexpress/integrations/miles.py` | Miles adapter; owns the cohort-keyed publisher session |
| `modelexpress_client/python/modelexpress/engines/sglang/refit/` | SGLang-side receiver and worker |
| `modelexpress_client/python/modelexpress/refit/reshard/receiver.py` | engine-agnostic pull-mode reshard receiver |
| `modelexpress_client/python/modelexpress/refit/reshard/megatron_publisher.py` | trainer-side publisher |
| `modelexpress_client/python/modelexpress/refit/reshard/rendezvous.py` | publisher/receiver handshake |
