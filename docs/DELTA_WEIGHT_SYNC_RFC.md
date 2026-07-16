# RFC: Delta-Compressed Weight Updates

## Summary

- Adds requirements for delta-compressed weight updates as a first-class ModelExpress source type.
- Frames the requirement around the pattern shown publicly by Cursor/Fireworks, Cognition, and Slime: full checkpoint anchors, compact deltas between anchors, durable shared storage, background fetch/process/stage, integrity checks, and short runtime apply windows.
- Keeps the proposal at the MX requirements layer instead of prescribing one delta format, storage backend, or framework implementation.
- Calls out the approval points for each requirement and acceptance criterion so reviewers can agree on the product boundary before we design APIs.

## Why

Async RL only works well if rollout workers can stay close enough to the trainer policy without repeatedly moving full checkpoints or pausing serving for long update windows.

The public Cursor/Fireworks and Cognition writeups point to the same system shape: move less data between trainer and rollout fleets, do most of the work before pausing inference, verify reconstruction, and apply the staged version through the serving runtime.

Slime gives a useful open-source version of the same pattern: non-colocated trainer/rollout weight sync, per-tensor deltas, shared storage, host-local checkpoint patching, per-tensor integrity checks, and reload through the ordinary serving-runtime disk update path.

MX should make this pattern framework-neutral by owning the versioning, readiness, tensor identity, target demand, fallback, and observability contract.

References:

- [Slime: Delta Weight Sync example](https://github.com/THUDM/slime/tree/main/examples/delta_weight_sync)
- [Fireworks: Cursor Composer 2 + Fireworks AI](https://fireworks.ai/blog/Cursor-Composer-2)
- [Fireworks: Frontier RL Is Cheaper Than You Think](https://fireworks.ai/blog/frontier-rl-is-cheaper-than-you-think)
- [Cognition: SWE-1.7](https://cognition.com/blog/swe-1-7)
- [Cursor: Improving Composer through real-time RL](https://cursor.com/blog/real-time-rl-for-composer)
- [Sequoia: How Cursor Trained Composer on Fireworks](https://sequoiacap.com/podcast/how-cursor-trained-composer-on-fireworks-distributed-infrastructure-for-high-performance-rl/)

## Requirements

- MX tracks a model-version chain made of full checkpoint anchors and delta updates between anchors. Approval note: approve this if MX needs to understand checkpoint lineage, not just advertise "latest weights available."
- MX stores enough delta metadata to validate model identity, base version, target version, tensor identity, layout compatibility, checksums, and reconstruction requirements. Approval note: approve this if MX should prevent wrong-version or wrong-layout updates before a runtime attempts to apply them.
- MX rejects a delta update when the target worker's current version does not match the delta's required base version, unless replay from an earlier anchor is available. Approval note: approve this if silent drift is unacceptable and stale workers must explicitly replay or fall back.
- MX exposes a pre-pause `fetch/process/stage` phase where workers download deltas, reconstruct required weights or shards, and verify checksums while continuing to serve. Approval note: approve this if the core customer value is reducing rollout downtime rather than only reducing bytes moved.
- MX exposes a pause-bound `apply/resume` phase where the runtime installs already-staged weights through framework-native hooks. Approval note: approve this if MX should separate expensive transfer/reconstruction work from the short runtime mutation window.
- MX supports target demand metadata so a worker can declare which tensors, ranges, TP/PP/EP shards, or local experts it needs for a target version. Approval note: approve this if MX should eventually avoid moving tensors, shards, or experts a worker does not need.
- MX supports ordinary-runtime reload paths, including the case where deltas patch a host-local checkpoint and the serving runtime reloads from disk. Approval note: approve this if MX should work with Slime/SGLang-style flows and not require every runtime to expose a delta-native GPU apply path in P0.
- MX makes partial-update legality explicit: which tensors can update independently, which must be grouped, which can be skipped, and which require full reload or framework fallback. Approval note: approve this if correctness rules should be visible in MX instead of hidden in per-framework naming conventions.
- MX supports shared storage as a durable source of truth for delta payloads and associated metadata, including object-store-backed filesystems that may need explicit visibility hooks. Approval note: approve this if the target use case includes wide-area async RL, not only same-cluster transport.
- MX supports cluster-local fanout after one controller or worker fetches a delta, including local disk, node-local cache, or tree broadcast. Approval note: approve this if MX should help reduce repeated shared-storage downloads and trainer egress at fleet scale.
- MX provides correctness checks, including post-reconstruction checksums and final runtime version confirmation. Approval note: approve this if MX readiness should mean "this worker is actually on the intended version," not best-effort loading.
- MX provides recovery behavior for missed versions, corrupted deltas, stale workers, and lost local checkpoint state. Approval note: approve this if restarted or delayed rollout workers are expected in normal async RL operation.
- MX emits per-update accounting for compressed bytes, reconstructed bytes, skipped bytes, fallback bytes, full-checkpoint bytes, trainer egress bytes, shared-storage egress bytes, and local fanout bytes. Approval note: approve this if we want to compare delta updates fairly against NIXL, NCCL M2N, full checkpoint reload, and framework-native broadcast.

## Acceptance Criteria

- A trainer or publisher can register a full checkpoint anchor and at least one compressed delta update with MX. This proves MX can ingest delta-compressed updates as a first-class source type.
- A rollout worker can ask MX for the next version, fetch/process/stage the required delta payload before pausing inference, and report staged readiness. This proves MX supports the Cursor/Fireworks/Cognition/Slime lifecycle where most work happens outside the serving pause.
- A rollout worker can pause, apply the staged version through a framework-native update hook, verify the final version, and resume. This proves the flow works against a real serving runtime boundary instead of only reconstructing offline weights.
- A disk-based path can patch a host-local checkpoint and reload through an ordinary runtime disk update API. This proves MX can support the open-source Slime/SGLang shape without requiring a new runtime-specific delta apply surface in P0.
- MX refuses to apply a delta if the base version, tensor metadata, layout tags, or checksum validation do not match. This proves the safety contract is enforced before runtime mutation.
- MX can recover a worker by replaying deltas from the latest checkpoint anchor or by routing it to a full checkpoint fallback. This proves the system handles normal async RL messiness: missed versions, restarts, and stale workers.
- MX reports timing split by `manifest_publish_ms`, `fetch_ms`, `process_ms`, `stage_ms`, `pause_ms`, `apply_ms`, and `resume_ms`. This proves reviewers can separate transfer/reconstruction cost from actual inference downtime.
- Reconstructing from the delta chain produces byte-equivalent or explicitly tolerance-equivalent weights compared with the target full checkpoint reference. This proves delta updates are not accumulating silent corruption or semantic drift.

## Open Questions

- What should MX own versus what should stay framework-owned in the delta update lifecycle?
- Should P0 be metadata/control plane only, or should MX include a reference fetch/reconstruct/stage client?
- Should the first MX integration target the disk/local-checkpoint path because it matches Slime's open-source design, or should we prioritize a runtime-native path for vLLM/SGLang?
- How should MX choose between delta artifacts, NIXL, NCCL M2N, full checkpoint fallback, and framework-native broadcast when multiple paths are available?
- What freshness target matters most for the first implementation: lowest trainer egress, lowest rollout pause, lowest policy staleness, or best end-to-end throughput?
- What level of partial update support is required for P0, especially for MoE experts, fused layers, quantized inference formats, and FP8 metadata?
- What cooperation do we need from vLLM, SGLang, Megatron, NeMo RL, and TRT-LLM to make this robust instead of adapter-specific?
- What correctness standard should MX enforce for reconstructed weights: byte-exact, tolerance-equivalent, per-tensor checksum, or end-to-end model-version attestation?

## Product Stance

Delta-compressed updates should not be a one-off loader path in MX. They should be a first-class versioned source type that reuses MX's strengths: tensor identity, target demand, partial-update legality, version fences, readiness, fallback policy, and observability.
