# cursor-code-review rubric: SGLang distributed trainer runtime proof

Date: 2026-06-05

Scope:

- Runtime bridge source-rank filtering for `distributed-trainer-loop`.
- Proof metadata changes for vLLM and SGLang runtime bridge source/target
  artifacts.
- SGLang cross-node, one-pod-per-source-rank runtime proof where the target
  consumes MX/NIXL endpoints published by real `torch.distributed` trainer
  source processes.

Tool status:

- `cursor-code-review` executable was not present in `PATH`.
- `/Users/amaddipoti/Desktop/repos/a3sh-dotfiles/agents/IMPORTS.md` confirms
  `cursor-code-review` was inspected but not imported.
- This artifact applies the same high-signal cursor review rubric manually
  and records the verification gates.

Findings:

1. No blocking correctness finding.
   - `resolve_source_filter_for_publisher` centralizes the only new branch:
     distributed trainer publishers default to the current distributed rank
     only when no explicit `source_id` or `source_worker_rank` is provided.
   - Explicit filters are preserved, so existing single-source and manual
     source-selection behavior is not removed.
   - Both vLLM and SGLang call the helper after distributed process-group
     initialization and before ownership filtering, which is the required
     ordering for rank-local source publication.

2. No functionality-loss finding.
   - The source artifact proof flag now sets `real_trainer_process_used` from
     `real_distributed_trainer_loop_used`.
   - The target artifact proof flag now sets `real_trainer_process_used` only
     when distributed trainer sources are actually observed.
   - Existing no-overclaim flags remain explicit:
     `real_rl_training_loop_used=false`,
     `synthetic_trainer_loop_smoke_used=false` for the distributed trainer
     source path, and `trainer_full_all_gather_used=false`.

3. Residual scope risk is documented, not hidden.
   - The proof is SGLang only, tiny single tensor, Gloo process group,
     synthetic `torch.optim.SGD` objective, and staging-copy install through
     `Engine.update_weights_from_tensor`.
   - It does not prove vLLM distributed-trainer runtime refit, direct NIXL
     landing into runtime-owned storage, FSDP/TP/PP/EP/RL ownership, or
     full-model refit.

Verification:

- nscale artifact assertions passed for
  `artifacts/resharding/nscale-live-sglang-mx-runtime-distributed-trainer-crossnode-20260605.json`.
- The target artifact records:
  `result=pass`, `cross_node=true`, `one_pod_per_source_rank=true`,
  `real_distributed_trainer_loop_used=true`,
  `real_trainer_process_used=true`, `actual_nixl_reads_used=true`,
  `trainer_full_all_gather_used=false`, staging/runtime allclose+checksum pass,
  `trainer_to_inference_bytes=16384`, and `segment_count=2`.
- Source artifacts assert one rank-local ownership each:
  rank 0 owns `[[0, 32], [0, 64]]`; rank 1 owns `[[32, 64], [0, 64]]`.
- Focused nscale tests:
  `PYTHONPATH=modelexpress_client/python python3 -m pytest modelexpress_client/python/tests/test_refit_vllm_mx_runtime.py modelexpress_client/python/tests/test_refit_sglang_mx_runtime.py modelexpress_client/python/tests/test_refit_trainer_step.py -q`
  passed with `26 passed`.
- `git diff --check` passed locally and in the nscale dev checkout.
