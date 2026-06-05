# cursor-code-review rubric: vLLM distributed trainer runtime proof

Date: 2026-06-05

Scope:

- vLLM MX runtime bridge proof with `distributed-trainer-loop` source
  publisher.
- Cross-node one-pod-per-source-rank placement where a live vLLM target reads
  MX-published NIXL endpoints from real `torch.distributed` trainer source
  processes.
- Artifact and claim boundary updates in `goal.md`.

Tool status:

- `cursor-code-review` executable was not present in `PATH`.
- `/Users/amaddipoti/Desktop/repos/a3sh-dotfiles/agents/IMPORTS.md` confirms
  `cursor-code-review` was inspected but not imported.
- This artifact applies the same high-signal cursor review rubric manually
  and records the verification gates.

Findings:

1. No blocking correctness finding.
   - No code path was changed for this proof; it exercised the existing
     distributed source-filter logic added for the SGLang proof.
   - The source artifacts show each `torchrun` process defaulted to its own
     rank-local `SliceOwnership`: rank 0 owns `[[0, 32], [0, 32]]`; rank 1
     owns `[[32, 64], [0, 32]]`.
   - The target artifact records MX endpoint discovery, two actual UCX/NIXL
     READs, vLLM `LLM.apply_model` install, original tensor restore, and
     allclose/checksum gates.

2. Blocked placement is separated from proof claim.
   - The first rank-1 pod landed on `bw4bt` and hung on a bounded Python
     import/CUDA probe.
   - That placement is banked as
     `nscale-live-vllm-mx-runtime-distributed-trainer-crossnode-bw4bt-import-hang-20260605.json`
     with `proof_claim_safe=false`.
   - The passing proof uses replacement rank 1 on `th9sn`, with source nodes
     `g2j7h` and `th9sn`, and target node `9c2x7`.

3. Residual scope risk is documented, not hidden.
   - The proof is vLLM only, tiny single tensor, Gloo process group,
     synthetic `torch.optim.SGD` objective, and staging-copy install through
     `LLM.apply_model`.
   - It does not prove direct NIXL landing into runtime-owned storage,
     FSDP/TP/PP/EP/RL ownership, production lifecycle integration, or
     full-model refit.

Verification:

- nscale artifact assertions passed for
  `artifacts/resharding/nscale-live-vllm-mx-runtime-distributed-trainer-crossnode-20260605.json`.
- The target artifact records:
  `result=pass`, `cross_node=true`, `one_pod_per_source_rank=true`,
  `real_distributed_trainer_loop_used=true`,
  `real_trainer_process_used=true`, `actual_nixl_reads_used=true`,
  `trainer_full_all_gather_used=false`, staging/runtime allclose+checksum pass,
  `trainer_to_inference_bytes=4096`, and `segment_count=2`.
- Source artifacts assert one rank-local ownership each:
  rank 0 owns `[[0, 32], [0, 32]]`; rank 1 owns `[[32, 64], [0, 32]]`.
- Focused nscale tests:
  `PYTHONPATH=modelexpress_client/python python3 -m pytest modelexpress_client/python/tests/test_refit_vllm_mx_runtime.py modelexpress_client/python/tests/test_refit_sglang_mx_runtime.py modelexpress_client/python/tests/test_refit_trainer_step.py -q`
  passed with `26 passed`.
