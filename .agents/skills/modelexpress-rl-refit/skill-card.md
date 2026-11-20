## Description: <br>
Wire an RL trainer (Megatron, FSDP, or Miles) and a vLLM or SGLang generator through ModelExpress for live weight refit. Use to choose between the NCCL M2N collective path, the NIXL reshard path, and the S3 delta path, integrate any of them, run the reference end-to-end refit, configure FSDP CPU staging or RDMA rail selection, or diagnose refit failures such as group admission, epoch changes, version leases, deadline expiry, and delta lineage or checkpoint-cache errors; not for initial model loading. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
RL framework integrators and ML infrastructure engineers use this skill to select a ModelExpress weight-refit transport, connect trainer ranks and vLLM or SGLang generator engines to the ModelExpress control plane, validate the integration with the reference end-to-end example, choose FSDP snapshot staging, diagnose RDMA rail contention, and triage refit failures into actionable categories. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: A failed refit can leave live generator weights partially updated, and some M2N failures require terminating the worker process. <br>
Mitigation: The skill instructs pausing generation before installation, verifying the installed version before resuming, restarting engines after any failure once installation has started, and running the bundled preflight script, which is read-only. <br>
Risk: Reused snapshot buffers or misleading NIC visibility can cause stale weights or invalid performance claims. <br>
Mitigation: Retire versions before reusing arenas, preserve dtype exceptions, validate the usable per-worker NIC map and physical traffic, and separate synthetic transport timings from live refit. <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NCCL M2N Collective Refit](references/m2n-collective-refit.md) <br>
- [NIXL Reshard Refit](references/nixl-reshard-refit.md) <br>
- [S3 Delta Refit](references/s3-delta-refit.md) <br>
- [FSDP CPU Staging](references/fsdp-cpu-staging.md) <br>
- [RDMA Rail Selection](references/rdma-rails.md) <br>
- [Failure Decision Tree](references/failure-decision-tree.md) <br>
- [ModelExpress Releases](https://github.com/ai-dynamo/modelexpress/releases/latest) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Python code, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash and Python code blocks; JSON from the preflight script] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Tasks: <br>
Evaluation cases are in `evals/evals.json`. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
0.8.0 (source: modelexpress_client/python/pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
