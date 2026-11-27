## Description: <br>
Set up ModelExpress P2P weight loading so new inference replicas pull weights from a running replica over RDMA instead of reading storage. Use to choose a metadata backend, install and health-check the ModelExpress server, enable P2P on vLLM, SGLang, or TensorRT-LLM workers, including vLLM and SGLang under NVIDIA Dynamo, and confirm the weights arrived over P2P; not for RL weight refit. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
ML infrastructure and inference platform engineers use this skill to choose a ModelExpress metadata backend, install and health-check the ModelExpress server on Kubernetes, configure vLLM, SGLang, or TensorRT-LLM workers (standalone or under NVIDIA Dynamo) to load weights over P2P, and confirm from worker logs that later replicas received their weights from a running source. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: A transfer can run over TCP or the wrong NIC without warning: NIXL's default UCX backend picks the transport from the configured resources available (such as `UCX_TLS`, `NIXL_UCX_TLS`, and `UCX_NET_DEVICES`), and the completion log line does not name it. <br>
Mitigation: The skill treats low throughput as a reason to check the transport, and points to NIC pinning, the libfabric backend on AWS EFA, and the AWS EFA example's steps to confirm the fabric. <br>
Risk: Workers can register with a different ModelExpress server than intended, because the deprecated `MODEL_EXPRESS_URL` takes precedence over `MX_SERVER_ADDRESS` and the Dynamo operator can inject it. <br>
Mitigation: The skill instructs setting both variables to the same address and lists the symptom in its troubleshooting table. <br>
Risk: The server's authentication is off by default and the example Redis manifest has no password. <br>
Mitigation: The skill notes both, points to the deployment docs for enabling authentication, and advises a password-protected Redis for production. <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Server and Backends](references/server-and-backends.md) <br>
- [Standalone Runtimes](references/runtimes.md) <br>
- [NVIDIA Dynamo](references/dynamo.md) <br>
- [ModelExpress Releases](https://github.com/ai-dynamo/modelexpress/releases/latest) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash and YAML code blocks] <br>
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
