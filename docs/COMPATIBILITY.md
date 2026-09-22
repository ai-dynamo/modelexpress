<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compatibility and version selection

Use the docs and examples from the ModelExpress revision you deploy. The server, Python client, and runtime image have separate versions. A newer server does not add an MX loader to an older inference image.

## Choose a runtime

The entries below describe the checked-in examples and active CI definitions, not a guarantee that every model or hardware combination works. Check the linked files for the current pins and the workflow results for execution evidence.

| Path | Example / CI coverage | Start here |
|---|---|---|
| vLLM inference | Public example uses vLLM `0.23.0` with its native MX load format; GPU CI uses `0.17.1` with the plugin. CI includes P2P, parallelism, rolling updates, and S3 loading. | [vLLM guide](integrations/runtimes/vllm.md), [example image](../examples/p2p_transfer_k8s/client/vllm/Dockerfile), [CI image](../ci/k8s/client/vllm/Dockerfile) |
| SGLang inference | `v0.5.13.post1` example and CI image; NIXL/Mooncake P2P and direct/fallback S3 loading are in the CI matrix. | [SGLang guide](SGLANG.md), [CI image](../ci/k8s/client/sgl/Dockerfile) |
| Dynamo + vLLM | CI builds on Dynamo's `1.2.1` runtime and installs this repository's MX client. Aggregated and disaggregated serving are covered. Other prebuilt Dynamo images can contain a different MX version. | [Dynamo guide](integrations/orchestrators/dynamo.md), [CI image](../ci/k8s/client/vllm/dynamo/Dockerfile) |
| TensorRT-LLM inference | Provisional native `checkpoint_format="MX"` example for `LlamaForCausalLM`. Its Dockerfile still requires qualification of an image containing the complete upstream integration; the default `1.3.0rc22` tag is not proof of support. Not in the active GPU CI matrix. | [TensorRT-LLM guide](integrations/runtimes/tensorrt-llm.md) |
| RL weight updates | Evolving APIs under `modelexpress_rl`; use the complete image and orchestration recipe for the selected example. The refit coordinator currently requires Redis. | [RL guide](guides/rl.md) |
| Standalone server / Rust CLI | Rust tests cover cache and RPC behavior; no GPU is required. | [Deployment](DEPLOYMENT.md), [CLI](CLI.md) |

[GPU CI workflow](../.github/workflows/modelexpress-ci-tests.yml) · [Rust/Python CI workflow](../.github/workflows/ci.yml) · [Detailed test plan](../ci/TEST_PLAN.md)

## Installation boundaries

- **Python client:** see [pyproject.toml](../modelexpress_client/python/pyproject.toml) for Python and dependency requirements. Installing it does not install an inference engine or the Rust server/CLI.
- **NIXL:** supplied by the runtime image or installed separately for the matching CUDA stack. It is not a core Python package dependency. Verify both the Python import and the required transport on your nodes.
- **Engine images:** preserve the runtime's CUDA, Torch, NIXL, and protobuf combination. The SGLang example uses `--no-deps` for this reason; follow its Dockerfile rather than applying that flag to a bare environment.
- **Rust:** follow [CONTRIBUTING.md](../CONTRIBUTING.md) and [rust-toolchain.toml](../rust-toolchain.toml), including the protobuf compiler prerequisite.
- **Helm:** the chart version and default `appVersion` are in [Chart.yaml](../helm/Chart.yaml). Confirm the corresponding [published server image](https://catalog.ngc.nvidia.com/orgs/nvidia/ai-dynamo/containers/modelexpress-server/-/tags) exists, or override the image with one built from your checkout. Set the metadata backend explicitly; the chart does not install Redis or inference workers.

P2P requires compatible model identity, parallelism, tensor layout, and accelerator configuration on source and target. Pin the actual checkpoint through the engine's revision option; `MX_MODEL_REVISION` only labels source identity. Qualify GPU/fabric combinations beyond the checked-in tests, including AWS EFA, GDS, and decentralized `k8s-service` discovery, on the target deployment.
