<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress RL package

The optional RL APIs add trainer publication, exact-version generator startup, and live weight updates to the ModelExpress Python distribution. Install the existing `modelexpress` distribution, including its runtime dependencies, then import RL APIs from `modelexpress_rl`. The source and installed package both retain the existing `modelexpress_rl` namespace.

```bash
pip install modelexpress
```

The RL APIs and base client are shipped by the same distribution. The RL package depends on the base package, and the base package does not depend on or import the RL package during ordinary inference loading. RL-only environment variables are documented in [RL configuration](CONFIGURATION.md). The detailed control plane, transfer planning, staging, caching, and recovery behavior are in [RL refit architecture](ARCHITECTURE.md). For S3 delta publication, use the [S3 delta refit guide](S3_DELTA_WEIGHT_REFIT.md).

## Quick start

A trainer publishes through the rank-local client after the framework has created a version and declared its source slots. The orchestrator and trainer responsibilities are described in the [S3 delta refit guide](S3_DELTA_WEIGHT_REFIT.md) and [trainer publication example](S3_DELTA_WEIGHT_REFIT.md#initialize-the-trainer-client).

```python
from modelexpress_rl import (
    MegatronTrainerContext,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    WeightVersionRef,
)

trainer = ModelExpressTrainerClient.initialize(
    ModelExpressTrainerConfig(engine_context=MegatronTrainerContext())
)
trainer.bind_tensors(megatron_tensor_specs)
trainer.publish_version(version=WeightVersionRef(version.version_id))
```

Generator clients use the same namespace. Engine setup and version preparation are specific to the serving runtime; see [vLLM refit installation](ARCHITECTURE.md#python-engine-integration) and the runtime integration pages.

## Runtime plugin setup

The distribution installs these automatic plugin entry points:

| Runtime | Entry-point group | Name | Target |
|---|---|---|---|
| vLLM | `vllm.general_plugins` | `modelexpress_rl` | `modelexpress_rl.inference.engines.vllm.registration:register` |
| SGLang | `sglang.srt.plugins` | `modelexpress_rl` | `modelexpress_rl.inference.engines.sglang.registration:register` |

With engine plugin autoload enabled, no manual registration call is needed. If plugin loading is filtered, include `modelexpress_rl` in `VLLM_PLUGINS` or `SGLANG_PLUGINS` as appropriate. For vLLM, the plugin exposes the `modelexpress` native weight-transfer backend; the regular base inference loader remains available independently.

Deployments that previously used only `VLLM_PLUGINS=modelexpress` for refit must include `modelexpress_rl`. For manual RL registration, call `modelexpress_rl.inference.engines.vllm.registration.register()` instead of the base `register_modelexpress_loaders()`. Set `MX_LOAD_STRATEGY_CHAIN` before registration. Existing native or custom weight-transfer registrations are preserved.

TensorRT-LLM support in the base package covers initial checkpoint P2P loading. It does not include RL refit or trainer/generator update APIs, and the RL-only settings are rejected by that integration.

## Support and tests

The RL package supports the explicitly implemented trainer and generator integrations documented in the architecture guide. vLLM and SGLang use automatic runtime entry points; the shared initial-load path and TensorRT-LLM P2P support remain in the base package.

Base and RL Python tests remain under `modelexpress_client/python/tests`. Run both suites with `python -m pytest tests/`.
