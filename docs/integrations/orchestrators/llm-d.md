<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# llm-d

Use llm-d's merged [ModelExpress P2P guide](https://github.com/llm-d/llm-d/tree/main/guides/modelexpress-p2p) for the end-to-end Optimized Baseline deployment. ModelExpress links to that guide instead of duplicating llm-d's router, Gateway API, Kustomize, and model-server manifests.

llm-d owns orchestration and routing. ModelExpress owns the vLLM load format, metadata contract, server, and NIXL transfer between workers.

## Version alignment

The guide merged in [llm-d PR #1608](https://github.com/llm-d/llm-d/pull/1608) currently pins ModelExpress `v0.5.0`; this repository and Helm chart are at `0.5.1`. Either reproduce the guide with its matching `v0.5.0` server, client, and CRDs, or update all ModelExpress image, package, and CRD references together and rerun the guide's validation. Do not mix versions implicitly.

ModelExpress does not currently carry an llm-d end-to-end CI job. The upstream guide validates the llm-d composition; this repository validates the underlying vLLM P2P path. ModelExpress settings are documented in [Configuration](../../CONFIGURATION.md).
