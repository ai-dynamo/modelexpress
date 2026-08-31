<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# llm-d

llm-d has a merged, runnable ModelExpress P2P guide: [llm-d ModelExpress P2P](https://github.com/llm-d/llm-d/tree/main/guides/modelexpress-p2p). It composes ModelExpress with llm-d's Optimized Baseline and uses the MX server as a metadata broker while vLLM workers transfer weights directly over NIXL/RDMA.

## What the llm-d guide provides

The upstream guide includes the end-to-end deployment, ModelExpress server and CRD manifests, Kustomize overlays, a ModelExpress-enabled image Dockerfile, a no-shared-storage setup, compile-cache transfer, security notes, benchmarks, validation, and CI metadata. ModelExpress should link to that guide rather than duplicate its llm-d-specific manifests.

The guide was merged in [llm-d PR #1608](https://github.com/llm-d/llm-d/pull/1608). Its current default pins ModelExpress `v0.5.0`. This repository's current `main` and Helm chart are `0.5.1`, so treat the upstream pin as a tested integration snapshot, not as an implicit instruction to mix a `0.5.0` server/client with `0.5.1` CRDs.

## Version alignment

When following the upstream guide, use one of these approaches:

- Reproduce the guide exactly with its `MX_VERSION=v0.5.0` pin and matching server/client image.
- Update the guide's `MX_VERSION`, image, and CRD references together to `v0.5.1`, then rerun its validation steps. Do not update only the Python package or only the server image.

ModelExpress's `origin/main` does not currently contain an llm-d end-to-end example or CI job. The upstream llm-d guide is the integration-level validation; ModelExpress CI validates the underlying vLLM P2P path and Dynamo topologies, not the llm-d router/operator composition.

## Integration boundary

llm-d owns the router, Gateway API, model-server deployment, and llm-d-specific overlays. ModelExpress owns the worker load format, P2P metadata contract, NIXL transfer, and the ModelExpress server. Use [P2P weight transfer](../loading-paths/p2p.md) for MX settings and the upstream guide for llm-d commands and manifests.
