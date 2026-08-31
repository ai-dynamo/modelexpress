<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Internals

Use these pages when you are implementing, debugging, or reviewing ModelExpress itself. They are not the recommended starting point for deploying a runtime.

- [Architecture](../ARCHITECTURE.md) covers the Rust workspace, Python client, gRPC services, metadata protocol, strategy implementation, NIXL, and transfer internals.
- [Metadata](../metadata.md) covers source identity, lifecycle, RPCs, Redis keys, and Kubernetes CRD shapes.
- [GCS provider](../GCS_PROVIDER.md) covers the standalone GCS model-cache provider implementation.
- [Kubernetes Service backend](../K8S_SERVICE_BACKEND.md) covers the decentralized discovery design and trade-offs.
- [Metrics](../METRICS.md) covers metric families and exposition behavior.
