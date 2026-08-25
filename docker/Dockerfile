# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM rust:1.90 AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire source code
COPY . .

# Build all available binaries
RUN cargo build --release --bin modelexpress-server && \
    cargo build --release --bin modelexpress-cli && \
    cargo build --release --bin test_client && \
    cargo build --release --bin fallback_test

# Create a minimal runtime image
FROM nvcr.io/nvidia/base/ubuntu:noble-20250619 AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libssl-dev gpgv && \
    rm -rf /var/lib/apt/lists/*

# Dedicated unprivileged account for the server. The Helm chart ships
# runAsNonRoot: true by default, and the kubelet refuses to start a container
# whose image would run as root -- so the image itself has to carry a non-root
# user, not just the chart. noble already ships a stock "ubuntu" account on
# 1000:1000; drop it so the server owns the conventional uid.
# userdel clears the matching primary group only while USERGROUPS_ENAB is yes,
# so free the gid explicitly rather than trusting that default -- and key on the
# gid, not the name, so a base image that renames the account is still caught.
# Keep 1000 in sync with podSecurityContext runAsUser/runAsGroup/fsGroup in
# helm/values.yaml.
RUN if getent passwd ubuntu >/dev/null; then userdel -r ubuntu; fi && \
    if getent group 1000 >/dev/null; then groupdel "$(getent group 1000 | cut -d: -f1)"; fi && \
    groupadd --gid 1000 modelexpress && \
    useradd --uid 1000 --gid 1000 --no-log-init --create-home \
            --home-dir /home/modelexpress --shell /usr/sbin/nologin modelexpress

# Copy all built binaries
COPY --from=builder /app/target/release/modelexpress-server .
COPY --from=builder /app/target/release/modelexpress-cli .
COPY --from=builder /app/target/release/test_client .
COPY --from=builder /app/target/release/fallback_test .

# Copy the NOTICES files
COPY NOTICES .

# The server writes downloaded models under MODEL_EXPRESS_CACHE_DIRECTORY, so the
# in-image default has to belong to the unprivileged account. Under the Helm
# chart this path is a mounted volume instead, and its ownership comes from
# podSecurityContext.fsGroup.
RUN mkdir -p /app/cache && chown modelexpress:modelexpress /app/cache

# Expose the default port
EXPOSE 8001

# Set default environment variables (can be overridden)
ENV HOME=/home/modelexpress
ENV MODEL_EXPRESS_SERVER_PORT=8001
ENV MODEL_EXPRESS_LOG_LEVEL=info
ENV MODEL_EXPRESS_CACHE_DIRECTORY=/app/cache
ENV HF_HUB_CACHE=/app/cache
# Distributed backend is required; operators must set MX_METADATA_BACKEND (drives both
# the P2P and registry backends) plus the relevant connection env vars (REDIS_URL for
# Redis, or POD_NAMESPACE for the Kubernetes backend).

# Numeric on purpose: the kubelet cannot resolve a user *name* against the image
# filesystem, so "USER modelexpress" would still fail the runAsNonRoot admission
# check with "container has runAsNonRoot and image has non-numeric user".
# Both listeners (8001 gRPC, 9401 metrics) are unprivileged, so no capability is
# needed to bind them.
USER 1000:1000

# Run the server by default
CMD ["./modelexpress-server"]
