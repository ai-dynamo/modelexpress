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

# Install runtime dependencies including Python for the sidecar
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gpgv \
        libssl-dev \
        python3 \
        python3-pip \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Create Python virtual environment and install sidecar package
COPY sidecar/pyproject.toml /app/sidecar/
COPY sidecar/src /app/sidecar/src
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir /app/sidecar

# Copy all built binaries
COPY --from=builder /app/target/release/modelexpress-server .
COPY --from=builder /app/target/release/modelexpress-cli .
COPY --from=builder /app/target/release/test_client .
COPY --from=builder /app/target/release/fallback_test .

# Copy the Attribution files
COPY ATTRIBUTIONS_Rust.md .

# Copy the entrypoint script
COPY entrypoint.sh .
RUN chmod +x /app/entrypoint.sh

# Expose the default ports (gRPC server and sidecar)
EXPOSE 8001
EXPOSE 8002

# Set default environment variables (can be overridden)
ENV MODEL_EXPRESS_SERVER_PORT=8001
ENV MODEL_EXPRESS_SIDECAR_PORT=8002
ENV MODEL_EXPRESS_SIDECAR_ENDPOINT=http://127.0.0.1:8002
ENV MODEL_EXPRESS_LOG_LEVEL=info
ENV MODEL_EXPRESS_DATABASE_PATH=/app/models.db
ENV MODEL_EXPRESS_CACHE_DIRECTORY=/app/cache
ENV HF_HUB_CACHE=/app/cache
ENV PYTHONPATH=/app/sidecar/src

# Run both services via entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
