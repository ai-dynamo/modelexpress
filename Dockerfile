# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM rust:1.88 AS builder

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
    cargo build --release --bin test_single_client && \
    cargo build --release --bin fallback_test

# Create a minimal runtime image
FROM nvcr.io/nvidia/base/ubuntu:noble-20250619 

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy all built binaries
COPY --from=builder /app/target/release/modelexpress-server .
COPY --from=builder /app/target/release/modelexpress-cli .
COPY --from=builder /app/target/release/test_client .
COPY --from=builder /app/target/release/test_single_client .
COPY --from=builder /app/target/release/fallback_test .

# Copy the Attribution files
COPY ATTRIBUTIONS_Rust.md .

# Expose the default port
EXPOSE 8001

# Set default environment variables (can be overridden)
ENV MODEL_EXPRESS_SERVER_PORT=8001
ENV MODEL_EXPRESS_LOGGING_LEVEL=info
ENV MODEL_EXPRESS_DATABASE_PATH=/app/models.db
ENV MODEL_EXPRESS_CACHE_DIRECTORY=/app/cache
ENV HF_HUB_CACHE=/app/cache

# Run the server by default
CMD ["./modelexpress-server"]
