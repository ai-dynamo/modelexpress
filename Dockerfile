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
RUN cargo build --release --bin modelexpress_server && \
    cargo build --release --bin model-express-cli && \
    cargo build --release --bin test_client && \
    cargo build --release --bin test_single_client && \
    cargo build --release --bin fallback_test

# Create a minimal runtime image
FROM debian:bookworm-slim 

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy all built binaries
COPY --from=builder /app/target/release/modelexpress-server .
COPY --from=builder /app/target/release/model-express-cli .
COPY --from=builder /app/target/release/test_client .
COPY --from=builder /app/target/release/test_single_client .
COPY --from=builder /app/target/release/fallback_test .

# Expose the default port
EXPOSE 8000

# Set default environment variables (can be overridden)
ENV MODEL_EXPRESS_SERVER_PORT=8000
ENV MODEL_EXPRESS_LOGGING_LEVEL=info
ENV MODEL_EXPRESS_DATABASE_PATH=/app/models.db
ENV MODEL_EXPRESS_CACHE_DIRECTORY=/app/cache

# Run the server by default
CMD ["./model_express_server"]
