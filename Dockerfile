FROM rust:1.84 AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire source code
COPY . .

# Build the application
RUN cargo build --release -p model_express_server

# Create a minimal runtime image
FROM debian:bullseye-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the built binary
COPY --from=builder /app/target/release/model_express_server .

# Expose the default port
EXPOSE 8000

# Set default environment variables (can be overridden)
ENV MODEL_EXPRESS_SERVER_PORT=8000
ENV MODEL_EXPRESS_LOGGING_LEVEL=info
ENV MODEL_EXPRESS_DATABASE_PATH=/app/models.db
ENV MODEL_EXPRESS_CACHE_DIRECTORY=/app/cache

# Run the server
CMD ["./model_express_server"]
