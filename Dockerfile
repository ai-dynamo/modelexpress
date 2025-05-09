FROM rust:1.76 as builder

WORKDIR /app

# Copy the Cargo.toml files first to cache dependencies
COPY Cargo.toml .
COPY model_express_common/Cargo.toml model_express_common/
COPY model_express_server/Cargo.toml model_express_server/

# Create dummy source files to build dependencies
RUN mkdir -p model_express_common/src model_express_server/src && \
    touch model_express_common/src/lib.rs && \
    touch model_express_server/src/main.rs && \
    cargo build --release -p model_express_server

# Copy the actual source code
COPY model_express_common/src model_express_common/src
COPY model_express_server/src model_express_server/src

# Build the application with the actual source code
RUN touch model_express_common/src/lib.rs model_express_server/src/main.rs && \
    cargo build --release -p model_express_server

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

# Set environment variables
ENV SERVER_PORT=8000
ENV LOG_LEVEL=info

# Run the server
CMD ["./model_express_server"]
