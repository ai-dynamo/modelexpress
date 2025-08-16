# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
set -e

# Print header
echo "======================================================"
echo "  Integration Test: Concurrent Model Download Test"
echo "  Using gRPC Protocol"
echo "======================================================"

# Function to check if Rust toolchain is stable
check_rust_toolchain() {
    echo "Checking Rust toolchain status..."

    # Check if we have a stable toolchain
    if ! rustup toolchain list | grep -q "stable"; then
        echo "Installing stable Rust toolchain..."
        rustup toolchain install stable
    fi

    # Set stable as default if not already set
    if [[ "$(rustup default)" != *"stable"* ]]; then
        echo "Setting stable as default toolchain..."
        rustup default stable
    fi

    # Verify toolchain is working
    if ! cargo --version > /dev/null 2>&1; then
        echo "ERROR: Cargo is not working properly. Please check your Rust installation."
        exit 1
    fi

    echo "Rust toolchain is ready: $(cargo --version)"
}

# Determine the target directory (honor CARGO_TARGET_DIR if set)
TARGET_DIR="${CARGO_TARGET_DIR:-target}"

# Function to build the project safely
build_project() {
    echo "Building ModelExpress project..."

    # Clean any previous builds to avoid stale artifacts
    cargo clean

    # Build in release mode for better performance
    if ! cargo build --release --bin model_express_server; then
        echo "ERROR: Failed to build model_express_server"
        exit 1
    fi

    if ! cargo build --release --bin test_client; then
        echo "ERROR: Failed to build test_client"
        exit 1
    fi

    echo "Build completed successfully"
}

# Check and prepare Rust toolchain
check_rust_toolchain

# Build the project before starting tests
build_project

# Start the server in the background using the built binary
echo "Starting model_express_server (gRPC) in the background..."
"$TARGET_DIR/release/model_express_server" > server.log 2>&1 &
SERVER_PID=$!

# Give the server time to start
echo "Waiting for gRPC server to start..."
sleep 3

# Trap to ensure server is killed on exit
function cleanup {
  echo "Cleaning up..."
  if [[ -n "$SERVER_PID" ]]; then
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
  fi
  echo "Server stopped."
}
trap cleanup EXIT

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null; then
  echo "ERROR: Server failed to start. Check server.log for details."
  cat server.log
  exit 1
fi

echo "gRPC Server started successfully with PID $SERVER_PID"

# Run the client test using the built binary
echo "Running concurrent model download test with gRPC..."
"$TARGET_DIR/release/test_client" --test-model "google-t5/t5-small"

echo "gRPC Test completed successfully!"
