#!/bin/bash
set -e

# Print header
echo "======================================================"
echo "  Integration Test: Concurrent Model Download Test"
echo "  Using gRPC Protocol"
echo "======================================================"

# Start the server in the background
echo "Starting model_express_server (gRPC) in the background..."
cargo run --bin model_express_server > server.log 2>&1 &
SERVER_PID=$!

# Give the server time to start
echo "Waiting for gRPC server to start..."
sleep 3

# Trap to ensure server is killed on exit
function cleanup {
  echo "Cleaning up..."
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
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

# Run the client test
echo "Running concurrent model download test with gRPC..."
cargo run --bin test_client -- --test-model "google-t5/t5-small"

echo "gRPC Test completed successfully!"
