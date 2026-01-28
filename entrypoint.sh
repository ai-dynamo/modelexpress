#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Entrypoint script that runs both the Model Streamer sidecar and the ModelExpress server.

set -e

# Configuration
SIDECAR_PORT="${MODEL_EXPRESS_SIDECAR_PORT:-8002}"
LOG_LEVEL="${MODEL_EXPRESS_LOG_LEVEL:-info}"

echo "Starting ModelExpress with Model Streamer sidecar..."

# Start the Python sidecar in the background
echo "Starting Model Streamer sidecar on port ${SIDECAR_PORT}..."
/app/venv/bin/python -m uvicorn \
    modelexpress_sidecar.main:app \
    --host 127.0.0.1 \
    --port "${SIDECAR_PORT}" \
    --log-level "${LOG_LEVEL}" &

SIDECAR_PID=$!

# Wait for sidecar to be ready
echo "Waiting for sidecar to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if /app/venv/bin/python -c "import httpx; httpx.get('http://127.0.0.1:${SIDECAR_PORT}/health', timeout=2)" > /dev/null 2>&1; then
        echo "Sidecar is ready!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 1
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Warning: Sidecar health check timed out, proceeding anyway..."
fi

# Trap to ensure cleanup on exit
cleanup() {
    echo "Shutting down..."
    if [ -n "$SIDECAR_PID" ]; then
        kill "$SIDECAR_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Start the main server (blocking)
echo "Starting ModelExpress server..."
exec /app/modelexpress-server "$@"
