#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Interop test: Rust node puts a record, Python node gets it, and vice versa.
# Usage: ./run_interop_test.sh
#
# Prerequisites:
#   - Rust toolchain (cargo)
#   - Python 3.10+ with: pip install libp2p
#   - Or use: uv pip install libp2p

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUST_DIR="$SCRIPT_DIR/rust_node"
PYTHON_SCRIPT="$SCRIPT_DIR/python_node.py"
TEST_KEY="mx:model:test-model:worker:0"
TEST_VALUE='{"rank":0,"tensors":[{"name":"layer.0.weight","size":1024}]}'
TIMEOUT=30

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

cleanup() {
    local pids=("$@")
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}

wait_for_addr() {
    local file="$1"
    local timeout="$2"
    local elapsed=0
    while [ $elapsed -lt "$timeout" ]; do
        if grep -q "LISTEN_ADDR=" "$file" 2>/dev/null; then
            grep "LISTEN_ADDR=" "$file" | head -1 | cut -d= -f2-
            return 0
        fi
        sleep 0.5
        elapsed=$((elapsed + 1))
    done
    return 1
}

echo -e "${YELLOW}=== libp2p Kademlia DHT Interop Test ===${NC}"
echo ""

# Build the Rust node
echo -e "${YELLOW}[1/4] Building Rust node...${NC}"
(cd "$RUST_DIR" && cargo build --release 2>&1) || {
    echo -e "${RED}Rust build failed!${NC}"
    exit 1
}
RUST_BIN="$RUST_DIR/target/release/kad-interop-test"
echo -e "${GREEN}  Rust node built.${NC}"

# Check Python dependencies
echo -e "${YELLOW}[2/4] Checking Python dependencies...${NC}"
python3 -c "from libp2p import new_host; from libp2p.kad_dht.kad_dht import KadDHT" 2>/dev/null || {
    echo -e "${RED}  py-libp2p not installed. Install with: pip install libp2p${NC}"
    exit 1
}
echo -e "${GREEN}  py-libp2p available.${NC}"

# --- Test 1: Rust puts, Python gets ---
echo ""
echo -e "${YELLOW}[3/4] Test 1: Rust node puts record -> Python node gets record${NC}"

RUST_OUT=$(mktemp)
PIDS_TO_CLEAN=()

# Start Rust node in put mode
RUST_LOG=info "$RUST_BIN" --mode put --key "$TEST_KEY" --value "$TEST_VALUE" --timeout-secs "$TIMEOUT" > "$RUST_OUT" 2>&1 &
RUST_PID=$!
PIDS_TO_CLEAN+=("$RUST_PID")

# Wait for Rust node to print its listen address
RUST_ADDR=$(wait_for_addr "$RUST_OUT" 20) || {
    echo -e "${RED}  Rust node failed to start. Output:${NC}"
    cat "$RUST_OUT"
    cleanup "${PIDS_TO_CLEAN[@]}"
    exit 1
}
echo "  Rust node listening: $RUST_ADDR"

# Start Python node in get mode
PY_OUT=$(mktemp)
python3 "$PYTHON_SCRIPT" --mode get --key "$TEST_KEY" --peer "$RUST_ADDR" --timeout-secs "$TIMEOUT" > "$PY_OUT" 2>&1
PY_EXIT=$?

if [ $PY_EXIT -eq 0 ] && grep -q "RESULT=OK" "$PY_OUT"; then
    RETRIEVED=$(grep "RECORD_VALUE=" "$PY_OUT" | head -1 | cut -d= -f2-)
    echo -e "${GREEN}  PASSED: Python retrieved value from Rust node${NC}"
    echo "  Value: $RETRIEVED"
else
    echo -e "${RED}  FAILED: Python could not retrieve value from Rust node${NC}"
    echo "  Python output:"
    cat "$PY_OUT"
    echo "  Rust output:"
    cat "$RUST_OUT"
fi

cleanup "${PIDS_TO_CLEAN[@]}"
rm -f "$RUST_OUT" "$PY_OUT"
PIDS_TO_CLEAN=()

# --- Test 2: Python puts, Rust gets ---
echo ""
echo -e "${YELLOW}[4/4] Test 2: Python node puts record -> Rust node gets record${NC}"

PY_OUT=$(mktemp)

# Start Python node in put mode
python3 "$PYTHON_SCRIPT" --mode put --key "$TEST_KEY" --value "$TEST_VALUE" --timeout-secs "$TIMEOUT" > "$PY_OUT" 2>&1 &
PY_PID=$!
PIDS_TO_CLEAN+=("$PY_PID")

# Wait for Python node to print its listen address
PY_ADDR=$(wait_for_addr "$PY_OUT" 20) || {
    echo -e "${RED}  Python node failed to start. Output:${NC}"
    cat "$PY_OUT"
    cleanup "${PIDS_TO_CLEAN[@]}"
    exit 1
}
echo "  Python node listening: $PY_ADDR"

# Start Rust node in get mode
RUST_OUT=$(mktemp)
RUST_LOG=info "$RUST_BIN" --mode get --key "$TEST_KEY" --peer "$PY_ADDR" --timeout-secs "$TIMEOUT" > "$RUST_OUT" 2>&1
RUST_EXIT=$?

if [ $RUST_EXIT -eq 0 ] && grep -q "RESULT=OK" "$RUST_OUT"; then
    RETRIEVED=$(grep "RECORD_VALUE=" "$RUST_OUT" | head -1 | cut -d= -f2-)
    echo -e "${GREEN}  PASSED: Rust retrieved value from Python node${NC}"
    echo "  Value: $RETRIEVED"
else
    echo -e "${RED}  FAILED: Rust could not retrieve value from Python node${NC}"
    echo "  Rust output:"
    cat "$RUST_OUT"
    echo "  Python output:"
    cat "$PY_OUT"
fi

cleanup "${PIDS_TO_CLEAN[@]}"
rm -f "$RUST_OUT" "$PY_OUT"

echo ""
echo -e "${YELLOW}=== Done ===${NC}"
