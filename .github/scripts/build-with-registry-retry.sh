#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

BUILD_LOG=$(mktemp)
trap 'rm -f "$BUILD_LOG"' EXIT

for attempt in 1 2 3; do
    if docker buildx build --progress=plain "$@" 2>&1 | tee "$BUILD_LOG"; then
        exit 0
    else
        status=$?
    fi
    if [ "$attempt" -eq 3 ] || ! grep -Eq \
        '^ERROR: failed to build: failed to solve: .*failed to (fetch (oauth|anonymous) token|resolve source metadata|do request).*(unexpected EOF|connection reset by peer|i/o timeout|TLS handshake timeout|502 Bad Gateway|503 Service Unavailable|504 Gateway Timeout)' \
        "$BUILD_LOG"; then
        exit "$status"
    fi
    echo "::warning::Registry transport failed on build attempt ${attempt}/3; retrying in 20s"
    sleep 20
done
