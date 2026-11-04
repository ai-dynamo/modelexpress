#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MANIFEST=$(mktemp)
APPLY_LOG=$(mktemp)
trap 'rm -f "$MANIFEST" "$APPLY_LOG"' EXIT
envsubst < "$1" > "$MANIFEST"

for attempt in 1 2 3 4 5 6; do
    if kubectl apply --request-timeout=30s -n "$2" -f "$MANIFEST" 2>&1 | tee "$APPLY_LOG"; then
        exit 0
    else
        status=$?
    fi
    if [ "$attempt" -eq 6 ] || ! grep -Eq \
        'failed calling webhook .*failed to call webhook:.*(connect: connection refused|no endpoints available|i/o timeout|TLS handshake timeout)' \
        "$APPLY_LOG"; then
        exit "$status"
    fi
    echo "::warning::Dynamo webhook unavailable on apply attempt ${attempt}/6; retrying in 10s"
    sleep 10
done
