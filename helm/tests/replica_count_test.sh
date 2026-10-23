#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

chart_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

assert_replicas() {
    local expected="$1"
    shift
    local rendered actual
    rendered="$(helm template mx "$chart_dir" --show-only templates/deployment.yaml "$@")"
    actual="$(awk '/^  replicas:/ { print $2 }' <<< "$rendered")"
    if [[ "$actual" != "$expected" ]]; then
        printf 'Expected %s replicas, got %s\n' "$expected" "$actual" >&2
        exit 1
    fi
    printf 'spec.replicas=%s\n' "$actual"
}

assert_replicas 1
assert_replicas 0 --set replicaCount=0
assert_replicas 3 --set replicaCount=3
