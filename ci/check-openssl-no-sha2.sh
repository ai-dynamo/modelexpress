#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The openssl build hashes through OpenSSL (modelexpress_common::digest), so the
# pure-Rust sha2 crate must not be a runtime dependency of that build. A direct
# `use sha2::...` in a crate built with --features openssl re-adds it; use
# modelexpress_common::digest::{sha256, sha256_hex} instead.
#
# Usage: ci/check-openssl-no-sha2.sh

set -euo pipefail

status=0
for pkg in modelexpress-common modelexpress-server; do
    # --target all: a platform-gated dependency still counts.
    rc=0
    out=$(cargo tree -p "${pkg}" --no-default-features --features openssl \
        --target all -e normal -i sha2 2>&1) || rc=$?
    if grep -q '^sha2 v' <<<"${out}"; then
        echo "FAIL: ${pkg} --features openssl depends on sha2:"
        sed 's/^/    /' <<<"${out}"
        status=1
    elif grep -qE 'nothing to print|did not match any packages' <<<"${out}"; then
        echo "ok: ${pkg} --features openssl has no sha2 dependency"
    else
        echo "ERROR: cargo tree failed for ${pkg} (exit ${rc}):"
        sed 's/^/    /' <<<"${out}"
        status=1
    fi
done
exit "${status}"
