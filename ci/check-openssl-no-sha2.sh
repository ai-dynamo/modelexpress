#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The openssl build hashes with OpenSSL, so sha2 must not be a runtime dependency of it.
# A direct `use sha2::...` re-adds it; call modelexpress_common::digest::sha256_hex instead.

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
