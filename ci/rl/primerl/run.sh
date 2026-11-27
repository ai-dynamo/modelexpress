#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /app
export PYTEST_OUTPUT_DIR="/models/mx-primerl-ci/${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}/outputs"
export USERNAME_CI=CI_RUNNER
mkdir -p "$PYTEST_OUTPUT_DIR"
python /opt/mx-ci/prepare.py
python -m pytest tests/integration/test_reverse_text_modelexpress.py -v \
    --junitxml="$(dirname "$PYTEST_OUTPUT_DIR")/junit.xml"
