# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AMD64 SGLang image for CI P2P tests.
# Layers ModelExpress client on the upstream lmsysorg/sglang image.
#
# Base image: lmsysorg/sglang:latest
#   - Public upstream SGLang from Docker Hub
#   - Pin the tag to a specific SGLang release once
#     https://github.com/sgl-project/sglang/pull/24723 lands in a tagged build,
#     so dev builds stay reproducible.
#
# REQUIRES UPSTREAM SGLANG PR #24723 — the delegation layer that wires the
# `remote_instance` load format to ModelExpress. Two specific flags must be
# present in `python3 -m sglang.launch_server --help`:
#
#   --remote-instance-weight-loader-backend  (choices include "modelexpress")
#   --modelexpress-config                    (JSON: url, transport)
#
# Note that vanilla `--load-format remote_instance` exists upstream already,
# but its default behavior is SGLang's own seed-instance P2P (using
# --remote-instance-weight-loader-seed-instance-ip / -service-port /
# -send-weights-group-ports), not ModelExpress. PR #24723 adds the backend
# selector that lets `remote_instance` hand off to ModelExpress. Without the
# PR, the manifest's args list will fail with "unrecognized arguments".
#
# Verify the base image has the PR with:
#   docker run --rm lmsysorg/sglang:latest python3 -m sglang.launch_server \
#       --help 2>&1 | grep -E "modelexpress|remote-instance-weight-loader-backend"
# Both lines must appear. As of writing (PR still open upstream), they don't.
#
# Workaround while waiting for upstream merge: build SGLang from the PR
# branch as a base image and `FROM` that here instead (see docs/SGLANG.md
# "build from local PR branch" recipe). The CI workflow's sglang matrix
# entry is intentionally commented out until this dependency clears.
#
# Build (from repo root):
#   docker buildx build --platform linux/amd64 \
#       -f ci/k8s/client/sgl/Dockerfile \
#       -t <IMAGE> --push --load .
FROM lmsysorg/sglang:latest

# Install ModelExpress Python client.
# Proto stubs (p2p_pb2.py, p2p_pb2_grpc.py) are checked-in source artifacts;
# regenerate with grpcio-tools when p2p.proto changes.
#
# Pip's default upgrade strategy (only-if-needed) keeps the base image's
# pre-installed nixl untouched as long as the constraint in modelexpress's
# pyproject.toml (`nixl[cu12]; sys_platform == 'linux'`) is unbounded.
# If lmsysorg/sglang ships a nixl that's incompatible with this image's UCX,
# see docs/NIXL_UCX_COMPATIBILITY.md for the resolution pattern.
COPY modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN pip uninstall -y modelexpress 2>/dev/null || true && \
    rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')"/modelexpress* && \
    pip install .

# Verify the client and the SGLang loader hook imported successfully.
RUN python3 -c "from modelexpress.client import MxClient; print('MxClient OK')" && \
    python3 -c "from modelexpress.engines.sglang import MxModelLoader; print('SGLang MxModelLoader OK')"

WORKDIR /workspace
