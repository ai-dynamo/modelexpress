# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AMD64 TRT-LLM image for Azure A100 CI P2P tests.
# Layers ModelExpress client + PRESHARDED patches on the public Dynamo
# TRT-LLM runtime image.
#
# Base image: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.1
#   - TRT-LLM 1.3.0rc13 (patches under trtllm_patches/v1.3.0rc5/ apply via
#     unchanged anchor context — the model_loader.py, linear.py, worker.py,
#     and llm_args.py code patterns we target are stable across rc5..rc13)
#   - AMD64 (x86_64)
#   - nixl 0.10.1 + nixl-cu12 0.10.1 pre-installed and pinned (matches
#     Dynamo's trtllm framework pin in container/context.yaml)
#   - UCX built from source against that NIXL version (no version skew)
#
# We use the Dynamo-curated runtime image rather than the upstream
# nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5 because every PyPI nixl wheel
# (1.1.0, 1.0.x, AND 0.10.1) is now built against UCX >= 1.21, while the
# upstream TRT-LLM image ships UCX 1.20.0. Loading the pip nixl on that
# image segfaults in ucp_init_version. The Dynamo runtime image builds NIXL
# and UCX from source together — same fix Dynamo applies to its own
# framework images — exposed on the public nvcr.io/nvidia/ai-dynamo/
# namespace alongside the vllm-runtime we already use for vLLM CI.
# See docs/NIXL_UCX_COMPATIBILITY.md for the full incident report.
#
# Build (from repo root):
#   docker buildx build --platform linux/amd64 \
#       -f ci/k8s/client/trt-llm/Dockerfile \
#       -t <IMAGE> --push .
FROM nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.1

# The Dynamo runtime image defaults to a non-root `dynamo` user. Build steps
# (pip install, file patches) need to write to root-owned directories under
# /opt/dynamo/venv and /usr/local, so switch to root for the build. The CMD
# inherits whatever user the base image's entrypoint switches to at runtime.
USER root

# Install ModelExpress Python client.
# Proto stubs (p2p_pb2.py, p2p_pb2_grpc.py) are checked-in source artifacts;
# regenerate with grpcio-tools when p2p.proto changes.
#
# MX does not pin a NIXL distribution; it imports `nixl._api` against
# whatever variant the base image provides. This image's pre-installed
# nixl-cu12 0.10.1 stays untouched and satisfies MX's runtime path,
# and the layer keeps working unchanged if Dynamo's tensorrtllm-runtime
# ever switches its bundled NIXL to a different version or to
# nixl-cu13. See docs/NIXL_UCX_COMPATIBILITY.md for the UCX ABI story.
COPY modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN pip uninstall -y modelexpress 2>/dev/null || true && \
    rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')"/modelexpress* && \
    pip install .

# Apply PRESHARDED + publish_from_worker patches to TRT-LLM.
# Patches: llm_args.py (PRESHARDED enum), model_loader.py (PRESHARDED branch),
#          linear.py (tp_size override), worker.py (publish_from_worker hook).
# Patches are named for 1.3.0rc5 but apply cleanly to 1.3.0rc13 too; if a
# future base image bump changes the anchor lines, apply_patches.py exits
# non-zero at build time with a clear "cannot find insertion point" error.
COPY trtllm_patches/v1.3.0rc5/apply_patches.py /tmp/apply_patches.py
RUN python3 /tmp/apply_patches.py && rm /tmp/apply_patches.py

# Verify key artifacts are present (GPU not available at build time, so we
# cannot test the full LLM import — but client imports do not need CUDA).
RUN python3 -c "from modelexpress.client import MxClient; print('MxClient OK')" && \
    python3 -c "from modelexpress.trtllm_live_transfer import MxLiveCheckpointLoader; print('MxLiveCheckpointLoader OK')"

# P2P launcher: selects source or target mode at container startup via MODEL_EXPRESS_SOURCE.
COPY ci/k8s/client/trt-llm/trtllm_p2p_launcher.py /opt/modelexpress/trtllm_p2p_launcher.py

WORKDIR /workspace
CMD ["python3", "/opt/modelexpress/trtllm_p2p_launcher.py"]
