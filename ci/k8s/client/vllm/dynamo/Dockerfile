# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dynamo + ModelExpress combined worker image for the row-7 CI test.
#
# Distinct from ci/k8s/client/vllm/Dockerfile (which extends vllm/vllm-openai)
# because the dynamo aggregated-serving path uses `python3 -m dynamo.vllm`
# (the dynamo-shipped vLLM entrypoint, not bare vllm.entrypoints.openai).
# That entrypoint lives in the dynamo `vllm-runtime` base image.
#
# Built and pushed by the `build` job in .github/workflows/modelexpress-ci-tests.yml
# under tag <MX_IMAGE_REPO>/mx-worker-vllm-dynamo:<sha>.
FROM nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1

# Layer the ModelExpress Python client on top. Build context is the repo root
# (matches the existing build job's build context), so the COPY path is
# unambiguous.
COPY --chown=dynamo:dynamo modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN uv pip install .

WORKDIR /workspace
