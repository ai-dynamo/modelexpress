# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG SERVER_IMAGE
FROM ${SERVER_IMAGE} AS server

# PrimeRL v0.10.1.dev140, merge 0790ebb5d856860d916b270bda4884184ce9d786.
FROM ghcr.io/primeintellect-ai/prime-rl@sha256:12c8550304a3daf12d10c68d6da7aadd7e4e1c573941a7ffa642555689e06220
USER root
RUN apt-get update && apt-get install -y --no-install-recommends redis-server \
    && rm -rf /var/lib/apt/lists/*
# Keep the qualified PrimeRL Torch/vLLM/NIXL dependency set; replace only the MX client.
RUN --mount=type=bind,source=modelexpress_client/python,target=/opt/mx-client,rw \
    uv pip install --python /app/.venv/bin/python --no-deps --reinstall /opt/mx-client
RUN uv pip install --python /app/.venv/bin/python 'pytest==8.4.1' 'toml==0.10.2'
COPY --from=server /app/modelexpress-server /app/third_party/modelexpress/bin/modelexpress-server
RUN ln -s /usr/bin/redis-server /app/third_party/modelexpress/bin/redis-server
COPY --from=prime-rl /tests /app/tests
COPY ci/rl/primerl/run.sh ci/rl/primerl/prepare.py /opt/mx-ci/
ENV UV_NO_SYNC=1 WANDB_MODE=offline PRIME_RUNS_MODE=disabled
WORKDIR /app
RUN python -c 'import modelexpress_rl; import prime_rl.transports.weights.modelexpress; import prime_rl.inference.vllm.worker.modelexpress'
RUN python -c 'from prime_rl.monitors.prime import prime_mode; assert prime_mode() == "disabled"'
ENTRYPOINT []
CMD ["bash", "/opt/mx-ci/run.sh"]
