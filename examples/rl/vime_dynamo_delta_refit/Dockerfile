# syntax=docker/dockerfile:1.7

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM rust:1.96.1-bookworm AS rust-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang cmake libprotobuf-dev pkg-config protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*
COPY --from=dynamo . /src/dynamo
COPY . /src/modelexpress
WORKDIR /src/dynamo
RUN cargo build --release --locked -p dynamo-vllm-sidecar
WORKDIR /src/modelexpress
RUN cargo build --release --locked --bin modelexpress-server

FROM vllm/vllm-openai@sha256:3578c1fa6a9676e1de068b9d75c777cc865d251fadfbe6175ae82278739c6674 AS worker
RUN --mount=type=bind,source=modelexpress_client/python,target=/opt/modelexpress,rw \
    python3 -m pip install --no-cache-dir --no-compile --no-deps --no-build-isolation \
      "zstandard>=0.22.0" /opt/modelexpress
COPY --from=rust-builder /src/dynamo/target/release/dynamo-vllm-sidecar /usr/local/bin/
COPY --from=rust-builder /src/modelexpress/target/release/modelexpress-server /usr/local/bin/
RUN python3 -B -c \
    "import boto3, modelexpress_rl, vllm, zstandard; from modelexpress import register_modelexpress; register_modelexpress()"

FROM vllm/vime@sha256:267dab8accfcdbbb31aa09ce8b74a31e13a9aaec3d7968164b3dad7ae865dd2c AS trainer
RUN rm -rf /root/vime
ADD https://github.com/nv-hwoo/vime.git#e5c82fde22e7bebf6ed0c98d5b38ab6e5e7cf280 /root/vime
ENV PYTHONPATH=/root/vime:/root/Megatron-LM
RUN --mount=type=bind,source=modelexpress_client/python,target=/opt/modelexpress,rw \
    MX_SKIP_EXT=1 python3 -m pip install --no-cache-dir --no-compile --no-deps \
      --no-build-isolation /opt/modelexpress
RUN python3 -B -c \
    "import modelexpress_rl, torch, vime, vllm; from megatron.bridge import AutoBridge; from megatron.training.arguments import parse_args; from vllm.entrypoints.openai.cli_args import FrontendArgs; from vime.backends.megatron_utils.update_weight.update_weight_from_modelexpress import UpdateWeightFromModelExpress; from vime.backends.vllm_utils.external import discover_dynamo_engine; assert torch.__version__.startswith('2.11.') and vllm.__version__ == '0.25.1'"
COPY --chmod=0755 examples/rl/vime_dynamo_delta_refit/trainer.sh /opt/delta-refit/trainer.sh
COPY examples/rl/vime_dynamo_delta_refit/prompts.jsonl /opt/delta-refit/prompts.jsonl
CMD ["/opt/delta-refit/trainer.sh"]
