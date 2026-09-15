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

FROM vllm/vllm-openai@sha256:3578c1fa6a9676e1de068b9d75c777cc865d251fadfbe6175ae82278739c6674
RUN --mount=type=bind,source=modelexpress_client/python,target=/opt/modelexpress,rw \
    python3 -m pip install --no-cache-dir --no-compile --no-deps \
      --no-build-isolation "zstandard>=0.22.0" /opt/modelexpress
COPY --from=rust-builder /src/dynamo/target/release/dynamo-vllm-sidecar /usr/local/bin/
COPY --from=rust-builder /src/modelexpress/target/release/modelexpress-server /usr/local/bin/
RUN python3 -B -c \
    "import boto3, modelexpress_rl, requests, transformers, vllm, zstandard; from modelexpress import register_modelexpress; register_modelexpress()"
