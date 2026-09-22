# syntax=docker/dockerfile:1.7
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG MILES_IMAGE=radixark/miles@sha256:2e155562fd57ead8bada9f37c0e4b009e283e4b0ffe74bbfa7ff8dcb2808ee1f
FROM ${MILES_IMAGE}

RUN rm -rf /root/miles

COPY --from=miles . /root/miles
COPY --from=modelexpress . /opt/modelexpress-client
COPY --from=megatron_bridge . /opt/megatron-bridge

RUN python3 -m pip install --no-cache-dir --no-deps -e /root/miles \
    && python3 -m pip install --no-cache-dir --no-deps /opt/modelexpress-client \
    && python3 -m pip install --no-cache-dir --no-deps --no-build-isolation /opt/megatron-bridge \
    && python3 -m pip install --no-cache-dir "nccl-extensions[cu13]" \
    && mkdir -p /opt/modelexpress/lib \
    && NCCL_LIBRARY="$(python3 -c 'import pathlib, site; matches = [path for root in site.getsitepackages() for path in pathlib.Path(root).glob("nvidia/nccl/lib/libnccl.so.2")]; assert matches, "nvidia-nccl-cu13 did not install libnccl.so.2"; print(matches[0])')" \
    && ln -sf "${NCCL_LIBRARY}" /opt/modelexpress/lib/libnccl.so.2

ENV LD_LIBRARY_PATH=/opt/modelexpress/lib:${LD_LIBRARY_PATH}
ENV LD_PRELOAD=/opt/modelexpress/lib/libnccl.so.2
ENV SGLANG_NCCL_SO_PATH=/opt/modelexpress/lib/libnccl.so.2
ENV SGLANG_PLUGINS=modelexpress_miles_collective
ENV NCCL_CUMEM_ENABLE=1
ENV PYTHONPATH=/opt/megatron-bridge/src:/root/miles:/root/Megatron-LM

COPY examples/rl/miles_m2n_collective/run_miles.py /opt/modelexpress-miles-m2n/run_miles.py
COPY examples/rl/miles_m2n_collective/verify_runtime.py /opt/modelexpress-miles-m2n/verify_runtime.py

RUN LD_PRELOAD=/opt/modelexpress/lib/libnccl.so.2 \
    python3 /opt/modelexpress-miles-m2n/verify_runtime.py

WORKDIR /root/miles
CMD ["python3", "/opt/modelexpress-miles-m2n/run_miles.py"]
