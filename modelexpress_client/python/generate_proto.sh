#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/../../modelexpress_common/proto"
OUT_DIR="${SCRIPT_DIR}/modelexpress"
RL_OUT_DIR="${SCRIPT_DIR}/modelexpress_rl"

YEAR="$(date +%Y)"
SPDX_HEADER="# SPDX-FileCopyrightText: Copyright (c) 2025-${YEAR} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#"

# Generate protobuf files. Keep the inference and RL surfaces in separate
# Python modules even though they are built from the same proto directory.
for package_proto in "${OUT_DIR}:p2p" "${OUT_DIR}:model" "${RL_OUT_DIR}:refit"; do
    package_dir="${package_proto%%:*}"
    proto="${package_proto##*:}"
    echo "Generating protobuf files from ${PROTO_DIR}/${proto}.proto..."
    python -m grpc_tools.protoc \
        "-I${PROTO_DIR}" \
        "--python_out=${package_dir}" \
        "--grpc_python_out=${package_dir}" \
        "${PROTO_DIR}/${proto}.proto"

    # Fix relative imports in gRPC files.
    grpc_file="${package_dir}/${proto}_pb2_grpc.py"
    echo "Fixing imports in ${proto}_pb2_grpc.py..."
    tmp_file="$(mktemp)"
    sed \
        -e "s/^import ${proto}_pb2 as/from . import ${proto}_pb2 as/" \
        -e "s/^        + f' but the generated code/        + ' but the generated code/" \
        "${grpc_file}" > "${tmp_file}"
    mv "${tmp_file}" "${grpc_file}"

    # Add SPDX headers to generated files.
    for file in "${package_dir}/${proto}_pb2.py" "${package_dir}/${proto}_pb2_grpc.py"; do
        echo "Adding SPDX header to ${file}..."
        tmp_file=$(mktemp)
        echo "${SPDX_HEADER}" > "${tmp_file}"
        cat "${file}" >> "${tmp_file}"
        mv "${tmp_file}" "${file}"
    done
done

echo "Done."
