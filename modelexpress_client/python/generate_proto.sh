#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/../../modelexpress_common/proto"
OUT_DIR="${SCRIPT_DIR}/modelexpress"

YEAR="$(date +%Y)"
SPDX_HEADER="# SPDX-FileCopyrightText: Copyright (c) 2025-${YEAR} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#"

PROTOS=(p2p model)

for proto in "${PROTOS[@]}"; do
    # Generate protobuf files
    echo "Generating protobuf files from ${PROTO_DIR}/${proto}.proto..."
    python -m grpc_tools.protoc \
        "-I${PROTO_DIR}" \
        "--python_out=${OUT_DIR}" \
        "--grpc_python_out=${OUT_DIR}" \
        "${PROTO_DIR}/${proto}.proto"

    # Fix relative import in grpc file
    echo "Fixing imports in ${proto}_pb2_grpc.py..."
    tmp_file="$(mktemp)"
    sed "s/^import ${proto}_pb2 as/from . import ${proto}_pb2 as/" \
        "${OUT_DIR}/${proto}_pb2_grpc.py" > "${tmp_file}"
    mv "${tmp_file}" "${OUT_DIR}/${proto}_pb2_grpc.py"

    # Add SPDX header to generated files
    for file in "${OUT_DIR}/${proto}_pb2.py" "${OUT_DIR}/${proto}_pb2_grpc.py"; do
        echo "Adding SPDX header to ${file}..."
        tmp_file=$(mktemp)
        echo "${SPDX_HEADER}" > "${tmp_file}"
        cat "${file}" >> "${tmp_file}"
        mv "${tmp_file}" "${file}"
    done
done

echo "Done."
