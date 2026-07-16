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

# Generate protobuf files
for proto_name in p2p m2n_bootstrap; do
    echo "Generating protobuf files from ${PROTO_DIR}/${proto_name}.proto..."
    python -m grpc_tools.protoc \
        "-I${PROTO_DIR}" \
        "--python_out=${OUT_DIR}" \
        "--grpc_python_out=${OUT_DIR}" \
        "${PROTO_DIR}/${proto_name}.proto"

    # grpc_tools emits absolute sibling imports; package modules need relative imports.
    echo "Fixing imports in ${proto_name}_pb2_grpc.py..."
    tmp_file="$(mktemp)"
    sed "s/^import ${proto_name}_pb2 as/from . import ${proto_name}_pb2 as/" \
        "${OUT_DIR}/${proto_name}_pb2_grpc.py" > "${tmp_file}"
    mv "${tmp_file}" "${OUT_DIR}/${proto_name}_pb2_grpc.py"

    # Generated files are checked in, so add the repository SPDX header.
    for suffix in pb2.py pb2_grpc.py; do
        file="${OUT_DIR}/${proto_name}_${suffix}"
        echo "Adding SPDX header to ${file}..."
        tmp_file="$(mktemp)"
        echo "${SPDX_HEADER}" > "${tmp_file}"
        cat "${file}" >> "${tmp_file}"
        mv "${tmp_file}" "${file}"
    done
done

echo "Done."
