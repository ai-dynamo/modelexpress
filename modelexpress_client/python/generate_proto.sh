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

PROTO_NAMES=(p2p revision)
PROTO_FILES=()
for name in "${PROTO_NAMES[@]}"; do
    PROTO_FILES+=("${PROTO_DIR}/${name}.proto")
done

# Generate protobuf files
echo "Generating protobuf files from ${PROTO_NAMES[*]}..."
python -m grpc_tools.protoc \
    "-I${PROTO_DIR}" \
    "--python_out=${OUT_DIR}" \
    "--grpc_python_out=${OUT_DIR}" \
    "${PROTO_FILES[@]}"

for name in "${PROTO_NAMES[@]}"; do
    grpc_file="${OUT_DIR}/${name}_pb2_grpc.py"
    echo "Fixing imports in ${name}_pb2_grpc.py..."
    tmp_file="$(mktemp)"
    sed -E 's/^import ([a-zA-Z0-9_]+_pb2) as/from . import \1 as/' "${grpc_file}" > "${tmp_file}"
    mv "${tmp_file}" "${grpc_file}"
    sed -i "s/+ f' but the generated code/+ ' but the generated code/" "${grpc_file}"
    sed -i '/^import warnings$/d' "${grpc_file}"

    for file in "${OUT_DIR}/${name}_pb2.py" "${grpc_file}"; do
        echo "Adding SPDX header to ${file}..."
        tmp_file="$(mktemp)"
        printf '%s\n' "${SPDX_HEADER}" > "${tmp_file}"
        cat "${file}" >> "${tmp_file}"
        mv "${tmp_file}" "${file}"
    done
done

echo "Done."
