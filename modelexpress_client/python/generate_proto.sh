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
echo "Generating protobuf files from ${PROTO_DIR}/p2p.proto..."
python -m grpc_tools.protoc \
    "-I${PROTO_DIR}" \
    "--python_out=${OUT_DIR}" \
    "--grpc_python_out=${OUT_DIR}" \
    "${PROTO_DIR}/p2p.proto"

# Fix relative import in grpc file
echo "Fixing imports in p2p_pb2_grpc.py..."
sed -i 's/^import p2p_pb2 as/from . import p2p_pb2 as/' "${OUT_DIR}/p2p_pb2_grpc.py"

# Add SPDX header to generated files
for file in "${OUT_DIR}/p2p_pb2.py" "${OUT_DIR}/p2p_pb2_grpc.py"; do
    echo "Adding SPDX header to ${file}..."
    tmp_file=$(mktemp)
    echo "${SPDX_HEADER}" > "${tmp_file}"
    cat "${file}" >> "${tmp_file}"
    mv "${tmp_file}" "${file}"
done

echo "Done."
