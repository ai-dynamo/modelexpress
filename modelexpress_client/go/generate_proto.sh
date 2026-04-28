#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/../../modelexpress_common/proto"
GO_MODULE="github.com/ai-dynamo/modelexpress/modelexpress_client/go"
GEN_DIR="${SCRIPT_DIR}/gen"
PROTO_FILES=(
    "health.proto"
    "api.proto"
    "model.proto"
    "p2p.proto"
)
GO_PACKAGE_OPTS=(
    "--go_opt=Mhealth.proto=${GO_MODULE}/gen/modelexpress/health;healthpb"
    "--go_opt=Mapi.proto=${GO_MODULE}/gen/modelexpress/api;apipb"
    "--go_opt=Mmodel.proto=${GO_MODULE}/gen/modelexpress/model;modelpb"
    "--go_opt=Mp2p.proto=${GO_MODULE}/gen/modelexpress/p2p;p2ppb"
)
GO_GRPC_PACKAGE_OPTS=(
    "--go-grpc_opt=Mhealth.proto=${GO_MODULE}/gen/modelexpress/health;healthpb"
    "--go-grpc_opt=Mapi.proto=${GO_MODULE}/gen/modelexpress/api;apipb"
    "--go-grpc_opt=Mmodel.proto=${GO_MODULE}/gen/modelexpress/model;modelpb"
    "--go-grpc_opt=Mp2p.proto=${GO_MODULE}/gen/modelexpress/p2p;p2ppb"
)

for tool in protoc protoc-gen-go protoc-gen-go-grpc gofmt; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "error: ${tool} is required but was not found on PATH" >&2
        exit 1
    fi
done

mkdir -p "${GEN_DIR}"

echo "Generating Go protobuf and gRPC bindings from ${PROTO_DIR}..."
(
    cd "${PROTO_DIR}"
    protoc \
        -I. \
        --experimental_allow_proto3_optional \
        "--go_out=${SCRIPT_DIR}" \
        "--go_opt=module=${GO_MODULE}" \
        "${GO_PACKAGE_OPTS[@]}" \
        "--go-grpc_out=${SCRIPT_DIR}" \
        "--go-grpc_opt=module=${GO_MODULE}" \
        "${GO_GRPC_PACKAGE_OPTS[@]}" \
        "${PROTO_FILES[@]}"
)

mapfile -d '' generated_files < <(find "${GEN_DIR}" -type f \( -name "*.pb.go" -o -name "*_grpc.pb.go" \) -print0 | sort -z)

if [[ "${#generated_files[@]}" -eq 0 ]]; then
    echo "error: no Go protobuf files were generated" >&2
    exit 1
fi

echo "Normalizing protoc version comments..."
for file in "${generated_files[@]}"; do
    tmp_file="$(mktemp)"
    sed -E \
        -e 's#^//([[:space:]]*)protoc([[:space:]]*)v[0-9][^[:space:]]*#//\1protoc\2(version normalized)#' \
        -e 's#^// - protoc([[:space:]]*)v[0-9][^[:space:]]*#// - protoc\1(version normalized)#' \
        "${file}" > "${tmp_file}"
    mv "${tmp_file}" "${file}"
done

echo "Adding SPDX headers..."
for file in "${generated_files[@]}"; do
    first_line=""
    IFS= read -r first_line < "${file}" || true
    if [[ "${first_line}" == "// SPDX-FileCopyrightText:"* ]]; then
        continue
    fi

    tmp_file="$(mktemp)"
    {
        printf '%s\n' "// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
        printf '%s\n\n' "// SPDX-License-Identifier: Apache-2.0"
        cat "${file}"
    } > "${tmp_file}"
    mv "${tmp_file}" "${file}"
done

echo "Formatting generated Go files..."
gofmt -w "${generated_files[@]}"

echo "Done."
