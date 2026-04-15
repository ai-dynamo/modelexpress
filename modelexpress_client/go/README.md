<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress Go Bindings

This module contains generated Go protobuf and gRPC bindings for the ModelExpress services.

## Import

```go
import (
    p2ppb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/p2p"
)
```

The generated packages are:

| Package | Service |
|---------|---------|
| `gen/modelexpress/health` | `HealthService` |
| `gen/modelexpress/api` | `ApiService` |
| `gen/modelexpress/model` | `ModelService` |
| `gen/modelexpress/p2p` | `P2pService`, `WorkerService` |

## Regenerate

Install the pinned protoc plugins and run the generator:

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.11
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.6.0

cd modelexpress_client/go
./generate_proto.sh
go test ./...
```
