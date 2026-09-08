---
name: add-grpc-service
description: Add a new gRPC service to ModelExpress. Use whenever creating a .proto file, wiring a tonic service into the server, or extending modelexpress_common's generated grpc module.
---

# Adding gRPC Services

1. Define the service in a `.proto` file under `modelexpress_common/proto/`
2. Add the proto file to `modelexpress_common/build.rs` compile list
3. Add the generated module to `modelexpress_common/src/lib.rs` under `pub mod grpc`
4. Implement the service trait in `modelexpress_server/src/` (new file or existing)
5. Register the service in `modelexpress_server/src/main.rs` during server startup
