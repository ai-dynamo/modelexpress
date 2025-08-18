<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<img src="ModelExpressTrainLogo.jpeg" alt="ModelExpress Logo" width="50%">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Dynamo Model Express

A Rust-based gRPC service for efficient model management and serving, capable of downloading and serving machine learning models from multiple providers, including HuggingFace and other sources.

## Project Overview

ModelExpress is a high-performance model serving platform built with Rust and gRPC. It provides efficient model downloading, caching, and serving capabilities with a focus on performance and reliability.

## Architecture

The project is organized as a Rust workspace with the following components:

- **`model_express_server`**: The main gRPC server that provides model services
- **`model_express_client`**: Client library for interacting with the server
- **`model_express_common`**: Shared code and constants between client and server
- **`workspace-tests`**: Integration tests and test utilities

### CLI Tool

The client library includes a comprehensive command-line interface:

- **`model-express-cli`**: A robust CLI tool for interacting with modelexpress server
  - Health monitoring and server status checks
  - Model downloads with multiple strategies (server, direct, fallback)
  - API operations with JSON payload support
  - Multiple output formats (human-readable, JSON, pretty JSON)
  - Comprehensive error handling and logging

See [docs/CLI.md](docs/CLI.md) for detailed CLI documentation.

## Prerequisites

- **Rust**: Latest stable version (recommended: 1.88)
- **Cargo**: Rust's package manager (included with Rust)
- **Docker** (optional): For containerized deployment

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd modelexpress
```

### 2. Build the Project

```bash
cargo build
```

### 3. Run the Server

```bash
cargo run --bin model_express_server
```

The server will start on `0.0.0.0:8001` by default.

## Running Options

### Option 1: Local Development

```bash
# Start the gRPC server
cargo run --bin model_express_server

# In another terminal, run tests
cargo test

# Run integration tests
./run_integration_tests.sh
```

### Option 2: Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build and run manually
docker build -t model-express .
docker run -p 8000:8000 model-express
```

### Option 3: Kubernetes Deployment

```bash
kubectl apply -f k8s-deployment.yaml
```

## Configuration

ModelExpress uses a layered configuration system that supports multiple sources in order of precedence:

1. **Command line arguments** (highest priority)
2. **Environment variables**
3. **Configuration files** (YAML)
4. **Default values** (lowest priority)

### Configuration File

Create a configuration file (supports YAML):

```bash
# Generate a sample configuration file
cargo run --bin config_gen -- --output model-express.yaml

# Or use the provided sample
cp model-express.yaml my-config.yaml
```

Start the server with a configuration file:

```bash
cargo run --bin model_express_server -- --config my-config.yaml
```

### Environment Variables

You can use structured environment variables with the `MODEL_EXPRESS_` prefix:

```bash
# Server settings
export MODEL_EXPRESS_SERVER_HOST="127.0.0.1"
export MODEL_EXPRESS_SERVER_PORT=8080

# Database settings
export MODEL_EXPRESS_DATABASE_PATH="/path/to/models.db"

# Cache settings
export MODEL_EXPRESS_CACHE_DIRECTORY="/path/to/cache"
export MODEL_EXPRESS_CACHE_EVICTION_ENABLED=true

# Logging settings
export MODEL_EXPRESS_LOGGING_LEVEL=debug
export MODEL_EXPRESS_LOGGING_FORMAT=json
```

### Command Line Arguments

```bash
# Basic usage
cargo run --bin model_express_server -- --port 8080 --log-level debug

# With configuration file
cargo run --bin model_express_server -- --config my-config.yaml --port 8080

# Validate configuration
cargo run --bin model_express_server -- --config my-config.yaml --validate-config
```

### Configuration Options

#### Server Settings

- `host`: Server host address (default: "0.0.0.0")
- `port`: Server port (default: 8001)
- `graceful_shutdown`: Enable graceful shutdown (default: true)
- `shutdown_timeout_seconds`: Shutdown timeout (default: 30)

#### Database Settings

- `path`: SQLite database file path (default: "./models.db")
- `wal_mode`: Enable WAL mode (default: true)
- `pool_size`: Connection pool size (default: 10)
- `connection_timeout_seconds`: Connection timeout (default: 30)

#### Cache Settings

- `directory`: Cache directory path (default: "./cache")
- `max_size_bytes`: Maximum cache size in bytes (default: null/unlimited)
- `eviction.enabled`: Enable cache eviction (default: true)
- `eviction.check_interval_seconds`: Eviction check interval (default: 3600)
- `eviction.policy.unused_threshold_seconds`: Unused threshold (default: 604800/7 days)
- `eviction.policy.max_models`: Maximum models to keep (default: null/unlimited)
- `eviction.policy.min_free_space_bytes`: Minimum free space (default: null/unlimited)

#### Logging Settings

- `level`: Log level - trace, debug, info, warn, error (default: "info")
- `format`: Log format - json, pretty, compact (default: "pretty")
- `file`: Log file path (default: null/stdout)
- `structured`: Enable structured logging (default: false)

### Default Settings

- **gRPC Port**: 8001
- **Server Address**: `0.0.0.0:8001` (listens on all interfaces)
- **Client Endpoint**: `http://localhost:8001`

## API Services

The server provides the following gRPC services:

- **HealthService**: Health check endpoints
- **ApiService**: General API endpoints
- **ModelService**: Model management and serving

## Testing

### Run All Tests

```bash
cargo test
```

### Run Specific Tests

```bash
# Integration tests
cargo test --test integration_tests

# Client tests with specific model
cargo run --bin test_client -- --test-model "google-t5/t5-small"

# Fallback tests
cargo run --bin fallback_test
```

### Test Coverage

```bash
# Run tests with coverage (requires cargo-tarpaulin)
cargo tarpaulin --out Html
```

## Development

### Project Structure

```
ModelExpress/
├── model_express_server/     # Main gRPC server
├── model_express_client/     # Client library
├── model_express_common/     # Shared code
├── workspace-tests/          # Integration tests
├── docker-compose.yml        # Docker configuration
├── Dockerfile                # Docker build file
├── k8s-deployment.yaml       # Kubernetes deployment
└── run_integration_tests.sh  # Test runner script
```

### Adding New Features

1. **Server Features**: Add to `model_express_server/src/`
2. **Client Features**: Add to `model_express_client/src/`
3. **Shared Code**: Add to `model_express_common/src/`
4. **Tests**: Add to appropriate directory under `workspace-tests/`

### Dependencies

Key dependencies include:

- `tokio`: Async runtime
- `tonic`: gRPC framework
- `axum`: Web framework (if needed)
- `serde`: Serialization
- `hf-hub`: Hugging Face Hub integration
- `rusqlite`: SQLite database

### Pre-commit Hooks

This repository uses pre-commit hooks to maintain code quality. In order to contribute effectively, please set up the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Performance

The project includes benchmarking capabilities:

```bash
# Run benchmarks
cargo bench
```

## Monitoring and Logging

The server uses structured logging with `tracing`:

```bash
# Set log level
RUST_LOG=debug cargo run --bin model_express_server
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Support

For issues and questions:

- Create an issue in the repository
- Check the integration tests for usage examples
- Review the client library documentation


## Dynamo 0.4.1 Release

**Includes:**
- Model Express being released as a CLI tool.
- Model weight caching within Kubernetes clusters using PVC.
- Database tracking of which models are stored on which nodes.
- Basic model download and storage management.
- Documentation for Kubernetes deployment and CLI usage.

---

## Tentative Feature Roadmap:

**Planned Features:**
- Swap Dynamo’s model download client with ModelExpress for AWS S3 and other model storage support.
- **Performance Goal:** Reduce latency from seconds to microseconds
- Integration of **NIXL** through Run:ai Model Streamer to facilitate peer-to-peer transfer for model weights
- Bypass kernel cache RAM in ModelExpress Cluster to reduce Kernel I/O costs
- Highest-tier GPU-to-GPU weight transfer caching (potentially via a sidecar process)
- Pre-compiled model weight transfer
- Transfer of weights for LoRA / NeMo RL workloads
- Support for LoRA / NeMo RL and related checkpoint files
- Peer-to-peer communication support in the Model Express Client library 
- Peer-to-peer network status querying for optimal node startup decisions
- Enhanced fault tolerance, allowing clients to operate independently from the server
- Enhanced model caching
- Performance optimizations
- Model versioning support
- Additional model format support
- Web UI for model management
