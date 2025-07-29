![ModelExpress Logo](ModelExpressTrainLogo.jpeg)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Dynamo Model Express

A Rust-based gRPC service for efficient model management and serving, with support for downloading and serving machine learning models from Hugging Face Hub.

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

- **`model-express-cli`**: A robust CLI tool for interacting with ModelExpress server
  - Health monitoring and server status checks
  - Model downloads with multiple strategies (server, direct, fallback)
  - API operations with JSON payload support
  - Multiple output formats (human-readable, JSON, pretty JSON)
  - Comprehensive error handling and logging

See [docs/CLI.md](docs/CLI.md) for detailed CLI documentation.

## Prerequisites

- **Rust**: Latest stable version (recommended: 1.70+)
- **Cargo**: Rust's package manager (included with Rust)
- **Docker** (optional): For containerized deployment

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ModelExpress
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

### Default Settings

- **gRPC Port**: 8001
- **Server Address**: `0.0.0.0:8001` (listens on all interfaces)
- **Client Endpoint**: `http://localhost:8001`

### Environment Variables

- `SERVER_PORT`: Override the default server port
- `LOG_LEVEL`: Set logging level (default: info)
- `RUST_LOG`: Configure tracing/logging output

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

# Client tests
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
├── tests/                    # Additional tests
├── benches/                  # Performance benchmarks
├── docker-compose.yml        # Docker configuration
├── Dockerfile               # Docker build file
├── k8s-deployment.yaml      # Kubernetes deployment
└── run_integration_tests.sh # Test runner script
```

### Adding New Features

1. **Server Features**: Add to `model_express_server/src/`
2. **Client Features**: Add to `model_express_client/src/`
3. **Shared Code**: Add to `model_express_common/src/`
4. **Tests**: Add to appropriate test directory

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

## Roadmap

- [ ] Enhanced model caching
- [ ] Model versioning support
- [ ] Performance optimizations
- [ ] Additional model format support
- [ ] Web UI for model management
