<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress CLI

A comprehensive command-line interface for interacting with ModelExpress server, providing easy access to model downloads, cache management, health checks, and API operations.

## Features

- **Health Monitoring**: Check server status, version, and uptime
- **Model Management**: Download, list, clear, validate, and manage models with automatic storage
- **API Operations**: Send custom API requests with JSON payloads
- **Multiple Output Formats**: Human-readable, JSON, or pretty-printed JSON
- **Flexible Configuration**: Server endpoint, timeouts, and logging levels
- **Robust Error Handling**: Clear error messages and proper exit codes

## Installation

Build the CLI from the ModelExpress workspace:

```bash
cargo build --bin modelexpress-cli
```

The compiled binary will be available at `target/debug/modelexpress-cli` (or `target/release/modelexpress-cli` for release builds).

## Usage

### Global Options

```bash
modelexpress-cli [OPTIONS] <COMMAND>
```

> **Note:** Global options must be placed **before** the subcommand (e.g., `modelexpress-cli --no-shared-storage model download ...`).

**Options:**
- `-e, --endpoint <ENDPOINT>`: Server endpoint (default: http://localhost:8001)
- `-t, --timeout <TIMEOUT>`: Request timeout in seconds (default: 30)
- `-f, --format <FORMAT>`: Output format: `human`, `json`, `json-pretty` (default: human)
- `-v, -vv, -vvv`: Verbose mode (info, debug, trace)
- `-q, --quiet`: Quiet mode (suppress all output except errors)
- `--cache-path <PATH>`: Model storage path override
- `--no-shared-storage`: Disable shared storage mode (will transfer files from server to client)
- `--transfer-chunk-size <SIZE>`: Chunk size in bytes for file transfer when shared storage is disabled (default: 32768)
- `-h, --help`: Print help information
- `-V, --version`: Print version

**Environment Variables:**
- `MODEL_EXPRESS_ENDPOINT`: Set the default server endpoint
- `MODEL_EXPRESS_CACHE_DIRECTORY`: Set the default model storage path
- `MODEL_EXPRESS_NO_SHARED_STORAGE`: Disable shared storage mode (set to 'true' to enable file transfers)
- `MODEL_EXPRESS_TRANSFER_CHUNK_SIZE`: Set the chunk size in bytes for file transfers

### Commands

#### Health Check

Check server health and status:

```bash
# Basic health check
modelexpress-cli health

# JSON output
modelexpress-cli --format json health
```

**Example output:**
```
Server Health Status
  Status: ok
  Version: 0.1.0
  Uptime: 120 seconds
```

#### Model Operations

Download and manage models with various strategies:

```bash
# Download with smart fallback (tries server first, then direct)
modelexpress-cli model download google-t5/t5-small

# Use specific provider and strategy
modelexpress-cli model download google-t5/t5-small \
  --provider hugging-face \
  --strategy server-only

# Direct download (bypass server)
modelexpress-cli model download microsoft/DialoGPT-medium \
  --strategy direct

# Download with file transfer when no shared storage exists
# Note: Global options must come before the subcommand
modelexpress-cli --no-shared-storage --transfer-chunk-size 65536 \
  model download google-t5/t5-small

# Initialize model storage configuration
modelexpress-cli model init

# Initialize with custom settings
modelexpress-cli model init \
  --storage-path /path/to/your/models \
  --server-endpoint http://localhost:8001

# List downloaded models
modelexpress-cli model list

# Show detailed model information
modelexpress-cli model list --detailed

# Check model storage status
modelexpress-cli model status

# Clear specific model from storage
modelexpress-cli model clear google-t5/t5-small

# Clear all models from storage (with confirmation)
modelexpress-cli model clear-all

# Clear all models without confirmation
modelexpress-cli model clear-all --yes

# Validate model integrity
modelexpress-cli model validate

# Validate specific model
modelexpress-cli model validate google-t5/t5-small

# Show model storage statistics
modelexpress-cli model stats

# Show detailed storage statistics
modelexpress-cli model stats --detailed
```

**Download Strategies:**
- `smart-fallback`: Try server first, fallback to direct download (default)
- `server-only`: Use server only (no fallback)
- `direct`: Direct download only (bypass server)

**Providers:**
- `hugging-face`: Hugging Face model hub (default)

**Model Commands:**
- `download`: Download model with automatic storage (use `--strategy` and `--provider` for options)
- `init`: Initialize model storage configuration
- `list`: List downloaded models (use `--detailed` for more info)
- `status`: Show model storage status and usage
- `clear`: Clear specific model from storage
- `clear-all`: Clear all models from storage (use `--yes` to skip confirmation)
- `validate`: Validate model integrity
- `stats`: Show model storage statistics (use `--detailed` for more info)

#### API Operations

Send custom API requests:

```bash
# Simple ping
modelexpress-cli api send ping

# Custom action with JSON payload
modelexpress-cli api send my-action \
  --payload '{"key": "value", "number": 42}'

# Read payload from file
modelexpress-cli api send process-data \
  --payload-file data.json

# Read payload from stdin
echo '{"input": "data"}' | modelexpress-cli api send process \
  --payload -
```

### Output Formats

#### Human-readable (default)
Colorized, structured output optimized for terminal viewing.

#### JSON
Compact JSON output suitable for scripting:
```bash
modelexpress-cli --format json health
# Output: {"version":"0.1.0","status":"ok","uptime":120}
```

#### Pretty JSON
Formatted JSON with indentation:
```bash
modelexpress-cli --format json-pretty health
# Output:
# {
#   "version": "0.1.0",
#   "status": "ok",
#   "uptime": 120
# }
```

## Examples

### Basic Usage

```bash
# Check if server is running
modelexpress-cli health

# Initialize model storage
modelexpress-cli model init

# Download a model with automatic storage
modelexpress-cli model download google-t5/t5-small

# Check model storage status
modelexpress-cli model status

# Test API connectivity
modelexpress-cli api send ping
```

### Advanced Usage

```bash
# Connect to remote server with custom timeout
modelexpress-cli --endpoint https://my-server.com:8001 \
  --timeout 60 \
  health

# Download model with verbose logging
modelexpress-cli -vv model download microsoft/DialoGPT-small \
  --strategy server-only

# Download model with custom storage path
modelexpress-cli --cache-path /custom/storage/path \
  model download google-t5/t5-small

# Send API request with file payload and JSON output
modelexpress-cli --format json api send process-batch \
  --payload-file batch-data.json

# Get model storage statistics in JSON format
modelexpress-cli --format json model stats --detailed
```

### Error Handling

The CLI provides clear error messages and appropriate exit codes:

- **Exit Code 0**: Success
- **Exit Code 1**: General error (network, server, validation)
- **Exit Code 2**: Invalid command line arguments

```bash
# Handle connection errors gracefully
if ! modelexpress-cli health >/dev/null 2>&1; then
    echo "Server is not available, trying direct download..."
    modelexpress-cli model download my-model --strategy direct
fi
```

## Integration Examples

### Shell Scripts

```bash
#!/bin/bash
# check-and-download.sh

MODEL_NAME="$1"
if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model-name>"
    exit 1
fi

# Check server health first
if modelexpress-cli --quiet health; then
    echo "Server is healthy, downloading via server..."
    modelexpress-cli model download "$MODEL_NAME" --strategy smart-fallback
else
    echo "Server unavailable, downloading directly..."
    modelexpress-cli model download "$MODEL_NAME" --strategy direct
fi

# Check if model was stored successfully
if modelexpress-cli --format json model validate "$MODEL_NAME" | jq -r '.exists' | grep -q true; then
    echo "Model '$MODEL_NAME' is now available in storage"
else
    echo "Warning: Model may not be properly stored"
fi
```

### JSON Processing with jq

```bash
# Get server uptime in a script
UPTIME=$(modelexpress-cli --format json health | jq -r '.uptime')
echo "Server has been running for $UPTIME seconds"

# Check if server is healthy
STATUS=$(modelexpress-cli --format json health | jq -r '.status')
if [ "$STATUS" = "ok" ]; then
    echo "Server is healthy"
else
    echo "Server is not healthy: $STATUS"
    exit 1
fi

# Get model storage statistics
TOTAL_MODELS=$(modelexpress-cli --format json model stats | jq -r '.total_models')
TOTAL_SIZE=$(modelexpress-cli --format json model stats | jq -r '.total_size')
echo "Storage contains $TOTAL_MODELS models using $TOTAL_SIZE"

# Check if specific model is stored
MODEL_EXISTS=$(modelexpress-cli --format json model validate "google-t5/t5-small" | jq -r '.exists')
if [ "$MODEL_EXISTS" = "true" ]; then
    echo "Model is available in storage"
else
    echo "Model not found in storage"
fi

# List all stored model names
modelexpress-cli --format json model list | jq -r '.models[].name'
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
- name: Test ModelExpress server
  run: |
    # Start server in background
    ./target/release/model-express-server &
    sleep 5

    # Test health endpoint
    ./target/release/modelexpress-cli health

    # Initialize model storage
    ./target/release/modelexpress-cli model init

    # Test model download
    ./target/release/modelexpress-cli model download google-t5/t5-small \
      --strategy server-only

    # Verify model was stored
    ./target/release/modelexpress-cli model validate google-t5/t5-small

    # Test API
    ./target/release/modelexpress-cli api send ping

    # Clean up storage
    ./target/release/modelexpress-cli model clear-all --yes
```

## Configuration

### Environment Variables

Set default values using environment variables:

```bash
# Set default server endpoint
export MODEL_EXPRESS_ENDPOINT="https://my-server.com:8001"

# Set default cache path
export MODEL_EXPRESS_CACHE_PATH="/path/to/storage"

# Use the CLI without specifying endpoint or storage path
modelexpress-cli health
modelexpress-cli model status
```

### Configuration File Support

While the CLI doesn't currently support configuration files, you can create wrapper scripts:

```bash
#!/bin/bash
# modelexpress-prod
exec modelexpress-cli --endpoint "https://prod-server.com:8001" "$@"
```

## Troubleshooting

### Connection Issues

```bash
# Test with verbose output
modelexpress-cli -vv health

# Check network connectivity
curl -v http://localhost:8001/health 2>&1 | grep -i connect
```

### Server Not Running

```bash
# Try direct download instead
modelexpress-cli model download my-model --strategy direct

# Check model storage status without server connection
modelexpress-cli model status
```

### Storage Issues

```bash
# Validate model storage integrity
modelexpress-cli model validate

# Check storage statistics to find problematic models
modelexpress-cli model stats --detailed

# Clear corrupted model and re-download
modelexpress-cli model clear problematic-model
modelexpress-cli model download problematic-model
```

### Debug Mode

```bash
# Maximum verbosity for debugging
modelexpress-cli -vvv model download my-model
```

## Development

### Building from Source

```bash
# Debug build
cargo build --bin modelexpress-cli

# Release build
cargo build --release --bin modelexpress-cli

# Run tests
cargo test --bin modelexpress-cli
```

### Adding New Commands

The CLI is built using the `clap` crate with derive macros. To add new commands:

1. Add the command to the `Commands` enum in `src/bin/cli.rs`
2. Implement the handler function
3. Add the handler to the main match statement

## License

This CLI is part of the ModelExpress project and follows the same license terms.
