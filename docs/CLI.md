# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
cargo build --bin model-express-cli
```

The compiled binary will be available at `target/debug/model-express-cli` (or `target/release/model-express-cli` for release builds).

## Usage

### Global Options

```bash
model-express-cli [OPTIONS] <COMMAND>
```

**Options:**
- `-e, --endpoint <ENDPOINT>`: Server endpoint (default: http://localhost:8001)
- `-t, --timeout <TIMEOUT>`: Request timeout in seconds (default: 30)
- `-f, --format <FORMAT>`: Output format: `human`, `json`, `json-pretty` (default: human)
- `-v, -vv, -vvv`: Verbose mode (info, debug, trace)
- `-q, --quiet`: Quiet mode (suppress all output except errors)
- `--cache-path <PATH>`: Model storage path override
- `-h, --help`: Print help information
- `-V, --version`: Print version

**Environment Variables:**
- `MODEL_EXPRESS_ENDPOINT`: Set the default server endpoint
- `MODEL_EXPRESS_CACHE_PATH`: Set the default model storage path

### Commands

#### Health Check

Check server health and status:

```bash
# Basic health check
model-express-cli health

# JSON output
model-express-cli --format json health
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
model-express-cli model download google-t5/t5-small

# Use specific provider and strategy
model-express-cli model download google-t5/t5-small \
  --provider hugging-face \
  --strategy server-only

# Direct download (bypass server)
model-express-cli model download microsoft/DialoGPT-medium \
  --strategy direct

# Initialize model storage configuration
model-express-cli model init

# Initialize with custom settings
model-express-cli model init \
  --storage-path /path/to/your/models \
  --server-endpoint http://localhost:8001 \
  --auto-mount

# List downloaded models
model-express-cli model list

# Show detailed model information
model-express-cli model list --detailed

# Check model storage status
model-express-cli model status

# Clear specific model from storage
model-express-cli model clear google-t5/t5-small

# Clear all models from storage (with confirmation)
model-express-cli model clear-all

# Clear all models without confirmation
model-express-cli model clear-all --yes

# Validate model integrity
model-express-cli model validate

# Validate specific model
model-express-cli model validate google-t5/t5-small

# Show model storage statistics
model-express-cli model stats

# Show detailed storage statistics
model-express-cli model stats --detailed
```

**Download Strategies:**
- `smart-fallback`: Try server first, fallback to direct download (default)
- `server-fallback`: Use server with fallback to direct download
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
model-express-cli api send ping

# Custom action with JSON payload
model-express-cli api send my-action \
  --payload '{"key": "value", "number": 42}'

# Read payload from file
model-express-cli api send process-data \
  --payload-file data.json

# Read payload from stdin
echo '{"input": "data"}' | model-express-cli api send process \
  --payload -
```

### Output Formats

#### Human-readable (default)
Colorized, structured output optimized for terminal viewing.

#### JSON
Compact JSON output suitable for scripting:
```bash
model-express-cli --format json health
# Output: {"version":"0.1.0","status":"ok","uptime":120}
```

#### Pretty JSON
Formatted JSON with indentation:
```bash
model-express-cli --format json-pretty health
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
model-express-cli health

# Initialize model storage
model-express-cli model init

# Download a model with automatic storage
model-express-cli model download google-t5/t5-small

# Check model storage status
model-express-cli model status

# Test API connectivity
model-express-cli api send ping
```

### Advanced Usage

```bash
# Connect to remote server with custom timeout
model-express-cli --endpoint https://my-server.com:8001 \
  --timeout 60 \
  health

# Download model with verbose logging
model-express-cli -vv model download microsoft/DialoGPT-small \
  --strategy server-only

# Download model with custom storage path
model-express-cli --cache-path /custom/storage/path \
  model download google-t5/t5-small

# Send API request with file payload and JSON output
model-express-cli --format json api send process-batch \
  --payload-file batch-data.json

# Get model storage statistics in JSON format
model-express-cli --format json model stats --detailed
```

### Error Handling

The CLI provides clear error messages and appropriate exit codes:

- **Exit Code 0**: Success
- **Exit Code 1**: General error (network, server, validation)
- **Exit Code 2**: Invalid command line arguments

```bash
# Handle connection errors gracefully
if ! model-express-cli health >/dev/null 2>&1; then
    echo "Server is not available, trying direct download..."
    model-express-cli model download my-model --strategy direct
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
if model-express-cli --quiet health; then
    echo "Server is healthy, downloading via server..."
    model-express-cli model download "$MODEL_NAME" --strategy server-fallback
else
    echo "Server unavailable, downloading directly..."
    model-express-cli model download "$MODEL_NAME" --strategy direct
fi

# Check if model was stored successfully
if model-express-cli --format json model validate "$MODEL_NAME" | jq -r '.exists' | grep -q true; then
    echo "Model '$MODEL_NAME' is now available in storage"
else
    echo "Warning: Model may not be properly stored"
fi
```

### JSON Processing with jq

```bash
# Get server uptime in a script
UPTIME=$(model-express-cli --format json health | jq -r '.uptime')
echo "Server has been running for $UPTIME seconds"

# Check if server is healthy
STATUS=$(model-express-cli --format json health | jq -r '.status')
if [ "$STATUS" = "ok" ]; then
    echo "Server is healthy"
else
    echo "Server is not healthy: $STATUS"
    exit 1
fi

# Get model storage statistics
TOTAL_MODELS=$(model-express-cli --format json model stats | jq -r '.total_models')
TOTAL_SIZE=$(model-express-cli --format json model stats | jq -r '.total_size')
echo "Storage contains $TOTAL_MODELS models using $TOTAL_SIZE"

# Check if specific model is stored
MODEL_EXISTS=$(model-express-cli --format json model validate "google-t5/t5-small" | jq -r '.exists')
if [ "$MODEL_EXISTS" = "true" ]; then
    echo "Model is available in storage"
else
    echo "Model not found in storage"
fi

# List all stored model names
model-express-cli --format json model list | jq -r '.models[].name'
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
    ./target/release/model-express-cli health

    # Initialize model storage
    ./target/release/model-express-cli model init --auto-mount

    # Test model download
    ./target/release/model-express-cli model download google-t5/t5-small \
      --strategy server-only

    # Verify model was stored
    ./target/release/model-express-cli model validate google-t5/t5-small

    # Test API
    ./target/release/model-express-cli api send ping

    # Clean up storage
    ./target/release/model-express-cli model clear-all --yes
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
model-express-cli health
model-express-cli model status
```

### Configuration File Support

While the CLI doesn't currently support configuration files, you can create wrapper scripts:

```bash
#!/bin/bash
# modelexpress-prod
exec model-express-cli --endpoint "https://prod-server.com:8001" "$@"
```

## Troubleshooting

### Connection Issues

```bash
# Test with verbose output
model-express-cli -vv health

# Check network connectivity
curl -v http://localhost:8001/health 2>&1 | grep -i connect
```

### Server Not Running

```bash
# Try direct download instead
model-express-cli model download my-model --strategy direct

# Check model storage status without server connection
model-express-cli model status
```

### Storage Issues

```bash
# Validate model storage integrity
model-express-cli model validate

# Check storage statistics to find problematic models
model-express-cli model stats --detailed

# Clear corrupted model and re-download
model-express-cli model clear problematic-model
model-express-cli model download problematic-model
```

### Debug Mode

```bash
# Maximum verbosity for debugging
model-express-cli -vvv model download my-model
```

## Development

### Building from Source

```bash
# Debug build
cargo build --bin model-express-cli

# Release build
cargo build --release --bin model-express-cli

# Run tests
cargo test --bin model-express-cli
```

### Adding New Commands

The CLI is built using the `clap` crate with derive macros. To add new commands:

1. Add the command to the `Commands` enum in `src/bin/cli.rs`
2. Implement the handler function
3. Add the handler to the main match statement

## License

This CLI is part of the ModelExpress project and follows the same license terms.
