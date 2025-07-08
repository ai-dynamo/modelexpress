# ModelExpress Cache Management CLI

The ModelExpress Cache Management CLI provides a comprehensive interface for managing cached models with automatic path discovery and configuration management.

## Installation

The CLI is included with the ModelExpress client library:

```bash
# Build the CLI
cargo build --bin cache_cli

# Or install globally
cargo install --path model_express_client --bin cache_cli
```

## Quick Start

### 1. Initialize Cache Configuration

```bash
# Interactive setup
model-express init

# Or with specific options
model-express init --cache-path ~/.model-express/cache --server-endpoint localhost:8001
```

This will:
- Prompt for cache path (defaults to `~/.model-express/cache`)
- Prompt for server endpoint (defaults to `localhost:8001`)
- Save configuration to `~/.model-express/config.yaml`

### 2. List Cached Models

```bash
# Basic list
model-express list

# Detailed information
model-express list --detailed
```

### 3. Check Cache Status

```bash
model-express status
```

## Command Reference

### `init` - Initialize Cache Configuration

Initialize or update cache configuration with interactive prompts.

```bash
model-express init [OPTIONS]
```

**Options:**
- `--cache-path <PATH>` - Set cache path directly
- `--server-endpoint <ENDPOINT>` - Set server endpoint directly
- `--auto-mount` - Enable auto-mount on startup
- `--no-auto-mount` - Disable auto-mount on startup

**Examples:**
```bash
# Interactive setup
model-express init

# Quick setup with defaults
model-express init --cache-path ~/.model-express/cache

# Full configuration
model-express init \
  --cache-path /shared/models \
  --server-endpoint localhost:8001 \
  --auto-mount
```

### `list` - List Cached Models

Display all cached models with size information.

```bash
model-express list [OPTIONS]
```

**Options:**
- `--detailed` - Show detailed information including file paths

**Examples:**
```bash
# Basic list
model-express list

# Detailed view
model-express list --detailed
```

**Output:**
```
Cached Models
=============
Total models: 3
Total size: 2.45 GB

Models:
  google-t5/t5-small (850.00 MB)
  bert-base-uncased (1.20 GB)
  gpt2 (400.00 MB)
```

### `status` - Show Cache Status

Display comprehensive cache status including connectivity and health.

```bash
model-express status
```

**Output:**
```
Cache Status
============
Cache path: "/home/user/.model-express/cache"
Server endpoint: "http://localhost:8001"
Auto-mount: true
Total models: 3
Total size: 2.45 GB
Cache directory: ✅ Accessible
Server connection: ✅ Available
```

### `clear` - Clear Specific Model

Remove a specific model from the cache.

```bash
model-express clear <MODEL_NAME>
```

**Examples:**
```bash
# Clear specific model
model-express clear google-t5/t5-small

# Clear with confirmation
model-express clear bert-base-uncased
```

### `clear-all` - Clear Entire Cache

Remove all cached models.

```bash
model-express clear-all [OPTIONS]
```

**Options:**
- `--yes` - Skip confirmation prompt

**Examples:**
```bash
# With confirmation
model-express clear-all

# Skip confirmation
model-express clear-all --yes
```

### `preload` - Pre-download Model

Download a model to cache before it's needed.

```bash
model-express preload <MODEL_NAME> [OPTIONS]
```

**Options:**
- `--provider <PROVIDER>` - Model provider (default: huggingface)

**Examples:**
```bash
# Preload HuggingFace model
model-express preload google-t5/t5-small

# Specify provider
model-express preload bert-base-uncased --provider huggingface
```

### `validate` - Validate Cache Integrity

Check cache integrity and model completeness.

```bash
model-express validate [MODEL_NAME]
```

**Examples:**
```bash
# Validate entire cache
model-express validate

# Validate specific model
model-express validate google-t5/t5-small
```

**Output:**
```
Validating cache...
Found 3 models in cache
  google-t5/t5-small (850.00 MB)
  bert-base-uncased (1.20 GB)
  gpt2 (400.00 MB)
```

### `stats` - Show Cache Statistics

Display detailed cache statistics.

```bash
model-express stats [OPTIONS]
```

**Options:**
- `--detailed` - Show detailed statistics for each model

**Examples:**
```bash
# Basic statistics
model-express stats

# Detailed statistics
model-express stats --detailed
```

## Configuration Management

### Configuration File Location

The CLI uses a YAML configuration file located at:
- **Linux/macOS**: `~/.model-express/config.yaml`
- **Windows**: `%USERPROFILE%\.model-express\config.yaml`

### Configuration Format

```yaml
local_path: ~/.model-express/cache
server_endpoint: http://localhost:8001
auto_mount: true
timeout_secs: 30
```

### Environment Variables

You can override configuration using environment variables:

```bash
# Set cache path
export MODEL_EXPRESS_CACHE_PATH=/path/to/cache

# Set server endpoint
export MODEL_EXPRESS_SERVER_ENDPOINT=http://localhost:8001

# Run CLI (will use environment variables)
model-express list
```

### Command Line Overrides

You can override configuration for individual commands:

```bash
# Override cache path for this command
model-express list --cache-path /different/cache/path

# Override server endpoint for this command
model-express status --server-endpoint localhost:9001
```

## Path Discovery Strategy

The CLI uses a hybrid approach to discover cache paths:

### Priority Order

1. **Command line argument** (`--cache-path`)
2. **Environment variable** (`MODEL_EXPRESS_CACHE_PATH`)
3. **Configuration file** (`~/.model-express/config.yaml`)
4. **Auto-detection** (common paths)
5. **Server query** (if server is reachable)
6. **User prompt** (fallback)

### Auto-detection Paths

The CLI automatically checks these common paths:

- `~/.model-express/cache`
- `~/.cache/huggingface/hub`
- `/cache`
- `/app/models`
- `./cache`
- `./models`

## Integration with ModelExpress Server

### Shared Volume Mounting

When using the CLI with a ModelExpress server that has persistent volumes:

```bash
# Server deployment with persistent volume
kubectl apply -f k8s-deployment.yaml

# CLI can access the same cache
model-express init --cache-path /shared/models
model-express list
```

### Docker Integration

```bash
# Run CLI with shared volume
docker run -v model-express-cache:/cache model-express-client \
  model-express init --cache-path /cache

# Or using the same volume as server
docker run -v model-express-cache:/cache model-express-client \
  model-express list
```

## Use Cases

### Development Workflow

```bash
# 1. Initialize cache for development
model-express init --cache-path ~/dev/models

# 2. Preload commonly used models
model-express preload bert-base-uncased
model-express preload gpt2

# 3. Check what's available
model-express list

# 4. Validate cache integrity
model-express validate
```

### Production Deployment

```bash
# 1. Initialize with production paths
model-express init --cache-path /shared/models --server-endpoint model-express:8001

# 2. Preload production models
model-express preload production-model-1
model-express preload production-model-2

# 3. Monitor cache usage
model-express stats --detailed
```

### Cache Maintenance

```bash
# 1. Check cache health
model-express status

# 2. Validate all models
model-express validate

# 3. Clear old models
model-express clear old-model-name

# 4. Monitor space usage
model-express stats
```

## Troubleshooting

### Common Issues

#### Cache Path Not Found
```bash
# Error: Cache path not configured
model-express list

# Solution: Initialize cache
model-express init
```

#### Permission Denied
```bash
# Error: Permission denied accessing cache
# Solution: Check directory permissions
ls -la ~/.model-express/cache
chmod 755 ~/.model-express/cache
```

#### Server Connection Failed
```bash
# Error: Server connection unavailable
# Solution: Check server status
model-express status
# Verify server is running on the configured endpoint
```

#### Configuration File Corrupted
```bash
# Error: Failed to parse config file
# Solution: Remove and recreate configuration
rm ~/.model-express/config.yaml
model-express init
```

### Debug Mode

Enable debug logging:

```bash
# Set debug environment variable
export RUST_LOG=debug

# Run CLI with debug output
model-express list
```

### Configuration Validation

Validate your configuration:

```bash
# Check configuration file syntax
cat ~/.model-express/config.yaml

# Test configuration loading
model-express status
```

## Best Practices

### 1. Use Consistent Paths
- Use the same cache path across all environments
- Consider using absolute paths for production

### 2. Regular Maintenance
- Periodically validate cache integrity
- Monitor cache size and clean up unused models
- Keep configuration files backed up

### 3. Security Considerations
- Restrict access to cache directories
- Use appropriate file permissions
- Consider encryption for sensitive models

### 4. Performance Optimization
- Use SSD storage for better I/O performance
- Consider network storage for shared access
- Monitor cache hit rates

### 5. Integration with CI/CD
- Preload models in CI/CD pipelines
- Validate cache before deployments
- Use cache statistics in monitoring

## Examples

### Complete Workflow Example

```bash
# 1. Set up development environment
model-express init --cache-path ~/dev/models

# 2. Preload development models
model-express preload bert-base-uncased
model-express preload gpt2

# 3. Check cache status
model-express status

# 4. List available models
model-express list --detailed

# 5. Validate cache integrity
model-express validate

# 6. Monitor usage
model-express stats

# 7. Clean up when done
model-express clear-all --yes
```

### Production Setup Example

```bash
# 1. Initialize production cache
model-express init \
  --cache-path /shared/models \
  --server-endpoint model-express.prod:8001 \
  --auto-mount

# 2. Preload production models
model-express preload production-bert-model
model-express preload production-gpt-model

# 3. Validate production setup
model-express validate

# 4. Monitor production cache
model-express stats --detailed
```

This CLI provides a comprehensive solution for managing ModelExpress model caches with automatic path discovery, configuration management, and integration with the server's persistent storage system. 