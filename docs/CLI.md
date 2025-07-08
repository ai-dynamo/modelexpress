# ModelExpress Cache CLI Documentation

The ModelExpress Cache CLI provides a powerful command-line interface for managing model downloads, cache operations, and model lifecycle management. This document covers how to use the CLI to download models and manage your local model cache.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Commands Reference](#commands-reference)
- [Downloading Models](#downloading-models)
- [Cache Management](#cache-management)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

The cache CLI is included with the ModelExpress client. Build it from source:

```bash
# Build the entire workspace
cargo build

# The cache CLI binary will be available at:
./target/debug/cache_cli
```

## Quick Start

### 1. Initialize Cache Configuration

First, set up your cache configuration:

```bash
# Initialize with default settings
./target/debug/cache_cli init

# Or specify custom settings
./target/debug/cache_cli init \
  --cache-path /path/to/your/cache \
  --server-endpoint http://localhost:8001 \
  --auto-mount true
```

### 2. Download Your First Model

```bash
# Download a model to cache
./target/debug/cache_cli preload "sentence-transformers/all-MiniLM-L6-v2"

# Download with specific provider
./target/debug/cache_cli preload "google/gemma-2b" --provider huggingface
```

### 3. Check Cache Status

```bash
# View cache status
./target/debug/cache_cli status

# List cached models
./target/debug/cache_cli list

# View detailed statistics
./target/debug/cache_cli stats
```

## Configuration

The CLI uses a hybrid configuration system that discovers settings from multiple sources in order of priority:

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Configuration file** (`~/.config/model_express/cache.toml`)
4. **Auto-detection** (default paths)
5. **Server query** (if server is available)
6. **User prompt** (interactive mode)

### Environment Variables

```bash
export MODEL_EXPRESS_CACHE_PATH="/path/to/cache"
export MODEL_EXPRESS_SERVER_ENDPOINT="http://localhost:8001"
export MODEL_EXPRESS_AUTO_MOUNT="true"
```

### Configuration File

The CLI automatically creates and manages a configuration file at `~/.config/model_express/cache.toml`:

```toml
local_path = "/home/user/.cache/model_express"
server_endpoint = "http://localhost:8001"
auto_mount = true
timeout_secs = 30
```

## Commands Reference

### Global Options

All commands support these global options:

```bash
--cache-path <PATH>           # Override cache path
--server-endpoint <ENDPOINT>  # Override server endpoint
-h, --help                    # Show help
-V, --version                 # Show version
```

### Available Commands

#### `init` - Initialize Cache Configuration

Sets up cache configuration interactively or with provided options.

```bash
# Interactive initialization
./target/debug/cache_cli init

# Non-interactive with options
./target/debug/cache_cli init \
  --cache-path /opt/models \
  --server-endpoint http://model-server:8001 \
  --auto-mount true
```

#### `preload` - Download Model to Cache

Downloads a model to the local cache for faster subsequent access.

```bash
# Basic model download
./target/debug/cache_cli preload "bert-base-uncased"

# Download with specific provider
./target/debug/cache_cli preload "google/gemma-2b" --provider huggingface

# Download with custom cache path
./target/debug/cache_cli --cache-path /custom/cache preload "roberta-base"
```

#### `list` - List Cached Models

Shows all models currently cached locally.

```bash
# List all cached models
./target/debug/cache_cli list

# List with custom cache path
./target/debug/cache_cli --cache-path /custom/cache list
```

#### `status` - Show Cache Status

Displays cache health, usage statistics, and configuration.

```bash
# Show cache status
./target/debug/cache_cli status

# Show status for specific cache
./target/debug/cache_cli --cache-path /custom/cache status
```

#### `stats` - Show Detailed Statistics

Provides detailed cache statistics including size, model count, and individual model information.

```bash
# Show detailed statistics
./target/debug/cache_cli stats

# Show stats for specific cache
./target/debug/cache_cli --cache-path /custom/cache stats
```

#### `validate` - Validate Cache Integrity

Checks cache integrity and validates cached models.

```bash
# Validate entire cache
./target/debug/cache_cli validate

# Validate specific cache
./target/debug/cache_cli --cache-path /custom/cache validate
```

#### `clear` - Clear Specific Model

Removes a specific model from the cache.

```bash
# Clear specific model
./target/debug/cache_cli clear "bert-base-uncased"

# Clear with custom cache path
./target/debug/cache_cli --cache-path /custom/cache clear "google/gemma-2b"
```

#### `clear-all` - Clear Entire Cache

Removes all cached models.

```bash
# Clear entire cache
./target/debug/cache_cli clear-all

# Clear with custom cache path
./target/debug/cache_cli --cache-path /custom/cache clear-all
```

## Downloading Models

### Understanding Server vs Client Cache

ModelExpress uses a **two-tier caching system**:

1. **Server Cache**: Models are downloaded and cached on the server in `~/.cache/huggingface/hub/`
2. **Client Cache**: Optional local cache for faster subsequent access

When you request a model, the system intelligently handles two scenarios:

### Scenario 1: Server Downloads New Model

When a model is **not cached** on the server:

```bash
# Request a model that's not on the server
./target/debug/cache_cli preload "google/gemma-7b"

# What happens:
# 1. Client connects to server via gRPC
# 2. Server checks ~/.cache/huggingface/hub/ - model not found
# 3. Server starts downloading from Hugging Face Hub
# 4. Server streams progress updates to client:
#    INFO: Model google/gemma-7b: Downloading model files...
#    INFO: Model google/gemma-7b: Downloaded config.json
#    INFO: Model google/gemma-7b: Downloaded pytorch_model-00001-of-00002.safetensors
#    INFO: Model google/gemma-7b: Downloaded pytorch_model-00002-of-00002.safetensors
#    INFO: Model google/gemma-7b: Downloaded tokenizer.json
#    INFO: Model google/gemma-7b: Model download completed successfully
# 5. Server saves model to ~/.cache/huggingface/hub/
# 6. Client receives confirmation
```

**Server-side process:**
```rust
// Server checks cache
if let Some(status) = MODEL_TRACKER.get_status("google/gemma-7b") {
    // Model not found, status is None
}

// Server downloads to HF cache
download::download_model("google/gemma-7b", ModelProvider::HuggingFace).await
// Downloads to: ~/.cache/huggingface/hub/models--google--gemma-7b--main/

// Server updates database
MODEL_TRACKER.set_status("google/gemma-7b", ModelStatus::DOWNLOADED)
```

### Scenario 2: Server Streams Existing Cached Model

When a model is **already cached** on the server:

```bash
# Request a model that's already on the server
./target/debug/cache_cli preload "google/gemma-7b"

# What happens:
# 1. Client connects to server via gRPC
# 2. Server checks ~/.cache/huggingface/hub/ - model found!
# 3. Server immediately returns cached status:
#    INFO: Model google/gemma-7b: Model already downloaded
# 4. No download occurs - instant response
# 5. Client receives immediate confirmation
```

**Server-side process:**
```rust
// Server checks cache
if let Some(status) = MODEL_TRACKER.get_status("google/gemma-7b") {
    // Model found, status is DOWNLOADED
    let update = ModelStatusUpdate {
        status: ModelStatus::DOWNLOADED,
        message: Some("Model already downloaded".to_string()),
        // ...
    };
    // Immediate response, no download needed
}
```

### Real-World Examples

#### Example 1: First-Time Model Download

```bash
# First time requesting a model
./target/debug/cache_cli preload "google/gemma-7b"

# Expected output:
# INFO: Pre-loading model: google/gemma-7b (provider: HuggingFace)
# INFO: Model google/gemma-7b: Starting download from Hugging Face Hub...
# INFO: Model google/gemma-7b: Downloading config.json (1.8 KB)
# INFO: Model google/gemma-7b: Downloading pytorch_model-00001-of-00002.safetensors (13.1 GB)
# INFO: Model google/gemma-7b: Downloading pytorch_model-00002-of-00002.safetensors (13.1 GB)
# INFO: Model google/gemma-7b: Downloading tokenizer.json (2.1 MB)
# INFO: Model google/gemma-7b: Model download completed successfully
# INFO: Model pre-loaded successfully!
```

#### Example 2: Subsequent Model Request

```bash
# Request the same model again
./target/debug/cache_cli preload "google/gemma-7b"

# Expected output:
# INFO: Pre-loading model: google/gemma-7b (provider: HuggingFace)
# INFO: Model google/gemma-7b: Model already downloaded
# INFO: Model pre-loaded successfully!
```

#### Example 3: Mixed Cache Status

```bash
# Check what's cached
./target/debug/cache_cli list

# Output:
# INFO: Cached Models
# INFO: =============
# INFO: Total models: 2
# INFO: Total size: 26.3 GB
# INFO: Models:
# INFO:   google/gemma-7b (26.2 GB)
# INFO:   google/gemma-2b (5.1 GB)

# Request cached model (instant)
./target/debug/cache_cli preload "google/gemma-7b"
# INFO: Model google/gemma-7b: Model already downloaded

# Request new model (downloads)
./target/debug/cache_cli preload "google/gemma-2b"
# INFO: Model google/gemma-2b: Starting download from Hugging Face Hub...
# INFO: Model google/gemma-2b: Model download completed successfully
```

### Monitoring Download Progress

The CLI provides real-time feedback during downloads:

```bash
# Watch download progress
./target/debug/cache_cli preload "google/gemma-7b"

# Real-time output:
# INFO: Pre-loading model: google/gemma-7b (provider: HuggingFace)
# INFO: Model google/gemma-7b: Starting download from Hugging Face Hub...
# INFO: Model google/gemma-7b: Downloading config.json (1.8 KB)
# INFO: Model google/gemma-7b: Downloading pytorch_model-00001-of-00002.safetensors (13.1 GB)
# INFO: Model google/gemma-7b: Downloading pytorch_model-00002-of-00002.safetensors (13.1 GB)
# INFO: Model google/gemma-7b: Downloading tokenizer.json (2.1 MB)
# INFO: Model google/gemma-7b: Model download completed successfully
# INFO: Model pre-loaded successfully!
```

### Cache Status Verification

Verify what's cached on the server:

```bash
# Check server cache status
./target/debug/cache_cli status

# Output shows:
# INFO: Cache Status
# INFO: ============
# INFO: Cache path: /home/user/.cache/model_express
# INFO: Server endpoint: http://localhost:8001
# INFO: Server connection: ✅ Available
# INFO: Total models: 3
# INFO: Total size: 31.4 GB

# List specific cached models
./target/debug/cache_cli list --detailed

# Output:
# INFO: Cached Models
# INFO: =============
# INFO: Total models: 3
# INFO: Total size: 31.4 GB
# INFO: Models:
# INFO:   google/gemma-7b (26.2 GB) - /home/user/.cache/model_express/google/gemma-7b
# INFO:   google/gemma-2b (5.1 GB) - /home/user/.cache/model_express/google/gemma-2b
# INFO:   microsoft/DialoGPT-medium (1.5 GB) - /home/user/.cache/model_express/microsoft/DialoGPT-medium
```

### Performance Comparison

| Scenario | Response Time | Network Usage | Server Load |
|----------|---------------|---------------|-------------|
| **New Model Download** | 5-15min | High (download) | High (processing) |
| **Cached Model Access** | <1s | Low (gRPC only) | Low (status check) |

### Best Practices

1. **Preload Frequently Used Models**: Download large models like Gemma-7B during off-peak hours
2. **Monitor Cache Status**: Regularly check what's cached, especially for large models
3. **Use Appropriate Models**: Choose smaller models like Gemma-2B for development
4. **Validate Downloads**: Verify model integrity after download, especially for multi-part models
5. **Consider Storage**: Large models like Gemma-7B require significant disk space (26+ GB)

### Supported Model Providers

Currently supported providers:
- `huggingface` (default) - Hugging Face Hub models
- Additional providers can be added in future versions

### Model Naming

Models can be specified using:
- **Full model names**: `"google/gemma-7b"`
- **Short names**: `"gemma-7b"` (resolves to `"google/gemma-7b"`)
- **Versioned models**: `"google/gemma-7b@v1.0"`

### Error Handling

The CLI handles various error scenarios:

```bash
# Network timeout for large model
./target/debug/cache_cli preload "invalid-model-name"
# ERROR: Model download failed: Failed to fetch model from HuggingFace

# Server unavailable
./target/debug/cache_cli preload "google/gemma-7b" --server-endpoint http://invalid:8001
# WARN: Server connection: ❌ Unavailable
# INFO: Server unavailable, pre-loading model google/gemma-7b directly

# Insufficient disk space for large model
./target/debug/cache_cli preload "google/gemma-7b"
# ERROR: No space left on device
# Solution: Ensure you have at least 30GB free space for Gemma-7B
```

## Cache Management

### Understanding Cache Structure

The cache is organized as follows:

```
~/.cache/model_express/
├── models/
│   ├── bert-base-uncased/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer.json
│   ├── google/gemma-2b/
│   │   ├── config.json
│   │   ├── pytorch_model-00001-of-00002.safetensors
│   │   ├── pytorch_model-00002-of-00002.safetensors
│   │   └── tokenizer.json
│   └── ...
└── cache.db
```

### Cache Statistics

View detailed cache information:

```bash
# Show cache statistics
./target/debug/cache_cli stats

# Example output:
# Total models: 5
# Total size: 2.34 GB
# Models:
#   - bert-base-uncased: 420.5 MB
#   - google/gemma-2b: 5.1 GB
#   - roberta-base: 1.37 GB
```

### Cache Validation

Validate cache integrity:

```bash
# Validate all cached models
./target/debug/cache_cli validate

# Check for corrupted files
./target/debug/cache_cli validate --check-integrity
```

## Examples

### Example 1: Setting Up Cache for Development

```bash
# 1. Initialize cache
./target/debug/cache_cli init --cache-path ~/dev_models

# 2. Download common models
./target/debug/cache_cli preload "bert-base-uncased"
./target/debug/cache_cli preload "google/gemma-2b"
./target/debug/cache_cli preload "sentence-transformers/all-MiniLM-L6-v2"

# 3. Check status
./target/debug/cache_cli status
```

### Example 2: Production Cache Setup

```bash
# 1. Initialize with production settings
./target/debug/cache_cli init \
  --cache-path /opt/model_cache \
  --server-endpoint http://model-server:8001 \
  --auto-mount true

# 2. Preload production models
./target/debug/cache_cli preload "microsoft/DialoGPT-medium"
./target/debug/cache_cli preload "facebook/bart-large-cnn"

# 3. Validate cache
./target/debug/cache_cli validate

# 4. Monitor usage
./target/debug/cache_cli stats
```

### Example 3: Cache Maintenance

```bash
# 1. Check cache usage
./target/debug/cache_cli stats

# 2. Remove unused models
./target/debug/cache_cli clear "old-model-name"

# 3. Clear entire cache if needed
./target/debug/cache_cli clear-all

# 4. Re-download essential models
./target/debug/cache_cli preload "essential-model"
```

### Example 4: Integration with ModelExpress Server

```bash
# 1. Configure for server integration
./target/debug/cache_cli init \
  --server-endpoint http://localhost:8001

# 2. Download models through server
./target/debug/cache_cli preload "bert-base-uncased"

# 3. Check server-cached models
./target/debug/cache_cli list
```

## Troubleshooting

### Common Issues

#### 1. Permission Denied

```bash
# Error: Permission denied when accessing cache
# Solution: Check cache directory permissions
ls -la ~/.cache/model_express/
chmod 755 ~/.cache/model_express/
```

#### 2. Network Connectivity Issues

```bash
# Error: Failed to download model
# Solution: Check network and try again
./target/debug/cache_cli preload "model-name" --timeout 60
```

#### 3. Insufficient Disk Space

```bash
# Error: No space left on device
# Solution: Check available space and clear cache if needed
df -h ~/.cache/model_express/
./target/debug/cache_cli clear-all
```

#### 4. Corrupted Cache

```bash
# Error: Cache validation failed
# Solution: Clear and re-download
./target/debug/cache_cli clear-all
./target/debug/cache_cli preload "model-name"
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Set debug logging
export RUST_LOG=debug

# Run CLI with debug output
./target/debug/cache_cli preload "model-name"
```

### Configuration Issues

If the CLI can't find your configuration:

```bash
# Check configuration file
cat ~/.config/model_express/cache.toml

# Re-initialize configuration
./target/debug/cache_cli init
```

## Best Practices

### 1. Cache Organization

- Use dedicated cache directories for different environments
- Regularly monitor cache usage with `stats` command
- Implement cache rotation policies for large deployments

### 2. Model Selection

- Preload frequently used models
- Use specific model versions for reproducibility
- Monitor model sizes before downloading

### 3. Performance Optimization

- Use SSD storage for cache directories
- Implement cache warming strategies
- Monitor download speeds and optimize network settings

### 4. Security

- Secure cache directories with appropriate permissions
- Use HTTPS endpoints for model downloads
- Regularly validate cache integrity

### 5. Monitoring

- Set up regular cache health checks
- Monitor disk usage and cache growth
- Implement alerting for cache issues

## Integration with ModelExpress

The cache CLI integrates seamlessly with the ModelExpress ecosystem:

- **Server Integration**: Models downloaded via CLI are available to the server
- **Client Integration**: Client applications can use cached models
- **Kubernetes**: Cache volumes can be mounted in containerized deployments
- **Fluid**: Advanced caching with distributed storage systems

For more information about server integration, see the [Server Documentation](../README.md#running-the-server).

## Support

For issues and questions:

1. Check this documentation
2. Review the [troubleshooting section](#troubleshooting)
3. Check the [ModelExpress repository](https://github.com/your-repo/model-express)
4. Open an issue with detailed error information

---

*Last updated: $(date)* 