# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Build the project
cargo build

# Build in release mode
cargo build --release

# Run the server
cargo run --bin modelexpress-server

# Run tests
cargo test

# Run integration tests (starts server, runs test client)
./run_integration_tests.sh

# Run a specific test client
cargo run --bin test_client -- --test-model "google-t5/t5-small"

# Run clippy (required before submitting code)
cargo clippy

# Generate sample configuration file
cargo run --bin config_gen -- --output model-express.yaml
```

## Architecture

ModelExpress is a Rust-based model cache management service that accelerates inference by caching HuggingFace models. It can be deployed standalone or as a sidecar alongside inference solutions like NVIDIA Dynamo.

### Workspace Structure

The project is a Rust workspace with three crates:

- **`modelexpress_server`** (`modelexpress-server`): gRPC server providing model services
  - `services.rs`: Implements `HealthService`, `ApiService`, and `ModelService` gRPC services
  - `database.rs`: SQLite-based model status persistence via `ModelDatabase`
  - `cache.rs`: Cache eviction and management
  - Uses global `MODEL_TRACKER` (`LazyLock<ModelDownloadTracker>`) for tracking download state

- **`modelexpress_client`** (`modelexpress-client`): Client library and CLI tool
  - `lib.rs`: Main `Client` struct with gRPC clients for health, API, and model services
  - `bin/cli.rs`: HuggingFace CLI replacement for model downloads
  - Supports automatic fallback to direct download when server unavailable

- **`modelexpress_common`** (`modelexpress-common`): Shared code and protobuf definitions
  - `grpc/` module contains generated proto code (health, api, model)
  - `providers/huggingface.rs`: HuggingFace download implementation
  - `download.rs`: Provider-agnostic download orchestration
  - `cache.rs`, `config.rs`, `client_config.rs`: Configuration types

### gRPC Services

Protocol definitions are in `modelexpress_common/proto/`:
- `health.proto`: Health check endpoint
- `api.proto`: Generic request/response API
- `model.proto`: Model download with streaming status updates

### Key Patterns

- Download status tracked in SQLite database with compare-and-swap for concurrent request handling
- Streaming gRPC responses for download progress updates via `ModelStatusUpdate`
- `CacheConfig::discover()` finds cache configuration from environment or config files
- Configuration layering: CLI args > environment variables > config files > defaults

### Adding CLI Arguments

Client CLI arguments and environment variables are defined in a shared struct to avoid duplication:

1. **`ClientArgs`** in `modelexpress_common/src/client_config.rs`:
   - Single source of truth for shared client arguments (endpoint, timeout, cache settings, etc.)
   - Add new arguments here with `#[arg(long, env = "MODEL_EXPRESS_...")]`
   - Avoid `-v` short flag (reserved for CLI's verbose)

2. **`ClientConfig::load()`** in the same file:
   - Apply the new argument to the config struct in the "APPLY CLI ARGUMENT OVERRIDES" section

3. **`Cli`** in `modelexpress_client/src/bin/modules/args.rs`:
   - Embeds `ClientArgs` via `#[command(flatten)]`
   - Only add CLI-specific arguments here (e.g., `--format`, `--verbose`)

4. **Tests**: Add tests in `client_config.rs` for argument parsing and config loading

## Code Standards

- **No `unwrap()`**: Strictly forbidden except in benchmarks. Use `match`, `?`, or `expect()` (tests only)
- **All dependencies in root `Cargo.toml`**: Sub-crates use workspace dependencies exclusively
- **Clippy enforced**: `cargo clippy` must pass with no warnings (multiple lints set to deny)
- **No emojis in code**
- **No markdown documentation files for code changes**

## Pre-commit Hooks

This repository uses pre-commit hooks to enforce code quality. **Run pre-commit after every code change**, even before creating commits:

```bash
# Run all pre-commit hooks on staged files
pre-commit run

# Run on all files (recommended after significant changes)
pre-commit run --all-files
```

The hooks include:
- `cargo fmt` - Code formatting
- `cargo clippy` - Linting with auto-fix
- `cargo check` - Compilation check
- File hygiene checks (trailing whitespace, end-of-file, YAML/TOML/JSON validation, etc.)

Running pre-commit hooks early and often catches issues before they accumulate. Do not wait until commit time to discover problems.

## AI Agent Instructions

When introducing new patterns, conventions, or architectural decisions that affect how code should be written, update ALL AI agent instruction files:
- `CLAUDE.md` (Claude Code)
- `.github/copilot-instructions.md` (GitHub Copilot)
- `.cursor/rules/rust.mdc` (Cursor)
