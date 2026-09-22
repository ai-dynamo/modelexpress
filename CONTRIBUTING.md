<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to ModelExpress

For technical architecture, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For AI coding-agent instructions, see [`AGENTS.md`](AGENTS.md).

## Development Setup

### Prerequisites

- **Rust**: Latest stable version (recommended: 1.90+)
- **Cargo**: Rust's package manager (included with Rust)
- **protoc**: Protocol Buffers compiler
- **Python 3.10+** (optional): For the P2P client library
- **C++ compiler** (optional, `g++` or any compiler in `$CXX`): To build the `modelexpress.vmm._alloc_ext` extension that enables the `MX_VMM_ARENA=1` fast path. Without a working compiler the Python install still succeeds and `MX_VMM_ARENA=1` falls back to the pool-reg path at runtime with a warning
- **Docker** (optional): For containerized deployment
- **Redis** (optional): For P2P metadata coordination
- **pre-commit**: For automated code quality checks

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/ai-dynamo/modelexpress.git
cd modelexpress

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Build the project
cargo build

# Run tests
cargo test

# (Optional) Install the Python P2P client for development
pip install -e './modelexpress_client/python[dev]'
```

### DevContainer

A devcontainer configuration is provided for VSCode in `.devcontainer/`. It includes:

- Ubuntu 24.04 with Rust toolchain
- protobuf-compiler, build-essential, libssl-dev
- Debugging tools: gdb, lldb, perf
- CLI tools: ripgrep, jq, tree, redis-tools
- Pre-configured rust-analyzer with clippy-on-save
- Port 8001 forwarded for gRPC

### Available Commands

| Command | Description |
|---------|-------------|
| `cargo build` | Build all crates |
| `cargo build --release` | Release build |
| `cargo test` | Run all tests |
| `cargo clippy` | Lint (must pass with no warnings) |
| `cargo bench` | Run Criterion benchmarks |
| `MX_METADATA_BACKEND=redis REDIS_URL=redis://localhost:6379 cargo run --bin modelexpress-server` | Run the server with an existing local Redis instance |
| `cargo run --bin modelexpress-cli` | Run the CLI client |
| `cargo run --bin config_gen -- --output model-express.yaml` | Generate server config |
| `cargo run --bin test_client -- --test-model "google-t5/t5-small"` | Run test client |
| `cargo run --bin fallback_test` | Run fallback tests |
| `./run_integration_tests.sh` | Concurrent-download integration test; requires backend environment and a reachable backend |
| `pytest modelexpress_client/python/tests/` | Run Python client tests |
| `modelexpress_client/python/generate_proto.sh` | Regenerate Python protobuf stubs |
| `modelexpress_client/go/generate_proto.sh` | Regenerate Go protobuf and gRPC bindings |
| `(cd modelexpress_client/go && go test ./...)` | Run Go binding compilation tests |
| `pre-commit run` | Run hooks on staged files |
| `pre-commit run --all-files` | Run hooks on all files |

Server and client integration commands require a reachable backend and the configuration below. `run_integration_tests.sh` starts a server but does not provision Redis or Kubernetes; it also runs `cargo clean` and can change the default Rust toolchain. Review those effects before using it. For configuration-only validation without a backend or GPU, use the commands in [AGENTS.md](AGENTS.md#work-and-validation).

### Pre-commit Hooks

The repository uses pre-commit hooks defined in `.pre-commit-config.yaml`:

**General hooks:**

- `trailing-whitespace` - Remove trailing whitespace (excludes `.md`)
- `end-of-file-fixer` - Ensure files end with newline
- `check-yaml` - Validate YAML syntax
- `check-toml` - Validate TOML syntax
- `check-json` - Validate JSON syntax
- `check-merge-conflict` - Detect merge conflict markers
- `check-added-large-files` - Fail if files > 1MB added
- `check-case-conflict` - Detect case-insensitive filename conflicts
- `mixed-line-ending` - Enforce consistent line endings

**Rust hooks:**

- `cargo fmt` - Format with rustfmt
- `cargo clippy` - Lint with `--fix` and `-D warnings`
- `cargo check` - Compilation check

Run pre-commit after every code change, even before creating commits. Do not wait until commit time to discover problems.

### Environment Variables

#### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_EXPRESS_SERVER_HOST` | `0.0.0.0` | Server bind address |
| `MODEL_EXPRESS_SERVER_PORT` | `8001` | Server port |
| `MODEL_EXPRESS_CACHE_DIRECTORY` | `./cache` | Model cache directory |
| `MODEL_EXPRESS_CACHE_EVICTION_ENABLED` | `true` | Enable cache eviction |
| `MODEL_EXPRESS_LOG_LEVEL` | `info` | Log level (trace, debug, info, warn, error) |
| `MODEL_EXPRESS_LOG_FORMAT` | `pretty` | Log format (json, pretty, compact) |
| `MX_METADATA_BACKEND` | (required) | `redis` or `kubernetes` — drives both the P2P metadata and model registry backends |
| `REDIS_URL` | Required for Redis, unless both host and port settings are supplied | Reachable Redis endpoint; there is no implicit localhost fallback |

For local development, start Redis in one shell and run the server in another or in the foreground:

```bash
docker run --rm -d -p 6379:6379 --name mx-redis redis:8-alpine
MX_METADATA_BACKEND=redis REDIS_URL=redis://localhost:6379 cargo run -p modelexpress-server
```

#### Client

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_EXPRESS_ENDPOINT` | `http://localhost:8001` | Server endpoint |
| `MODEL_EXPRESS_TIMEOUT` | `30` | Request timeout in seconds |
| `MODEL_EXPRESS_CACHE_DIRECTORY` | (see below) | Cache path override |
| `MODEL_EXPRESS_NO_SHARED_STORAGE` | `false` | Disable shared storage mode |

Cache directory resolution order: `MODEL_EXPRESS_CACHE_DIRECTORY` -> `HF_HUB_CACHE` -> `~/.cache/huggingface/hub`.

#### P2P / NIXL

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_SERVER_ADDRESS` | `localhost:8001` | gRPC server address (recommended) |
| `MODEL_EXPRESS_URL` | `localhost:8001` | Legacy address; takes precedence over `MX_SERVER_ADDRESS` when both are set. Prefer `MX_SERVER_ADDRESS`; keep both equal only when an older integration requires it. |
| `MX_POOL_REG` | `0` | Allocation-level NIXL registration (registers cudaMalloc blocks instead of individual tensors) |

### Docker

```bash
# Build production image
docker build -f docker/Dockerfile -t model-express .

# Run with docker-compose
docker compose -f docker/docker-compose.yml up --build

# Build P2P client image
docker build -f examples/p2p_transfer_k8s/client/vllm/Dockerfile \
  -t your-registry/IMAGE_NAME:TAG .
```

The SGLang build and TensorRT-LLM deployment paths are documented under [`examples/p2p_transfer_k8s/client/`](examples/p2p_transfer_k8s/client/).

### Helm

The chart deploys the server and requires an explicit metadata backend, its connection settings, and a compatible server image. Follow [Helm installation](helm/README.md#installation) for working Redis or Kubernetes backend commands. The chart does not deploy Redis or inference workers. If you use `helm/deploy.sh`, supply a values file with those settings; its bare default invocation does not select a backend.

Values files: `values.yaml` (base defaults), `values-development.yaml`, `values-production.yaml`, `values-local-storage.yaml`. Review their storage, image, and backend settings for your environment.

## Contribution Guidelines

Contributions that fix documentation errors or that make small changes to existing code can be contributed directly by following the rules below and submitting a PR.

Contributions intended to add significant new functionality must follow a more collaborative path. Before submitting a large PR, submit a GitHub issue describing the proposed change so the ModelExpress team can provide feedback:

- A design for your change will be agreed upon to ensure consistency with ModelExpress's architecture.
- The ModelExpress project is spread across multiple repositories. The team will provide guidance about how and where your enhancement should be implemented.
- Testing is critical. Plan on spending significant time on creating tests. The team will help design testing compatible with existing infrastructure.
- User-visible features need documentation.

## Contribution Rules

- Code style is enforced by `cargo fmt` and `cargo clippy`. Follow existing conventions.
- Avoid introducing unnecessary complexity.
- Keep PRs concise and focused on a single concern.
- Build log must be clean: no warnings or errors.
- All tests must pass.
- No license or patent conflicts. You must certify compliance with the [license terms](https://github.com/ai-dynamo/modelexpress/blob/main/LICENSE) and sign off on the [Developer Certificate of Origin (DCO)](https://developercertificate.org).

## Git Workflow

Feature branches use `<username>/feature-name` format, forked from `main`.

## Running GitHub Actions Locally

You can use the `act` tool to run GitHub Actions locally. See [act usage](https://nektosact.com/introduction.html).

```bash
act -j pre-merge-rust
```

You can also use the VSCode extension [GitHub Local Actions](https://marketplace.visualstudio.com/items?itemName=SanjulaGanepola.github-local-actions).

## Developer Certificate of Origin

ModelExpress is open source under the Apache 2.0 license (see [the Apache site](https://www.apache.org/licenses/LICENSE-2.0) or [LICENSE](./LICENSE)).

We respect intellectual property rights and want to ensure all contributions are correctly attributed and licensed. A Developer Certificate of Origin (DCO) is a lightweight mechanism to do that.

The DCO is a declaration attached to every contribution. In the commit message, the developer adds a `Signed-off-by` statement and thereby agrees to the DCO, which you can find at [DeveloperCertificate.org](http://developercertificate.org/).

We require that every contribution is signed with a DCO, verified by a required CI check. Please use your real name. We do not accept anonymous contributors or pseudonyms.

Each commit must include:

```
Signed-off-by: Jane Smith <jane.smith@email.com>
```

You can use `-s` or `--signoff` to add the `Signed-off-by` line automatically.

If your pull request fails the DCO check, inspect the affected commits and preserve existing sign-offs. Check `git config user.name` and `git config user.email`, and add your sign-off only to contributions you can certify. The following repair is appropriate when every commit being rebased is yours to certify; otherwise arrange sign-off by each contributor or repair only the relevant commits:

```bash
git rebase --signoff origin/main
git push --force-with-lease
```

See the [DCO skill](.agents/skills/dco/SKILL.md) for the shared agent procedure. Do not add tool-attribution or `Co-Authored-By` trailers.
