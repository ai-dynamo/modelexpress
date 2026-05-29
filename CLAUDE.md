# ModelExpress

Rust-based model cache management service and GPU-to-GPU weight transfer system using NVIDIA NIXL over RDMA.

**Reference documentation:**
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Project structure, crate catalog, gRPC services, server internals, Python client, NIXL integration
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) - Configuration reference, Docker, Kubernetes, Helm, P2P transfer setup, debugging
- [`CONTRIBUTING.md`](CONTRIBUTING.md) - Development setup, available commands, pre-commit hooks, environment variables, DCO
- [`docs/CLI.md`](docs/CLI.md) - CLI tool usage, commands, output formats, integration examples

## Coding Standards

- `unwrap()` is **strictly forbidden** except in benchmarks. `expect()` is allowed in tests. Always handle errors with `match`, `?`, or custom error types.
- All cargo dependencies go in the root `Cargo.toml`. Sub-crates use workspace dependencies exclusively. Never edit `Cargo.toml` or `Cargo.lock` by hand to add or update dependencies - use `cargo add` so you always get the latest version.
- Python dependencies go in `pyproject.toml`. Never edit dependency files by hand - use `uv add` so you always get the latest version.
- `cargo clippy` must pass with no warnings.
- No emojis in code or comments.
- Do not create markdown files to document code changes or decisions.
- Do not over-comment code. Removing code is fine without adding comments to explain why.
- Use mermaid diagrams instead of ASCII art in markdown files.
- Prefer established crates over hand-rolled implementations. Check existing workspace dependencies before adding new ones.

## Build and Test Commands

```bash
cargo build                          # Build
cargo build --release                # Release build
cargo test                           # Run all tests
cargo clippy                         # Lint (must pass, no warnings)
cargo run --bin modelexpress-server  # Run server
cargo run --bin config_gen -- --output model-express.yaml  # Generate config
cargo run --bin test_client -- --test-model "google-t5/t5-small"  # Test client
cargo run --bin fallback_test        # Fallback test
cargo bench                          # Criterion benchmarks
./run_integration_tests.sh           # Integration tests (starts server)
```

## Pre-commit Hooks

Run pre-commit after every code change, even before creating commits:

```bash
pre-commit run              # Staged files only
pre-commit run --all-files  # All files (recommended after significant changes)
```

Hooks: `cargo fmt`, `cargo clippy` (--fix), `cargo check`, trailing whitespace, end-of-file, YAML/TOML/JSON validation, merge conflict detection, large file check.

## Adding CLI Arguments

Client CLI arguments are defined in a shared struct to avoid duplication:

1. **Add to `ClientArgs`** in `modelexpress_common/src/client_config.rs`
   - Single source of truth for shared arguments
   - Use `#[arg(long, env = "MODEL_EXPRESS_...")]` for environment variable support
   - Do NOT use `-v` short flag (reserved for CLI's verbose)

2. **Update `ClientConfig::load()`** in the same file
   - Add override logic in the "APPLY CLI ARGUMENT OVERRIDES" section

3. **Do NOT duplicate in `Cli`** (`modelexpress_client/src/bin/modules/args.rs`)
   - `Cli` embeds `ClientArgs` via `#[command(flatten)]`
   - Only add CLI-specific arguments there (e.g., `--format`, `--verbose`)

4. **Add tests** in the `tests` module of `client_config.rs`

## Adding gRPC Services

1. Define the service in a `.proto` file under `modelexpress_common/proto/`
2. Add the proto file to `modelexpress_common/build.rs` compile list
3. Add the generated module to `modelexpress_common/src/lib.rs` under `pub mod grpc`
4. Implement the service trait in `modelexpress_server/src/` (new file or existing)
5. Register the service in `modelexpress_server/src/main.rs` during server startup

## Bumping the ModelExpress Version

ModelExpress version strings split into two categories with different bump
cadences. **Confusing them is the most common mistake** — public-image
references on `main` pointing at a tag that doesn't yet exist sends users
at a broken pull.

### On `main` (e.g. 0.4.0 → 0.5.0)

Bump the workspace/chart version and the protocol-version fixtures. **Do
not** touch any reference to the public release container image.

Files to bump:

- `Cargo.toml` — `[workspace.package].version` and the three internal
  `modelexpress-{common,client,server}` path-dep `version =` entries.
- `Cargo.lock` — regenerate with `cargo update --workspace` (touches only
  our four crates: `modelexpress-{common,client,server}` plus
  `model-express-workspace-tests`).
- `modelexpress_client/python/pyproject.toml` — `[project].version`.
- `modelexpress_client/python/uv.lock` — regenerate with `uv lock`
  (from `modelexpress_client/python/`).
- `helm/Chart.yaml` — `version` and `appVersion` (chart version tracks
  the workspace; chart image tag does not — see below).
- `docs/metadata.md` — `mx_version` example values (search the file for
  the old version literal; expect ~2 hits).
- `modelexpress_server/src/p2p/{source_identity,state,service}.rs` and
  `modelexpress_server/src/p2p/backend/redis.rs` — `mx_version` strings
  in `mod tests` fixtures.
- `modelexpress_client/python/tests/test_source_id.py` and
  `test_k8s_service_client.py` — `mx_version` strings in test fixtures.

`mx_version` is part of the `SourceIdentity` proto, which is hashed into
the `mx_source_id`. After bumping the version literals, the pinned
source-id assertions break. Capture the new hashes by running the
failing tests:

```bash
cd modelexpress_client/python && .venv/bin/python -m pytest \
  tests/test_source_id.py -v -k "pinned_hash or case_colliding"
```

Update the three Python assertions in `tests/test_source_id.py` and the
three matching Rust assertions in
`modelexpress_server/src/p2p/source_identity.rs` (search for
`test_python_cross_check_*`) so both sides cross-check on the new hashes.

Verify end-to-end before committing:

```bash
cargo check --workspace --tests
cd modelexpress_client/python && .venv/bin/python -m pytest tests/
```

### Public-image tag references — separate cadence

These references point at the public release container on NGC. They lag
behind the workspace version because the container has to be built and
published first:

- `examples/**/*.yaml` — `image: nvcr.io/nvidia/ai-dynamo/modelexpress-server:<tag>`
- `helm/values*.yaml` — `image.tag`
- `helm/README.md` — `image.tag` row + `docker pull` example
- Any CI manifest comment that contrasts against `nvcr.io/<old_tag>` for
  context (e.g. `ci/k8s/client/vllm/dynamo/manifest-azure-aggregated.yaml`)

Lifecycle:

1. Bump the workspace/Python/Helm-chart version on `main` (this section's
   primary procedure). Image-tag references are left untouched.
2. Cut the `release/<X.Y.Z>` branch from `main`. The branch inherits the
   still-stale image-tag references.
3. After the release tag is set and the container is published on NGC,
   bump the image-tag references on the release branch
   (e.g. `harrison/bump-examples-<X.Y.Z>` against `release/<X.Y.Z>`).
4. Cherry-pick that image-tag-bump commit back to `main`. This is the
   only time image-tag refs change on `main`, and only ever after the
   container actually exists on NGC.

Pulling references at a tag that doesn't yet exist on NGC sends users
at a broken `docker pull`, which is why these never move ahead of the
container publish.

## Git Workflow

Feature branches use `<username>/feature-name` format, forked from `main`.

## Tips

- Always read files to understand context before making changes.
- Do not implement changes eagerly. When discussing a problem or new feature, investigate thoroughly first, report findings, propose changes, and ask if they are acceptable before writing code.
- Flush Redis on redeploy: stale metadata causes P2P transfer failures.
- Long startup times are normal: DeepSeek-V3 takes ~40 min to warm up.
- Set `UCX_LOG_LEVEL=DEBUG` for NIXL/RDMA diagnostics.
- NIXL agents must match ranks: source rank 0 -> target rank 0.

## Documentation Updates

When making changes, update the appropriate documentation files:

| Change type | Files to update |
|---|---|
| Architecture, components, NIXL, gRPC services | `docs/ARCHITECTURE.md` |
| Coding standards, build commands, new patterns | `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/rust.mdc` |
| CLI arguments or commands | `docs/CLI.md` + all 3 agent files (Adding CLI Arguments section) |
| Configuration, environment variables | `docs/DEPLOYMENT.md` |
| Deployment (Docker, K8s, Helm, P2P) | `docs/DEPLOYMENT.md` |
| Known issues, FP8 handling | `docs/ARCHITECTURE.md` |
| Dev setup, scripts, pre-commit hooks | `CONTRIBUTING.md` |
| Contribution process, DCO | `CONTRIBUTING.md` |
| New binary targets, crates, Python modules | `docs/ARCHITECTURE.md` |
| Version-bump procedure changes | `CLAUDE.md` (Bumping the ModelExpress Version section) |

**A feature is incomplete until documentation is updated.**
