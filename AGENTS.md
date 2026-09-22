# ModelExpress

Rust-based model cache management service and GPU-to-GPU weight transfer system using NVIDIA NIXL over RDMA.

This file holds the always-on rules for every AI coding agent working in this repository. Multi-step procedures live as skills under `.agents/skills/` and load on demand. Both are shared across tools:

| Tool | Always-on rules | Skills |
|---|---|---|
| Codex | `AGENTS.md` (native) | `.agents/skills/` (native) |
| Cursor | `AGENTS.md` (native) | `.agents/skills/` (native) |
| Claude Code | `CLAUDE.md` imports this file via `@AGENTS.md` | `.claude/skills/<name>` symlinks into `.agents/skills/<name>` |
| GitHub Copilot | Coding agent reads `AGENTS.md`; Copilot Chat is pointed here by `.github/copilot-instructions.md` | via `AGENTS.md` pointers below |

Edit `AGENTS.md` or the skill; never add a tool-specific copy. When a section here grows into a procedure, move it to a skill and list it below.

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

## Procedures (skills)

Load the matching skill before starting any of these. Each is a `SKILL.md` under `.agents/skills/`:

| Task | Skill |
|---|---|
| Add or change a client CLI argument or env var | `add-cli-argument` |
| Add a gRPC service | `add-grpc-service` |
| Bump the release version or public-image tags | `bump-version` |
| Commit sign-off and DCO repair | `dco` |

## Git Workflow

Feature branches use `<username>/feature-name` format, forked from `main`.

### Commits and DCO

- Every commit must carry a `Signed-off-by: Real Name <email>` trailer. Always commit with `git commit -s`. The DCO check is required CI and fails the PR otherwise.
- Use the contributor's real name and the email configured in `git config user.name` / `user.email`. Check both before committing on an unfamiliar machine.
- Preserve existing trailers when amending, rebasing, squashing, or cherry-picking. Use `git rebase --signoff` or `git cherry-pick --signoff` only when the person running the command is the one certifying the change.
- Do not add `Co-Authored-By` or tool-attribution trailers.
- See the DCO section of `CONTRIBUTING.md` for the full policy.

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
| Coding standards, build commands, new patterns, agent rules | `AGENTS.md` (the only agent-instruction file; `CLAUDE.md` and `.github/copilot-instructions.md` are pointers) |
| CLI arguments or commands | `docs/CLI.md` + `.agents/skills/add-cli-argument/SKILL.md` |
| Configuration, environment variables | `docs/DEPLOYMENT.md` |
| Deployment (Docker, K8s, Helm, P2P) | `docs/DEPLOYMENT.md` |
| Known issues, FP8 handling | `docs/ARCHITECTURE.md` |
| Dev setup, scripts, pre-commit hooks | `CONTRIBUTING.md` |
| Contribution process, DCO | `CONTRIBUTING.md` |
| New binary targets, crates, Python modules | `docs/ARCHITECTURE.md` |
| Version-bump procedure changes | `.agents/skills/bump-version/SKILL.md` |
| Agent procedures (multi-step how-tos) | `.agents/skills/<name>/SKILL.md` + the Procedures table in `AGENTS.md` |

**A feature is incomplete until documentation is updated.**
