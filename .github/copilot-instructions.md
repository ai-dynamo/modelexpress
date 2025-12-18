<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Repository Coding Standards

- In this repository, the use of `unwrap` in Rust code is **strictly forbidden** except in benchmarks. Using expect is allowed in tests. Always handle errors using proper error handling patterns such as `match`, `?`, or custom error types.
- All pull requests will be reviewed for accidental use of `unwrap`, and shouldn't pass clippy checks. Trying to circumvent these rules except in benchmark code will result in immediate rejection of the pull request.
- All cargo dependencies should be in the root `Cargo.toml` file, and any dependency for the sub crates should be workspace exclusively.

# General Guidelines

- Follow idiomatic Rust practices as outlined in the Rust Book.
- Prefer descriptive error messages when returning errors.

# Testing

- Write tests for all new features and bug fixes.
- Use the built-in testing framework and follow the conventions outlined in the Rust Book.
- Aim for high test coverage, but prioritize testing critical functionality.
- Always try running the ./run_integration_tests.sh script at the root of the repository before claiming work is done.
- Always run `cargo test` to ensure all tests pass before claiming work is done.
- Always run `cargo clippy` to catch common mistakes and improve the code quality, and fix any warnings or errors it reports.

# Architecture

The project is split in 3 separate crates:
1. `client`: Contains a test client, a CLI tool, and a client library.
2. `common`: Provides shared utilities and data structures for the model. Any constant definitions should be placed here. As much as possible, any shared logic should also be placed here.
3. `server`: Implements the server-side logic and API endpoints for ModelExpress in a stand alone server.

## Adding CLI Arguments

Client CLI arguments are defined in a shared struct to avoid duplication:

1. **Add to `ClientArgs`** in `modelexpress_common/src/client_config.rs`:
   - This is the single source of truth for shared arguments
   - Use `#[arg(long, env = "MODEL_EXPRESS_...")]` for environment variable support
   - Do NOT use `-v` short flag (reserved for CLI's verbose)

2. **Update `ClientConfig::load()`** in the same file:
   - Add override logic in the "APPLY CLI ARGUMENT OVERRIDES" section

3. **Do NOT duplicate in `Cli`** (`modelexpress_client/src/bin/modules/args.rs`):
   - `Cli` embeds `ClientArgs` via `#[command(flatten)]`
   - Only add CLI-specific arguments there (e.g., `--format`, `--verbose`)

4. **Add tests** in the `tests` module of `client_config.rs`

# Code quality

- Do **NOT** use emojis. These are unprofessional.
- Do not create markdown files to document code changes or decisions.
- Do not over-comment code. Removing code is fine without adding new comments to explain why.

# Pre-commit Hooks

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

# AI Agent Instructions

When introducing new patterns, conventions, or architectural decisions that affect how code should be written, update ALL AI agent instruction files:
- `CLAUDE.md` (Claude Code)
- `.github/copilot-instructions.md` (GitHub Copilot)
- `.cursor/rules/rust.mdc` (Cursor)
