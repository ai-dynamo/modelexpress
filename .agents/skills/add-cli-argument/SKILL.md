---
name: add-cli-argument
description: Add or change a ModelExpress client CLI argument or environment variable. Use whenever touching ClientArgs, ClientConfig::load(), the Cli struct, or envs.rs/envs.py.
---

# Adding CLI Arguments

Client CLI arguments are defined in a shared struct to avoid duplication:

1. **Add to `ClientArgs`** in `modelexpress_common/src/client_config.rs`
   - Single source of truth for shared arguments
   - Register the variable name in `modelexpress_common/src/envs.rs` and reference the constant: `#[arg(long, env = crate::envs::MODEL_EXPRESS_...)]` (never a bare string literal, so the CLI and the `envs` getters cannot drift). All env-var names live in `modelexpress_common/src/envs.rs` (Rust) and `modelexpress/envs.py` (Python).
   - Do NOT use `-v` short flag (reserved for CLI's verbose)

2. **Update `ClientConfig::load()`** in the same file
   - Add override logic in the "APPLY CLI ARGUMENT OVERRIDES" section

3. **Do NOT duplicate in `Cli`** (`modelexpress_client/src/bin/modules/args.rs`)
   - `Cli` embeds `ClientArgs` via `#[command(flatten)]`
   - Only add CLI-specific arguments there (e.g., `--format`, `--verbose`)

4. **Add tests** in the `tests` module of `client_config.rs`
