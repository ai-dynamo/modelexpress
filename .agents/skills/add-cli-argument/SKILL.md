---
name: add-cli-argument
description: Add or change a ModelExpress client CLI argument or environment setting in Rust shared configuration, Python inference settings, or Python RL settings.
---

# Client arguments and environment settings

Choose the procedure for the component that reads the setting. A Python environment variable does not need a Rust CLI argument.

## Python settings

- Inference and shared client settings live in `modelexpress_client/python/modelexpress/envs.py`. RL also uses this module for shared model identity, server connectivity, worker endpoints, and heartbeat timing.
- RL-specific deployment policy lives in `modelexpress_client/python/modelexpress_rl/envs.py`, including trainer staging, payload format, desired version, checkpoint replay, and S3 transfer settings. Inspect existing readers before choosing a module; the variable prefix alone does not determine ownership.
- Add or update the `environment_variables` reader and its `TYPE_CHECKING` annotation. Define parsing, defaults, and validation in the owning module. Both registries read values live through `__getattr__`; use `envs.NAME` at the call site instead of copying the value at import time. Environment writes remain at their call sites.
- Cover the setting's default, overrides, and relevant invalid values in `tests/test_envs.py` or `tests/test_refit_envs.py` under `modelexpress_client/python/`, and test the behavior that consumes it when applicable.

## Rust client CLI and settings

1. Add shared arguments to `ClientArgs` in `modelexpress_common/src/client_config.rs`. Register environment names in `modelexpress_common/src/envs.rs` and reference their constants, for example `#[arg(long, env = crate::envs::MODEL_EXPRESS_...)]`.
2. Update `ClientConfig::load()` to apply CLI overrides. For a Rust environment-only setting, use the existing reader/configuration path; do not add a CLI option unless the requested interface needs one.
3. Keep shared arguments out of `Cli` in `modelexpress_client/src/bin/modules/args.rs`; it already flattens `ClientArgs`. Only CLI-specific options such as output format and verbosity belong there. `-v` is reserved for verbosity.
4. Add parsing and precedence tests in the relevant Rust module; shared client configuration tests live in `client_config.rs`.

Update `docs/CONFIGURATION.md` for settings and `docs/CLI.md` for CLI changes. Keep defaults, precedence, and examples consistent with the implementation; link specialized RL settings from the relevant RL guide rather than duplicating whole tables.
