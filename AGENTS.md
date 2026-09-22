# ModelExpress agent guide

ModelExpress loads inference weights, transfers weights between peers, and updates RL rollout workers. The Rust server manages caches and metadata; Python runtime integrations move and install weights.

This is the shared instruction file. Keep `CLAUDE.md` and `.github/copilot-instructions.md` as pointers; put reusable procedures in `.agents/skills/`. Claude's skill entries under `.claude/skills/` are symlinks to those shared skills.

## Find the relevant path

| Task | Start here | Implementation |
|---|---|---|
| Inference startup or P2P | [Loading paths](docs/guides/choose-a-path.md), then the runtime guide | [`modelexpress/engines/`](modelexpress_client/python/modelexpress/engines/) and [`load_strategy/`](modelexpress_client/python/modelexpress/load_strategy/) |
| RL weight updates | [RL guide](docs/guides/rl.md) | [`modelexpress_rl/`](modelexpress_client/python/modelexpress_rl/); shared geometry in [`refit/`](modelexpress_client/python/modelexpress/refit/) |
| Server, cache, or deployment | [Configuration](docs/CONFIGURATION.md), [Deployment](docs/DEPLOYMENT.md) | [`modelexpress_server/src/`](modelexpress_server/src/), [`helm/`](helm/) |
| CLI or shared RPC types | [CLI](docs/CLI.md), [Architecture](docs/ARCHITECTURE.md) | [`modelexpress_client/src/`](modelexpress_client/src/), [`modelexpress_common/`](modelexpress_common/) |

Inspect the current implementation before editing. Rust environment names live in [`envs.rs`](modelexpress_common/src/envs.rs); Python inference settings live in [`modelexpress/envs.py`](modelexpress_client/python/modelexpress/envs.py), with RL settings also in [`modelexpress_rl/envs.py`](modelexpress_client/python/modelexpress_rl/envs.py). Check [Compatibility](docs/COMPATIBILITY.md) before choosing images: example pins, CI pins, and published releases can differ.

## Work and validation

- Carry out the requested changes after checking existing behavior. For significant new functionality, follow the design discussion process in [CONTRIBUTING.md](CONTRIBUTING.md); routine fixes and documentation edits do not need a separate proposal approval.
- Start with checks that fit the change. Server configuration and many client unit tests need no GPU. A GPU serving response alone does not prove a particular transfer path ran; verify its completion logs and, for RL, the installed version and update lifecycle.
- Use isolated namespaces and cache paths for deployment tests. Inspect stale source records before changing them; never treat a blanket Redis flush as routine redeployment cleanup.
- Update the relevant user guide or reference with behavior changes. Use existing documents for explanations; do not add change diaries or decision reports to the repository unless requested. Keep Markdown paragraphs and list items on one line, relying on soft wrapping.
- Report what you checked and any missing runtime evidence. Do not imply that CPU tests validate a GPU or fabric combination.

From the repository root, these configuration checks do not require a running backend or GPU:

```bash
cargo run --bin config_gen -- --output /tmp/mx-agent-config.yaml
cargo run --bin modelexpress-server -- --config /tmp/mx-agent-config.yaml --validate-config
cargo run --bin modelexpress-cli -- --help
```

Config validation does not test backend connectivity. Starting a server requires an existing backend, for example `MX_METADATA_BACKEND=redis REDIS_URL=redis://localhost:6379 cargo run --bin modelexpress-server`. RL's refit service currently requires Redis. See [Deployment](docs/DEPLOYMENT.md) for backend setup.

For Python changes, install the client in a virtual environment with `pip install -e './modelexpress_client/python[dev]'`, then run the relevant tests with `python -m pytest`. For example, `python -m pytest modelexpress_client/python/tests/test_envs.py -q` checks environment parsing without a GPU. For Rust changes, run the affected tests plus formatting and lint checks; [CONTRIBUTING.md](CONTRIBUTING.md) and the [CI workflow](.github/workflows/ci.yml) describe the broader checks.

Run pre-commit on changed files before handing off work:

```bash
pre-commit run --files path/to/changed-file
```

The [hook configuration](.pre-commit-config.yaml) selects applicable checks. Rust hooks run workspace formatting, Clippy with `--fix` and `-D warnings`, and compilation; inspect any changes they make.

## Coding standards

- Never use `unwrap()` outside benchmarks. `expect()` is allowed in tests; handle other errors with `match`, `?`, or error types. Clippy must pass with no warnings.
- Keep Rust dependencies in the root `Cargo.toml`; member crates use workspace dependencies. Use `cargo add` for dependency changes rather than editing dependency entries or lockfiles by hand.
- Keep Python dependencies in `pyproject.toml` and use `uv add` for dependency changes. Preserve the runtime's compatible dependency stack.
- Prefer existing dependencies and established libraries over new implementations. Keep comments useful; do not add comments merely to narrate removed code.
- No emojis in code or comments. Use Mermaid instead of ASCII diagrams in Markdown. Preserve applicable copyright/SPDX headers and [license requirements](LICENSE).

## Procedures

Read the matching skill before doing the work:

| Task | Shared skill |
|---|---|
| Add or change a client CLI argument or environment variable | [add-cli-argument](.agents/skills/add-cli-argument/SKILL.md) |
| Add a gRPC service | [add-grpc-service](.agents/skills/add-grpc-service/SKILL.md) |
| Bump versions or public-image tags | [bump-version](.agents/skills/bump-version/SKILL.md) |
| Create, rewrite, or publish commits; open a PR; repair DCO | [dco](.agents/skills/dco/SKILL.md) |

## Commits

Feature branches use `<username>/feature-name`, based on `main`. Reuse the user's existing branch or draft PR when that is the requested starting point.

Every commit requires `Signed-off-by: Real Name <email>`; use `git commit -s` and check `git config user.name` and `git config user.email` first. Preserve existing sign-offs when rewriting commits, and only certify contributions you are authorized to certify. Do not add `Co-Authored-By` or tool-attribution trailers. Follow the [DCO policy](CONTRIBUTING.md#developer-certificate-of-origin) and the linked skill.
