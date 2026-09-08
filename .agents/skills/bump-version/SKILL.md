---
name: bump-version
description: Bump the ModelExpress workspace, Python, and Helm chart version, and update the pinned source-id hashes. Use for any release version bump, and for the separate public-image tag bump on release branches.
---

# Bumping the ModelExpress Version

ModelExpress version strings split into two categories with different bump
cadences. **Confusing them is the most common mistake** — public-image
references on `main` pointing at a tag that doesn't yet exist sends users
at a broken pull.

## On `main` (e.g. 0.4.0 → 0.5.0)

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
- `modelexpress_server/src/p2p/{source_identity,state,service}.rs`,
  `modelexpress_server/src/p2p/backend/{redis,memory}.rs` and
  `workspace-tests/tests/artifact_transfer_contract.rs` — `mx_version`
  strings in `mod tests` fixtures. `redis.rs` has two, a fixture and its
  `assert_eq!`.
- `modelexpress_client/python/tests/{test_source_id,test_k8s_service_client,
  test_vllm_artifacts,test_artifact_transfer}.py` — `mx_version` strings in
  test fixtures. `test_vllm_artifacts.py` has three, `test_artifact_transfer.py`
  has four.

Do not work from that list alone. It has been incomplete before, and a
missed fixture is silent: `backend/memory.rs` sat at `"0.3.0"` through
several bumps because it was not named here. Bump by search, not by list:

```bash
grep -rIn --exclude-dir=.git '<old_version>' .
```

Search every file type, not just `*.rs`/`*.py`/`*.toml`: the version also
lives in Markdown, YAML and the Helm chart. Then classify the hits, because
the public-image tag references below are stale on purpose and must not be
swept up by the same pass.

`mx_version` is part of the `SourceIdentity` proto, which is hashed into
the `mx_source_id`. After bumping the version literals, the pinned
source-id assertions break. Capture the new hashes by running the
failing tests:

```bash
cd modelexpress_client/python && .venv/bin/python -m pytest \
  tests/test_source_id.py -v
```

There are **four** pinned assertions per side, not three, carrying three
distinct hash values. Update all four in `tests/test_source_id.py` and the
four matching ones in `modelexpress_server/src/p2p/source_identity.rs` so
both sides cross-check on the new hashes:

- `test_empty_artifact_fields_preserve_existing_id` (both sides) — shares
  the base-identity hash literal. Run the whole file rather than filtering
  on `-k "pinned_hash or case_colliding"`, which does not select this one
  and leaves it red on both sides.
- `test_pinned_hash_base_identity` / `test_python_cross_check_base_identity`
- `test_pinned_hash_with_revision` / `test_python_cross_check_with_revision`
- `test_case_colliding_extra_parameters_are_deterministic` /
  `test_python_cross_check_case_colliding_extra`

If the Rust and Python values ever disagree, stop rather than pinning two
numbers — the canonicalizers in `metadata/source_id.py` and
`p2p/source_identity.rs` have diverged and that is the bug.

Verify end-to-end before committing. The Python venv needs the `dev` extra
(`uv pip install -e ".[dev]"`); without it `tests/test_tracing.py` aborts
collection on a missing `opentelemetry` and pytest runs zero tests while
reporting no failures.

```bash
cargo check --workspace --tests
cargo test --workspace
cd modelexpress_client/python && .venv/bin/python -m pytest tests/
```

`cargo check --workspace --tests` compiles the test targets, it does not run
their assertions, so on its own it cannot tell you a pinned hash is wrong.
`cargo test --workspace` is the step that actually executes them. Read the
per-test ok lines rather than the exit code alone.

## Public-image tag references — separate cadence

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
