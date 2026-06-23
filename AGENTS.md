---
name: modelexpress
description: Work on the public ModelExpress repository, including Rust server/client code, Python engine integrations, docs, Helm charts, examples, CI, and release/version alignment.
license: Apache-2.0
compatibility: Applies to the public ai-dynamo/modelexpress repository.
metadata:
  author: ModelExpress maintainers
  version: "1.0"
---

# ModelExpress guide for agents

ModelExpress is a Rust-based model weight management service with Python
engine integrations for LLM inference. It handles model acquisition, cache
coordination, metadata, and GPU-to-GPU weight transfer.

## Non-negotiables

- **Do not guess flags, defaults, release versions, image tags, or support
  status.** Verify against source files, examples, CI, or published release
  artifacts before documenting them.
- **Keep public docs public.** Do not add internal runner names, internal
  registries, private repo details, secrets, or company-only release mechanics
  to public documentation.
- **Do not bump public image references ahead of a published container.** If a
  doc or example references `nvcr.io/nvidia/ai-dynamo/modelexpress-server:<tag>`,
  verify that tag is the intended public image tag for the branch/release.
- **Protect user work.** Check `git status --short --branch` before editing,
  stage explicit paths, and never revert unrelated changes.

## Source of truth hierarchy

1. **This repository**
   - `Cargo.toml` and `Cargo.lock` for Rust workspace versions.
   - `modelexpress_client/python/pyproject.toml` and `uv.lock` for Python
     client versions and dependencies.
   - `helm/Chart.yaml`, `helm/values*.yaml`, and `helm/README.md` for chart
     and server-image defaults.
   - `ci/`, `.github/workflows/`, and integration Dockerfiles for tested
     runtime combinations.
   - `examples/` for user-facing deployment patterns.
2. **Public release artifacts**
   - GitHub release tags and public container images for release/image claims.
3. **Upstream integrations**
   - vLLM, SGLang, Dynamo, TensorRT-LLM, NIXL, Kubernetes, and cloud-provider
     docs when documenting behavior owned by those projects.

## Before you edit

- Fetch current main for docs or release work: `git fetch origin main`.
- Read the nearby file(s) before changing structure or tone.
- Search before adding new docs; prefer updating an existing page or adding a
  link over duplicating content.
- Keep README focused on orientation. Put detailed support/status/version
  information in `docs/COMPATIBILITY.md`.

## Documentation standards

- Position ModelExpress as usable standalone and with integrations such as
  vLLM, SGLang, Dynamo, llm-d, Prime-RL, and similar systems. Dynamo is an
  integration path, not a requirement.
- Clearly label experimental or beta paths, especially TensorRT-LLM P2P,
  GPUDirect Storage, SGLang ModelStreamer, and evolving RL/live-refit flows.
- Keep commands copy/pasteable. Use consistent placeholders such as
  `<namespace>`, `<your-registry>`, `<tag>`, and `<your-token>`.
- For compatibility/version docs, derive pins from CI, Dockerfiles, Helm, and
  examples. Do not invent untested combinations.
- Use Mermaid for diagrams rather than ASCII art.

## Version and compatibility checks

When changing release docs, compatibility docs, Helm, or examples, check for
drift across:

- Rust workspace version in `Cargo.toml`.
- Python package version in `modelexpress_client/python/pyproject.toml`.
- Helm chart/app version in `helm/Chart.yaml`.
- Helm image tags in `helm/values*.yaml` and `helm/README.md`.
- Public server image references in `examples/**/*.yaml`.
- Runtime pins in `ci/k8s/client/**/Dockerfile*`,
  `examples/**/Dockerfile*`, and `.github/workflows/*.yml`.
- Support coverage in `ci/TEST_PLAN.md` and `docs/COMPATIBILITY.md`.

For detailed version-bump mechanics, read `CLAUDE.md` before editing release
metadata or source-id fixtures.

## Validation commands

Run the narrowest relevant checks for the change. For docs/release metadata,
prefer:

```bash
git diff --check
find README.md docs CONTRIBUTING.md helm/README.md examples -name '*.md' -print0 \
  | xargs -0 awk 'BEGIN{bad=0} /^```/{count[FILENAME]++} END{for (f in count) if (count[f] % 2) {print f ": odd fence count " count[f]; bad=1} exit bad}'
OLD_TAG="REPLACE_WITH_OLD_TAG"
rg -n "modelexpress-server:${OLD_TAG}|image\\.tag.*${OLD_TAG}|${OLD_TAG}" \
  README.md docs helm examples .github ci -g '*.md' -g '*.yaml' -g '*.yml'
helm lint helm
```

For Rust/Python code changes, use the build and test commands in `CLAUDE.md`
and `CONTRIBUTING.md`.
