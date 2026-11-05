#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and package ModelExpress's publishable workspace crates for staging.

Adapted from ai-dynamo/dynamo's .github/scripts/stage_crates.py: the crate set
and dependency order are derived from `cargo metadata` (not a hardcoded list),
crates are processed leaves-first, an expect-version gate aborts before any
work if a crate carries an un-bumped version, and staging is fail-soft with
machine-readable STAGED_CRATE=/FAILED_CRATE= lines for the calling workflow.

Two modes:

* Default (nightly): the staging destination is a plain folder in Artifactory,
  so each crate is validated with `cargo check` and packaged with
  `cargo package --no-verify` (--no-verify because intra-workspace deps resolve
  by version, and the staged versions exist in no registry), and the .crate
  files are collected into --output-dir for the workflow to upload.
* `--registry <alias>` (RC): crates are published, leaves first, to the
  Artifactory cargo registry with `cargo publish`, so the staged versions land
  in the registry index and resolve for `cargo` consumers. Artifactory only,
  never crates.io. The registry comes from ARTIFACTORY_CARGO_INDEX, or is
  derived from ARTIFACTORY_URL + ARTIFACTORY_CARGO_REPO_NAME; auth is
  ARTIFACTORY_TOKEN. Idempotent: versions already on the registry are skipped.

Version stamping: --stage-version rewrites `version = "<current>"` across the
workspace manifests — [workspace.package].version and the internal path-dep
pins in [workspace.dependencies] share the same literal — then resyncs
Cargo.lock, so the staged crates carry the nightly prerelease version.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Artifactory < 7.148 rejects the `Content-Type: application/octet-stream` header
# cargo >= 1.96 sends on publish with a 415 (rust-lang/cargo#17086), so the
# upload runs on the last pre-header toolchain. Remove once the registry host
# runs Artifactory 7.148+.
PUBLISH_TOOLCHAIN = "1.95.0"
# No bare "HTTP/2" alternation: cargo prints the raw status line on every HTTP
# error, which would make deterministic 4xx failures retry as transient.
RETRYABLE = re.compile(
    r"stream error|INTERNAL_ERROR|connection error|timed out|\b(429|50[0-9])\b"
)
ALREADY = re.compile(r"already exists|409 Conflict|already uploaded")
# Sparse-index propagation lag after publishing a dep; retried before failing.
DEP_NOT_INDEXED = re.compile(r"no matching package named")


def cargo_metadata(root: Path) -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def publishable(meta: dict) -> dict[str, dict]:
    members = set(meta["workspace_members"])
    pkgs: dict[str, dict] = {}
    for p in meta["packages"]:
        if p["id"] not in members:
            continue
        if p.get("publish") == []:  # `publish = false` serializes to []
            continue
        pkgs[p["name"]] = p
    return pkgs


def topo_order(pkgs: dict[str, dict]) -> list[str]:
    names = set(pkgs)
    incoming = {
        n: {d["name"] for d in pkgs[n]["dependencies"]
            if d["name"] in names and d["name"] != n and d.get("kind") != "dev"}
        for n in names
    }
    order: list[str] = []
    ready = sorted(n for n in names if not incoming[n])
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in names:
            if n in incoming[m]:
                incoming[m].discard(n)
                if not incoming[m] and m not in order and m not in ready:
                    ready.append(m)
        ready.sort()
    if len(order) != len(names):
        raise RuntimeError(f"dependency cycle among crates: {sorted(names - set(order))}")
    return order


def workspace_manifests(root: Path, pkgs: dict[str, dict]) -> list[Path]:
    return [root / "Cargo.toml", *sorted(Path(p["manifest_path"]) for p in pkgs.values())]


def rewrite_versions(root: Path, pkgs: dict[str, dict], old: str, new: str) -> int:
    """Stamp the staging version across the workspace manifests.

    Two, and only two, kinds of literal are rewritten:

    1. `[workspace.package].version` — the version every member inherits.
    2. The `version = "<old>"` field of an internal dependency entry, i.e. a
       line whose key is a workspace member (`modelexpress-common = { path =
       ..., version = "0.5.0" }`). These are rewritten to an EXACT `=`
       requirement so a consumer of the staged .crate can never satisfy the
       dependency with a published release instead of the nightly sibling.

    A blanket `version = "<old>"` substitution is deliberately avoided: a
    third-party dependency that happens to be pinned at the same literal as
    the workspace version would be silently repointed at a version that does
    not exist, corrupting the dependency graph in a way that only surfaces at
    consumer build time.

    Returns the number of manifests changed.
    """
    members = sorted(pkgs, key=len, reverse=True)
    member_alt = "|".join(re.escape(m) for m in members)
    dep_pat = re.compile(
        rf'(?m)^(\s*(?:{member_alt})\s*=\s*\{{[^}}\n]*?\bversion\s*=\s*")=?{re.escape(old)}(")'
    )
    wp_pat = re.compile(
        rf'(?ms)(^\[workspace\.package\](?:(?!^[ \t]*\[).)*?\n[ \t]*version[ \t]*=[ \t]*"){re.escape(old)}(")'
    )
    pkg_pat = re.compile(
        rf'(?ms)(^\[package\](?:(?!^[ \t]*\[).)*?\n[ \t]*version[ \t]*=[ \t]*"){re.escape(old)}(")'
    )

    changed = 0
    for manifest in workspace_manifests(root, pkgs):
        text = manifest.read_text()
        out = wp_pat.sub(lambda m: f"{m.group(1)}{new}{m.group(2)}", text, count=1)
        out = pkg_pat.sub(lambda m: f"{m.group(1)}{new}{m.group(2)}", out, count=1)
        out = dep_pat.sub(lambda m: f"{m.group(1)}={new}{m.group(2)}", out)
        if out != text:
            manifest.write_text(out)
            changed += 1
    return changed


def registry_index() -> str:
    """Sparse index URL: ARTIFACTORY_CARGO_INDEX, else built from ARTIFACTORY_URL
    and ARTIFACTORY_CARGO_REPO_NAME. Empty when neither is configured."""
    index = os.environ.get("ARTIFACTORY_CARGO_INDEX", "").strip()
    if index:
        return index
    base = os.environ.get("ARTIFACTORY_URL", "").strip().rstrip("/")
    repo = os.environ.get("ARTIFACTORY_CARGO_REPO_NAME", "").strip().strip("/")
    if not base or not repo:
        return ""
    base = base.removesuffix("/artifactory")
    return f"sparse+{base}/artifactory/api/cargo/{repo}/index/"


def write_cargo_config(root: Path, alias: str, index: str) -> None:
    """Append the registry tables to .cargo/config.toml, preserving any build
    settings already there. The marker makes re-runs idempotent."""
    marker = "# --- stage_crates registry (generated) ---"
    cfg = root / ".cargo" / "config.toml"
    cfg.parent.mkdir(exist_ok=True)
    text = cfg.read_text() if cfg.exists() else ""
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    # Checked after stripping our own block so a re-run cannot mask a foreign table.
    if re.search(r"(?m)^\s*\[registr(?:y|ies)[\].]", text):
        raise RuntimeError(
            ".cargo/config.toml already defines registry tables; refusing to append conflicting ones"
        )
    cfg.write_text(
        f"{text}\n{marker}\n"
        f'[registries]\n{alias} = {{ index = "{index}" }}\n\n'
        f'[registry]\ndefault = "{alias}"\n'
        'global-credential-providers = ["cargo:token"]\n'
    )


def inject_registry(
    manifests: list[Path], names: list[str], alias: str, versions: dict[str, str]
) -> None:
    """Give every internal path dependency a registry (and a version, if it has
    none): `cargo publish` otherwise defaults them to crates.io or rejects them.
    `workspace = true` deps inherit from the root table, which is also rewritten."""
    pat = re.compile(
        r"^(" + "|".join(re.escape(n) for n in names) + r")(\s*=\s*)\{([^}]*)\}",
        re.MULTILINE,
    )

    def repl(m: re.Match) -> str:
        name, inner = m.group(1), m.group(3).strip()
        if "registry" in inner or "workspace" in inner:
            return m.group(0)
        parts = inner.rstrip(",")
        if "version" not in inner and name in versions:
            parts = f'{parts}, version = "{versions[name]}"'
        return f'{m.group(1)}{m.group(2)}{{ {parts}, registry = "{alias}" }}'

    for manifest in manifests:
        text = manifest.read_text()
        new = pat.sub(repl, text)
        if new != text:
            manifest.write_text(new)


def crate_exists(raw_base: str, name: str, version: str, token: str) -> bool:
    """200 -> present, 404 -> absent; anything else (auth, outage) is a hard error."""
    url = f"{raw_base}/{name}/{name}-{version}.crate"
    req = urllib.request.Request(
        url, method="HEAD", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        # Path only: the registry host is secret-derived and error strings end
        # up in public Actions logs.
        raise RuntimeError(
            f"registry HEAD {urllib.parse.urlparse(url).path} failed: HTTP {e.code} (bad ARTIFACTORY_TOKEN or registry outage?)"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"registry HEAD {urllib.parse.urlparse(url).path} unreachable: {e.reason}"
        ) from e


def publish(manifest: str, alias: str, env: dict) -> str:
    """Returns 'published' or 'already' (both staged); raises on definitive failure."""
    for attempt in range(1, 4):
        r = subprocess.run(
            ["cargo", f"+{PUBLISH_TOOLCHAIN}", "publish", "--allow-dirty", "--no-verify",
             "--manifest-path", manifest, "--registry", alias],
            env=env, capture_output=True, text=True,
        )
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        out = r.stdout + r.stderr
        if r.returncode == 0:
            return "published"
        if ALREADY.search(out):
            print(f"  -> {manifest}: skip (already present)")
            return "already"
        if attempt < 3 and DEP_NOT_INDEXED.search(out):
            print(f"  -> dep not yet indexed (sparse-index lag?), retry in 15s ({attempt}/3)")
            time.sleep(15)
            continue
        if attempt < 3 and RETRYABLE.search(out):
            print(f"  -> transient error, retry in 10s ({attempt}/3)")
            time.sleep(10)
            continue
        raise RuntimeError(f"cargo publish failed for {manifest}")
    raise RuntimeError(f"exhausted retries for {manifest}")


def publish_to_registry(
    root: Path, pkgs: dict[str, dict], order: list[str], alias: str, index: str, token: str
) -> int:
    """Publish `order` (leaves first) to the Artifactory cargo registry, fail-soft:
    a failed crate skips its dependents but not the rest. Emits STAGED_CRATE= and
    FAILED_CRATE=<name> <reason> lines for the workflow."""
    m = re.match(r"sparse\+(https://.+?)/api/cargo/([^/]+)/index/?$", index)
    if not m:
        print("::error::cargo index must be 'sparse+https://<host>/.../api/cargo/<repo>/index/'",
              file=sys.stderr)
        return 1
    raw_base = f"{m.group(1)}/{m.group(2)}/crates"

    manifests = [root / "Cargo.toml", *(Path(pkgs[n]["manifest_path"]) for n in order)]
    write_cargo_config(root, alias, index)
    inject_registry(manifests, order, alias, {n: pkgs[n]["version"] for n in order})

    env = dict(os.environ)
    env[f"CARGO_REGISTRIES_{alias.upper().replace('-', '_')}_TOKEN"] = f"Bearer {token}"
    r = subprocess.run(["rustup", "toolchain", "install", "--profile", "minimal",
                        "--no-self-update", PUBLISH_TOOLCHAIN], env=env)
    if r.returncode != 0:
        print(f"::error::rustup install of publish toolchain {PUBLISH_TOOLCHAIN} failed",
              file=sys.stderr)
        return 1

    deps_of = {
        n: {d["name"] for d in pkgs[n]["dependencies"]
            if d["name"] in pkgs and d["name"] != n and d.get("kind") != "dev"}
        for n in order
    }
    published = already = 0
    staged: list[str] = []
    failed: dict[str, str] = {}
    for name in order:
        manifest, version = pkgs[name]["manifest_path"], pkgs[name]["version"]
        print(f"=== {name} {version} ===")
        bad_deps = sorted(deps_of[name] & set(failed))
        if bad_deps:
            failed[name] = f"dependency {','.join(bad_deps)} failed"
            print(f"::error::{name} {version} skipped: {failed[name]}", file=sys.stderr)
            print(f"FAILED_CRATE={name} {failed[name]}", flush=True)
            continue
        try:
            if crate_exists(raw_base, name, version, token):
                print(f"  -> {name} {version} already on the registry (skip)")
                already += 1
                staged.append(name)
                print(f"STAGED_CRATE={name}", flush=True)
                continue
            if subprocess.run(["cargo", "check", "--manifest-path", manifest],
                              cwd=root, env=env).returncode != 0:
                raise RuntimeError(f"cargo check failed for {manifest}")
            status = publish(manifest, alias, env)
        except RuntimeError as e:
            failed[name] = str(e)
            print(f"::error::{name} {version} not staged: {e}", file=sys.stderr)
            print(f"FAILED_CRATE={name} {e}", flush=True)
            continue
        if status == "published":
            published += 1
        else:
            already += 1
        staged.append(name)
        print(f"STAGED_CRATE={name}", flush=True)

    print(f"Done: {len(order)} crates ({published} published, {already} already present, "
          f"{len(staged)} staged, {len(failed)} failed).")
    print(f"STAGED_CRATES={','.join(staged)}")
    if failed:
        print(f"FAILED_CRATES={','.join(failed)}")
        print(f"::error::{len(failed)} crate(s) not staged: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--registry", default="",
                    help="publish to this Artifactory cargo registry alias (RC mode) instead of "
                         "packaging .crate files into --output-dir (nightly mode)")
    ap.add_argument("--output-dir", default="crate-dist",
                    help="directory the packaged .crate files are collected into")
    ap.add_argument("--expect-version", default=os.environ.get("EXPECT_VERSION", ""),
                    help="fail fast if any publishable crate's version differs from this "
                         "(evaluated after --stage-version stamping)")
    ap.add_argument("--stage-version", default=os.environ.get("STAGE_VERSION", ""),
                    help="rewrite every publishable crate's version to this exact value "
                         "before packaging (e.g. 0.7.0-nightly.20261002.gf2d68fa); empty = as-is")
    ap.add_argument("--stamp-only", action="store_true",
                    help="stop after stamping and the version check; no cargo check or package")
    args = ap.parse_args()

    # Artifactory only: crates.io must never be a publish target.
    if args.registry in ("crates-io", "crates.io", "crates_io"):
        print("::error::refusing to publish to crates.io; this stages to Artifactory only",
              file=sys.stderr)
        return 1
    token = os.environ.get("ARTIFACTORY_TOKEN", "")
    index = registry_index() if args.registry else ""
    if args.registry and not (token and index):
        print("::error::--registry needs ARTIFACTORY_TOKEN and ARTIFACTORY_CARGO_INDEX "
              "(or ARTIFACTORY_URL + ARTIFACTORY_CARGO_REPO_NAME)", file=sys.stderr)
        return 1

    root = Path(args.root).resolve()
    out_dir = Path(args.output_dir).resolve()
    if not args.registry:
        out_dir.mkdir(parents=True, exist_ok=True)

    pkgs = publishable(cargo_metadata(root))
    if not pkgs:
        print("::error::no publishable workspace crates found", file=sys.stderr)
        return 1
    order = topo_order(pkgs)
    print("Package order:", " ".join(order))

    stage_version = args.stage_version.strip()
    if stage_version:
        wm = re.search(
            r'(?ms)^\[workspace\.package\](?:(?!^[ \t]*\[).)*?\n[ \t]*version[ \t]*=[ \t]*"([^"]+)"',
            (root / "Cargo.toml").read_text())
        if not wm:
            print("::error::--stage-version: cannot read [workspace.package].version", file=sys.stderr)
            return 1
        cur = wm.group(1)
        if cur != stage_version:
            n = rewrite_versions(root, pkgs, cur, stage_version)
            print(f"stage-version: rewrote {cur} -> {stage_version} across {n} manifest(s)")
            for nm in pkgs:
                if pkgs[nm]["version"] == cur:
                    pkgs[nm]["version"] = stage_version
            r = subprocess.run(["cargo", "update", "--workspace"], cwd=root)
            if r.returncode != 0:
                print("::error::cargo update --workspace failed after the stage-version rewrite",
                      file=sys.stderr)
                return 1

    if args.expect_version:
        mismatched = [(n, pkgs[n]["version"]) for n in order if pkgs[n]["version"] != args.expect_version]
        if mismatched:
            for n, v in mismatched:
                print(f"::error::crate {n} is at version {v}, expected {args.expect_version}",
                      file=sys.stderr)
            return 1

    if args.stamp_only:
        return 0

    if args.registry:
        return publish_to_registry(root, pkgs, order, args.registry, index, token)

    print("=== cargo check --workspace ===", flush=True)
    if subprocess.run(["cargo", "check", "--workspace"], cwd=root).returncode != 0:
        print("::error::cargo check failed for the workspace at the staging version",
              file=sys.stderr)
        for name in order:
            print(f"FAILED_CRATE={name} cargo check failed", flush=True)
        return 1

    # All crates in one cargo package call: each resolves its siblings from the others' just-packaged .crate.
    print(f"=== cargo package ({len(order)} crates, leaves first) ===", flush=True)
    pkg_args = []
    for name in order:
        pkg_args += ["-p", name]
    rc = subprocess.run(
        ["cargo", "package", "--no-verify", "--allow-dirty", *pkg_args], cwd=root
    ).returncode

    staged: list[str] = []
    failed: dict[str, str] = {}
    for name in order:
        version = pkgs[name]["version"]
        crate_file = root / "target" / "package" / f"{name}-{version}.crate"
        if not crate_file.is_file():
            failed[name] = f"expected package output missing: {crate_file.name}"
            print(f"::error::{name} {version} not staged: {failed[name]}", file=sys.stderr)
            print(f"FAILED_CRATE={name} {failed[name]}", flush=True)
            continue
        shutil.copy2(crate_file, out_dir / crate_file.name)
        staged.append(name)
        print(f"STAGED_CRATE={name}", flush=True)

    print(f"Done: {len(order)} crates ({len(staged)} packaged, {len(failed)} failed).")
    print(f"STAGED_CRATES={','.join(staged)}")
    if failed or rc != 0:
        if failed:
            print(f"FAILED_CRATES={','.join(failed)}")
        print(f"::error::crate packaging did not fully succeed (cargo rc={rc}, "
              f"{len(failed)} crate(s) missing)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
