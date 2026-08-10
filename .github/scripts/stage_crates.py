#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and package ModelExpress's publishable workspace crates for staging.

Adapted from ai-dynamo/dynamo's .github/scripts/stage_crates.py: the crate set
and dependency order are derived from `cargo metadata` (not a hardcoded list),
crates are processed leaves-first, an expect-version gate aborts before any
work if a crate carries an un-bumped version, and staging is fail-soft with
machine-readable STAGED_CRATE=/FAILED_CRATE= lines for the calling workflow.

Unlike the Dynamo original this does NOT publish to a cargo registry index: the
nightly staging destination is a plain folder in Artifactory, so each crate is
validated with `cargo check` and packaged with `cargo package --no-verify`
(--no-verify because intra-workspace deps resolve by version, and the staged
versions exist in no registry), and the .crate files are collected into
--output-dir for the workflow to upload.

Version stamping: --stage-version rewrites `version = "<current>"` across the
workspace manifests — [workspace.package].version and the internal path-dep
pins in [workspace.dependencies] share the same literal — then resyncs
Cargo.lock, so every staged nightly carries a distinct prerelease version.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


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
    # `<member> = { ... version = "<old>" ... }` on a single line. Note `=?`
    # rather than a `{0,1}` quantifier: inside an f-string, braces would be
    # parsed as a replacement field.
    dep_pat = re.compile(
        rf'(?m)^(\s*(?:{member_alt})\s*=\s*\{{[^}}\n]*?\bversion\s*=\s*")=?{re.escape(old)}(")'
    )
    # `version = "<old>"` inside the [workspace.package] table only.
    wp_pat = re.compile(
        rf'(?ms)(\[workspace\.package\].*?\n\s*version\s*=\s*"){re.escape(old)}(")'
    )
    # `version = "<old>"` inside a member manifest's [package] table.
    pkg_pat = re.compile(
        rf'(?ms)(\[package\].*?\n\s*version\s*=\s*"){re.escape(old)}(")'
    )

    changed = 0
    for manifest in workspace_manifests(root, pkgs):
        text = manifest.read_text()
        out = wp_pat.sub(lambda m: f"{m.group(1)}{new}{m.group(2)}", text, count=1)
        out = pkg_pat.sub(lambda m: f"{m.group(1)}{new}{m.group(2)}", out, count=1)
        # Exact-match requirement for internal deps.
        out = dep_pat.sub(lambda m: f"{m.group(1)}={new}{m.group(2)}", out)
        if out != text:
            manifest.write_text(out)
            changed += 1
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--output-dir", default="crate-dist",
                    help="directory the packaged .crate files are collected into")
    ap.add_argument("--expect-version", default=os.environ.get("EXPECT_VERSION", ""),
                    help="fail fast if any publishable crate's version differs from this "
                         "(evaluated after --stage-version stamping)")
    ap.add_argument("--stage-version", default=os.environ.get("STAGE_VERSION", ""),
                    help="rewrite every publishable crate's version to this exact value "
                         "before packaging (e.g. 0.5.0-nightly.20260806); empty = as-is")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pkgs = publishable(cargo_metadata(root))
    if not pkgs:
        print("::error::no publishable workspace crates found", file=sys.stderr)
        return 1
    order = topo_order(pkgs)
    print("Package order:", " ".join(order))

    stage_version = args.stage_version.strip()
    if stage_version:
        wm = re.search(r'\[workspace\.package\][^\[]*?\n\s*version\s*=\s*"([^"]+)"',
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
            # Resync the lockfile; fail loudly here rather than as a confusing
            # package error later.
            r = subprocess.run(["cargo", "update", "--workspace"], cwd=root)
            if r.returncode != 0:
                print("::error::cargo update --workspace failed after the stage-version rewrite",
                      file=sys.stderr)
                return 1

    # Fail fast BEFORE any build if a crate carries an unexpected version (e.g. a
    # hardcoded version the bump missed) — never silently stage wrong versions.
    if args.expect_version:
        mismatched = [(n, pkgs[n]["version"]) for n in order if pkgs[n]["version"] != args.expect_version]
        if mismatched:
            for n, v in mismatched:
                print(f"::error::crate {n} is at version {v}, expected {args.expect_version}",
                      file=sys.stderr)
            return 1

    # Validate the whole workspace compiles at the stamped version before
    # producing anything.
    print("=== cargo check --workspace ===", flush=True)
    if subprocess.run(["cargo", "check", "--workspace"], cwd=root).returncode != 0:
        print("::error::cargo check failed for the workspace at the staging version",
              file=sys.stderr)
        for name in order:
            print(f"FAILED_CRATE={name} cargo check failed", flush=True)
        return 1

    # Package ALL crates in ONE invocation. This is load-bearing, not a
    # micro-optimisation: with several `-p` targets cargo registers each
    # just-packaged .crate into a temporary overlay registry that the others
    # resolve against. Packaging crate-by-crate instead makes cargo resolve
    # `modelexpress-common = "=<staging version>"` against crates.io, where
    # that version does not exist, so every dependent crate fails (or, worse,
    # silently embeds a lockfile pointing at the last public release).
    print(f"=== cargo package ({len(order)} crates, leaves first) ===", flush=True)
    pkg_args = []
    for name in order:
        pkg_args += ["-p", name]
    rc = subprocess.run(
        ["cargo", "package", "--no-verify", "--allow-dirty", *pkg_args], cwd=root
    ).returncode

    # Report per crate on what actually landed on disk, so a partial failure
    # still tells the workflow (and Slack) exactly which crates are staged.
    #   STAGED_CRATE=<name>            packaged into --output-dir
    #   FAILED_CRATE=<name> <reason>   not produced
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
