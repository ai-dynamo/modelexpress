#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bump the ModelExpress workspace version end to end.

This automates the "Bumping the ModelExpress Version" procedure documented in
CLAUDE.md (the on-`main` cadence). It deliberately does NOT touch public
release image tags (nvcr.io/.../modelexpress-server:<tag>) - those follow a
separate release-branch cadence.

The pipeline runs five stages, each skippable:

  1. literals  - edit workspace/chart/python version strings and every
                 mx_version fixture/example literal.
  2. cargo lock - regenerate Cargo.lock via `cargo update --workspace`.
  3. uv lock    - regenerate uv.lock via `uv lock`.
  4. hashes     - the version feeds the SHA256-derived mx_source_id, so the
                 pinned-hash assertions change. We run the pinned tests to
                 capture the new hashes and patch both the Python and Rust
                 assertions, then re-run to confirm.
  5. verify     - `cargo check --workspace --tests`, the Rust source-id tests,
                 and the Python test suite.

Usage:
    python scripts/bump_version.py <new_version> [options]

Run with --help for the full option list. Use --dry-run first to preview.

Stdlib only; the lock/hash/verify stages shell out to the existing cargo, uv,
and pytest toolchain.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Repo-relative paths.
PYTHON_DIR = Path("modelexpress_client/python")
CARGO_TOML = Path("Cargo.toml")
PYPROJECT = PYTHON_DIR / "pyproject.toml"
CHART = Path("helm/Chart.yaml")
PY_SOURCE_ID_TEST = PYTHON_DIR / "tests/test_source_id.py"
RS_SOURCE_ID = Path("modelexpress_server/src/p2p/source_identity.rs")

# Directories pruned from the mx_version discovery walk.
PRUNE_DIRS = {".git", "target", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}
# File extensions scanned for mx_version literals.
SCAN_SUFFIXES = {".rs", ".py", ".md", ".yaml", ".yml", ".json"}

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+([.\-+].+)?$")
HEX16 = r"[0-9a-f]{16}"


def fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Stage 1: literals
# ---------------------------------------------------------------------------


def mx_version_pattern(old: str) -> re.Pattern[str]:
    """Match a quoted `old` version that sits next to an `mx_version` token.

    Handles every fixture/example form in the tree (shown with a placeholder
    version so this docstring is never itself rewritten):
      - Rust struct field:  mx_version: "X.Y.Z".to_string()
      - Rust assert:         assert_eq!(attr.mx_version, "X.Y.Z")
      - Python kwarg:        mx_version="X.Y.Z"
      - Markdown table:      | mx_version | "X.Y.Z" |
      - JSON example:        "mx_version":"X.Y.Z"

    The bounded separator class only spans punctuation/quotes/whitespace, so it
    never jumps across an identifier and never matches a bare version literal.
    """
    return re.compile(r'(mx_version[`"\s:=,|]{0,8}")' + re.escape(old) + r'(")')


def replace_in_toml_section(text: str, header: str, pattern: re.Pattern[str], repl) -> tuple[str, int]:
    """Apply `pattern` only within the `[header]` TOML section."""
    idx = text.find(header)
    if idx == -1:
        return text, 0
    start = idx + len(header)
    nxt = re.search(r"\n\[", text[start:])
    end = start + nxt.start() if nxt else len(text)
    new_segment, count = pattern.subn(repl, text[start:end])
    return text[:start] + new_segment + text[end:], count


def edit_file(repo_root: Path, rel: Path, transform, dry_run: bool) -> int:
    """Run `transform(text) -> (new_text, count)` on a file; write unless dry-run."""
    path = repo_root / rel
    if not path.is_file():
        fail(f"expected file is missing: {rel}")
    original = path.read_text()
    new_text, count = transform(original)
    if count and not dry_run:
        path.write_text(new_text)
    return count


def apply_literals(repo_root: Path, old: str, new: str, dry_run: bool) -> list[tuple[str, int]]:
    """Edit the structured version files and every mx_version literal."""
    changes: list[tuple[str, int]] = []
    sub = lambda m: m.group(1) + new + m.group(2)

    # Cargo.toml: [workspace.package] version (section-scoped) + the three
    # internal path-dep version entries.
    ws_version = re.compile(r'(version\s*=\s*")' + re.escape(old) + r'(")')
    path_dep = re.compile(
        r'(modelexpress-(?:common|client|server)\s*=\s*\{[^}]*version\s*=\s*")' + re.escape(old) + r'(")'
    )

    def cargo_transform(text: str) -> tuple[str, int]:
        text, n1 = replace_in_toml_section(text, "[workspace.package]", ws_version, sub)
        text, n2 = path_dep.subn(sub, text)
        return text, n1 + n2

    changes.append((str(CARGO_TOML), edit_file(repo_root, CARGO_TOML, cargo_transform, dry_run)))

    # pyproject.toml: [project] version (section-scoped).
    proj_version = re.compile(r'(version\s*=\s*")' + re.escape(old) + r'(")')
    changes.append((
        str(PYPROJECT),
        edit_file(
            repo_root,
            PYPROJECT,
            lambda t: replace_in_toml_section(t, "[project]", proj_version, sub),
            dry_run,
        ),
    ))

    # helm/Chart.yaml: version (unquoted) + appVersion (quoted).
    chart_version = re.compile(r'^(version:\s*)' + re.escape(old) + r'(\s*)$', re.MULTILINE)
    chart_app = re.compile(r'^(appVersion:\s*")' + re.escape(old) + r'(")', re.MULTILINE)

    def chart_transform(text: str) -> tuple[str, int]:
        text, n1 = chart_version.subn(sub, text)
        text, n2 = chart_app.subn(sub, text)
        return text, n1 + n2

    changes.append((str(CHART), edit_file(repo_root, CHART, chart_transform, dry_run)))

    # mx_version literals: discovered across the whole tree so fixtures added
    # later (e.g. workspace-tests) are picked up automatically.
    mx_re = mx_version_pattern(old)
    for path in walk_source_files(repo_root):
        rel = path.relative_to(repo_root)
        original = path.read_text()
        new_text, count = mx_re.subn(sub, original)
        if count:
            if not dry_run:
                path.write_text(new_text)
            changes.append((str(rel), count))

    return changes


def walk_source_files(repo_root: Path):
    """Yield scannable source files, pruning build/VCS/cache dirs and this tool's own dir."""
    self_dir = Path(__file__).resolve().parent
    stack = [repo_root]
    while stack:
        current = stack.pop()
        for child in sorted(current.iterdir()):
            if child.is_dir():
                if child.name not in PRUNE_DIRS and child.resolve() != self_dir:
                    stack.append(child)
            elif child.suffix in SCAN_SUFFIXES:
                yield child


# ---------------------------------------------------------------------------
# Version detection / validation
# ---------------------------------------------------------------------------


def detect_workspace_version(repo_root: Path) -> str:
    text = (repo_root / CARGO_TOML).read_text()
    match = re.search(r"\[workspace\.package\][^\[]*?version\s*=\s*\"([^\"]+)\"", text, re.DOTALL)
    if not match:
        fail("could not find [workspace.package] version in Cargo.toml")
    return match.group(1)


def validate_semver(label: str, value: str) -> None:
    if not SEMVER_RE.match(value):
        fail(f"{label} is not a valid version: {value!r}")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str], cwd: Path, dry_run: bool, capture: bool = False) -> subprocess.CompletedProcess:
    printable = " ".join(cmd)
    print(f"      $ {printable}  (cwd: {cwd})")
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        fail(f"required tool {name!r} not found on PATH (skip its stage with the matching --skip flag)")


def python_interpreter(repo_root: Path) -> str:
    """Prefer the project's dev venv so the modelexpress package + pytest resolve."""
    venv = repo_root / PYTHON_DIR / ".venv/bin/python"
    return str(venv) if venv.is_file() else sys.executable


# ---------------------------------------------------------------------------
# Stage 4: hashes
# ---------------------------------------------------------------------------


def regenerate_hashes(repo_root: Path, dry_run: bool) -> None:
    """Recompute the pinned mx_source_id hashes and patch Python + Rust asserts.

    mx_version feeds the canonical JSON that is SHA256-hashed into mx_source_id,
    so bumping it changes every pinned hash. We run the pinned tests (which now
    fail), parse the (new, old) pairs from the assertion output, and replace each
    old hash literal with its new value across both the Python test and the Rust
    cross-check. The mandatory re-run guarantees correctness regardless of how the
    parse went: a bad map leaves the tests red and aborts the bump.
    """
    py = python_interpreter(repo_root)
    py_dir = repo_root / PYTHON_DIR

    if dry_run:
        print("      would run pinned-hash tests, parse new hashes, patch py + rs, then re-run")
        run([py, "-m", "pytest", "tests/test_source_id.py", "-q"], py_dir, dry_run=True)
        return

    proc = run([py, "-m", "pytest", "tests/test_source_id.py"], py_dir, dry_run=False, capture=True)
    output = (proc.stdout or "") + (proc.stderr or "")

    # pytest renders `assert <computed> == <pinned>` as 'new' == 'old'.
    pairs = re.findall(r"'(" + HEX16 + r")'\s*==\s*'(" + HEX16 + r")'", output)
    mapping = {old: new for new, old in pairs}

    if not mapping:
        if proc.returncode == 0:
            print("      pinned-hash tests already pass; no hashes to regenerate")
            return
        print(output)
        fail(
            "pinned-hash tests failed but no 'new == old' hash pairs were parsed; "
            "regenerate the assertions manually (see scripts/README.md)"
        )

    for old_hash, new_hash in mapping.items():
        print(f"      hash {old_hash} -> {new_hash}")

    for rel in (PY_SOURCE_ID_TEST, RS_SOURCE_ID):
        path = repo_root / rel
        text = path.read_text()
        for old_hash, new_hash in mapping.items():
            text = text.replace(old_hash, new_hash)
        path.write_text(text)

    confirm = run([py, "-m", "pytest", "tests/test_source_id.py", "-q"], py_dir, dry_run=False, capture=True)
    if confirm.returncode != 0:
        print((confirm.stdout or "") + (confirm.stderr or ""))
        fail("pinned-hash tests still fail after patching; the regenerated hashes are wrong")
    print("      patched pinned hashes; Python pinned-hash tests pass")


# ---------------------------------------------------------------------------
# Stage 5: verify
# ---------------------------------------------------------------------------


def verify(repo_root: Path, dry_run: bool) -> None:
    require_tool("cargo")
    py = python_interpreter(repo_root)
    py_dir = repo_root / PYTHON_DIR

    steps = [
        (["cargo", "check", "--workspace", "--tests"], repo_root),
        (["cargo", "test", "-p", "modelexpress-server", "--", "p2p::source_identity::tests"], repo_root),
        ([py, "-m", "pytest", "tests/"], py_dir),
    ]
    for cmd, cwd in steps:
        result = run(cmd, cwd, dry_run)
        if not dry_run and result.returncode != 0:
            fail(f"verification command failed: {' '.join(cmd)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bump_version.py",
        description="Bump the ModelExpress workspace version (on-main cadence).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("new_version", help="target version, e.g. 0.6.0")
    parser.add_argument("--from", dest="old_version", default=None,
                        help="current version (default: read from Cargo.toml [workspace.package])")
    parser.add_argument("--repo-root", default=None,
                        help="repository root (default: inferred from this script's location)")
    parser.add_argument("--dry-run", action="store_true", help="print planned edits and commands; write nothing")
    parser.add_argument("--skip-locks", action="store_true", help="skip Cargo.lock / uv.lock regeneration")
    parser.add_argument("--skip-hashes", action="store_true", help="skip source-id pinned-hash regeneration")
    parser.add_argument("--skip-verify", action="store_true", help="skip the cargo/pytest verification stage")
    return parser.parse_args(argv)


def resolve_repo_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    # scripts/bump_version.py -> repo root is one parent up.
    return Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    if not (repo_root / CARGO_TOML).is_file():
        fail(f"{CARGO_TOML} not found under repo root {repo_root}; pass --repo-root")

    new = args.new_version
    old = args.old_version or detect_workspace_version(repo_root)
    validate_semver("new version", new)
    validate_semver("current version", old)
    if old == new:
        print(f"current version is already {new}; nothing to do")
        return 0

    mode = " (dry-run)" if args.dry_run else ""
    print(f"bumping ModelExpress {old} -> {new}{mode}")
    print(f"repo root: {repo_root}")

    print("[1/5] literals")
    changes = apply_literals(repo_root, old, new, args.dry_run)
    touched = [(f, n) for f, n in changes if n]
    for f, n in touched:
        print(f"      {f}: {n}")
    total = sum(n for _, n in touched)
    if total == 0:
        fail(f"no version literals matched {old!r}; is --from correct?")
    print(f"      {len(touched)} files, {total} replacements")

    if args.skip_locks:
        print("[2/5] cargo.lock  (skipped)")
        print("[3/5] uv.lock     (skipped)")
    else:
        print("[2/5] cargo.lock")
        require_tool("cargo")
        run(["cargo", "update", "--workspace"], repo_root, args.dry_run)
        print("[3/5] uv.lock")
        require_tool("uv")
        run(["uv", "lock"], repo_root / PYTHON_DIR, args.dry_run)

    if args.skip_hashes:
        print("[4/5] hashes      (skipped)")
    else:
        print("[4/5] hashes")
        regenerate_hashes(repo_root, args.dry_run)

    if args.skip_verify:
        print("[5/5] verify      (skipped)")
    else:
        print("[5/5] verify")
        verify(repo_root, args.dry_run)

    print(f"done: {old} -> {new}" + (" (dry-run, nothing written)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
