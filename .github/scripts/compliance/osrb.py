#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the OSRB dependency CSVs consumed by release-automation's nvbug-attach-*.py."""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

FIELDS = ["package_name", "version", "type", "spdx_license"]
PUBLISHED_CRATES = ["modelexpress-common", "modelexpress-client", "modelexpress-server"]
LINUX_TARGETS = {"amd64": "x86_64-unknown-linux-gnu", "arm64": "aarch64-unknown-linux-gnu"}
CARGO_PKG_RE = re.compile(r"^(?P<name>\S+) v(?P<version>\S+)(?: \((?P<source>[^)]*)\))?(?: \(\*\))?$")


def fail(msg: str) -> None:
    print(f"::error::{msg}", file=sys.stderr)
    sys.exit(1)


def read_tsv(path: Path, pkg_type: str) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            rows.append(dict(zip(FIELDS, [parts[0], parts[1], pkg_type, parts[2]])))
    return rows


def image_packages(directory: Path) -> list[dict]:
    return read_tsv(directory / "dpkg.tsv", "dpkg") + read_tsv(directory / "python.tsv", "python")


def cargo_packages(manifest: Path, targets: list[str]) -> list[dict]:
    cmd = ["cargo", "tree", "--locked", "--manifest-path", str(manifest),
           "-e", "normal,build", "--prefix", "none", "--format", "{p}\t{l}"]
    for crate in PUBLISHED_CRATES:
        cmd += ["-p", crate]
    for target in targets:
        cmd += ["--target", target]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        pkg, _, lic = line.partition("\t")
        m = CARGO_PKG_RE.match(pkg.strip())
        if not m:
            fail(f"unparseable cargo tree line: {line!r}")
        source = m.group("source") or ""
        if source.startswith("/"):
            continue  # workspace member
        key = (m.group("name"), m.group("version"))
        rows[key] = dict(zip(FIELDS, [key[0], key[1], "cargo", lic.strip() or "UNKNOWN"]))
    return list(rows.values())


def python_license(meta: dict) -> str:
    if meta.get("license_expression"):
        return meta["license_expression"]
    lic = (meta.get("license") or "").strip()
    if lic and "\n" not in lic and len(lic) <= 100:
        return lic
    classifiers = [c.split(" :: ")[-1] for c in meta.get("classifier", []) if c.startswith("License ::")]
    return " AND ".join(classifiers) or "UNKNOWN"


def python_packages(report: Path, own_name: str) -> list[dict]:
    data = json.loads(report.read_text())
    rows = []
    for item in data.get("install", []):
        meta = item.get("metadata", {})
        if meta.get("name", "").lower() == own_name:
            continue
        rows.append(dict(zip(FIELDS, [meta["name"], meta["version"], "python", python_license(meta)])))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["type"], r["package_name"].lower(), r["version"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}", file=sys.stderr)


def cmd_container(args: argparse.Namespace) -> None:
    target = image_packages(Path(args.target_dir))
    base = image_packages(Path(args.base_dir))
    cargo = cargo_packages(Path(args.manifest), [LINUX_TARGETS[args.arch]])
    if not target:
        fail(f"no packages extracted from the image in {args.target_dir}")
    if not base:
        fail(f"no packages extracted from the base image in {args.base_dir}")
    if not cargo:
        fail("cargo tree returned no dependencies")
    base_versions = {(r["package_name"], r["type"]): r["version"] for r in base}
    added = [r for r in target if base_versions.get((r["package_name"], r["type"])) != r["version"]]
    stem = f"osrb-modelexpress-server-{args.arch}-{args.sha[:8]}"
    out = Path(args.out_dir)
    write_csv(out / f"{stem}.csv", target + cargo)
    write_csv(out / f"{stem}.diff.csv", added + cargo)


def cmd_source(args: argparse.Namespace) -> None:
    cargo = cargo_packages(Path(args.manifest), list(LINUX_TARGETS.values()))
    python = python_packages(Path(args.pip_report), args.wheel_name.lower())
    if not cargo:
        fail("cargo tree returned no dependencies")
    if not python:
        fail(f"no packages in the pip report {args.pip_report}")
    write_csv(Path(args.out_dir) / f"osrb-modelexpress-deps-{args.sha[:8]}.csv", cargo + python)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("container")
    c.add_argument("--target-dir", required=True)
    c.add_argument("--base-dir", required=True)
    c.add_argument("--arch", required=True, choices=["amd64", "arm64"])
    s = sub.add_parser("source")
    s.add_argument("--pip-report", required=True)
    s.add_argument("--wheel-name", default="modelexpress")
    for p in (c, s):
        p.add_argument("--manifest", default="Cargo.toml")
        p.add_argument("--sha", required=True)
        p.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        fail(f"--sha must be a 40-hex commit SHA, got {args.sha!r}")
    {"container": cmd_container, "source": cmd_source}[args.cmd](args)


if __name__ == "__main__":
    main()
