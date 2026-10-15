#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the OSRB dependency CSVs consumed by release-automation's nvbug-attach-*.py."""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import diff_osrb_csv  # noqa: E402

FIELDS = ["ecosystem", "name", "version", "spdx", "source_url", "notes"]
PUBLISHED_CRATES = ["modelexpress-common", "modelexpress-client", "modelexpress-server"]
LINUX_TARGETS = {"amd64": "x86_64-unknown-linux-gnu", "arm64": "aarch64-unknown-linux-gnu"}
CARGO_PKG_RE = re.compile(r"^(?P<name>\S+) v(?P<version>\S+)(?: \(proc-macro\))?(?: \((?P<source>[^)]*)\))?$")
OVERRIDES_FILE = Path(__file__).resolve().parent / "license_overrides.toml"
LICENSE_TEXT = {
    "mit license": "MIT",
    "apache 2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "3-clause bsd license": "BSD-3-Clause",
}
LICENSE_CLASSIFIERS = {
    "MIT License": "MIT",
    "Apache Software License": "Apache-2.0",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "ISC License (ISCL)": "ISC",
    "The Unlicense (Unlicense)": "Unlicense",
}


def fail(msg: str) -> None:
    print(f"::error::{msg}", file=sys.stderr)
    sys.exit(1)


def canonical_spdx(expr: str) -> str | None:
    from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression

    try:
        return canonicalize_license_expression(expr) if expr and expr.strip() else None
    except InvalidLicenseExpression:
        return None


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_overrides() -> dict:
    data = tomllib.loads(OVERRIDES_FILE.read_text())
    out = {}
    for entry in data.get("override", []):
        if not entry.get("source"):
            fail(f"{OVERRIDES_FILE.name}: {entry.get('name')} has no source")
        if canonical_spdx(entry["license"]) is None:
            fail(f"{OVERRIDES_FILE.name}: {entry['name']} has an invalid license {entry['license']!r}")
        if entry["ecosystem"] not in ("dpkg", "python", "rust"):
            fail(f"{OVERRIDES_FILE.name}: {entry['name']} has an unknown ecosystem {entry['ecosystem']!r}")
        key = (entry["ecosystem"], canonical_name(entry["name"]))
        if key in out:
            fail(f"{OVERRIDES_FILE.name}: {entry['name']} is listed twice for {entry['ecosystem']}")
        out[key] = entry
    return out


OVERRIDES: dict = {}


def resolve(ecosystem: str, name: str, declared: list[str], fallback: list[str]) -> tuple[str, str]:
    """Declared SPDX, else the override, else a recognised free-text value, else UNKNOWN. Returns (spdx, notes)."""
    for value in declared:
        spdx = canonical_spdx(value)
        if spdx:
            return spdx, ""
    entry = OVERRIDES.get((ecosystem, canonical_name(name)))
    if entry:
        return canonical_spdx(entry["license"]), f"license from license_overrides.toml: {entry['source']}"
    for value in fallback:
        spdx = canonical_spdx(value)
        if spdx:
            return spdx, ""
    return "UNKNOWN", ""


def row(ecosystem: str, name: str, version: str, declared: list[str], fallback: list[str], source_url: str = "") -> dict:
    spdx, notes = resolve(ecosystem, name, declared, fallback)
    return dict(zip(FIELDS, [ecosystem, name, version, spdx, source_url, notes]))


def read_tsv(path: Path, ecosystem: str) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            name = canonical_name(parts[0]) if ecosystem == "python" else parts[0]
            url = f"https://pypi.org/project/{name}/{parts[1]}/" if ecosystem == "python" else ""
            # The helpers' values are heuristic, so an override wins for image rows.
            rows.append(row(ecosystem, name, parts[1], [], [parts[2]], url))
    return rows


def image_packages(directory: Path) -> list[dict]:
    return read_tsv(directory / "dpkg.tsv", "dpkg") + read_tsv(directory / "python.tsv", "python")


def cargo_license(raw: str) -> str:
    lic = raw.strip()
    if "/" in lic and " OR " not in lic and " AND " not in lic:
        lic = " OR ".join(part.strip() for part in lic.split("/"))
    return lic


def cargo_packages(manifest: Path, targets: list[str], edges: str, all_features: bool) -> list[dict]:
    cmd = ["cargo", "tree", "--locked", "--manifest-path", str(manifest),
           "-e", edges, "--prefix", "none", "--format", "{p}\t{l}"]
    if all_features:
        cmd.append("--all-features")
    for crate in PUBLISHED_CRATES:
        cmd += ["-p", crate]
    for target in targets:
        cmd += ["--target", target]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        line = line.rstrip().removesuffix(" (*)")
        if not line:
            continue
        pkg, _, lic = line.partition("\t")
        m = CARGO_PKG_RE.match(pkg.strip())
        if not m:
            fail(f"unparseable cargo tree line: {line!r}")
        if (m.group("source") or "").startswith("/"):
            continue  # workspace member
        name, version = m.group("name"), m.group("version")
        rows[(name, version)] = row("rust", name, version, [cargo_license(lic)], [], f"pkg:cargo/{name}@{version}")
    return list(rows.values())


def python_fallback(meta: dict) -> list[str]:
    out = []
    lic = (meta.get("license") or "").strip()
    if lic and "\n" not in lic and len(lic) <= 100 and lic.upper() != "UNKNOWN":
        out += [lic, LICENSE_TEXT.get(lic.lower(), "")]
    classifiers = [c.split(" :: ")[-1] for c in meta.get("classifier", []) if c.startswith("License ::")]
    if len(classifiers) == 1:
        out.append(LICENSE_CLASSIFIERS.get(classifiers[0], ""))
    return out


def python_packages(reports: list[Path], own_name: str) -> list[dict]:
    rows = {}
    for report in reports:
        for item in json.loads(report.read_text()).get("install", []):
            meta = item.get("metadata", {})
            name, version = canonical_name(meta["name"]), meta["version"]
            if name == canonical_name(own_name):
                continue
            new = row("python", name, version, [meta.get("license_expression") or ""], python_fallback(meta),
                      f"https://pypi.org/project/{name}/{version}/")
            key = (name, version)
            if key in rows and rows[key]["spdx"] != new["spdx"]:
                fail(f"{name} {version} resolves to different licenses across pip reports")
            rows[key] = new
    return list(rows.values())


def check_licenses(rows: list[dict]) -> None:
    bad = [r for r in rows if r["spdx"] == "UNKNOWN"]
    if bad:
        listing = ", ".join(f"{r['ecosystem']}:{r['name']} {r['version']}" for r in bad)
        fail(f"no valid SPDX license for {len(bad)} package(s): {listing}. "
             f"Add an entry with a source to {OVERRIDES_FILE.name}.")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["ecosystem"], r["name"].lower(), r["version"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}", file=sys.stderr)


def find_baseline_csv(info: dict, pattern: str) -> tuple[Path | None, str]:
    if not info.get("available"):
        return None, info.get("reason") or "no baseline lookup was run"
    root = Path(info["dir"])
    found = [p for p in root.glob(pattern)
             if not p.name.endswith(".diff.csv") and "baseline" not in p.relative_to(root).parts[:-1]]
    if len(found) != 1:
        return None, f"artifact {info['artifact']} has {len(found)} files matching {pattern}"
    with open(found[0], newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header[:4] != FIELDS[:4]:
        return None, f"artifact {info['artifact']} uses an older CSV layout"
    return found[0], ""


def write_diff(out_dir: Path, stem: str, rows: list[dict], baseline: str, pattern: str, copy_baseline: bool) -> None:
    info = json.loads(Path(baseline).read_text()) if baseline else {}
    base_csv, reason = find_baseline_csv(info, pattern)
    if base_csv:
        label = f"scheduled nightly run {info['run_id']} at {info['head_sha'][:8]} ({info['created_at']})"
        diff_rows = diff_osrb_csv.compute_diff(rows, diff_osrb_csv.read_osrb_csv(base_csv))
    else:
        label = reason
        diff_rows = diff_osrb_csv.baseline_unavailable_rows(reason)
    diff_osrb_csv.write_diff_csv(diff_rows, out_dir / f"{stem}.diff.csv")
    print(f"wrote {len(diff_rows)} diff rows against: {label}", file=sys.stderr)

    repo = info.get("repo", "ai-dynamo/modelexpress")
    lines = [
        "# OSRB diff baseline",
        "",
        "The `*.diff.csv` next to this folder compares this build's OSRB",
        "dependency CSV against the baseline described below.",
        "",
        "## How the baseline is chosen",
        "- The newest earlier scheduled `nightly-ci.yml` run on `main` that still has this artifact",
        "",
        "## This build's baseline",
        f"- Label: {label}",
    ]
    if base_csv:
        lines += [
            f"- Baseline commit SHA: `{info['head_sha']}`",
            f"- Commit: https://github.com/{repo}/commit/{info['head_sha']}",
            f"- Generating workflow run: {info['run_url']}",
        ]
        if copy_baseline:
            lines.append(f"- Baseline CSV (copied into this folder): `{base_csv.name}`")
        else:
            lines.append(f"- Baseline CSV: `{base_csv.name}` in artifact `{info['artifact']}`")
    else:
        lines += ["- Baseline commit SHA: (none — no baseline was available)",
                  f"- Baseline CSV: unavailable ({reason})"]
    folder = out_dir / "baseline"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "BASELINE.md").write_text("\n".join(lines) + "\n")
    if base_csv and copy_baseline:
        shutil.copy2(base_csv, folder / base_csv.name)


def check_arch(directory: Path, arch: str) -> None:
    arch_file = directory / "arch.txt"
    found = arch_file.read_text().strip() if arch_file.is_file() else ""
    if found != arch:
        fail(f"{directory}: image arch is {found or 'unknown'}, expected {arch}")


def cmd_container(args: argparse.Namespace) -> None:
    target_dir, base_dir = Path(args.target_dir), Path(args.base_dir)
    check_arch(target_dir, args.arch)
    check_arch(base_dir, args.arch)
    target = image_packages(target_dir)
    base = image_packages(base_dir)
    if not target:
        fail(f"no packages extracted from the image in {target_dir}")
    if not base:
        fail(f"no packages extracted from the base image in {base_dir}")
    base_versions: dict = {}
    for r in base:
        base_versions.setdefault((r["ecosystem"], r["name"]), set()).add(r["version"])
    added = [r for r in target if r["version"] not in base_versions.get((r["ecosystem"], r["name"]), set())]
    cargo = cargo_packages(Path(args.manifest), [LINUX_TARGETS[args.arch]], "normal", all_features=False)
    if not cargo:
        fail("cargo tree returned no dependencies")
    rows = added + cargo
    check_licenses(rows)
    stem = f"osrb-modelexpress-server-{args.arch}-{args.sha[:8]}"
    out = Path(args.out_dir)
    write_csv(out / f"{stem}.csv", rows)
    write_diff(out, stem, rows, args.baseline, f"linux_{args.arch}/osrb-modelexpress-server-{args.arch}-*.csv",
               copy_baseline=True)


def cmd_source(args: argparse.Namespace) -> None:
    cargo = cargo_packages(Path(args.manifest), list(LINUX_TARGETS.values()), "normal,build", all_features=True)
    python = python_packages([Path(p) for p in args.pip_report], args.wheel_name)
    if not cargo:
        fail("cargo tree returned no dependencies")
    if not python:
        fail(f"no packages in the pip reports {args.pip_report}")
    rows = cargo + python
    check_licenses(rows)
    stem = f"osrb-modelexpress-deps-{args.sha[:8]}"
    out = Path(args.out_dir)
    write_csv(out / f"{stem}.csv", rows)
    write_diff(out, stem, rows, args.baseline, "osrb-modelexpress-deps-*.csv", copy_baseline=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("container")
    c.add_argument("--target-dir", required=True)
    c.add_argument("--base-dir", required=True)
    c.add_argument("--arch", required=True, choices=sorted(LINUX_TARGETS))
    s = sub.add_parser("source")
    s.add_argument("--pip-report", required=True, action="append")
    s.add_argument("--wheel-name", default="modelexpress")
    for p in (c, s):
        p.add_argument("--manifest", default="Cargo.toml")
        p.add_argument("--sha", required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--baseline", default="", help="JSON written by baseline.py")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        fail(f"--sha must be a 40-hex commit SHA, got {args.sha!r}")
    OVERRIDES.update(load_overrides())
    {"container": cmd_container, "source": cmd_source}[args.cmd](args)


if __name__ == "__main__":
    main()
