#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract installed dpkg packages and their licenses to CSV.

Usage:
    python3 calculate_dpkg_deps.py [--baseline base_packages.txt] [output_path]

When --baseline is provided, only packages not in that file are included.
This allows capturing only packages installed on top of a base image.

Outputs:
    dpkg_deps.csv - Summary with type, name, version, license
"""

import argparse
import csv
import os
import re
import subprocess


def get_installed_packages():
    """Query dpkg for all installed packages with their versions."""
    result = subprocess.run(
        ["dpkg-query", "-W", "-f", "${Package}\t${Version}\n"],
        capture_output=True,
        text=True,
        check=True,
    )
    packages = []
    for line in result.stdout.strip().split("\n"):
        if "\t" in line:
            name, version = line.split("\t", 1)
            packages.append((name, version))
    return packages


def load_baseline(path):
    """Load a baseline package list (one package name per line)."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def extract_license(package_name):
    """Extract license identifier(s) from a package's copyright file."""
    copyright_path = f"/usr/share/doc/{package_name}/copyright"
    if not os.path.isfile(copyright_path):
        return "Unknown"

    try:
        with open(copyright_path) as f:
            content = f.read()
    except OSError:
        return "Unknown"

    licenses = set()
    for match in re.finditer(r"^License:\s*(.+)$", content, re.MULTILINE):
        lic = match.group(1).strip()
        if lic:
            licenses.add(lic)

    if licenses:
        return " / ".join(sorted(licenses))

    return "Unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Extract installed dpkg packages and their licenses to CSV."
    )
    parser.add_argument(
        "--baseline",
        help="Path to a file listing base image package names (one per line). "
        "Only packages not in this list will be included.",
    )
    parser.add_argument(
        "output", nargs="?", default="dpkg_deps.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    packages = get_installed_packages()

    if args.baseline:
        baseline = load_baseline(args.baseline)
        packages = [(name, ver) for name, ver in packages if name not in baseline]

    packages.sort(key=lambda p: p[0].lower())

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "name", "version", "license"])
        for name, version in packages:
            license_id = extract_license(name)
            writer.writerow(["dpkg", name, version, license_id])


if __name__ == "__main__":
    main()
