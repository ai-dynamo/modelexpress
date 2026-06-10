#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract the dpkg packages a container adds on top of its base image to CSV.

Usage:
    python3 calculate_dpkg_deps.py [--baseline base_packages.txt] [output_path]

Run inside the final runtime image. dpkg-query enumerates every installed
package (transitive dependencies included, not just the apt-get install list).
When --baseline lists the base image's packages (one name per line), those are
subtracted so the output is only what this image adds.

To avoid counting tooling (e.g. a python3 installed solely to run this script),
capture the package set with `dpkg-query -W -f '${Package}\t${Version}\n'`
BEFORE installing that tooling and pass it via --installed; copyright files for
those packages remain readable from the live image.

No NOTICES/license-text file is produced for dpkg packages: their copyright
files live at the well-known /usr/share/doc/<pkg>/copyright path inside the
image, which is sufficient for OSRB.

Output:
    dpkg_deps.csv - OSRB rows: type,name,version,spdx_license (type=dpkg).

The spdx_license column is best-effort: it is parsed from the Debian copyright
file's `License:` stanzas, which are not always valid SPDX. Review and
normalize before submission.
"""

import argparse
import csv
import os
import re
import subprocess

OSRB_HEADER = ["type", "name", "version", "spdx_license"]


def parse_package_tsv(text):
    """Parse `${Package}\t${Version}` lines into (name, version) tuples."""
    packages = []
    for line in text.strip().split("\n"):
        if "\t" in line:
            name, version = line.split("\t", 1)
            packages.append((name, version))
    return packages


def get_installed_packages(installed_path=None):
    """Installed packages with versions, from a snapshot file or live dpkg."""
    if installed_path:
        with open(installed_path) as f:
            return parse_package_tsv(f.read())
    result = subprocess.run(
        ["dpkg-query", "-W", "-f", "${Package}\t${Version}\n"],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_package_tsv(result.stdout)


def load_baseline(path):
    """Load a baseline package list (one package name per line)."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def extract_license(package_name):
    """Best-effort license identifier(s) from a package's copyright file."""
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
        "Only packages not in this list are included.",
    )
    parser.add_argument(
        "--installed",
        help="Path to a pre-captured `${Package}\\t${Version}` snapshot of the "
        "runtime image, taken before any extraction tooling was installed. "
        "Defaults to a live dpkg-query.",
    )
    parser.add_argument(
        "output", nargs="?", default="dpkg_deps.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    packages = get_installed_packages(args.installed)

    if args.baseline:
        baseline = load_baseline(args.baseline)
        packages = [(name, ver) for name, ver in packages if name not in baseline]

    packages.sort(key=lambda p: p[0].lower())

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OSRB_HEADER)
        for name, version in packages:
            writer.writerow(["dpkg", name, version, extract_license(name)])


if __name__ == "__main__":
    main()
