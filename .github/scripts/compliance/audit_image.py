#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-check the container OSRB CSV against independent syft scans of the image and its base.

Fails when the image adds a package the CSV does not list, or a file that no
package owns and that is not one of ModelExpress's own files.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Files COPY'd from the build stage; the binaries' crates are in the CSV via cargo.
FIRST_PARTY_FILES = {
    "/app/NOTICES",
    "/app/fallback_test",
    "/app/modelexpress-cli",
    "/app/modelexpress-server",
    "/app/test_client",
}
# State written by apt, dpkg, debconf, ldconfig and update-ca-certificates.
GENERATED = re.compile(
    r"^(/var/lib/(dpkg|apt)/|/var/cache/(debconf|ldconfig)/|/var/log/"
    r"|/etc/ld\.so\.cache$|/etc/ca-certificates\.conf$|/etc/ssl/certs/)"
)


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load(path: Path) -> tuple[set, dict, set]:
    doc = json.loads(path.read_text())
    packages = {(a["type"], a["name"], a["version"]) for a in doc["artifacts"]}
    paths = {f["id"]: f["location"]["path"] for f in doc.get("files", [])}
    files = {
        f["location"]["path"]: next((d["value"] for d in f.get("digests") or []), "")
        for f in doc.get("files", [])
        if f.get("metadata", {}).get("type") == "RegularFile"
    }
    owned = {paths[r["child"]] for r in doc["artifactRelationships"]
             if r["type"] == "contains" and r["child"] in paths}
    return packages, files, owned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-sbom", required=True, help="syft-json of the image, with every file cataloged")
    parser.add_argument("--base-sbom", required=True, help="syft-json of the base image, cataloged the same way")
    parser.add_argument("--csv", required=True, help="the image's OSRB CSV from osrb.py container")
    args = parser.parse_args()

    packages, files, owned = load(Path(args.image_sbom))
    base_packages, base_files, _ = load(Path(args.base_sbom))
    if not packages or not files or not base_files:
        print("::error::a syft scan came back empty", file=sys.stderr)
        sys.exit(1)
    with open(args.csv, newline="", encoding="utf-8") as f:
        listed = {canonical_name(r["name"]) for r in csv.DictReader(f)}

    findings = [f"{kind} package {name} {version} is not in the CSV"
                for kind, name, version in sorted(packages - base_packages)
                if canonical_name(name) not in listed]
    findings += [f"{path} is not owned by any package"
                 for path, digest in sorted(files.items())
                 if base_files.get(path) != digest and path not in owned
                 and path not in FIRST_PARTY_FILES and not GENERATED.match(path)]

    print(f"{args.csv}: {len(packages - base_packages)} packages above the base, {len(findings)} findings")
    for finding in findings:
        print(f"::error::{finding}")
    if findings:
        sys.exit(1)


if __name__ == "__main__":
    main()
