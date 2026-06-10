#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the Python section of NOTICES and the pypi OSRB CSV.

Enumerates the distributions installed in the current environment and emits
their licenses. The intended usage is to create a clean virtualenv, install
ONLY the wheel's runtime dependencies into it, then run this script with that
environment's interpreter -- every installed distribution is then a runtime
dependency (direct or transitive) and is in scope.

Usage:
    python -m venv /tmp/runtime-env
    /tmp/runtime-env/bin/pip install <built-wheel>      # pulls runtime deps
    /tmp/runtime-env/bin/python generate_pypi_attributions.py \
        --md-output NOTICES_Python.md --csv-output pypi_deps.csv

The dependency closure -- and therefore this output -- is platform specific:
torch pulls different nvidia-* CUDA wheels on amd64 vs arm64, so run this once
per target architecture.

Outputs:
    NOTICES_Python.md - Full license texts for the Python runtime dependencies.
    pypi_deps.csv     - OSRB rows: type,name,version,spdx_license (type=pypi).
"""

import argparse
import csv
import re
from importlib import metadata

OSRB_HEADER = ["type", "name", "version", "spdx_license"]

# First-party package plus environment bootstrap tools, excluded by default.
DEFAULT_EXCLUDES = {"modelexpress", "pip", "setuptools", "wheel"}

# Trove classifier -> SPDX, for distributions that predate License-Expression
# (PEP 639) and only declare an OSI-approved classifier.
CLASSIFIER_TO_SPDX = {
    "Apache Software License": "Apache-2.0",
    "MIT License": "MIT",
    "MIT No Attribution License (MIT-0)": "MIT-0",
    "BSD License": "BSD-3-Clause",
    "ISC License (ISCL)": "ISC",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "GNU General Public License v2 (GPLv2)": "GPL-2.0-only",
    "GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "GNU Lesser General Public License v2 (LGPLv2)": "LGPL-2.0-only",
    "GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.0-or-later",
    "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "Python Software Foundation License": "PSF-2.0",
    "The Unlicense (Unlicense)": "Unlicense",
    "zlib/libpng License": "Zlib",
    "Historical Permission Notice and Disclaimer (HPND)": "HPND",
}

# Common legacy free-text `License:` field values -> SPDX. Applied only when a
# distribution declares neither License-Expression nor a License classifier.
LEGACY_TO_SPDX = {
    "bsd": "BSD-3-Clause",
    "bsd license": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "3-clause bsd license": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "2-clause bsd license": "BSD-2-Clause",
    "mit": "MIT",
    "mit license": "MIT",
    "apache": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "isc": "ISC",
    "isc license": "ISC",
    "mpl 2.0": "MPL-2.0",
    "mpl-2.0": "MPL-2.0",
    "psf": "PSF-2.0",
}

LICENSE_FILE_RE = re.compile(r"(LICEN[CS]E|COPYING|NOTICE)", re.IGNORECASE)


def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


# Proprietary/unknown markers that, on an NVIDIA component, all mean the same
# thing. The CUDA wheels declare this inconsistently across metadata fields.
_NVIDIA_PROPRIETARY_MARKERS = {
    "other/proprietary license",
    "nvidia proprietary software",
    "licenseref-nvidia-proprietary",
    "licenseref-nvidia-software-license",
    "licenseref-nvidia-software",
    "proprietary",
    "unknown",
    "",
}


def canonicalize(norm_name, raw):
    """Normalize messy proprietary/unknown license strings to stable labels."""
    low = raw.strip().lower()
    is_nvidia = (
        norm_name.startswith("nvidia-")
        or norm_name.startswith("cuda-")
        or "nvshmem" in norm_name
    )
    if is_nvidia and low in _NVIDIA_PROPRIETARY_MARKERS:
        return "LicenseRef-NVIDIA-Proprietary"
    if low == "other/proprietary license":
        return "LicenseRef-Proprietary"
    if low == "unknown":
        return "Unknown"
    return raw


def _raw_license(md):
    """Resolve a license string from metadata, before canonicalization."""
    expr = md.get("License-Expression")
    if expr:
        return expr.strip()

    spdx_from_classifiers = []
    for classifier in md.get_all("Classifier") or []:
        if not classifier.startswith("License ::"):
            continue
        leaf = classifier.split("::")[-1].strip()
        spdx_from_classifiers.append(CLASSIFIER_TO_SPDX.get(leaf, leaf))
    if spdx_from_classifiers:
        return " AND ".join(dict.fromkeys(spdx_from_classifiers))

    # Legacy free-text License field: keep only if it is a short identifier,
    # not a dumped full license body.
    legacy = (md.get("License") or "").strip()
    if legacy and "\n" not in legacy and len(legacy) <= 64:
        return LEGACY_TO_SPDX.get(legacy.lower(), legacy)

    return "Unknown"


def spdx_license(dist):
    """Best-effort, canonicalized SPDX identifier for a distribution."""
    md = dist.metadata
    return canonicalize(normalize_name(md["Name"]), _raw_license(md))


def license_texts(dist):
    """Return [(filename, text)] for the distribution's bundled license files."""
    declared = {fn.strip() for fn in (dist.metadata.get_all("License-File") or [])}

    candidates = []
    for path in dist.files or []:
        parts = path.parts
        if not any(p.endswith(".dist-info") for p in parts):
            continue
        if path.name in declared or LICENSE_FILE_RE.search(path.name):
            candidates.append(path)

    seen = set()
    texts = []
    for path in candidates:
        if path.name in seen:
            continue
        seen.add(path.name)
        try:
            # read_text resolves relative to the .dist-info directory.
            rel = "/".join(path.parts[path.parts.index(next(
                p for p in path.parts if p.endswith(".dist-info"))) + 1:])
            text = dist.read_text(rel)
        except (OSError, StopIteration, ValueError):
            text = None
        if text:
            texts.append((path.name, text.replace("\r\n", "\n").rstrip()))
    return texts


HEADER = """\
<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Software Attributions (Python)

This file lists the Python runtime dependencies of the ModelExpress client
wheel, resolved for a single target platform. Each library is licensed under
the terms indicated below.

This file is automatically generated. Please do not edit it directly.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate Python NOTICES section and OSRB CSV from the "
        "installed environment."
    )
    parser.add_argument(
        "--md-output", default="NOTICES_Python.md", help="Markdown output path."
    )
    parser.add_argument(
        "--csv-output", default="pypi_deps.csv", help="OSRB CSV output path."
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional distribution name to exclude (repeatable).",
    )
    args = parser.parse_args()

    excludes = {normalize_name(n) for n in DEFAULT_EXCLUDES}
    excludes.update(normalize_name(n) for n in args.exclude)

    dists = []
    seen = set()
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        key = normalize_name(name)
        if key in excludes or key in seen:
            continue
        seen.add(key)
        dists.append(dist)

    dists.sort(key=lambda d: normalize_name(d.metadata["Name"]))

    output = [HEADER]
    rows = []
    for dist in dists:
        name = dist.metadata["Name"]
        version = dist.version
        lic = spdx_license(dist)
        rows.append(["pypi", name, version, lic])

        repo = dist.metadata.get("Home-page") or ""
        for key in ("Project-URL",):
            for entry in dist.metadata.get_all(key) or []:
                if not repo and entry:
                    repo = entry.split(",", 1)[-1].strip()

        output.append(f"## {name} - {version}")
        if repo:
            output.append(f"**Project URL**: {repo}")
        output.append(f"**License Type(s)**: {lic}")
        for filename, text in license_texts(dist):
            output.append(f"### License: {filename}")
            output.append(f"```\n{text}\n```")
            output.append("")
        output.append("")

    with open(args.md_output, "w") as f:
        f.write("\n".join(output).rstrip())
        f.write("\n")

    with open(args.csv_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OSRB_HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
