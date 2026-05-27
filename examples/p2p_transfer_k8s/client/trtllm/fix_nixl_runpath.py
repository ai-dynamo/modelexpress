#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the NIXL Python binding use TRT-LLM's system NIXL stack."""

from __future__ import annotations

import glob
import os
import site
import subprocess
import sys
import sysconfig


def _run(args: list[str]) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def _site_package_dirs() -> list[str]:
    candidates = []
    candidates.extend(site.getsitepackages())
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        candidates.append(purelib)
    candidates.extend(sys.path)

    dirs = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(candidate)
        if path in seen or not os.path.isdir(path):
            continue
        seen.add(path)
        dirs.append(path)
    return dirs


def main() -> None:
    nixl_lib_dir = os.environ["NIXL_LIB_DIR"]
    bindings = []
    for site_pkg in _site_package_dirs():
        bindings.extend(
            glob.glob(os.path.join(site_pkg, "nixl_cu13", "_bindings*.so"))
        )
    if len(bindings) != 1:
        raise RuntimeError(f"Expected one nixl_cu13 binding, found {bindings}")

    binding = bindings[0]
    rpath = "$ORIGIN/../.nixl_cu13.mesonpy.libs:$ORIGIN/../nixl_cu13.libs"
    _run(["patchelf", "--set-rpath", rpath, binding])

    dynamic = _run(["readelf", "-d", binding])
    if "RUNPATH" not in dynamic:
        raise RuntimeError(f"{binding} does not use DT_RUNPATH after patching")

    linked = _run(["ldd", binding])
    expected = f"{nixl_lib_dir}/libnixl.so"
    if expected not in linked:
        raise RuntimeError(f"{binding} does not resolve libnixl.so from {nixl_lib_dir}")


if __name__ == "__main__":
    main()
