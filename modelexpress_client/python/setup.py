# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build script for ModelExpress C++ extensions.

Project metadata lives in pyproject.toml. This setup.py only adds the
ext_modules entries that pyproject.toml can't express. Setuptools merges
the two when both are present.

Pure setuptools build, no CUDA or PyTorch dependency in the extensions:
they use only the Python C API, plus libzstd for the delta checker, so they
build anywhere with a C++ toolchain.

Build-time failure of a C extension is non-fatal: pip install succeeds
without it, and the runtime detects the missing extension. Without
modelexpress.vmm._alloc_ext the loader takes the pool-reg path instead of
the arena allocator; without modelexpress_rl.inference._delta_ext refit
checks deltas in Python instead.
"""

import os
import subprocess
import sys
import warnings

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError


_OPTIONAL_EXT_ERRORS = (
    CCompilerError,
    DistutilsExecError,
    DistutilsPlatformError,
    FileNotFoundError,
    PermissionError,
)


class BuildExtension(build_ext):
    """Custom build extension that respects $CXX and is best-effort."""

    def build_extensions(self):
        cxx = os.environ.get("CXX", "g++")
        self.compiler.set_executable("compiler_so", cxx)
        self.compiler.set_executable("compiler_cxx", cxx)
        # macOS extensions resolve Python's symbols when the interpreter loads them.
        lookup = " -undefined dynamic_lookup" if sys.platform == "darwin" else ""
        self.compiler.set_executable("linker_so", f"{cxx} -shared{lookup}")
        try:
            build_ext.build_extensions(self)
        except _OPTIONAL_EXT_ERRORS as e:
            for ext in self.extensions:
                self._warn_optional_skip(e, ext_name=ext.name)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except _OPTIONAL_EXT_ERRORS as e:
            self._warn_optional_skip(e, ext_name=ext.name)

    @staticmethod
    def _warn_optional_skip(exc, ext_name):
        msg = (
            f"Failed to build optional C extension {ext_name}: "
            f"{type(exc).__name__}: {exc}. "
            f"ModelExpress will install without it; {_SKIPPED.get(ext_name, '')}"
        )
        warnings.warn(msg, stacklevel=2)
        print(f"WARNING: {msg}", file=sys.stderr)


# What each optional extension's absence costs, and how to build it.
_SKIPPED = {
    "modelexpress.vmm._alloc_ext": (
        "MX_VMM_ARENA will be a runtime no-op. Install a C++ compiler (g++ or "
        "set $CXX) and reinstall to enable the arena allocator fast path."
    ),
    "modelexpress_rl.inference._delta_ext": (
        "refit will check deltas in Python. Install a C++ compiler and the "
        "zstd development files (libzstd-dev or libzstd-devel) and reinstall "
        "to check them natively."
    ),
}


def _zstd_dirs():
    """Include and library directories of libzstd, when pkg-config knows them."""
    try:
        flags = subprocess.run(
            ["pkg-config", "--cflags-only-I", "--libs-only-L", "libzstd"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {
        "include_dirs": [flag[2:] for flag in flags if flag.startswith("-I")],
        "library_dirs": [flag[2:] for flag in flags if flag.startswith("-L")],
    }


def _ext_modules():
    if os.environ.get("MX_SKIP_EXT"):
        return []
    extra_compile_args = ["-std=c++17", "-O3", "-fPIC"]
    return [
        Extension(
            name="modelexpress.vmm._alloc_ext",
            sources=["modelexpress/vmm/_alloc_ext.cpp"],
            extra_compile_args=extra_compile_args,
            optional=True,
        ),
        Extension(
            name="modelexpress_rl.inference._delta_ext",
            sources=["modelexpress_rl/inference/_delta_ext.cpp"],
            extra_compile_args=[*extra_compile_args, "-pthread"],
            extra_link_args=["-pthread"],
            libraries=["zstd"],
            optional=True,
            **_zstd_dirs(),
        ),
    ]


setup(
    ext_modules=_ext_modules(),
    cmdclass={"build_ext": BuildExtension},
)
