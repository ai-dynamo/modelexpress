# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build script for ModelExpress C++ extensions.

Project metadata lives in pyproject.toml. This setup.py only adds the
ext_modules entry that pyproject.toml can't express. Setuptools merges
the two when both are present.

Pure setuptools build. The VMM shim only uses the Python C API. The optional
NCCL bootstrap shim includes ``nccl.h`` so ``ncclConfig_t`` always matches the
NCCL installation used at runtime.

Build-time failure of an optional extension is non-fatal. VMM detects a
missing allocator shim and falls back to pool registration. NCCL M2N bootstrap
detects a missing bootstrap shim and fails closed; it never falls back to
blocking communicator initialization.
"""

import os
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
    """Custom build extension that respects $CXX and is best-effort.

    ModelExpress C extensions are OPTIONAL. If
    compilation fails for any reason (no compiler installed, header
    mismatch, sandbox restrictions), this build_ext catches the error
    and emits a warning instead of aborting the install. The resulting
    wheel is pure-Python; the runtime detects the missing .so and falls
    back gracefully.
    """

    def build_extensions(self):
        cxx = os.environ.get("CXX", "g++")
        self.compiler.set_executable("compiler_so", cxx)
        self.compiler.set_executable("compiler_cxx", cxx)
        self.compiler.set_executable("linker_so", f"{cxx} -shared")
        try:
            build_ext.build_extensions(self)
        except _OPTIONAL_EXT_ERRORS as e:
            self._warn_optional_skip(e)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except _OPTIONAL_EXT_ERRORS as e:
            self._warn_optional_skip(e, ext_name=ext.name)

    @staticmethod
    def _warn_optional_skip(exc, ext_name="ModelExpress extension"):
        msg = (
            f"Failed to build optional C extension {ext_name}: "
            f"{type(exc).__name__}: {exc}. "
            "ModelExpress will install without this native feature. Install "
            "its headers and a C++ compiler (g++ or set $CXX), then reinstall."
        )
        warnings.warn(msg, stacklevel=2)
        print(f"WARNING: {msg}", file=sys.stderr)


def _ext_modules():
    if os.environ.get("MX_SKIP_EXT"):
        return []
    extra_compile_args = ["-std=c++17", "-O3", "-fPIC"]
    extensions = [
        Extension(
            name="modelexpress.vmm._alloc_ext",
            sources=["modelexpress/vmm/_alloc_ext.cpp"],
            extra_compile_args=extra_compile_args,
            optional=True,
        ),
    ]

    if sys.platform.startswith("linux"):
        include_dirs = []
        explicit_include = os.environ.get("NCCL_INCLUDE_DIR")
        nccl_home = os.environ.get("NCCL_HOME") or os.environ.get("NCCL_ROOT")
        if explicit_include:
            include_dirs.append(explicit_include)
        if nccl_home:
            include_dirs.append(os.path.join(nccl_home, "include"))
        extensions.append(
            Extension(
                name=(
                    "modelexpress.weight_transfer.transport._nccl_bootstrap_ext"
                ),
                sources=[
                    "modelexpress/weight_transfer/transport/_nccl_bootstrap_ext.cpp"
                ],
                include_dirs=include_dirs,
                libraries=["dl"],
                extra_compile_args=extra_compile_args,
                optional=True,
            )
        )
    return extensions


setup(
    ext_modules=_ext_modules(),
    cmdclass={"build_ext": BuildExtension},
)
