# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build script for ModelExpress C++ extensions.

Project metadata lives in pyproject.toml. This setup.py only adds the
ext_modules entry that pyproject.toml can't express. Setuptools merges
the two when both are present.

Pattern follows dynamo's GMS setup.py - pure setuptools build, no CUDA
or PyTorch dependency in the extension itself (the shim only uses the
Python C API and dispatches to Python callbacks).

Build-time failure of the C extension is non-fatal: pip install
succeeds without it, and modelexpress.vmm.hook detects the missing
extension at runtime and disables the arena allocator. Users without a
working C++ compiler get the pool-reg path; users with one get the
arena+pool-reg fast path.
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

    The modelexpress.vmm._alloc_ext C extension is OPTIONAL. If
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
    def _warn_optional_skip(exc, ext_name="modelexpress.vmm._alloc_ext"):
        msg = (
            f"Failed to build optional C extension {ext_name}: "
            f"{type(exc).__name__}: {exc}. "
            "ModelExpress will install pure-Python; MX_VMM_ARENA will be a "
            "runtime no-op. Install a C++ compiler (g++ or set $CXX) and "
            "reinstall to enable the arena allocator fast path."
        )
        warnings.warn(msg, stacklevel=2)
        print(f"WARNING: {msg}", file=sys.stderr)


def _ext_modules():
    extra_compile_args = ["-std=c++17", "-O3", "-fPIC"]
    return [
        Extension(
            name="modelexpress.vmm._alloc_ext",
            sources=["modelexpress/vmm/_alloc_ext.cpp"],
            extra_compile_args=extra_compile_args,
            optional=True,
        ),
    ]


setup(
    ext_modules=_ext_modules(),
    cmdclass={"build_ext": BuildExtension},
)
