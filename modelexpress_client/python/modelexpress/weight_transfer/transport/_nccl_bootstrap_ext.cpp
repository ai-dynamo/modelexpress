// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Small ABI-safe bridge for NCCL's versioned ncclConfig_t. Each call resolves
// symbols against the exact ctypes-loaded libnccl handle supplied by Python.

#define PY_SSIZE_T_CLEAN
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <Python.h>

#include <dlfcn.h>
#include <nccl.h>

#include <cstdint>
#include <cstring>

namespace {

template <typename Function>
Function resolve(void* library_handle, const char* name) {
  if (library_handle == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "NCCL library handle is null");
    return nullptr;
  }
  dlerror();
  auto* symbol = dlsym(library_handle, name);
  const char* error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    PyErr_Format(PyExc_RuntimeError, "required NCCL symbol %s is unavailable: %s", name,
                 error == nullptr ? "unknown error" : error);
    return nullptr;
  }
  return reinterpret_cast<Function>(symbol);
}
bool validate_nccl_identity(void* nccl_handle, void* m2n_handle) {
  if (nccl_handle == nullptr || m2n_handle == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "NCCL or M2N library handle is null");
    return false;
  }

  dlerror();
  void* selected_symbol = dlsym(nccl_handle, "ncclGetVersion");
  const char* selected_error = dlerror();
  dlerror();
  void* m2n_symbol = dlsym(m2n_handle, "ncclGetVersion");
  const char* m2n_error = dlerror();
  dlerror();
  void* process_symbol = dlsym(RTLD_DEFAULT, "ncclGetVersion");
  const char* process_error = dlerror();
  if (selected_error != nullptr || selected_symbol == nullptr ||
      m2n_error != nullptr || m2n_symbol == nullptr ||
      process_error != nullptr || process_symbol == nullptr) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "cannot verify selected, M2N dependency, and process-wide NCCL identity");
    return false;
  }

  Dl_info selected_info{};
  Dl_info m2n_info{};
  Dl_info process_info{};
  if (dladdr(selected_symbol, &selected_info) == 0 ||
      dladdr(m2n_symbol, &m2n_info) == 0 ||
      dladdr(process_symbol, &process_info) == 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "cannot resolve loaded NCCL library paths");
    return false;
  }
  if (selected_info.dli_fbase != m2n_info.dli_fbase ||
      selected_info.dli_fbase != process_info.dli_fbase) {
    PyErr_Format(
        PyExc_RuntimeError,
        "NCCL identity mismatch: selected=%s m2n=%s process=%s",
        selected_info.dli_fname, m2n_info.dli_fname, process_info.dli_fname);
    return false;
  }
  return true;
}

bool validate_nccl_version(void* nccl_handle, void* m2n_handle) {
  if (!validate_nccl_identity(nccl_handle, m2n_handle)) {
    return false;
  }
  using GetVersionFunction = ncclResult_t (*)(int*);
  auto get_version = resolve<GetVersionFunction>(nccl_handle, "ncclGetVersion");
  if (get_version == nullptr) {
    return false;
  }

  int runtime_version = 0;
  ncclResult_t result;
  Py_BEGIN_ALLOW_THREADS
  result = get_version(&runtime_version);
  Py_END_ALLOW_THREADS
  if (result != ncclSuccess) {
    PyErr_Format(PyExc_RuntimeError,
                 "ncclGetVersion failed with status %d",
                 static_cast<int>(result));
    return false;
  }
  if (runtime_version != NCCL_VERSION_CODE) {
    PyErr_Format(PyExc_RuntimeError,
                 "NCCL header/runtime version mismatch: compiled=%d runtime=%d",
                 NCCL_VERSION_CODE, runtime_version);
    return false;
  }
  return true;
}

PyObject* comm_init_rank_config(PyObject*, PyObject* args) {
  unsigned long long nccl_handle_value = 0;
  unsigned long long m2n_handle_value = 0;
  int nranks = 0;
  int rank = 0;
  PyObject* uid_object = nullptr;
  if (!PyArg_ParseTuple(args, "KKiO!i", &nccl_handle_value, &m2n_handle_value,
                        &nranks, &PyBytes_Type, &uid_object, &rank)) {
    return nullptr;
  }
  if (nranks <= 0 || rank < 0 || rank >= nranks) {
    PyErr_SetString(PyExc_ValueError, "invalid NCCL world size or rank");
    return nullptr;
  }

  char* uid_bytes = nullptr;
  Py_ssize_t uid_size = 0;
  if (PyBytes_AsStringAndSize(uid_object, &uid_bytes, &uid_size) < 0) {
    return nullptr;
  }
  if (uid_size != static_cast<Py_ssize_t>(sizeof(ncclUniqueId))) {
    PyErr_Format(PyExc_ValueError, "NCCL unique ID must be %zu bytes", sizeof(ncclUniqueId));
    return nullptr;
  }

  using InitFunction = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);
  auto* nccl_handle =
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(nccl_handle_value));
  auto* m2n_handle =
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(m2n_handle_value));
  if (!validate_nccl_version(nccl_handle, m2n_handle)) {
    return nullptr;
  }
  auto init = resolve<InitFunction>(nccl_handle, "ncclCommInitRankConfig");
  if (init == nullptr) {
    return nullptr;
  }

  ncclUniqueId uid{};
  std::memcpy(&uid, uid_bytes, sizeof(uid));
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  ncclComm_t comm = nullptr;
  ncclResult_t result;
  Py_BEGIN_ALLOW_THREADS
  result = init(&comm, nranks, uid, rank, &config);
  Py_END_ALLOW_THREADS
  return Py_BuildValue("(Ki)", static_cast<unsigned long long>(
                                reinterpret_cast<std::uintptr_t>(comm)),
                       static_cast<int>(result));
}

PyObject* comm_get_async_error(PyObject*, PyObject* args) {
  unsigned long long library_handle_value = 0;
  unsigned long long comm_value = 0;
  if (!PyArg_ParseTuple(args, "KK", &library_handle_value, &comm_value)) {
    return nullptr;
  }
  auto comm = reinterpret_cast<ncclComm_t>(static_cast<std::uintptr_t>(comm_value));
  if (comm == nullptr) {
    PyErr_SetString(PyExc_ValueError, "NCCL communicator is null");
    return nullptr;
  }

  using AsyncErrorFunction = ncclResult_t (*)(ncclComm_t, ncclResult_t*);
  auto* library_handle = reinterpret_cast<void*>(static_cast<std::uintptr_t>(library_handle_value));
  auto get_async_error = resolve<AsyncErrorFunction>(library_handle, "ncclCommGetAsyncError");
  if (get_async_error == nullptr) {
    return nullptr;
  }
  ncclResult_t async_error = ncclInProgress;
  ncclResult_t result;
  Py_BEGIN_ALLOW_THREADS
  result = get_async_error(comm, &async_error);
  Py_END_ALLOW_THREADS
  return Py_BuildValue("(ii)", static_cast<int>(result), static_cast<int>(async_error));
}

PyObject* comm_abort(PyObject*, PyObject* args) {
  unsigned long long library_handle_value = 0;
  unsigned long long comm_value = 0;
  if (!PyArg_ParseTuple(args, "KK", &library_handle_value, &comm_value)) {
    return nullptr;
  }
  auto comm = reinterpret_cast<ncclComm_t>(static_cast<std::uintptr_t>(comm_value));
  if (comm == nullptr) {
    PyErr_SetString(PyExc_ValueError, "NCCL communicator is null");
    return nullptr;
  }

  using AbortFunction = ncclResult_t (*)(ncclComm_t);
  auto* library_handle = reinterpret_cast<void*>(static_cast<std::uintptr_t>(library_handle_value));
  auto abort = resolve<AbortFunction>(library_handle, "ncclCommAbort");
  if (abort == nullptr) {
    return nullptr;
  }
  ncclResult_t result;
  Py_BEGIN_ALLOW_THREADS
  result = abort(comm);
  Py_END_ALLOW_THREADS
  return PyLong_FromLong(static_cast<long>(result));
}

PyMethodDef methods[] = {
    {"comm_init_rank_config", comm_init_rank_config, METH_VARARGS,
     "Start nonblocking ncclCommInitRankConfig."},
    {"comm_get_async_error", comm_get_async_error, METH_VARARGS,
     "Return (call_status, async_status) for a communicator."},
    {"comm_abort", comm_abort, METH_VARARGS, "Abort a communicator."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_nccl_bootstrap_ext",
    "ABI-safe NCCL nonblocking bootstrap bridge.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__nccl_bootstrap_ext() { return PyModule_Create(&module); }
