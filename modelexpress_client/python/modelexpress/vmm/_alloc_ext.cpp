// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CUDAPluggableAllocator shim for the VmmArena allocator.
//
// PyTorch calls mx_malloc / mx_free through CUDAPluggableAllocator. Those
// symbols cannot carry a Python object argument, so the module keeps one
// process-global "active arena" pointer. The arena state itself is not global:
// every Python VmmArena owns an ArenaState capsule with its own bump counter,
// allocation map, callbacks, and lifecycle state.
//
// The extension intentionally does not link against CUDA. cuMem* calls stay in
// Python backends so non-CUDA StubBackend tests exercise the same C hot path.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr const char* kArenaCapsuleName = "modelexpress.VmmArenaState";
constexpr int kStateOpen = 0;
constexpr int kStateClosed = 1;

struct AllocRecord {
  size_t size;
  uint64_t handle;
};

struct ArenaState {
  uintptr_t base;
  size_t total_bytes;
  size_t granularity;
  std::atomic<size_t> next_offset;
  std::atomic<size_t> mapped_bytes;
  std::atomic<int> state;
  // in_flight counts threads currently inside a malloc/free call that have
  // passed the OPEN check. close_and_drain CASes state -> CLOSED then spins
  // on in_flight to reach zero before snapshotting the map, so a malloc
  // that won the OPEN check cannot insert into a map the closer has already
  // cleared, and a free that has erased a record cannot decrement
  // mapped_bytes after the closer has reset it.
  std::atomic<int> in_flight;
  PyObject* alloc_callback;
  PyObject* dealloc_callback;
  std::mutex allocations_mutex;
  std::unordered_map<uintptr_t, AllocRecord> allocations;
};

// RAII guard for the in_flight counter. Increments on construction;
// decrements on destruction. Use at the top of every public hot-path
// function on a per-arena state.
class InFlightGuard {
 public:
  explicit InFlightGuard(ArenaState* s) : state_(s) {
    state_->in_flight.fetch_add(1, std::memory_order_seq_cst);
  }
  ~InFlightGuard() {
    state_->in_flight.fetch_sub(1, std::memory_order_seq_cst);
  }
  InFlightGuard(const InFlightGuard&) = delete;
  InFlightGuard& operator=(const InFlightGuard&) = delete;
 private:
  ArenaState* state_;
};

std::mutex g_active_mutex;
PyObject* g_active_capsule = nullptr;

ArenaState*
get_state(PyObject* capsule)
{
  return static_cast<ArenaState*>(PyCapsule_GetPointer(capsule, kArenaCapsuleName));
}

bool
is_power_of_two(size_t value)
{
  return value != 0 && (value & (value - 1)) == 0;
}

bool
round_up(size_t value, size_t granularity, size_t* out)
{
  if (granularity == 0) {
    return false;
  }
  size_t remainder = value % granularity;
  if (remainder == 0) {
    *out = value;
    return true;
  }
  size_t add = granularity - remainder;
  if (value > std::numeric_limits<size_t>::max() - add) {
    return false;
  }
  *out = value + add;
  return true;
}

void
arena_capsule_destructor(PyObject* capsule)
{
  ArenaState* state = get_state(capsule);
  if (!state) {
    PyErr_Clear();
    return;
  }

  PyGILState_STATE gstate = PyGILState_Ensure();
  Py_XDECREF(state->alloc_callback);
  Py_XDECREF(state->dealloc_callback);
  PyGILState_Release(gstate);

  delete state;
}

PyObject*
call_alloc_callback(ArenaState* state, uintptr_t va, size_t size)
{
  PyObject* cb = state->alloc_callback;
  Py_INCREF(cb);
  PyObject* result = PyObject_CallFunction(
      cb, "KK", static_cast<unsigned long long>(va),
      static_cast<unsigned long long>(size));
  Py_DECREF(cb);
  return result;
}

bool
call_dealloc_callback(
    ArenaState* state, uintptr_t va, size_t size, uint64_t handle,
    bool write_unraisable)
{
  PyObject* cb = state->dealloc_callback;
  Py_INCREF(cb);
  PyObject* result = PyObject_CallFunction(
      cb, "KKK", static_cast<unsigned long long>(va),
      static_cast<unsigned long long>(size),
      static_cast<unsigned long long>(handle));
  if (!result) {
    if (write_unraisable) {
      PyErr_WriteUnraisable(cb);
    } else {
      PyErr_Clear();
    }
    Py_DECREF(cb);
    return false;
  }
  Py_DECREF(result);
  Py_DECREF(cb);
  return true;
}

PyObject*
arena_malloc_impl(ArenaState* state, Py_ssize_t requested_size)
{
  if (requested_size <= 0) {
    PyErr_SetString(PyExc_ValueError, "size must be positive");
    return nullptr;
  }
  // Increment in_flight BEFORE checking state, so a concurrent
  // close_and_drain that CASes state to CLOSED will see this thread's
  // increment when it spins on in_flight, and will wait for the malloc
  // body to complete (callback + map insert) before draining.
  InFlightGuard guard(state);
  if (state->state.load(std::memory_order_seq_cst) != kStateOpen) {
    PyErr_SetString(PyExc_RuntimeError, "arena is closed");
    return nullptr;
  }

  size_t aligned_size = 0;
  if (!round_up(static_cast<size_t>(requested_size), state->granularity, &aligned_size)) {
    PyErr_SetString(PyExc_OverflowError, "allocation size overflow during alignment");
    return nullptr;
  }

  // The CAS bump loop touches no Python C API. Release attached state so
  // other threads can run Python code while we spin on the atomic. Overflow
  // is signaled via a local flag; the Python error is raised after
  // re-attaching.
  size_t offset = state->next_offset.load(std::memory_order_relaxed);
  bool overflow = false;
  Py_BEGIN_ALLOW_THREADS
  while (true) {
    if (offset > state->total_bytes || aligned_size > state->total_bytes - offset) {
      overflow = true;
      break;
    }
    size_t next = offset + aligned_size;
    if (state->next_offset.compare_exchange_weak(
            offset, next, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      break;
    }
  }
  Py_END_ALLOW_THREADS
  if (overflow) {
    PyErr_Format(
        PyExc_RuntimeError,
        "allocation of %zd bytes (aligned %zu) at offset %zu would exceed reserved range of %zu bytes",
        requested_size, aligned_size, offset, state->total_bytes);
    return nullptr;
  }

  uintptr_t va = state->base + offset;
  PyObject* result = call_alloc_callback(state, va, aligned_size);
  if (!result) {
    // Intentional: failed backend allocation leaks this VA slot in the 16 TiB
    // reserve, but no physical memory or map entry exists. Recycling failed
    // slots would add synchronization and rollback complexity to the hot path.
    return nullptr;
  }

  unsigned long long handle_value = PyLong_AsUnsignedLongLong(result);
  Py_DECREF(result);
  if (PyErr_Occurred()) {
    return nullptr;
  }

  // Map insert and mapped_bytes accounting touch no Python C API. Release
  // attached state across the mutex + atomic block.
  Py_BEGIN_ALLOW_THREADS
  {
    std::lock_guard<std::mutex> lock(state->allocations_mutex);
    state->allocations.emplace(
        va, AllocRecord{aligned_size, static_cast<uint64_t>(handle_value)});
  }
  state->mapped_bytes.fetch_add(aligned_size, std::memory_order_acq_rel);
  Py_END_ALLOW_THREADS

  return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(va));
}

void
arena_free_impl(ArenaState* state, uintptr_t ptr, bool write_unraisable)
{
  // Take an in-flight reference so close_and_drain cannot clear the map
  // and reset mapped_bytes between our erase and our fetch_sub. If we are
  // already closed, the close drain has captured our record into the
  // returned list, so doing nothing here is correct.
  InFlightGuard guard(state);

  // State check, map lookup/erase, and mapped_bytes accounting are all
  // pure-C. Release attached state across the block; reattach to call the
  // Python dealloc callback below.
  AllocRecord record{};
  bool found = false;
  Py_BEGIN_ALLOW_THREADS
  if (state->state.load(std::memory_order_seq_cst) == kStateOpen) {
    {
      std::lock_guard<std::mutex> lock(state->allocations_mutex);
      auto it = state->allocations.find(ptr);
      if (it != state->allocations.end()) {
        record = it->second;
        state->allocations.erase(it);
        found = true;
      }
    }
    if (found) {
      state->mapped_bytes.fetch_sub(record.size, std::memory_order_acq_rel);
    }
  }
  Py_END_ALLOW_THREADS

  if (!found) {
    return;
  }

  call_dealloc_callback(
      state, ptr, record.size, record.handle, write_unraisable);
}

}  // namespace

extern "C" {

void*
mx_malloc(ssize_t size, int device, void* stream)
{
  (void)device;
  (void)stream;

  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* capsule = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_active_mutex);
    capsule = g_active_capsule;
    Py_XINCREF(capsule);
  }

  if (!capsule) {
    PyGILState_Release(gstate);
    return nullptr;
  }

  ArenaState* state = get_state(capsule);
  if (!state) {
    PyErr_WriteUnraisable(capsule);
    Py_DECREF(capsule);
    PyGILState_Release(gstate);
    return nullptr;
  }

  void* ptr = nullptr;
  PyObject* result = arena_malloc_impl(state, static_cast<Py_ssize_t>(size));
  if (result) {
    unsigned long long val = PyLong_AsUnsignedLongLong(result);
    if (!PyErr_Occurred()) {
      ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(val));
    }
    Py_DECREF(result);
  }
  if (PyErr_Occurred()) {
    PyErr_WriteUnraisable(capsule);
  }

  Py_DECREF(capsule);
  PyGILState_Release(gstate);
  return ptr;
}

void
mx_free(void* ptr, ssize_t size, int device, void* stream)
{
  (void)size;
  (void)device;
  (void)stream;

  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* capsule = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_active_mutex);
    capsule = g_active_capsule;
    Py_XINCREF(capsule);
  }

  if (capsule) {
    ArenaState* state = get_state(capsule);
    if (state) {
      arena_free_impl(state, reinterpret_cast<uintptr_t>(ptr), true);
    } else {
      PyErr_WriteUnraisable(capsule);
    }
    Py_DECREF(capsule);
  }

  PyGILState_Release(gstate);
}

}  // extern "C"

static PyObject*
py_arena_create(PyObject* self, PyObject* args)
{
  (void)self;
  unsigned long long base = 0;
  unsigned long long total_bytes = 0;
  unsigned long long granularity = 0;
  PyObject* alloc_cb = nullptr;
  PyObject* dealloc_cb = nullptr;

  if (!PyArg_ParseTuple(
          args, "KKKOO", &base, &total_bytes, &granularity, &alloc_cb,
          &dealloc_cb)) {
    return nullptr;
  }
  if (!PyCallable_Check(alloc_cb) || !PyCallable_Check(dealloc_cb)) {
    PyErr_SetString(PyExc_TypeError, "allocation callbacks must be callable");
    return nullptr;
  }
  if (!is_power_of_two(static_cast<size_t>(granularity))) {
    PyErr_SetString(PyExc_ValueError, "granularity must be a positive power of 2");
    return nullptr;
  }

  auto* state = new ArenaState{
      static_cast<uintptr_t>(base),
      static_cast<size_t>(total_bytes),
      static_cast<size_t>(granularity),
      0,
      0,
      kStateOpen,
      0,  // in_flight
      alloc_cb,
      dealloc_cb,
      {},
      {}};
  Py_INCREF(alloc_cb);
  Py_INCREF(dealloc_cb);

  PyObject* capsule = PyCapsule_New(state, kArenaCapsuleName, arena_capsule_destructor);
  if (!capsule) {
    Py_DECREF(alloc_cb);
    Py_DECREF(dealloc_cb);
    delete state;
    return nullptr;
  }
  return capsule;
}

static PyObject*
py_arena_malloc(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* capsule = nullptr;
  Py_ssize_t size = 0;
  if (!PyArg_ParseTuple(args, "On", &capsule, &size)) {
    return nullptr;
  }
  ArenaState* state = get_state(capsule);
  if (!state) {
    return nullptr;
  }
  return arena_malloc_impl(state, size);
}

static PyObject*
py_arena_free(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* capsule = nullptr;
  unsigned long long ptr = 0;
  if (!PyArg_ParseTuple(args, "OK", &capsule, &ptr)) {
    return nullptr;
  }
  ArenaState* state = get_state(capsule);
  if (!state) {
    return nullptr;
  }
  // write_unraisable=true so dealloc errors surface via sys.unraisablehook
  // instead of being silently cleared. Direct-Python callers (arena.free)
  // get the same error visibility as the PyTorch CUDAPluggableAllocator
  // path (mx_free).
  arena_free_impl(state, static_cast<uintptr_t>(ptr), true);
  Py_RETURN_NONE;
}

static PyObject*
py_arena_registered_range(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  ArenaState* state = get_state(capsule);
  if (!state) {
    return nullptr;
  }
  return Py_BuildValue(
      "KK", static_cast<unsigned long long>(state->base),
      static_cast<unsigned long long>(
          state->next_offset.load(std::memory_order_acquire)));
}

static PyObject*
py_arena_get_used_bytes(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  ArenaState* state = get_state(capsule);
  if (!state) {
    return nullptr;
  }
  return PyLong_FromUnsignedLongLong(
      static_cast<unsigned long long>(state->next_offset.load(std::memory_order_acquire)));
}

static PyObject*
py_arena_get_mapped_bytes(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  ArenaState* state = get_state(capsule);
  if (!state) {
    return nullptr;
  }
  return PyLong_FromUnsignedLongLong(
      static_cast<unsigned long long>(state->mapped_bytes.load(std::memory_order_acquire)));
}

static PyObject*
py_arena_get_live_count(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  ArenaState* state = get_state(capsule);
  if (!state) {
    return nullptr;
  }
  size_t live_count = 0;
  Py_BEGIN_ALLOW_THREADS
  {
    std::lock_guard<std::mutex> lock(state->allocations_mutex);
    live_count = state->allocations.size();
  }
  Py_END_ALLOW_THREADS
  return PyLong_FromSize_t(live_count);
}

static PyObject*
py_arena_is_closed(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  ArenaState* state = get_state(capsule);
  if (!state) {
    return nullptr;
  }
  if (state->state.load(std::memory_order_acquire) == kStateClosed) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject*
py_arena_close_and_drain(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  ArenaState* state = get_state(capsule);
  if (!state) {
    return nullptr;
  }

  int expected = kStateOpen;
  if (!state->state.compare_exchange_strong(
          expected, kStateClosed, std::memory_order_seq_cst,
          std::memory_order_seq_cst)) {
    Py_RETURN_NONE;
  }

  // Wait for any in-flight malloc/free that started before our CAS to
  // CLOSED. Operations that took an InFlightGuard before the CAS will
  // complete their callback + map insert (malloc) or fetch_sub (free);
  // operations that take the guard after the CAS see CLOSED state and
  // bail without touching the map. The spin is bounded by the longest
  // in-flight allocator callback, typically sub-millisecond.
  //
  // Release the GIL while spinning: the in-flight operations whose
  // completion we are waiting on dispatch their cuMem* work through
  // Python backend callbacks, so they need the GIL to make progress.
  // Holding the GIL here would deadlock against those callbacks.
  Py_BEGIN_ALLOW_THREADS
  while (state->in_flight.load(std::memory_order_seq_cst) > 0) {
    std::this_thread::yield();
  }
  Py_END_ALLOW_THREADS

  // Snapshot the map under lock. We deliberately do NOT clear the map
  // yet: if PyList_New or Py_BuildValue fails below we want the records
  // to remain so a retry path can recover them via rollback.
  std::vector<std::pair<uintptr_t, AllocRecord>> drained;
  {
    std::lock_guard<std::mutex> lock(state->allocations_mutex);
    drained.reserve(state->allocations.size());
    for (const auto& item : state->allocations) {
      drained.emplace_back(item.first, item.second);
    }
  }

  // Build the Python list before clearing C-side state. On any failure
  // (PyList_New or Py_BuildValue), roll the state back to OPEN so the
  // arena and its records survive the failure and the caller can retry
  // close_and_drain to recover the live allocations.
  PyObject* list = PyList_New(static_cast<Py_ssize_t>(drained.size()));
  if (!list) {
    state->state.store(kStateOpen, std::memory_order_seq_cst);
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(drained.size()); ++i) {
    PyObject* tuple = Py_BuildValue(
        "KKK", static_cast<unsigned long long>(drained[i].first),
        static_cast<unsigned long long>(drained[i].second.size),
        static_cast<unsigned long long>(drained[i].second.handle));
    if (!tuple) {
      Py_DECREF(list);
      state->state.store(kStateOpen, std::memory_order_seq_cst);
      return nullptr;
    }
    PyList_SET_ITEM(list, i, tuple);
  }

  // Commit: the list owns copies of every record. Now it is safe to
  // clear the authoritative C-side state.
  {
    std::lock_guard<std::mutex> lock(state->allocations_mutex);
    state->allocations.clear();
  }
  state->mapped_bytes.store(0, std::memory_order_release);

  return list;
}

static PyObject*
py_set_active_arena(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }

  PyObject* new_capsule = nullptr;
  if (capsule != Py_None) {
    ArenaState* state = get_state(capsule);
    if (!state) {
      return nullptr;
    }
    new_capsule = capsule;
    Py_INCREF(new_capsule);
  }

  PyObject* old_capsule = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_active_mutex);
    old_capsule = g_active_capsule;
    g_active_capsule = new_capsule;
  }
  Py_XDECREF(old_capsule);

  Py_RETURN_NONE;
}

static PyObject*
py_init_module(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* malloc_cb = nullptr;
  PyObject* free_cb = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &malloc_cb, &free_cb)) {
    return nullptr;
  }
  if (!PyCallable_Check(malloc_cb) || !PyCallable_Check(free_cb)) {
    PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
    return nullptr;
  }
  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"arena_create", py_arena_create, METH_VARARGS, "Create a VmmArena state capsule"},
    {"arena_malloc", py_arena_malloc, METH_VARARGS, "Allocate from an arena capsule"},
    {"arena_free", py_arena_free, METH_VARARGS, "Free an arena capsule allocation"},
    {"arena_registered_range", py_arena_registered_range, METH_VARARGS, "Return (base, used_bytes)"},
    {"arena_get_used_bytes", py_arena_get_used_bytes, METH_VARARGS, "Return arena bump pointer"},
    {"arena_get_mapped_bytes", py_arena_get_mapped_bytes, METH_VARARGS, "Return live mapped bytes"},
    {"arena_get_live_count", py_arena_get_live_count, METH_VARARGS, "Return live allocation count"},
    {"arena_is_closed", py_arena_is_closed, METH_VARARGS, "Return whether arena is closed"},
    {"arena_close_and_drain", py_arena_close_and_drain, METH_VARARGS, "Close and drain live allocations"},
    {"set_active_arena", py_set_active_arena, METH_VARARGS, "Set active arena capsule for mx_malloc/mx_free"},
    {"init_module", py_init_module, METH_VARARGS, "Compatibility callback validator"},
    {nullptr, nullptr, 0, nullptr}};

#if PY_VERSION_HEX >= 0x030D0000 && defined(Py_mod_gil)
static PyModuleDef_Slot module_slots[] = {
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
    {0, nullptr},
};
#endif

static struct PyModuleDef vmm_alloc_module = {
    PyModuleDef_HEAD_INIT,
    "_alloc_ext",
    "CUDAPluggableAllocator shim for the VmmArena",
    0,
    module_methods,
#if PY_VERSION_HEX >= 0x030D0000 && defined(Py_mod_gil)
    module_slots,
#else
    nullptr,
#endif
    nullptr,
    nullptr,
    nullptr};

PyMODINIT_FUNC
PyInit__alloc_ext(void)
{
#if PY_VERSION_HEX >= 0x030D0000 && defined(Py_mod_gil)
  return PyModuleDef_Init(&vmm_alloc_module);
#else
  return PyModule_Create(&vmm_alloc_module);
#endif
}
