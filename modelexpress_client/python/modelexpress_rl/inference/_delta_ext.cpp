// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Checks of zstd-compressed XOR deltas against checkpoint files, off the GIL.
//
// A check decompresses a tensor's delta in chunks, records the delta's nonzero
// bytes for staging, and derives the target's Adler-32. A sparse check reads
// the checkpoint file only where the delta changes it and updates the base
// checksum: Adler-32 is linear in the bytes. A full check reads every byte.
// Bytes of deltas staged on the files are XORed in where they are read.
// Tensors run on a pool of threads.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <zstd.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <system_error>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kAdlerBase = 65521;
// The most bytes Adler-32's sums take before they must be reduced.
constexpr size_t kAdlerBlock = 5552;
constexpr size_t kChunk = size_t(1) << 20;

// Outcomes of one tensor's check.
enum Status : uint8_t {
  kSparse = 0,    // a sparse check hit the target checksum
  kFull = 1,      // a full check hit it, without a base checksum
  kFallback = 2,  // a sparse check missed, and a full check hit it
  kMissed = 3,    // a full check missed it
  kError = 4,     // unchecked: the delta does not decompress to the tensor's size,
                  // or the tensor lies outside its file
};

struct Layer {
  const uint32_t* positions;
  const uint8_t* values;
  size_t count;
};

struct Tensor {
  size_t file;
  size_t offset;
  size_t size;
  const uint8_t* frame;
  size_t frame_size;
  int64_t base;
  uint32_t expected;
  std::vector<Layer> layers;
};

struct Result {
  Status status = kError;
  std::vector<uint32_t> positions;
  std::vector<uint8_t> values;
};

struct Mapping {
  const uint8_t* data = nullptr;
  size_t size = 0;
};

// One thread's decompression context and buffers, reused across tensors.
struct Scratch {
  ZSTD_DCtx* context = ZSTD_createDCtx();
  std::vector<uint8_t> chunk = std::vector<uint8_t>(kChunk);
  std::vector<uint8_t> target = std::vector<uint8_t>(kChunk);
  ~Scratch() { ZSTD_freeDCtx(context); }
};

uint32_t adler32(uint32_t value, const uint8_t* data, size_t length) {
  uint32_t low = value & 0xFFFF;
  uint32_t high = value >> 16;
  while (length > 0) {
    const size_t block = std::min(length, kAdlerBlock);
    length -= block;
    for (size_t index = 0; index < block; ++index) {
      low += data[index];
      high += low;
    }
    data += block;
    low %= kAdlerBase;
    high %= kAdlerBase;
  }
  return (high << 16) | low;
}

// Staged deltas' bytes, walked forward along one tensor.
struct Overlay {
  const Layer* layer;
  size_t cursor = 0;

  // The XOR of this layer at `position`; positions must not decrease.
  uint8_t at(size_t position) {
    while (cursor < layer->count && layer->positions[cursor] < position) {
      ++cursor;
    }
    return cursor < layer->count && layer->positions[cursor] == position ? layer->values[cursor] : 0;
  }

  // XOR this layer into `bytes`, the tensor's bytes from `start`.
  void apply(uint8_t* bytes, size_t start, size_t length) {
    while (cursor < layer->count && layer->positions[cursor] < start) {
      ++cursor;
    }
    for (; cursor < layer->count && layer->positions[cursor] < start + length; ++cursor) {
      bytes[layer->positions[cursor] - start] ^= layer->values[cursor];
    }
  }
};

// Call `found(position, value)` for each nonzero byte of `block`, at tensor offset `start`.
template <typename Found>
void scan(const uint8_t* block, size_t length, size_t start, Found&& found) {
  size_t index = 0;
  for (; index + 8 <= length; index += 8) {
    uint64_t word;
    std::memcpy(&word, block + index, 8);
    if (word == 0) {
      continue;
    }
    for (size_t byte = 0; byte < 8; ++byte) {
      if (block[index + byte] != 0) {
        found(start + index + byte, block[index + byte]);
      }
    }
  }
  for (; index < length; ++index) {
    if (block[index] != 0) {
      found(start + index, block[index]);
    }
  }
}

// Decompress into `out` until it is full or the frames end; false if the frames are malformed.
bool decompress(ZSTD_DCtx* context, ZSTD_inBuffer& input, uint8_t* out, size_t capacity,
                size_t& written, bool& ended) {
  ZSTD_outBuffer output{out, capacity, 0};
  for (;;) {
    const size_t hint = ZSTD_decompressStream(context, &output, &input);
    if (ZSTD_isError(hint)) {
      return false;
    }
    written = output.pos;
    // The last frame decoded and flushed, with no input left, is the end.
    ended = hint == 0 && input.pos == input.size;
    if (ended || output.pos == output.size) {
      return true;
    }
    if (input.pos == input.size) {
      return false;
    }
  }
}

// Run one check of `tensor` and set `checksum`; false if the delta is malformed.
bool run(const Tensor& tensor, const uint8_t* old, bool sparse, Scratch& scratch, Result& result,
         uint32_t& checksum) {
  ZSTD_DCtx_reset(scratch.context, ZSTD_reset_session_only);
  ZSTD_inBuffer input{tensor.frame, tensor.frame_size, 0};
  std::vector<Overlay> overlays;
  overlays.reserve(tensor.layers.size());
  for (const Layer& layer : tensor.layers) {
    overlays.push_back(Overlay{&layer});
  }
  result.positions.clear();
  result.values.clear();
  // Sparse sums of byte changes, or the running checksum of the target.
  int64_t low = 0;
  int64_t high = 0;
  checksum = 1;
  size_t done = 0;
  for (;;) {
    // Past the tensor's end, one more byte shows a delta that runs over.
    const size_t want = std::clamp(tensor.size - done, size_t(1), kChunk);
    size_t written = 0;
    bool ended = false;
    if (!decompress(scratch.context, input, scratch.chunk.data(), want, written, ended) ||
        done + written > tensor.size) {
      return false;
    }
    const uint8_t* delta = scratch.chunk.data();
    if (sparse) {
      scan(delta, written, done, [&](size_t position, uint8_t value) {
        uint8_t before = old[position];
        for (Overlay& overlay : overlays) {
          before ^= overlay.at(position);
        }
        const int64_t difference = int64_t(uint8_t(before ^ value)) - int64_t(before);
        low += difference;
        high += int64_t((tensor.size - position) % kAdlerBase) * difference;
        result.positions.push_back(uint32_t(position));
        result.values.push_back(value);
      });
    } else if (written > 0) {
      uint8_t* target = scratch.target.data();
      std::memcpy(target, old + done, written);
      for (Overlay& overlay : overlays) {
        overlay.apply(target, done, written);
      }
      for (size_t index = 0; index < written; ++index) {
        target[index] ^= delta[index];
      }
      checksum = adler32(checksum, target, written);
      scan(delta, written, done, [&](size_t position, uint8_t value) {
        result.positions.push_back(uint32_t(position));
        result.values.push_back(value);
      });
    }
    done += written;
    if (ended) {
      break;
    }
  }
  if (done != tensor.size) {
    return false;
  }
  if (sparse) {
    const int64_t base = tensor.base;
    int64_t sum = ((base & 0xFFFF) + low) % kAdlerBase;
    int64_t weighted = ((base >> 16) + high) % kAdlerBase;
    sum += sum < 0 ? kAdlerBase : 0;
    weighted += weighted < 0 ? kAdlerBase : 0;
    checksum = uint32_t((weighted << 16) | sum);
  }
  return true;
}

void check(const Tensor& tensor, const std::vector<Mapping>& maps, Scratch& scratch, Result& result) {
  const Mapping& map = maps[tensor.file];
  if (tensor.offset > map.size || tensor.size > map.size - tensor.offset) {
    return;
  }
  const uint8_t* old = map.data + tensor.offset;
  uint32_t checksum = 0;
  if (tensor.base >= 0) {
    if (!run(tensor, old, true, scratch, result, checksum)) {
      return;
    }
    if (checksum == tensor.expected) {
      result.status = kSparse;
      return;
    }
  }
  if (!run(tensor, old, false, scratch, result, checksum)) {
    return;
  }
  result.status = checksum != tensor.expected ? kMissed : tensor.base >= 0 ? kFallback : kFull;
}

// Run `work(index)` for each index of `order` on up to `threads` threads, the caller's among them.
template <typename Work>
void parallel(size_t threads, const std::vector<size_t>& order, Work&& work) {
  std::atomic<size_t> next{0};
  auto drain = [&] {
    for (size_t claimed = next.fetch_add(1); claimed < order.size(); claimed = next.fetch_add(1)) {
      work(order[claimed]);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(std::max<size_t>(threads, 1) - 1);
  for (size_t worker = 1; worker < threads; ++worker) {
    try {
      pool.emplace_back(drain);
    } catch (const std::system_error&) {
      // Fewer threads still drain every index.
      break;
    }
  }
  drain();
  for (std::thread& thread : pool) {
    thread.join();
  }
}

// Buffers of Python objects, released when the call returns.
class Views {
 public:
  ~Views() {
    for (Py_buffer& view : views_) {
      PyBuffer_Release(&view);
    }
  }

  // The contiguous buffer of `object`, as items of `itemsize` bytes.
  const void* take(PyObject* object, Py_ssize_t itemsize, size_t& count) {
    Py_buffer view;
    if (PyObject_GetBuffer(object, &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) {
      return nullptr;
    }
    views_.push_back(view);
    if (view.itemsize != itemsize) {
      PyErr_Format(PyExc_ValueError, "buffer holds %zd-byte items, not %zd", view.itemsize, itemsize);
      return nullptr;
    }
    count = size_t(view.len / view.itemsize);
    return view.buf != nullptr ? view.buf : "";
  }

 private:
  std::vector<Py_buffer> views_;
};

// Read-only mappings of checkpoint files, unmapped when the call returns.
class Files {
 public:
  ~Files() {
    for (const Mapping& map : maps_) {
      if (map.data != nullptr) {
        munmap(const_cast<uint8_t*>(map.data), map.size);
      }
    }
  }

  // Map `path`. A mapping stays valid after its descriptor is closed.
  bool open(const char* path) {
    const int descriptor = ::open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
      PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
      return false;
    }
    Mapping map;
    struct stat info;
    bool mapped = fstat(descriptor, &info) == 0;
    map.size = mapped ? size_t(info.st_size) : 0;
    if (map.size > 0) {
      void* data = mmap(nullptr, map.size, PROT_READ, MAP_SHARED, descriptor, 0);
      mapped = data != MAP_FAILED;
      map.data = mapped ? static_cast<const uint8_t*>(data) : nullptr;
    }
    // Set the error first, since close() may change errno.
    if (!mapped) {
      PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
    }
    ::close(descriptor);
    if (mapped) {
      maps_.push_back(map);
    }
    return mapped;
  }

  const std::vector<Mapping>& maps() const { return maps_; }

 private:
  std::vector<Mapping> maps_;
};

using Owned = std::unique_ptr<PyObject, decltype(&Py_DecRef)>;

PyObject* py_check(PyObject*, PyObject* args) {
  PyObject *paths, *files, *offsets, *sizes, *frames, *bases, *expected, *overlays;
  Py_ssize_t threads;
  if (!PyArg_ParseTuple(args, "OOOOOOOOn", &paths, &files, &offsets, &sizes, &frames, &bases,
                        &expected, &overlays, &threads)) {
    return nullptr;
  }
  Views views;
  size_t count = 0;
  const auto* file_index = static_cast<const int64_t*>(views.take(files, 8, count));
  if (file_index == nullptr) {
    return nullptr;
  }
  // Per-tensor arrays, each as long as `files`.
  auto column = [&](PyObject* object, Py_ssize_t itemsize) -> const void* {
    size_t length = 0;
    const void* data = views.take(object, itemsize, length);
    if (data != nullptr && length != count) {
      PyErr_SetString(PyExc_ValueError, "per-tensor arguments differ in length");
      return nullptr;
    }
    return data;
  };
  const auto* offset = static_cast<const int64_t*>(column(offsets, 8));
  const auto* size = offset != nullptr ? static_cast<const int64_t*>(column(sizes, 8)) : nullptr;
  const auto* base = size != nullptr ? static_cast<const int64_t*>(column(bases, 8)) : nullptr;
  const auto* target = base != nullptr ? static_cast<const uint32_t*>(column(expected, 4)) : nullptr;
  if (target == nullptr) {
    return nullptr;
  }
  Owned frame_list(PySequence_Fast(frames, "frames must be a sequence"), Py_DecRef);
  Owned overlay_list(PySequence_Fast(overlays, "overlays must be a sequence"), Py_DecRef);
  Owned path_list(PySequence_Fast(paths, "paths must be a sequence"), Py_DecRef);
  if (frame_list == nullptr || overlay_list == nullptr || path_list == nullptr) {
    return nullptr;
  }
  if (PySequence_Fast_GET_SIZE(frame_list.get()) != Py_ssize_t(count) ||
      PySequence_Fast_GET_SIZE(overlay_list.get()) != Py_ssize_t(count)) {
    PyErr_SetString(PyExc_ValueError, "per-tensor arguments differ in length");
    return nullptr;
  }
  Files mapped;
  for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(path_list.get()); ++index) {
    const char* path = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(path_list.get(), index));
    if (path == nullptr || !mapped.open(path)) {
      return nullptr;
    }
  }
  std::vector<Tensor> tensors(count);
  for (size_t index = 0; index < count; ++index) {
    Tensor& tensor = tensors[index];
    if (file_index[index] < 0 || size_t(file_index[index]) >= mapped.maps().size() || offset[index] < 0 ||
        size[index] < 0 || size[index] > (int64_t(1) << 32)) {
      PyErr_SetString(PyExc_ValueError, "tensor file, offset, or size out of range");
      return nullptr;
    }
    tensor.file = size_t(file_index[index]);
    tensor.offset = size_t(offset[index]);
    tensor.size = size_t(size[index]);
    tensor.base = base[index];
    tensor.expected = target[index];
    tensor.frame = static_cast<const uint8_t*>(
        views.take(PySequence_Fast_GET_ITEM(frame_list.get(), Py_ssize_t(index)), 1, tensor.frame_size));
    if (tensor.frame == nullptr) {
      return nullptr;
    }
    Owned layers(PySequence_Fast(PySequence_Fast_GET_ITEM(overlay_list.get(), Py_ssize_t(index)),
                                 "each tensor's overlays must be a sequence"),
                 Py_DecRef);
    if (layers == nullptr) {
      return nullptr;
    }
    for (Py_ssize_t layer_index = 0; layer_index < PySequence_Fast_GET_SIZE(layers.get()); ++layer_index) {
      PyObject* pair = PySequence_Fast_GET_ITEM(layers.get(), layer_index);
      if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2) {
        PyErr_SetString(PyExc_ValueError, "an overlay is a (positions, values) pair");
        return nullptr;
      }
      Layer layer{nullptr, nullptr, 0};
      size_t values = 0;
      layer.positions = static_cast<const uint32_t*>(views.take(PyTuple_GET_ITEM(pair, 0), 4, layer.count));
      layer.values =
          layer.positions != nullptr ? static_cast<const uint8_t*>(views.take(PyTuple_GET_ITEM(pair, 1), 1, values))
                                     : nullptr;
      if (layer.values == nullptr) {
        return nullptr;
      }
      if (values != layer.count) {
        PyErr_SetString(PyExc_ValueError, "staged offsets and values differ in length");
        return nullptr;
      }
      tensor.layers.push_back(layer);
    }
  }
  // Large tensors first, so one does not finish last.
  std::vector<size_t> order(count);
  std::iota(order.begin(), order.end(), size_t(0));
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t left, size_t right) { return tensors[left].size > tensors[right].size; });
  const size_t workers = size_t(std::max<Py_ssize_t>(1, std::min<Py_ssize_t>(threads, Py_ssize_t(count))));
  std::vector<Result> results(count);
  Py_BEGIN_ALLOW_THREADS;
  parallel(workers, order, [&](size_t index) {
    thread_local Scratch scratch;
    try {
      check(tensors[index], mapped.maps(), scratch, results[index]);
    } catch (const std::exception&) {
      results[index] = Result{};
    }
  });
  Py_END_ALLOW_THREADS;
  // Each checked tensor's staged bytes, concatenated in argument order.
  std::vector<int64_t> counts(count);
  std::vector<size_t> starts(count);
  std::vector<uint8_t> statuses(count);
  size_t total = 0;
  for (size_t index = 0; index < count; ++index) {
    Result& result = results[index];
    if (result.status == kMissed || result.status == kError) {
      result.positions = {};
      result.values = {};
    }
    statuses[index] = result.status;
    starts[index] = total;
    counts[index] = int64_t(result.positions.size());
    total += result.positions.size();
  }
  Owned positions(PyByteArray_FromStringAndSize(nullptr, Py_ssize_t(total * sizeof(uint32_t))), Py_DecRef);
  Owned values(PyByteArray_FromStringAndSize(nullptr, Py_ssize_t(total)), Py_DecRef);
  if (positions == nullptr || values == nullptr) {
    return nullptr;
  }
  auto* position_bytes = reinterpret_cast<uint8_t*>(PyByteArray_AS_STRING(positions.get()));
  auto* value_bytes = reinterpret_cast<uint8_t*>(PyByteArray_AS_STRING(values.get()));
  Py_BEGIN_ALLOW_THREADS;
  parallel(workers, order, [&](size_t index) {
    Result& result = results[index];
    if (!result.positions.empty()) {
      std::memcpy(position_bytes + starts[index] * sizeof(uint32_t), result.positions.data(),
                  result.positions.size() * sizeof(uint32_t));
      std::memcpy(value_bytes + starts[index], result.values.data(), result.values.size());
    }
    result.positions = {};
    result.values = {};
  });
  Py_END_ALLOW_THREADS;
  return Py_BuildValue("(y#y#OO)", reinterpret_cast<const char*>(statuses.data()), Py_ssize_t(count),
                       reinterpret_cast<const char*>(counts.data()), Py_ssize_t(count * sizeof(int64_t)),
                       positions.get(), values.get());
}

PyMethodDef methods[] = {
    {"check", py_check, METH_VARARGS,
     "check(paths, files, offsets, sizes, frames, bases, expected, overlays, threads)\n\n"
     "Check zstd XOR deltas against checkpoint files. Returns each tensor's status,\n"
     "the counts of their nonzero delta bytes, and those bytes' offsets and values\n"
     "concatenated."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_delta_ext", nullptr, -1, methods,
                      nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__delta_ext(void) { return PyModule_Create(&module); }
