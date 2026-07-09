# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ctypes binding over libnccl_m2n.so -- the reshard transport surface.

Vendored from the nccl MR-2984 PoC (``m2n_binding.py``) where NCCL M2N lives
before migrating into the nccl-extensions repo.  Wraps the public C API in
``nccl_m2n.h``::

    ncclM2nInit / ncclM2nFinalize
    ncclReshardWithWindow(comm, window, src, dst, stream)

plus the NCCL symmetric-window calls the reshard requires (``ncclMemAlloc``,
``ncclCommWindowRegister`` / ``ncclCommWindowDeregister``, resolved from
``libnccl.so``), and -- added here for the ModelExpress spike -- the fresh-comm
bootstrap (``ncclGetUniqueId`` + ``ncclCommInitRank`` brokered over
``torch.distributed``) and a device-to-device staging copy.

Window contract (from the header + tests/basic_api_test_core.h):
  * Buffer must be allocated with ncclMemAlloc and registered on the comm with
    NCCL_WIN_COLL_SYMMETRIC; both src and dst dataPtr must sit at the window base
    (zero-offset contract).
  * src/dst descriptors are required on EVERY rank; a non-participating side sets
    dataPtr=NULL but still carries its mesh, ndims, dtype.
"""

from __future__ import annotations

import ctypes
import ctypes.util

# ---- NCCL enums (from nccl.h) ----------------------------------------------
ncclInt8, ncclUint8 = 0, 1
ncclInt32, ncclUint32 = 2, 3
ncclInt64, ncclUint64 = 4, 5
ncclFloat16, ncclFloat32, ncclFloat64 = 6, 7, 8
ncclBfloat16 = 9
ncclFloat8e4m3, ncclFloat8e5m2 = 10, 11  # available in recent NCCL

ncclSuccess = 0
NCCL_WIN_COLL_SYMMETRIC = 2  # window mode flag (see nccl.h / device API)
NCCL_UNIQUE_ID_BYTES = 128

# placement helpers (mirror nccl_m2n.h)
NCCL_RESHARD_REPLICATE = -1


def NCCL_RESHARD_SHARD(d: int) -> int:
    return d


NCCL_RESHARD_MAX_TENSOR_DIMS = 3
NCCL_M2N_API_MAGIC = 0x4D324E00
NCCL_M2N_CONFIG_UNDEF_INT = -(2**31)

# cudaMemcpyKind
_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3


# ---- struct layouts (must match nccl_m2n.h / nccl.h byte-for-byte) ---------
class ncclMesh_t(ctypes.Structure):
    _fields_ = [
        ("dims", ctypes.c_int * 2),
        ("startRank", ctypes.c_int),
        ("placement", ctypes.c_int * 2),
    ]


class ncclDistTensor_t(ctypes.Structure):
    _fields_ = [
        ("dataPtr", ctypes.c_void_p),
        ("localShape", ctypes.c_size_t * NCCL_RESHARD_MAX_TENSOR_DIMS),
        ("ndims", ctypes.c_int),
        ("dtype", ctypes.c_int),  # ncclDataType_t
        ("mesh", ctypes.POINTER(ncclMesh_t)),
    ]


class ncclM2nConfig_t(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("maxCta", ctypes.c_int),
    ]


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * NCCL_UNIQUE_ID_BYTES)]


def make_config(max_cta: int | None = None) -> ncclM2nConfig_t:
    c = ncclM2nConfig_t()
    c.size = ctypes.sizeof(ncclM2nConfig_t)
    c.magic = NCCL_M2N_API_MAGIC
    c.maxCta = NCCL_M2N_CONFIG_UNDEF_INT if max_cta is None else max_cta
    return c


# ---- library loading -------------------------------------------------------
def _load(name, explicit=None):
    if explicit:
        return ctypes.CDLL(explicit, mode=ctypes.RTLD_GLOBAL)
    path = ctypes.util.find_library(name)
    return ctypes.CDLL(path or f"lib{name}.so", mode=ctypes.RTLD_GLOBAL)


class M2N:
    """Loaded libnccl_m2n.so + libnccl.so (+ libcudart) with argtypes wired up."""

    def __init__(
        self,
        nccl_path: str | None = None,
        m2n_path: str | None = None,
        cudart_path: str | None = None,
    ):
        self.nccl = _load("nccl", nccl_path)
        self.m2n = _load("nccl_m2n", m2n_path)
        self.cudart = _load("cudart", cudart_path)

        # nccl symmetric-memory + window API
        self.nccl.ncclMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.nccl.ncclMemAlloc.restype = ctypes.c_int
        self.nccl.ncclMemFree.argtypes = [ctypes.c_void_p]
        self.nccl.ncclMemFree.restype = ctypes.c_int
        # ncclCommWindowRegister(comm, ptr, size, ncclWindow_t*, winFlags)
        self.nccl.ncclCommWindowRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        self.nccl.ncclCommWindowRegister.restype = ctypes.c_int
        self.nccl.ncclCommWindowDeregister.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.nccl.ncclCommWindowDeregister.restype = ctypes.c_int

        # nccl comm bootstrap (fresh dedicated comm over a broadcast uniqueId)
        self.nccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(ncclUniqueId)]
        self.nccl.ncclGetUniqueId.restype = ctypes.c_int
        # ncclCommInitRank(ncclComm_t*, int nranks, ncclUniqueId commId, int rank)
        self.nccl.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ncclUniqueId,
            ctypes.c_int,
        ]
        self.nccl.ncclCommInitRank.restype = ctypes.c_int
        self.nccl.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        self.nccl.ncclCommDestroy.restype = ctypes.c_int

        # cuda device-to-device staging copy (window base <-> param tile)
        self.cudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.cudart.cudaMemcpy.restype = ctypes.c_int
        self.cudart.cudaSetDevice.argtypes = [ctypes.c_int]
        self.cudart.cudaSetDevice.restype = ctypes.c_int
        self.cudart.cudaDeviceSynchronize.argtypes = []
        self.cudart.cudaDeviceSynchronize.restype = ctypes.c_int

        # m2n API
        self.m2n.ncclM2nInit.argtypes = [ctypes.POINTER(ncclM2nConfig_t)]
        self.m2n.ncclM2nInit.restype = ctypes.c_int
        self.m2n.ncclM2nFinalize.argtypes = []
        self.m2n.ncclM2nFinalize.restype = ctypes.c_int
        self.m2n.ncclReshardWithWindow.argtypes = [
            ctypes.c_void_p,  # ncclComm_t
            ctypes.c_void_p,  # ncclWindow_t
            ctypes.POINTER(ncclDistTensor_t),  # src
            ctypes.POINTER(ncclDistTensor_t),  # dst
            ctypes.c_void_p,  # cudaStream_t
        ]
        self.m2n.ncclReshardWithWindow.restype = ctypes.c_int

    # -- thin wrappers --
    def init(self, max_cta: int | None = None):
        cfg = make_config(max_cta)
        rc = self.m2n.ncclM2nInit(ctypes.byref(cfg))
        assert rc == ncclSuccess, f"ncclM2nInit rc={rc}"

    def finalize(self):
        self.m2n.ncclM2nFinalize()

    def mem_alloc(self, nbytes: int) -> int:
        p = ctypes.c_void_p()
        rc = self.nccl.ncclMemAlloc(ctypes.byref(p), nbytes)
        assert rc == ncclSuccess, f"ncclMemAlloc rc={rc}"
        return p.value

    def mem_free(self, ptr: int):
        self.nccl.ncclMemFree(ptr)

    def window_register(self, comm: int, ptr: int, nbytes: int) -> int:
        win = ctypes.c_void_p()
        rc = self.nccl.ncclCommWindowRegister(
            comm, ptr, nbytes, ctypes.byref(win), NCCL_WIN_COLL_SYMMETRIC
        )
        assert rc == ncclSuccess, f"ncclCommWindowRegister rc={rc}"
        return win.value

    def window_deregister(self, comm: int, win: int):
        self.nccl.ncclCommWindowDeregister(comm, win)

    def memcpy_dtod(self, dst_ptr: int, src_ptr: int, nbytes: int):
        rc = self.cudart.cudaMemcpy(dst_ptr, src_ptr, nbytes, _CUDA_MEMCPY_DEVICE_TO_DEVICE)
        assert rc == 0, f"cudaMemcpy rc={rc}"

    def device_synchronize(self):
        rc = self.cudart.cudaDeviceSynchronize()
        assert rc == 0, f"cudaDeviceSynchronize rc={rc}"

    def comm_init_rank(self, nranks: int, uid: ncclUniqueId, rank: int) -> int:
        comm = ctypes.c_void_p()
        rc = self.nccl.ncclCommInitRank(ctypes.byref(comm), nranks, uid, rank)
        assert rc == ncclSuccess, f"ncclCommInitRank rc={rc}"
        return comm.value

    def comm_destroy(self, comm: int):
        self.nccl.ncclCommDestroy(comm)

    def reshard(
        self,
        comm: int,
        window: int,
        src: ncclDistTensor_t,
        dst: ncclDistTensor_t,
        stream: int = 0,
    ) -> int:
        return self.m2n.ncclReshardWithWindow(
            comm, window, ctypes.byref(src), ctypes.byref(dst), stream
        )


def make_tensor_desc(data_ptr, local_shape, ndims, dtype, mesh: ncclMesh_t) -> ncclDistTensor_t:
    t = ncclDistTensor_t()
    t.dataPtr = ctypes.c_void_p(data_ptr if data_ptr else None)
    for i in range(NCCL_RESHARD_MAX_TENSOR_DIMS):
        t.localShape[i] = local_shape[i] if i < len(local_shape) else 0
    t.ndims = ndims
    t.dtype = dtype
    t.mesh = ctypes.pointer(mesh)
    return t


def make_mesh(dims, start_rank, placement) -> ncclMesh_t:
    m = ncclMesh_t()
    m.dims[0], m.dims[1] = dims[0], dims[1]
    m.startRank = start_rank
    m.placement[0], m.placement[1] = placement[0], placement[1]
    return m


def bootstrap_comm_from_torch(m2n: M2N, tp_src: int, tp_dst: int, device_id: int | None = None) -> int:
    """Build a fresh ncclComm_t spanning all src+dst ranks.

    The "bootstrap fresh" path: rank 0 generates an ncclUniqueId, broadcasts its
    128 bytes over the already-initialized ``torch.distributed`` world, and every
    rank calls ncclCommInitRank.  Rank layout: ``[0, tp_src)`` = source (trainer)
    mesh, ``[tp_src, tp_src + tp_dst)`` = destination (generator) mesh.

    Requires a torch.distributed process group that already spans both groups
    (the spike's broadcast channel); the MX-server-brokered uniqueId broadcast is
    the follow-up for a truly disaggregated deployment.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("bootstrap_comm_from_torch requires an initialized torch.distributed world")

    world = dist.get_world_size()
    rank = dist.get_rank()
    if world != tp_src + tp_dst:
        raise ValueError(f"world_size {world} != tp_src {tp_src} + tp_dst {tp_dst}")

    if device_id is None:
        device_id = torch.cuda.current_device()
    m2n.cudart.cudaSetDevice(device_id)

    uid = ncclUniqueId()
    if rank == 0:
        rc = m2n.nccl.ncclGetUniqueId(ctypes.byref(uid))
        assert rc == ncclSuccess, f"ncclGetUniqueId rc={rc}"

    # Broadcast the 128 uniqueId bytes from rank 0 over torch.distributed.  Use a
    # CPU tensor so the control-plane PG can be gloo -- keeping torch from
    # standing up its own NCCL communicator alongside the reshard comm.
    buf = torch.frombuffer(bytearray(uid.internal), dtype=torch.uint8).clone()
    dist.broadcast(buf, src=0)
    ctypes.memmove(ctypes.byref(uid), buf.numpy().tobytes(), NCCL_UNIQUE_ID_BYTES)

    return m2n.comm_init_rank(world, uid, rank)
