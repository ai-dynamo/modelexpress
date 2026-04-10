# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Patch TRT-LLM communicator.py to use chunked safe_allgather.

Based on TRT-LLM PR #12174 by chienchunhung:
  https://github.com/NVIDIA/TensorRT-LLM/pull/12174

Replaces raw comm.allgather(obj) in tp_allgather with safe_allgather()
that chunks transfers via MPI.Allgatherv to avoid 32-bit int overflow
and MPI_ERR_TRUNCATE with ob1 TCP BTL on GB200.
"""

import os
import sys

COMM_PATH = "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/distributed/communicator.py"

SAFE_ALLGATHER = '''
import math as _sa_math

def safe_allgather(comm, obj, chunk_size: int = 4 * 1024 * 1024):
    """Safely allgather potentially large objects by splitting into
    fixed-size chunks using raw-byte MPI.Allgatherv.

    Based on TRT-LLM PR #12174. Avoids mpi4py 32-bit int overflow
    in counts/displacements and pickle5 out-of-band buffer issues.
    """
    if not ENABLE_MULTI_DEVICE:
        return [obj]
    if ENABLE_MULTI_DEVICE and MPI is None:
        raise RuntimeError("mpi4py is required when ENABLE_MULTI_DEVICE is True")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    rank = comm.Get_rank()
    size = comm.Get_size()

    max_safe_chunk = np.iinfo(np.int32).max // size if size > 0 else chunk_size
    if chunk_size > max_safe_chunk:
        chunk_size = max_safe_chunk

    try:
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        my_n = len(payload)
    except Exception as e:
        _ = comm.allgather(-1)
        raise RuntimeError(f"Rank {rank} serialization failed: {e}") from e

    lengths = np.array(comm.allgather(my_n), dtype=np.int64)
    if (lengths < 0).any():
        raise RuntimeError("Serialization failed on at least one rank")

    displs = np.zeros(size, dtype=np.int64)
    if size > 1:
        displs[1:] = np.cumsum(lengths[:-1])

    sendbuf_full = np.frombuffer(payload, dtype=np.uint8, count=my_n)
    recvbuf = np.empty(lengths.sum(), dtype=np.uint8)

    max_len = lengths.max() if size > 0 else 0
    num_rounds = _sa_math.ceil(max_len / chunk_size) if max_len > 0 else 0

    for r in range(num_rounds):
        round_offs = r * chunk_size
        counts_this_round = np.minimum(
            np.maximum(lengths - round_offs, 0), chunk_size
        ).astype(np.int32)
        sent_so_far = np.minimum(lengths, round_offs)

        round_recvbuf = np.empty(counts_this_round.sum(), dtype=np.uint8)
        round_displs = np.zeros(size, dtype=np.int32)
        if size > 1:
            round_displs[1:] = np.cumsum(counts_this_round[:-1])

        send_part = sendbuf_full[
            sent_so_far[rank]:sent_so_far[rank] + counts_this_round[rank]
        ]

        comm.Allgatherv(
            [send_part, MPI.BYTE],
            [round_recvbuf, counts_this_round, round_displs, MPI.BYTE],
        )

        src_offset = 0
        for i in range(size):
            n = counts_this_round[i]
            if n > 0:
                dst_start = displs[i] + sent_so_far[i]
                recvbuf[dst_start:dst_start + n] = (
                    round_recvbuf[src_offset:src_offset + n]
                )
                src_offset += n

    out = []
    for i in range(size):
        sz = lengths[i]
        if sz == 0:
            out.append(None)
            continue
        start = displs[i]
        blob = recvbuf[start:start + sz].tobytes()
        try:
            out.append(pickle.loads(blob))
        except Exception as e:
            raise RuntimeError(f"Deserialization failed for rank {i}: {e}") from e

    return out
'''

REPLACEMENT = (
    "def tp_allgather(self, obj):\n        return self.tp_comm.allgather(obj)",
    "def tp_allgather(self, obj, chunk_size: int = 4 * 1024 * 1024):\n        return safe_allgather(self.tp_comm, obj, chunk_size=chunk_size)",
)


def main():
    if not os.path.exists(COMM_PATH):
        print(f"communicator.py not found at {COMM_PATH}")
        sys.exit(1)

    with open(COMM_PATH) as f:
        src = f.read()

    if "safe_allgather" in src:
        print("communicator.py already has safe_allgather — skipping")
        return

    old, new = REPLACEMENT
    if old in src:
        src = src.replace(old, new)
        print("Patched tp_allgather to use safe_allgather")
    else:
        print("WARNING: tp_allgather pattern not found — TRT-LLM version may differ")
        sys.exit(1)

    lines = src.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i + 1

    lines.insert(insert_idx, SAFE_ALLGATHER)
    src = "\n".join(lines)

    with open(COMM_PATH, "w") as f:
        f.write(src)

    print("Applied safe_allgather from TRT-LLM PR #12174 (chunk_size=4MB)")


if __name__ == "__main__":
    main()
