"""
Patch TRT-LLM communicator.py to use chunked allgather.

Replaces raw comm.allgather(obj) in tp_allgather/cp_allgather/pp_allgather
with a safe chunked version that avoids MPI_ERR_TRUNCATE with ob1 TCP BTL.
"""

import os
import sys

COMM_PATH = "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/distributed/communicator.py"

SAFE_ALLGATHER = '''
import pickle as _sa_pickle
import numpy as _sa_np

def safe_allgather(comm, obj, chunk_size=65536):
    """Chunked allgather — avoids MPI_ERR_TRUNCATE with ob1 TCP BTL."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    my_bytes = _sa_pickle.dumps(obj)
    my_n = len(my_bytes)
    lengths = _sa_np.array(comm.allgather(my_n), dtype=_sa_np.int64)
    max_len = int(max(lengths))
    all_data = [bytearray() for _ in range(size)]
    for start in range(0, max_len, chunk_size):
        end = min(start + chunk_size, my_n)
        my_chunk = my_bytes[start:end] if start < my_n else b""
        chunks = comm.allgather(bytes(my_chunk))
        for i, c in enumerate(chunks):
            all_data[i] += c
    return [_sa_pickle.loads(bytes(d[:l])) for d, l in zip(all_data, lengths)]
'''

REPLACEMENTS = [
    (
        "def tp_allgather(self, obj):\n        return self.tp_comm.allgather(obj)",
        "def tp_allgather(self, obj):\n        return safe_allgather(self.tp_comm, obj)",
    ),
    (
        "def cp_allgather(self, obj):\n        return self.cp_comm.allgather(obj)",
        "def cp_allgather(self, obj):\n        return safe_allgather(self.cp_comm, obj)",
    ),
    (
        "def pp_allgather(self, obj):\n        return self.pp_comm.allgather(obj)",
        "def pp_allgather(self, obj):\n        return safe_allgather(self.pp_comm, obj)",
    ),
]


def main():
    if not os.path.exists(COMM_PATH):
        print(f"communicator.py not found at {COMM_PATH}")
        sys.exit(1)

    with open(COMM_PATH) as f:
        src = f.read()

    if "safe_allgather" in src:
        print("communicator.py already patched")
        return

    # Apply replacements
    patched = 0
    for old, new in REPLACEMENTS:
        if old in src:
            src = src.replace(old, new)
            patched += 1

    if patched == 0:
        print("WARNING: no allgather patterns matched — TRT-LLM version may differ")
        sys.exit(1)

    # Insert safe_allgather after last top-level import
    lines = src.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i + 1

    lines.insert(insert_idx, SAFE_ALLGATHER)
    src = "\n".join(lines)

    with open(COMM_PATH, "w") as f:
        f.write(src)

    print(f"Patched {patched} allgather methods in communicator.py (chunk_size=64KB)")


if __name__ == "__main__":
    main()
