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

def safe_allgather(comm, obj, chunk_size=65536):
    """Allgather with size check — only chunks if message exceeds threshold."""
    my_bytes = _sa_pickle.dumps(obj)
    if len(my_bytes) <= chunk_size:
        return comm.allgather(obj)
    # Large object: use gather + broadcast to avoid allgather truncation
    rank = comm.Get_rank()
    size = comm.Get_size()
    gathered = comm.gather(my_bytes, root=0)
    if rank == 0:
        result = [_sa_pickle.loads(g) for g in gathered]
    else:
        result = None
    result = comm.bcast(result, root=0)
    return result
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
