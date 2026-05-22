# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM P2P CI launcher.

SOURCE mode (MODEL_EXPRESS_SOURCE=1):
  Creates LLM with normal loading. The patched executor/worker.py calls
  publish_from_worker() automatically when MODEL_EXPRESS_SOURCE is set,
  registering per-rank GPU buffers with NIXL.

TARGET mode (default, MODEL_EXPRESS_SOURCE unset or empty):
  Creates LLM with MxLiveCheckpointLoader and LoadFormat.PRESHARDED,
  which receives weights directly from the source via NIXL RDMA.

Both modes start an OpenAI-compatible HTTP server on WORKER_PORT.

Environment variables:
  MODEL_NAME            — HuggingFace model id or local path
  WORKER_PORT           — port to serve on (default: 8000)
  MODEL_EXPRESS_URL     — mx-server gRPC address (default: mx-server:8000)
  TP_SIZE               — tensor parallel size (default: 1)
  MODEL_EXPRESS_SOURCE  — set to any non-empty value for source mode

The `if __name__ == "__main__"` guard is required: TRT-LLM uses MpiPoolSession
to spawn MPI worker subprocesses, which re-import this module.  Without the
guard, the spawned workers would re-execute the LLM setup and abort with
"main script attempted to spawn new MPI worker processes".
"""

import asyncio
import logging
import os
import sys

# Configure logging at MODULE LEVEL (not inside main()) so MPI worker
# subprocesses re-importing this module also configure their loggers.
# Without this, all modelexpress logger.info() emissions inside workers
# (e.g. nixl_transfer's "[TIMING] add_remote_agent" line that the
# test_per_rank_source_agents test greps for) silently drop because the
# default Python level is WARNING.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)


def main() -> None:
    logger = logging.getLogger(__name__)

    model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B")
    port = int(os.environ.get("WORKER_PORT", "8000"))
    mx_server = os.environ.get("MODEL_EXPRESS_URL", "mx-server:8000")
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    # On MIG-sliced GPUs the default ~85% KV cache fraction OOMs the slice
    # and trips an NVML probe in PyTorch's allocator that the MIG cgroup
    # blocks. Capping to 20% on 10GB slices avoids both. Set via env var so
    # the same launcher works on full-GPU nodes by raising/unsetting it.
    kv_fraction = float(os.environ.get("TRTLLM_KV_CACHE_FRACTION", "0.85"))

    from tensorrt_llm.llmapi import LLM, KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import LoadFormat
    from tensorrt_llm.serve import OpenAIServer

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=kv_fraction)

    if os.environ.get("MODEL_EXPRESS_SOURCE"):
        logger.info("Source mode: loading '%s' (tp=%d, kv_cache_fraction=%.2f) — will publish via worker hook",
                    model, tp_size, kv_fraction)
        llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            backend="pytorch",
            kv_cache_config=kv_cache_config,
        )
        logger.info("Source loaded and published. Starting server on port %d", port)
    else:
        logger.info("Target mode: receiving '%s' via PRESHARDED RDMA (tp=%d, mx=%s, kv_cache_fraction=%.2f)",
                    model, tp_size, mx_server, kv_fraction)
        from modelexpress.trtllm_live_transfer import MxLiveCheckpointLoader
        loader = MxLiveCheckpointLoader(mx_server=mx_server)
        llm = LLM(
            model=model,
            checkpoint_loader=loader,
            load_format=LoadFormat.PRESHARDED,
            tensor_parallel_size=tp_size,
            backend="pytorch",
            kv_cache_config=kv_cache_config,
        )
        print("ModelExpress P2P transfer complete, starting server", flush=True)
        logger.info("P2P transfer complete. Starting server on port %d", port)

    # OpenAIServer is the canonical TRT-LLM 1.3.x serving entry point used by
    # the `trtllm-serve` CLI internally (see tensorrt_llm/commands/serve.py:315).
    # `metadata_server_cfg=None` skips the optional etcd-based metadata server.
    # Start via __call__ as an asyncio coroutine.
    server = OpenAIServer(
        generator=llm,
        model=model,
        tool_parser=None,
        server_role=None,
        metadata_server_cfg=None,
    )
    asyncio.run(server(host="0.0.0.0", port=port))


if __name__ == "__main__":
    main()
