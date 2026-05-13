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
  MX_SERVER_ADDRESS     — mx-server gRPC address (default: mx-server:8000)
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger(__name__)

    model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B")
    port = int(os.environ.get("WORKER_PORT", "8000"))
    mx_server = os.environ.get("MX_SERVER_ADDRESS", "mx-server:8000")
    tp_size = int(os.environ.get("TP_SIZE", "1"))

    from tensorrt_llm.llmapi import LLM
    from tensorrt_llm.llmapi.llm_args import LoadFormat
    from tensorrt_llm.serve import OpenAIServer

    if os.environ.get("MODEL_EXPRESS_SOURCE"):
        logger.info("Source mode: loading '%s' (tp=%d) — will publish via worker hook", model, tp_size)
        llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            backend="pytorch",
        )
        logger.info("Source loaded and published. Starting server on port %d", port)
    else:
        logger.info("Target mode: receiving '%s' via PRESHARDED RDMA (tp=%d, mx=%s)", model, tp_size, mx_server)
        from modelexpress.trtllm_live_transfer import MxLiveCheckpointLoader
        loader = MxLiveCheckpointLoader(mx_server=mx_server)
        llm = LLM(
            model=model,
            checkpoint_loader=loader,
            load_format=LoadFormat.PRESHARDED,
            tensor_parallel_size=tp_size,
            backend="pytorch",
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
