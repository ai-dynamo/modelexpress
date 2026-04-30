# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone entrypoint that runs a Python ModelService bench peer.

Boots a gRPC server hosting :class:`SyntheticBenchServicer`, which serves
the same `bench:<bytes>:<files>` model_name encoding as the Rust bench
server. The existing Rust bench client can target it directly so we get
apples-to-apples gRPC throughput numbers across language implementations.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from concurrent import futures

import grpc

from modelexpress import model_pb2_grpc

from .synthetic_servicer import SyntheticBenchServicer


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ModelExpress Python bench peer")
    p.add_argument("--addr", default="[::]:8001", help="bind address (default %(default)s)")
    p.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="threadpool size for concurrent streaming RPCs (default %(default)s)",
    )
    p.add_argument(
        "--source-buf-size",
        type=int,
        default=16 * 1024 * 1024,
        help="synthetic source-buffer size in bytes (default %(default)s)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("modelexpress_bench.server")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    servicer = SyntheticBenchServicer(source_buf_size=args.source_buf_size)
    model_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)
    bound = server.add_insecure_port(args.addr)
    server.start()
    log.info(
        "bench peer listening on %s (max_workers=%d, source_buf=%d)",
        args.addr if not args.addr.endswith(":0") else f"[::]:{bound}",
        args.max_workers,
        args.source_buf_size,
    )

    stop_event = threading.Event()

    def _shutdown(signum, _frame):
        log.info("received signal %d, shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    stop_event.wait()
    server.stop(grace=5).wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
