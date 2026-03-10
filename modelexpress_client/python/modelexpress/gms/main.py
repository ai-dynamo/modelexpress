# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MX GMS CLI entry point.

Thin dispatcher that parses config and delegates to engine-specific launcher.

Usage:
    python -m modelexpress.gms --model <model> --engine vllm --mode source
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import EngineType, GmsConfig, GmsMode, WeightSourceType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_launcher(engine: EngineType):
    """Import and return engine-specific launcher module.

    Args:
        engine: Engine type to get launcher for.

    Returns:
        Launcher module with a ``run(config)`` function.

    Raises:
        ValueError: If engine type is not supported.
    """
    if engine == EngineType.VLLM:
        from .launchers import vllm

        return vllm
    raise ValueError(f"Unsupported engine: {engine}")


def parse_args() -> GmsConfig:
    """Parse command-line arguments into a GmsConfig."""
    parser = argparse.ArgumentParser(
        description="MX GMS: multi-GPU model loading with GMS + NIXL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--engine",
        type=str,
        default="vllm",
        choices=[e.value for e in EngineType],
        help="Inference engine type",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="source",
        choices=[m.value for m in GmsMode],
        help="GMS operating mode",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallel size")
    parser.add_argument(
        "--device", type=int, default=0, help="Base device (single-GPU only)"
    )
    parser.add_argument(
        "--mx-server", type=str, default="localhost:8001", help="MX Server address"
    )
    parser.add_argument(
        "--model-name", type=str, default=None, help="Override model name for MX Server"
    )
    parser.add_argument("--dtype", type=str, default="auto", help="Model data type")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    parser.add_argument(
        "--revision", type=str, default=None, help="Model revision to use"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None, help="Maximum model context length"
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Enable expert parallelism for MoE models",
    )

    # Weight source options
    parser.add_argument(
        "--weight-source",
        type=str,
        default="disk",
        choices=[s.value for s in WeightSourceType],
        help="Weight loading source",
    )
    parser.add_argument(
        "--s3-bucket", type=str, default=None, help="S3 bucket for weight source"
    )
    parser.add_argument(
        "--s3-prefix", type=str, default=None, help="S3 key prefix for weight source"
    )
    parser.add_argument(
        "--cache-endpoint",
        type=str,
        default=None,
        help="Cache endpoint for weight source",
    )

    args = parser.parse_args()
    return GmsConfig(
        model=args.model,
        engine=EngineType(args.engine),
        mode=GmsMode(args.mode),
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        device=args.device,
        mx_server=args.mx_server,
        model_name=args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        max_model_len=args.max_model_len,
        enable_expert_parallel=args.enable_expert_parallel,
        weight_source=WeightSourceType(args.weight_source),
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        cache_endpoint=args.cache_endpoint,
    )


def main() -> int:
    """Main entry point for MX GMS."""
    try:
        config = parse_args()
    except (ValueError, SystemExit) as e:
        if isinstance(e, SystemExit) and e.code == 0:
            return 0
        logger.error("Configuration error: %s", e)
        return 1

    logger.info(
        "MX GMS: model=%s engine=%s mode=%s tp=%d ep=%d source=%s",
        config.model,
        config.engine,
        config.mode,
        config.tp_size,
        config.ep_size,
        config.weight_source,
    )

    try:
        launcher = get_launcher(config.engine)
    except ValueError as e:
        logger.error("Launcher error: %s", e)
        return 1

    # launcher.run() blocks until all worker processes are terminated.
    # Each worker calls signal.pause() after loading to keep its NIXL agent
    # and GPU memory alive. mp.spawn(join=True) keeps the main process
    # waiting on the workers. When the container receives SIGTERM (e.g. from
    # kubelet), child processes are killed, mp.spawn returns, and we exit.
    launcher.run(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
