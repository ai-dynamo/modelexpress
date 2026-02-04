# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS sidecar for loading model weights into GMS.

This script loads model weights from disk and writes them to GMS, enabling
vLLM engines to import weights without loading from disk.

Usage:
    python -m modelexpress.gms_sidecar --model <model> --device <id>

Example:
    python -m modelexpress.gms_sidecar \
        --model meta-llama/Llama-3.2-1B \
        --device 0
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from enum import StrEnum
from typing import Optional

import torch
from pydantic import BaseModel, Field, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DType(StrEnum):
    """Supported model data types."""

    AUTO = "auto"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class MxGmsSidecarConfig(BaseModel):
    """Configuration for the GMS sidecar.

    Attributes:
        model: HuggingFace model name or local path.
        device: CUDA device index.
        socket_path: GMS socket path. Auto-derived from device if not provided.
        dtype: Model data type.
        trust_remote_code: Whether to trust remote code from HuggingFace.
        stay_running: Keep process alive after commit for debugging.
        revision: Model revision to use.
        max_model_len: Maximum model context length (None = auto).
    """

    model: str = Field(..., description="HuggingFace model name or local path")
    device: int = Field(default=0, ge=0, description="CUDA device index")
    socket_path: Optional[str] = Field(
        default=None, description="GMS socket path (auto-derived if None)"
    )
    dtype: DType = Field(default=DType.AUTO, description="Model data type")
    trust_remote_code: bool = Field(
        default=False, description="Trust remote code from HuggingFace"
    )
    stay_running: bool = Field(
        default=False, description="Keep process alive after commit"
    )
    revision: Optional[str] = Field(default=None, description="Model revision")
    max_model_len: Optional[int] = Field(
        default=None, ge=1, description="Maximum model context length"
    )

    @model_validator(mode="after")
    def validate_device(self) -> "MxGmsSidecarConfig":
        """Validate CUDA device is available."""
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        device_count = torch.cuda.device_count()
        if self.device >= device_count:
            raise ValueError(
                f"Device {self.device} not available. "
                f"Only {device_count} devices found."
            )
        return self


def parse_args() -> MxGmsSidecarConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GMS sidecar for loading model weights into GMS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index",
    )
    parser.add_argument(
        "--socket-path",
        type=str,
        default=None,
        help="GMS socket path (auto-derived from device if not provided)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=[d.value for d in DType],
        default=DType.AUTO.value,
        help="Model data type",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    parser.add_argument(
        "--stay-running",
        action="store_true",
        help="Keep process alive after commit (for debugging)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length (None = auto)",
    )

    args = parser.parse_args()

    return MxGmsSidecarConfig(
        model=args.model,
        device=args.device,
        socket_path=args.socket_path,
        dtype=DType(args.dtype),
        trust_remote_code=args.trust_remote_code,
        stay_running=args.stay_running,
        revision=args.revision,
        max_model_len=args.max_model_len,
    )


def build_vllm_configs(config: MxGmsSidecarConfig):
    """Build minimal vLLM configuration objects.

    Args:
        config: Sidecar configuration.

    Returns:
        Tuple of (vllm_config, model_config, load_config).
    """
    from vllm.engine.arg_utils import AsyncEngineArgs

    # Map dtype string to torch dtype
    dtype_map = {
        DType.AUTO: "auto",
        DType.FLOAT16: "float16",
        DType.BFLOAT16: "bfloat16",
        DType.FLOAT32: "float32",
    }

    # Build engine args with minimal config
    engine_args = AsyncEngineArgs(
        model=config.model,
        dtype=dtype_map[config.dtype],
        trust_remote_code=config.trust_remote_code,
        revision=config.revision,
        max_model_len=config.max_model_len,
        # Disable unnecessary features for weight loading
        disable_log_stats=True,
        enable_prefix_caching=False,
    )

    # Create vLLM config
    vllm_config = engine_args.create_engine_config()
    model_config = vllm_config.model_config
    load_config = vllm_config.load_config

    return vllm_config, model_config, load_config


def main() -> int:
    """Main entry point for the GMS sidecar."""
    try:
        config = parse_args()
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1

    logger.info("Starting GMS sidecar for model: %s", config.model)
    logger.info("Using CUDA device: %d", config.device)

    # Set CUDA device before any CUDA operations
    torch.cuda.set_device(config.device)

    # Build vLLM configs
    try:
        vllm_config, model_config, load_config = build_vllm_configs(config)
    except Exception as e:
        logger.error("Failed to build vLLM configs: %s", e)
        return 1

    # Import loader here to allow graceful failure if GMS not installed
    from modelexpress.gms_loader import MxGmsSourceLoader

    # Create loader and load model
    loader = MxGmsSourceLoader(load_config)

    try:
        model = loader.load_model(vllm_config, model_config)
        logger.info("Model loaded and committed to GMS successfully")

        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %.2fB", param_count / 1e9)

    except Exception as e:
        logger.error("Failed to load model: %s", e)
        loader.close()
        return 1

    if config.stay_running:
        logger.info("Staying running (--stay-running). Press Ctrl+C to exit.")

        # Set up signal handler for graceful shutdown
        shutdown_event = False

        def signal_handler(signum, frame):
            nonlocal shutdown_event
            logger.info("Received signal %d, shutting down...", signum)
            shutdown_event = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Wait for shutdown signal
        while not shutdown_event:
            signal.pause()

        loader.close()
        logger.info("Shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
