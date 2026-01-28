# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application for the ModelExpress sidecar."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from modelexpress_sidecar import __version__
from modelexpress_sidecar.api.routes import router

# Configure logging
log_level = os.environ.get("MODEL_EXPRESS_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("ModelExpress Sidecar starting up (version %s)", __version__)
    logger.info("Log level: %s", log_level)
    yield
    # Shutdown (if needed)


app = FastAPI(
    title="ModelExpress Sidecar",
    description="REST API for Model Streamer integration with ModelExpress",
    version=__version__,
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MODEL_EXPRESS_SIDECAR_PORT", "8002"))
    uvicorn.run(app, host="127.0.0.1", port=port)
