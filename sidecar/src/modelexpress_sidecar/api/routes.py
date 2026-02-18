# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""REST API routes for the ModelExpress sidecar."""

import logging
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query

from modelexpress_sidecar.api.schemas import (
    DeleteResponse,
    DownloadRequest,
    DownloadResponse,
    GetModelResponse,
)
from modelexpress_sidecar.services.downloader import ModelDownloader

logger = logging.getLogger(__name__)

router = APIRouter()
downloader = ModelDownloader()


@router.post("/download", response_model=DownloadResponse)
async def download_model(request: DownloadRequest) -> DownloadResponse:
    """Download a model from S3/GCS to local cache."""
    logger.info("Received download request for model: %s", request.model_path)

    try:
        result = await downloader.download(
            model_path=request.model_path,
            cache_dir=request.cache_dir,
            ignore_weights=request.ignore_weights,
            s3_credentials=request.s3_credentials,
            gcs_credentials=request.gcs_credentials,
        )
        return result
    except Exception as e:
        logger.exception("Failed to download model: %s", request.model_path)
        return DownloadResponse(
            success=False,
            error=str(e),
            error_code="DOWNLOAD_ERROR",
        )


@router.get("/models/{model_id:path}", response_model=GetModelResponse)
async def get_model(
    model_id: str,
    cache_dir: str = Query(None, description="Cache directory to search"),
) -> GetModelResponse:
    """Get information about a cached model."""
    # URL decode the model_id
    model_id = unquote(model_id)
    logger.info("Getting model info for: %s", model_id)

    try:
        result = await downloader.get_model(model_id, cache_dir)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except Exception as e:
        logger.exception("Failed to get model: %s", model_id)
        return GetModelResponse(
            exists=False,
            error=str(e),
        )


@router.delete("/models/{model_id:path}", response_model=DeleteResponse)
async def delete_model(
    model_id: str,
    cache_dir: str = Query(None, description="Cache directory to search"),
) -> DeleteResponse:
    """Delete a model from the local cache."""
    # URL decode the model_id
    model_id = unquote(model_id)
    logger.info("Deleting model: %s", model_id)

    try:
        result = await downloader.delete_model(model_id, cache_dir)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except Exception as e:
        logger.exception("Failed to delete model: %s", model_id)
        return DeleteResponse(
            success=False,
            error=str(e),
        )
