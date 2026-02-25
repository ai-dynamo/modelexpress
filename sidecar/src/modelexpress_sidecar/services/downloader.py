# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model Streamer integration for downloading models from S3/GCS."""

import asyncio
import logging
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Optional
from urllib.parse import urlparse

from modelexpress_sidecar.api.schemas import (
    DeleteResponse,
    DownloadResponse,
    GcsCredentials,
    GetModelResponse,
    S3Credentials,
)

logger = logging.getLogger(__name__)

# Cache subdirectory name used by the sidecar
CACHE_SUBDIR = "model-streamer"

# Weight file extensions to filter when ignore_weights is True
WEIGHT_EXTENSIONS = {".bin", ".safetensors", ".h5", ".msgpack", ".ckpt"}

# Files to ignore during download
IGNORED_FILES = {".gitattributes", ".gitignore", "README.md"}

# Image file extensions to ignore
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".svg", ".ico", ".bmp", ".tiff", ".tif",
}


def is_weight_file(filename: str) -> bool:
    """Check if a file is a model weight file."""
    filename = filename.lower()
    return any(filename.endswith(ext) for ext in WEIGHT_EXTENSIONS)


def is_ignored_file(filename: str) -> bool:
    """Check if a file should be ignored."""
    return filename in IGNORED_FILES


def is_image_file(filename: str) -> bool:
    """Check if a file is an image."""
    filename = filename.lower()
    return any(filename.endswith(ext) for ext in IMAGE_EXTENSIONS)


class ModelDownloader:
    """Service for downloading models using Model Streamer."""

    def __init__(self):
        """Initialize the downloader."""
        self.default_cache_dir = Path(
            os.environ.get("MODEL_EXPRESS_CACHE_DIRECTORY", "/app/cache")
        )
        self._credentials_lock = asyncio.Lock()

    def _parse_model_uri(self, model_path: str) -> tuple[str, str, str]:
        """
        Parse a model URI into (scheme, bucket, path).

        Supported formats:
        - s3://bucket/path/to/model
        - gs://bucket/path/to/model
        - s3+http://endpoint/bucket/path (MinIO)
        """
        parsed = urlparse(model_path)

        if parsed.scheme in ("s3", "gs"):
            bucket = parsed.netloc
            path = parsed.path.lstrip("/")
            return parsed.scheme, bucket, path
        elif parsed.scheme in ("s3+http", "s3+https"):
            # For MinIO: s3+http://endpoint:port/bucket/path
            # The netloc is the endpoint, first path component is bucket
            bucket, _, path = parsed.path.lstrip("/").partition("/")
            return "s3", bucket, path
        else:
            raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")

    def _get_cache_path(self, model_path: str, cache_dir: str) -> Path:
        """Get the local cache path for a model."""
        scheme, bucket, path = self._parse_model_uri(model_path)
        cache_base = Path(cache_dir) if cache_dir else self.default_cache_dir
        return cache_base / CACHE_SUBDIR / scheme / bucket / path

    def _resolve_model_id(self, model_id: str, cache_dir: Optional[str]) -> Path:
        """Convert a model_id (e.g. s3/bucket/path) to a local cache path."""
        parts = model_id.split("/", 2)
        if len(parts) < 2:
            raise FileNotFoundError(f"Invalid model_id format: {model_id}")

        scheme = parts[0]
        bucket = parts[1]
        path = parts[2] if len(parts) > 2 else ""

        cache_base = Path(cache_dir) if cache_dir else self.default_cache_dir
        resolved = (cache_base / CACHE_SUBDIR / scheme / bucket / path).resolve()

        # Prevent path traversal (e.g. model_id = "../../etc/passwd")
        if not str(resolved).startswith(str((cache_base / CACHE_SUBDIR).resolve())):
            raise ValueError(f"Invalid model_id: path escapes cache directory")

        return resolved

    def _setup_credentials(
        self,
        s3_credentials: Optional[S3Credentials],
        gcs_credentials: Optional[GcsCredentials],
    ) -> dict[str, str]:
        """Set up environment variables for credentials and return original values."""
        original_env = {}

        if s3_credentials:
            if s3_credentials.access_key_id:
                original_env["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "")
                os.environ["AWS_ACCESS_KEY_ID"] = s3_credentials.access_key_id
            if s3_credentials.secret_access_key:
                original_env["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
                os.environ["AWS_SECRET_ACCESS_KEY"] = s3_credentials.secret_access_key
            if s3_credentials.region:
                original_env["AWS_REGION"] = os.environ.get("AWS_REGION", "")
                os.environ["AWS_REGION"] = s3_credentials.region
            if s3_credentials.endpoint:
                original_env["AWS_ENDPOINT_URL"] = os.environ.get("AWS_ENDPOINT_URL", "")
                os.environ["AWS_ENDPOINT_URL"] = s3_credentials.endpoint

        if gcs_credentials:
            if gcs_credentials.credentials_file:
                original_env["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get(
                    "GOOGLE_APPLICATION_CREDENTIALS", ""
                )
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcs_credentials.credentials_file

        return original_env

    def _restore_credentials(self, original_env: dict[str, str]) -> None:
        """Restore original environment variables."""
        for key, value in original_env.items():
            if value:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    async def download(
        self,
        model_path: str,
        cache_dir: str,
        ignore_weights: bool = False,
        s3_credentials: Optional[S3Credentials] = None,
        gcs_credentials: Optional[GcsCredentials] = None,
    ) -> DownloadResponse:
        """
        Download a model from S3/GCS to local cache.

        Uses Model Streamer to efficiently stream files from cloud storage.
        """
        logger.info("Starting download of model: %s", model_path)

        # Hold the lock while credentials are in the environment so
        # concurrent requests don't overwrite each other's values.
        async with self._credentials_lock:
            original_env = self._setup_credentials(s3_credentials, gcs_credentials)

            try:
                # Parse the model URI
                scheme, bucket, path = self._parse_model_uri(model_path)
                logger.info("Parsed URI - scheme: %s, bucket: %s, path: %s", scheme, bucket, path)

                # Determine local cache path
                local_path = self._get_cache_path(model_path, cache_dir)
                local_path.mkdir(parents=True, exist_ok=True)
                logger.info("Local cache path: %s", local_path)

                # Download using Model Streamer
                downloaded_files = await self._download_with_model_streamer(
                    model_path=model_path,
                    local_path=local_path,
                    ignore_weights=ignore_weights,
                )

                # Filter to files that actually exist on disk
                downloaded_files = [
                    f for f in downloaded_files if (local_path / f).exists()
                ]

                total_size = sum(
                    (local_path / f).stat().st_size for f in downloaded_files
                )

                logger.info(
                    "Successfully downloaded %d files (%d bytes) to %s",
                    len(downloaded_files),
                    total_size,
                    local_path,
                )

                return DownloadResponse(
                    success=True,
                    local_path=str(local_path),
                    files=downloaded_files,
                    total_size=total_size,
                )

            except Exception as e:
                logger.exception("Failed to download model %s", model_path)
                return DownloadResponse(
                    success=False,
                    error=str(e),
                    error_code="DOWNLOAD_ERROR",
                )
            finally:
                self._restore_credentials(original_env)

    async def _download_with_model_streamer(
        self,
        model_path: str,
        local_path: Path,
        ignore_weights: bool,
    ) -> list[str]:
        """
        Download files using Model Streamer SDK.

        Model Streamer is primarily designed to stream tensors to GPU,
        but we can use it to download files to local storage.
        """
        downloaded_files = []

        try:
            # Try to import Model Streamer
            from runai_model_streamer import SafetensorsStreamer

            # For now, we'll use boto3 for actual file downloads
            # Model Streamer is better suited for streaming to GPU
            # In phase 2, we can use Model Streamer's streaming capabilities
            downloaded_files = await asyncio.to_thread(
                self._download_with_boto3,
                model_path=model_path,
                local_path=local_path,
                ignore_weights=ignore_weights,
            )

        except ImportError:
            logger.warning("Model Streamer not available, falling back to boto3")
            downloaded_files = await asyncio.to_thread(
                self._download_with_boto3,
                model_path=model_path,
                local_path=local_path,
                ignore_weights=ignore_weights,
            )

        return downloaded_files

    def _download_with_boto3(
        self,
        model_path: str,
        local_path: Path,
        ignore_weights: bool,
    ) -> list[str]:
        """
        Download files from S3 using boto3.

        This is a fallback when Model Streamer is not available or for
        non-SafeTensors files.
        """
        import boto3
        from botocore.config import Config

        scheme, bucket, prefix = self._parse_model_uri(model_path)

        if scheme != "s3":
            raise ValueError(f"boto3 download only supports S3, got: {scheme}")

        # Create S3 client
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        config = Config(signature_version="s3v4")

        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            config=config,
        )

        # List objects in the bucket with the given prefix
        paginator = s3_client.get_paginator("list_objects_v2")
        downloaded_files = []

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Get relative path from prefix
                rel_path = key[len(prefix):].lstrip("/") if key.startswith(prefix) else key
                filename = PurePosixPath(rel_path).name

                # Skip ignored files
                if is_ignored_file(filename):
                    logger.debug("Skipping ignored file: %s", filename)
                    continue

                # Skip image files
                if is_image_file(filename):
                    logger.debug("Skipping image file: %s", filename)
                    continue

                # Skip weight files if requested
                if ignore_weights and is_weight_file(filename):
                    logger.debug("Skipping weight file: %s", filename)
                    continue

                # Download the file
                local_file = local_path / rel_path
                local_file.parent.mkdir(parents=True, exist_ok=True)

                logger.info("Downloading: %s -> %s", key, local_file)
                s3_client.download_file(bucket, key, str(local_file))
                downloaded_files.append(rel_path)

        return downloaded_files

    async def get_model(
        self,
        model_id: str,
        cache_dir: Optional[str],
    ) -> GetModelResponse:
        """Get information about a cached model."""
        local_path = self._resolve_model_id(model_id, cache_dir)

        if not local_path.exists():
            raise FileNotFoundError(f"Model not found at {local_path}")

        # List files in the model directory
        files = []
        total_size = 0
        for f in local_path.rglob("*"):
            if f.is_file():
                rel_path = str(f.relative_to(local_path))
                files.append(rel_path)
                total_size += f.stat().st_size

        return GetModelResponse(
            exists=True,
            local_path=str(local_path),
            files=files,
            total_size=total_size,
        )

    async def delete_model(self, model_id: str, cache_dir: Optional[str] = None) -> DeleteResponse:
        """Delete a model from the local cache."""
        local_path = self._resolve_model_id(model_id, cache_dir)

        if not local_path.exists():
            raise FileNotFoundError(f"Model not found at {local_path}")

        # Delete the directory
        shutil.rmtree(local_path)
        logger.info("Deleted model at: %s", local_path)

        return DeleteResponse(
            success=True,
            deleted_path=str(local_path),
        )
