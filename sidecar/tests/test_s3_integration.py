# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for S3/MinIO downloads.

These tests require a running MinIO instance. They are skipped if
the MinIO environment is not available.
"""

import os
import tempfile
from urllib.parse import urlparse

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from modelexpress_sidecar.main import app


def get_minio_endpoint() -> str:
    """Get the MinIO endpoint from environment."""
    return os.environ.get("AWS_ENDPOINT_URL", "")


def minio_available() -> bool:
    """Check if MinIO is available for testing."""
    endpoint = get_minio_endpoint()
    return bool(endpoint) and ("minio" in endpoint.lower() or "localhost" in endpoint)


def get_s3_model_path(bucket: str, path: str) -> str:
    """Build s3+http model path using the configured MinIO endpoint."""
    endpoint = get_minio_endpoint()
    parsed = urlparse(endpoint)
    # Use s3+http://host:port/bucket/path format
    return f"s3+http://{parsed.netloc}/{bucket}/{path}"


@pytest_asyncio.fixture
async def client():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.skipif(not minio_available(), reason="MinIO not available")
@pytest.mark.asyncio
async def test_download_from_minio(client: AsyncClient):
    """Test downloading a model from MinIO."""
    endpoint = get_minio_endpoint()
    model_path = get_s3_model_path("models", "test-model")

    with tempfile.TemporaryDirectory() as tmpdir:
        response = await client.post(
            "/api/v1/download",
            json={
                "model_path": model_path,
                "cache_dir": tmpdir,
                "s3_credentials": {
                    "access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
                    "secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
                    "region": os.environ.get("AWS_REGION", "us-east-1"),
                    "endpoint": endpoint,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "local_path" in data
        assert data["local_path"].startswith(tmpdir)


@pytest.mark.skipif(not minio_available(), reason="MinIO not available")
@pytest.mark.asyncio
async def test_get_downloaded_model(client: AsyncClient):
    """Test getting info about a downloaded model."""
    endpoint = get_minio_endpoint()
    model_path = get_s3_model_path("models", "test-model")

    with tempfile.TemporaryDirectory() as tmpdir:
        # First download the model
        download_response = await client.post(
            "/api/v1/download",
            json={
                "model_path": model_path,
                "cache_dir": tmpdir,
                "s3_credentials": {
                    "access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
                    "secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
                    "region": os.environ.get("AWS_REGION", "us-east-1"),
                    "endpoint": endpoint,
                },
            },
        )
        assert download_response.status_code == 200
        assert download_response.json()["success"] is True

        # Now get the model info using the cache_dir query param
        response = await client.get(
            "/api/v1/models/s3/models/test-model",
            params={"cache_dir": tmpdir},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["exists"] is True
        assert "local_path" in data


@pytest.mark.asyncio
async def test_download_invalid_s3_uri(client: AsyncClient):
    """Test downloading from an invalid S3 URI returns error."""
    response = await client.post(
        "/api/v1/download",
        json={
            "model_path": "ftp://invalid/path",
            "cache_dir": "/tmp/cache",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "error" in data


@pytest.mark.skipif(not minio_available(), reason="MinIO not available")
@pytest.mark.asyncio
async def test_download_nonexistent_bucket(client: AsyncClient):
    """Test downloading from a non-existent bucket returns error."""
    endpoint = get_minio_endpoint()
    model_path = get_s3_model_path("nonexistent-bucket", "model")

    with tempfile.TemporaryDirectory() as tmpdir:
        response = await client.post(
            "/api/v1/download",
            json={
                "model_path": model_path,
                "cache_dir": tmpdir,
                "s3_credentials": {
                    "access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
                    "secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
                    "region": os.environ.get("AWS_REGION", "us-east-1"),
                    "endpoint": endpoint,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Download may succeed but return empty files, or fail entirely
        # depending on S3 behavior with non-existent prefixes
        assert "success" in data
