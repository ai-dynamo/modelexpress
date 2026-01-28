# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sidecar REST API."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from modelexpress_sidecar.main import app


@pytest_asyncio.fixture
async def client():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Test the health endpoint returns healthy status."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_get_nonexistent_model(client: AsyncClient):
    """Test getting a model that doesn't exist returns 404."""
    response = await client.get("/api/v1/models/s3/nonexistent/bucket/model")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_model(client: AsyncClient):
    """Test deleting a model that doesn't exist returns 404."""
    response = await client.delete("/api/v1/models/s3/nonexistent/bucket/model")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_download_invalid_uri(client: AsyncClient):
    """Test downloading with an invalid URI returns error."""
    response = await client.post(
        "/api/v1/download",
        json={
            "model_path": "invalid://bucket/path",
            "cache_dir": "/tmp/cache",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "error" in data
