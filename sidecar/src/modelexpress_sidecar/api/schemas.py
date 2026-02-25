# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for the sidecar REST API."""

from typing import Optional

from pydantic import BaseModel, Field


class S3Credentials(BaseModel):
    """S3 credentials for accessing S3 or S3-compatible storage."""

    access_key_id: Optional[str] = Field(None, description="AWS access key ID")
    secret_access_key: Optional[str] = Field(None, description="AWS secret access key")
    region: Optional[str] = Field("us-east-1", description="AWS region")
    endpoint: Optional[str] = Field(
        None, description="Custom S3 endpoint (for MinIO or other S3-compatible storage)"
    )


class GcsCredentials(BaseModel):
    """GCS credentials for accessing Google Cloud Storage."""

    credentials_file: Optional[str] = Field(
        None, description="Path to GCS credentials JSON file"
    )
    credentials_json: Optional[str] = Field(
        None, description="Base64-encoded GCS credentials JSON"
    )


class DownloadRequest(BaseModel):
    """Request to download a model from S3/GCS."""

    model_path: str = Field(
        ..., description="Model path URI (e.g., s3://bucket/path or gs://bucket/path)"
    )
    cache_dir: str = Field(..., description="Local cache directory")
    ignore_weights: bool = Field(
        False, description="Whether to ignore weight files during download"
    )
    s3_credentials: Optional[S3Credentials] = Field(
        None, description="S3 credentials (if not using environment variables)"
    )
    gcs_credentials: Optional[GcsCredentials] = Field(
        None, description="GCS credentials (if not using environment variables)"
    )


class DownloadResponse(BaseModel):
    """Response from a model download request."""

    success: bool = Field(..., description="Whether the download was successful")
    local_path: Optional[str] = Field(
        None, description="Local path where the model was downloaded"
    )
    files: Optional[list[str]] = Field(None, description="List of downloaded files")
    total_size: Optional[int] = Field(
        None, description="Total size of downloaded files in bytes"
    )
    error: Optional[str] = Field(None, description="Error message if download failed")
    error_code: Optional[str] = Field(None, description="Error code for categorization")


class GetModelResponse(BaseModel):
    """Response for getting model information."""

    exists: bool = Field(..., description="Whether the model exists in cache")
    local_path: Optional[str] = Field(None, description="Local path to the model")
    files: Optional[list[str]] = Field(None, description="List of files in the model")
    total_size: Optional[int] = Field(
        None, description="Total size of model files in bytes"
    )
    error: Optional[str] = Field(None, description="Error message if lookup failed")


class DeleteResponse(BaseModel):
    """Response from a model deletion request."""

    success: bool = Field(..., description="Whether the deletion was successful")
    deleted_path: Optional[str] = Field(None, description="Path that was deleted")
    error: Optional[str] = Field(None, description="Error message if deletion failed")
