# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Azure Blob transfers for canonical checkpoint artifacts."""

from __future__ import annotations

from urllib.parse import urlsplit
from uuid import uuid4

from modelexpress_rl import envs as rl_envs

_MAX_SINGLE_PUT_BYTES = 64 * 1024**2
_BLOCK_BYTES = 4 * 1024**2
_MAX_BLOCKS = 50_000


class ImmutableAzureConflict(RuntimeError):
    """An immutable Blob already contains different bytes."""


def _parse_uri(uri: str) -> tuple[str, str]:
    """Parse an az:// URI without percent-decoding the blob name."""
    if uri[:5].lower() != "az://" or any(char in uri for char in "\r\n\t"):
        raise ValueError("invalid Azure Blob URI: expected az://container/blob")
    try:
        parsed = urlsplit(uri)
    except ValueError:
        raise ValueError("invalid Azure Blob URI: expected az://container/blob") from None
    if (
        parsed.scheme != "az"
        or not parsed.netloc.strip()
        or "@" in parsed.netloc
        or ":" in parsed.netloc
        or not parsed.path.startswith("/")
        or parsed.path.startswith("//")
        or len(parsed.path) == 1
        or "?" in uri
        or "#" in uri
    ):
        raise ValueError("invalid Azure Blob URI: expected az://container/blob")
    return parsed.netloc, parsed.path[1:]


class AzureBlobReader:
    """Small Azure Blob reader for canonical checkpoint artifacts."""

    def __init__(self) -> None:
        try:
            from azure.storage.blob import BlobServiceClient
        except ModuleNotFoundError as error:
            if error.name not in {"azure", "azure.storage", "azure.storage.blob"}:
                raise
            raise ImportError(
                "Azure Blob reads require the modelexpress[azure] extra"
            ) from error

        from azure.core.pipeline.transport import RequestsTransport
        from requests import Session
        from requests.adapters import HTTPAdapter

        self._download_concurrency = rl_envs.MX_AZURE_DOWNLOAD_CONCURRENCY
        retries = rl_envs.MX_AZURE_MAX_ATTEMPTS - 1
        options = {
            "max_single_get_size": rl_envs.MX_AZURE_DOWNLOAD_RANGE_THRESHOLD_BYTES,
            "max_chunk_get_size": rl_envs.MX_AZURE_DOWNLOAD_RANGE_BYTES,
            "retry_total": retries,
            "retry_connect": retries,
            "retry_read": retries,
            "retry_status": retries,
        }
        pool_size = rl_envs.MX_AZURE_MAX_POOL_CONNECTIONS
        connect_timeout = rl_envs.MX_AZURE_CONNECTION_TIMEOUT_SECONDS
        read_timeout = rl_envs.MX_AZURE_READ_TIMEOUT_SECONDS
        session = Session()
        # Pool size controls connection reuse; shard and per-blob workers bound
        # concurrent requests. Leave request retries to the Blob SDK policy.
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=pool_size, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        self._transport = RequestsTransport(
            session=session, session_owner=True,
            connection_timeout=connect_timeout, read_timeout=read_timeout,
        )
        self._credential: DefaultAzureCredential | None = None
        try:
            self._client = self._create_client(BlobServiceClient, options)
        except Exception:
            try:
                self._transport.close()
            finally:
                if self._credential is not None:
                    self._credential.close()
            raise

    def _create_client(self, factory, options):
        connection_string = rl_envs.AZURE_STORAGE_CONNECTION_STRING
        if connection_string:
            return factory.from_connection_string(
                connection_string, transport=self._transport, **options,
            )

        account_url = rl_envs.AZURE_STORAGE_ACCOUNT_URL
        if not account_url:
            account_name = rl_envs.AZURE_STORAGE_ACCOUNT_NAME
            if not account_name:
                raise ValueError(
                    "AZURE_STORAGE_ACCOUNT_URL or AZURE_STORAGE_ACCOUNT_NAME "
                    "is required when AZURE_STORAGE_CONNECTION_STRING is not set"
                )
            account_url = f"https://{account_name}.blob.core.windows.net"

        try:
            from azure.identity import DefaultAzureCredential
        except ModuleNotFoundError as error:
            if error.name not in {"azure", "azure.identity"}:
                raise
            raise ImportError(
                "Azure Blob reads require the modelexpress[azure] extra"
            ) from error

        self._credential = DefaultAzureCredential()
        return factory(
            account_url=account_url, credential=self._credential,
            transport=self._transport, **options,
        )

    def get(self, uri: str) -> bytes:
        """Read one Blob object."""
        container, blob = _parse_uri(uri)
        client = self._client.get_blob_client(container=container, blob=blob)
        # Preserve stored bytes so downloads match the size reported by Blob properties.
        return client.download_blob(
            decompress=False, max_concurrency=self._download_concurrency,
        ).readall()

    def size(self, uri: str) -> int:
        """Return one Blob object's byte size without downloading its payload."""
        container, blob = _parse_uri(uri)
        client = self._client.get_blob_client(container=container, blob=blob)
        return int(client.get_blob_properties().size)

    def close(self) -> None:
        """Close the underlying SDK client and credential."""
        try:
            self._client.close()
        finally:
            if self._credential is not None:
                self._credential.close()


class AzureBlobClient(AzureBlobReader):
    """Read and immutably publish canonical checkpoint objects."""

    def put(self, *, uri: str, data: bytes) -> None:
        """Conditionally create an object, accepting an identical retry."""
        from azure.core import MatchConditions
        from azure.core.exceptions import ResourceExistsError, ResourceModifiedError

        container, blob = _parse_uri(uri)
        client = self._client.get_blob_client(container=container, blob=blob)
        block_count = (len(data) + _BLOCK_BYTES - 1) // _BLOCK_BYTES
        if block_count > _MAX_BLOCKS:
            raise ValueError("Azure upload exceeds 50,000 blocks")
        try:
            if len(data) <= _MAX_SINGLE_PUT_BYTES:
                client.upload_blob(data, overwrite=False)
            else:
                # SDK upload_blob uses deterministic block IDs. Unique IDs keep
                # concurrent conditional writers from mixing uncommitted blocks.
                attempt = uuid4().hex
                blocks = []
                for index in range(block_count):
                    block_id = f"{attempt}{index:08d}"
                    offset = index * _BLOCK_BYTES
                    client.stage_block(block_id, data[offset : offset + _BLOCK_BYTES])
                    blocks.append(block_id)
                client.commit_block_list(
                    blocks, match_condition=MatchConditions.IfMissing,
                )
        except (ResourceExistsError, ResourceModifiedError) as error:
            if error.error_code not in {"BlobAlreadyExists", "ConditionNotMet"}:
                raise
            if self.size(uri) != len(data) or self.get(uri) != data:
                raise ImmutableAzureConflict(
                    "immutable Azure Blob object contains different bytes"
                ) from error


__all__ = ["AzureBlobReader", "AzureBlobClient", "ImmutableAzureConflict"]
