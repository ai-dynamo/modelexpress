# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import gzip
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
pytest.importorskip('azure.storage.blob')
pytest.importorskip('azure.identity')
from azure.core.exceptions import (
    HttpResponseError,
    ResourceNotFoundError,
    ServiceResponseError,
)
import azure.identity
import azure.storage.blob
from azure.storage.blob import BlobServiceClient
import requests
from urllib3.response import HTTPResponse
from modelexpress_rl.azure import AzureBlobReader


class _BlobService:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []
        self.error = None
        self.fail_get_once = None
        self.close = Mock()

    def get_blob_client(self, *, container, blob):
        key = (container, blob)

        def read(operation):
            self.calls.append((operation, key))
            if self.error is not None:
                raise self.error
            if operation == "get" and key == self.fail_get_once:
                self.fail_get_once = None
                raise ServiceResponseError("injected Blob download interruption")
            if key not in self.objects:
                raise ResourceNotFoundError("BlobNotFound")
            return self.objects[key]

        return SimpleNamespace(
            download_blob=lambda **_kwargs: SimpleNamespace(readall=lambda: read("get")),
            get_blob_properties=lambda: SimpleNamespace(size=len(read("size"))),
        )


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_URL", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    client = _BlobService({("models", "v2/shard.safetensors"): b"weights"})
    factory = Mock(return_value=client)
    factory.from_connection_string.return_value = client
    credential = Mock()
    credential_factory = Mock(return_value=credential)
    monkeypatch.setattr(azure.storage.blob, "BlobServiceClient", factory)
    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", credential_factory)
    return SimpleNamespace(
        client=client,
        factory=factory,
        credential=credential,
        credential_factory=credential_factory,
    )


@pytest.fixture
def reader(sdk, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    reader = AzureBlobReader()
    yield reader
    reader.close()


@pytest.mark.parametrize("scheme", ["az", "AZ", "Az"])
@pytest.mark.parametrize("data", [b"", b"checkpoint\x00\xff"])
@pytest.mark.parametrize(
    "blob",
    [
        "model.safetensors.index.json", "V2/Shard.safetensors", "v2/weights%20x",
        "v2/my weights ", " ",
    ],
)
def test_reader_returns_exact_object_bytes_and_size(reader, sdk, scheme, blob, data):
    sdk.client.objects = {("models", blob): data}

    assert reader.get(f"{scheme}://models/{blob}") == data
    assert reader.size(f"{scheme}://models/{blob}") == len(data)
    assert sdk.client.calls == [("get", ("models", blob)), ("size", ("models", blob))]


def test_reader_preserves_gzip_bytes_and_uses_head_for_size(monkeypatch):
    stored = gzip.compress(b"checkpoint\x00\xff" * 128, mtime=0)
    monkeypatch.setenv(
        "AZURE_STORAGE_CONNECTION_STRING",
        "BlobEndpoint=https://testaccount.blob.core.windows.net;"
        "SharedAccessSignature=sig=test",
    )

    methods = []

    def request(_session, method, url, **kwargs):
        assert method in {"GET", "HEAD"}
        methods.append(method)
        response = requests.Response()
        response.status_code = 206 if method == "GET" else 200
        response.headers.update(
            {"Content-Length": str(len(stored)), "Content-Encoding": "gzip"}
        )
        if method == "GET":
            response.headers["Content-Range"] = f"bytes 0-{len(stored) - 1}/{len(stored)}"
        response.raw = HTTPResponse(
            body=BytesIO(stored if method == "GET" else b""),
            headers=response.headers,
            preload_content=False,
            decode_content=False,
            request_method=method,
        )
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    reader = AzureBlobReader()
    try:
        assert reader.get("az://models/checkpoint.gz") == stored
        methods.clear()
        assert reader.size("az://models/checkpoint.gz") == len(stored)
        assert methods == ["HEAD"]
    finally:
        reader.close()


def test_reader_rejects_short_http_response(monkeypatch):
    monkeypatch.setenv(
        "AZURE_STORAGE_CONNECTION_STRING",
        "BlobEndpoint=https://testaccount.blob.core.windows.net;"
        "SharedAccessSignature=sig=test",
    )
    create = BlobServiceClient.from_connection_string
    monkeypatch.setattr(
        BlobServiceClient, "from_connection_string",
        lambda value, **kwargs: create(value, **{**kwargs, "retry_total": 0}),
    )
    attempts = []

    def request(_session, method, url, **kwargs):
        assert method == "GET"
        attempts.append(url)
        assert len(attempts) <= 3
        response = requests.Response()
        response.status_code = 206
        response.headers.update({
            "Content-Length": "5", "Content-Range": "bytes 0-4/5",
        })
        response.raw = HTTPResponse(
            body=BytesIO(b"four"),
            headers=response.headers,
            preload_content=False,
            decode_content=False,
            enforce_content_length=True,
            request_method=method,
        )
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    reader = AzureBlobReader()
    try:
        with pytest.raises((HttpResponseError, ServiceResponseError)):
            reader.get("az://models/checkpoint.safetensors")
        assert attempts
    finally:
        reader.close()


@pytest.mark.parametrize("operation", ["get", "size"])
@pytest.mark.parametrize(
    "uri",
    [
        "",
        "models/shard",
        "s3://models/shard",
        "https://account.blob.core.windows.net/models/shard",
        "az:///shard",
        "az://models",
        "az://models/",
        "az://models//shard",
        "az://models/shard?",
        "az://models/shard#",
        "az://models/shard?sig=secret",
        "az://models/shard#fragment",
        " az://models/shard",
        "\x00az://models/shard",
        "\x1faz://models/shard",
        "junkaz://models/shard",
        "\raz://models/shard",
        "az:\n//models/shard",
        "az://mo\tdels/shard",
        "az://models/sh\rard",
        "az://models/sh\nard",
        "az://models/sh\tard",
        "az://models/shard\n",
        "az://user:secret@models/shard",
        "az://secret@models/shard",
        "az://models:8080/shard",
        "az://models:/shard",
        "az:// /shard",
        "az://\u2003/shard",
    ],
)
def test_malformed_uri_fails_before_storage_access(reader, sdk, uri, operation):
    with pytest.raises(ValueError, match="invalid Azure Blob URI") as error:
        getattr(reader, operation)(uri)

    assert sdk.client.calls == []
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("operation", ["get", "size"])
def test_missing_blob_preserves_not_found_error(reader, operation):
    with pytest.raises(ResourceNotFoundError):
        getattr(reader, operation)("az://models/missing")


@pytest.mark.parametrize("operation", ["get", "size"])
def test_authorization_failure_is_not_treated_as_missing_or_empty(reader, sdk, operation):
    sdk.client.error = HttpResponseError("AuthorizationPermissionMismatch")

    with pytest.raises(HttpResponseError) as error:
        getattr(reader, operation)("az://models/shard")

    assert error.value is sdk.client.error


def test_interrupted_download_does_not_return_partial_contents(reader, sdk):
    sdk.client.error = ServiceResponseError("connection interrupted during readall")

    with pytest.raises(ServiceResponseError) as error:
        reader.get("az://models/shard")

    assert error.value is sdk.client.error


def test_connection_string_takes_precedence_without_loading_identity(sdk, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "test-connection-string")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "unusedaccount")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://unused.example.test")
    original_import = builtins.__import__

    def import_without_identity(name, *args, **kwargs):
        if name == "azure.identity":
            raise ModuleNotFoundError("No module named 'azure.identity'", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_identity)

    reader = AzureBlobReader()

    assert sdk.factory.from_connection_string.call_args.args == ("test-connection-string",)
    sdk.factory.from_connection_string.assert_called_once()
    sdk.factory.assert_not_called()
    sdk.credential_factory.assert_not_called()
    assert reader.get("az://models/v2/shard.safetensors") == b"weights"
    reader.close()
    sdk.client.close.assert_called_once_with()
    sdk.credential.close.assert_not_called()


def test_account_name_uses_default_credential_and_closes_resources(sdk, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", " testaccount ")
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", " ")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", " ")

    reader = AzureBlobReader()

    sdk.credential_factory.assert_called_once_with()
    sdk.factory.assert_called_once()
    assert sdk.factory.call_args.kwargs["account_url"] == (
        "https://testaccount.blob.core.windows.net"
    )
    assert sdk.factory.call_args.kwargs["credential"] is sdk.credential
    sdk.factory.from_connection_string.assert_not_called()
    assert reader.size("az://models/v2/shard.safetensors") == len(b"weights")
    reader.close()
    sdk.client.close.assert_called_once_with()
    sdk.credential.close.assert_called_once_with()


@pytest.mark.parametrize("account_name", [None, "ignoredaccount"])
def test_account_url_uses_default_credential_without_requiring_account_name(
    sdk, monkeypatch, account_name
):
    if account_name is not None:
        monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", account_name)
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", " https://blob.example.test/account ")
    reader = AzureBlobReader()
    try:
        sdk.factory.assert_called_once()
        assert sdk.factory.call_args.kwargs["account_url"] == (
            "https://blob.example.test/account"
        )
        assert sdk.factory.call_args.kwargs["credential"] is sdk.credential
        sdk.credential_factory.assert_called_once_with()
        sdk.factory.from_connection_string.assert_not_called()
        assert reader.get("az://models/v2/shard.safetensors") == b"weights"
    finally:
        reader.close()
    sdk.client.close.assert_called_once()
    sdk.credential.close.assert_called_once()


@pytest.mark.parametrize("account_name", [None, "", " "])
def test_missing_account_configuration_fails_before_credential_creation(
    sdk, monkeypatch, account_name
):
    if account_name is not None:
        monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", account_name)

    with pytest.raises(ValueError, match="AZURE_STORAGE_ACCOUNT_NAME is required"):
        AzureBlobReader()

    sdk.factory.assert_not_called()
    sdk.credential_factory.assert_not_called()


def test_invalid_connection_string_does_not_fall_back_to_identity(sdk, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "invalid")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    sdk.factory.from_connection_string.side_effect = ValueError("invalid connection string")

    with pytest.raises(ValueError, match="invalid connection string"):
        AzureBlobReader()

    sdk.factory.assert_not_called()
    sdk.credential_factory.assert_not_called()


def test_client_close_failure_still_closes_credential(sdk, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    sdk.client.close.side_effect = OSError("client close failed")
    reader = AzureBlobReader()

    with pytest.raises(OSError, match="client close failed"):
        reader.close()

    sdk.client.close.assert_called_once_with()
    sdk.credential.close.assert_called_once_with()


def test_reader_close_is_idempotent_with_real_sdk(monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("AZURE_TENANT_ID", "00000000-0000-0000-0000-000000000001")
    monkeypatch.setenv("AZURE_CLIENT_ID", "00000000-0000-0000-0000-000000000002")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "test-only-secret")
    monkeypatch.delenv("AZURE_TOKEN_CREDENTIALS", raising=False)
    monkeypatch.setattr(
        requests.Session,
        "request",
        Mock(side_effect=AssertionError("unexpected network request")),
    )

    reader = AzureBlobReader()
    # Open real transports, including the environment client-secret credential,
    # so repeated close exercises initialized resources without network requests.
    with reader._client, reader._credential:
        reader.close()
        reader.close()


@pytest.mark.parametrize("module", ["azure.storage.blob", "azure.identity"])
def test_missing_sdk_dependency_explains_azure_extra(sdk, monkeypatch, module):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    original_import = builtins.__import__
    missing = ModuleNotFoundError(f"No module named '{module}'", name=module)

    def import_without_sdk(name, *args, **kwargs):
        if name == module:
            raise missing
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sdk)

    with pytest.raises(ImportError, match=r"modelexpress\[azure\]") as error:
        AzureBlobReader()

    assert error.value.__cause__ is missing


@pytest.mark.parametrize("module", ["azure.storage.blob", "azure.identity"])
@pytest.mark.parametrize(
    "failure",
    [
        ModuleNotFoundError("missing transitive dependency", name="cryptography"),
        ImportError("broken SDK dependency"),
    ],
)
def test_internal_sdk_import_errors_propagate(sdk, monkeypatch, module, failure):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    original_import = builtins.__import__

    def import_with_failure(name, *args, **kwargs):
        if name == module:
            raise failure
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_failure)

    with pytest.raises(ImportError) as error:
        AzureBlobReader()

    assert error.value is failure
