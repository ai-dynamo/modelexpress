# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verified immutable filesystem and S3 CANONICAL transport tests."""

from __future__ import annotations

import base64
import io
import os
import threading

import boto3
import pytest

import modelexpress.refit.transport.filesystem as filesystem_module
from modelexpress.refit.codec import crc32c_hex
from modelexpress.refit.manifest import DeltaLocation, FilesystemLocation, S3Location
from modelexpress.refit.transport import (
    ImmutableObjectConflict,
    ObjectVerificationError,
    StoredObject,
    TransportClosedError,
    canonical_object_key,
)
from modelexpress.refit.transport.base import validate_relative_key
from modelexpress.refit.transport.filesystem import FilesystemCanonicalTransport
from modelexpress.refit.transport.s3 import S3CanonicalTransport


@pytest.fixture
def managed_transport(request):
    resources = []

    def manage(resource):
        resources.append(resource)
        return resource

    def close_resources():
        for resource in reversed(resources):
            resource.close()

    request.addfinalizer(close_resources)
    return manage


def test_canonical_object_keys_are_identity_bound_and_path_safe():
    first = canonical_object_key(
        "org/model", "base/1", "target/2", "bucket-00000000.mxcd"
    )
    repeated = canonical_object_key(
        "org/model", "base/1", "target/2", "bucket-00000000.mxcd"
    )

    assert first == repeated
    assert first.startswith("canonical/")
    assert "org/model" not in first
    assert first.endswith("/bucket-00000000.mxcd")
    with pytest.raises(ValueError, match="object name"):
        canonical_object_key("m", "b", "t", "../root.json")
    for malformed in ("a//b", "a/./b", "a/../b", "a\\b"):
        with pytest.raises(ValueError, match="relative object key"):
            validate_relative_key(malformed)


def test_filesystem_transport_is_create_only_verified_and_retry_safe(
    tmp_path, managed_transport
):
    transport = managed_transport(FilesystemCanonicalTransport(tmp_path / "objects"))
    data = b"immutable canonical bytes"
    checksum = crc32c_hex(data)

    stored = transport.publish("revision/bucket.mxcd", data, checksum)
    repeated = transport.publish("revision/bucket.mxcd", data, checksum)

    assert repeated == stored
    assert stored.location.filesystem is not None
    assert stored.location.filesystem.path.endswith("revision/bucket.mxcd")
    assert transport.fetch(stored) == data
    transport.verify(stored)

    with pytest.raises(ImmutableObjectConflict, match="immutable object conflict"):
        replacement = b"different canonical bytes"
        transport.publish("revision/bucket.mxcd", replacement, crc32c_hex(replacement))
    with pytest.raises(ObjectVerificationError, match="payload checksum"):
        transport.publish("revision/bad.mxcd", b"payload", "00000000")
    with pytest.raises(ValueError, match="relative object key"):
        transport.publish("../escape", data, checksum)


def test_filesystem_transport_detects_tampering_and_close_keeps_published_bytes(
    tmp_path, managed_transport
):
    transport = managed_transport(FilesystemCanonicalTransport(tmp_path / "objects"))
    data = b"root index"
    stored = transport.publish("revision/root.json", data, crc32c_hex(data))
    path = tmp_path / "objects" / "revision" / "root.json"
    path.write_bytes(b"tampered")

    with pytest.raises(ObjectVerificationError, match="verification failed"):
        transport.fetch(stored)

    transport.close()
    assert path.exists()
    with pytest.raises(TransportClosedError):
        transport.verify(stored)


def test_filesystem_fetch_reads_at_most_the_declared_size_plus_one(
    tmp_path, monkeypatch, managed_transport
):
    transport = managed_transport(FilesystemCanonicalTransport(tmp_path / "objects"))
    stored = transport.publish("revision/root.json", b"root", crc32c_hex(b"root"))
    path = tmp_path / "objects" / "revision" / "root.json"
    path.write_bytes(b"x" * 100_000)
    requested_sizes = []
    real_read = filesystem_module.os.read

    def bounded_read(descriptor, size):
        requested_sizes.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(filesystem_module.os, "read", bounded_read)
    try:
        with pytest.raises(ObjectVerificationError, match="verification failed"):
            transport.fetch(stored)
        assert requested_sizes and max(requested_sizes) <= stored.size + 1
    finally:
        transport.close()


def test_filesystem_resolves_manifest_location_with_verified_bounded_size(
    tmp_path, monkeypatch, managed_transport
):
    transport = managed_transport(FilesystemCanonicalTransport(tmp_path / "objects"))
    data = b"manifest-addressed root"
    stored = transport.publish("revision/root.json", data, crc32c_hex(data))
    requested_sizes = []
    real_read = filesystem_module.os.read

    def bounded_read(descriptor, size):
        requested_sizes.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(filesystem_module.os, "read", bounded_read)

    resolved = transport.resolve(stored.location, stored.checksum, len(data))

    assert resolved == stored
    assert requested_sizes and max(requested_sizes) <= len(data) + 1

    requested_sizes.clear()
    with pytest.raises(ObjectVerificationError, match="maximum_size"):
        transport.resolve(stored.location, stored.checksum, len(data) - 1)
    assert requested_sizes == []

    with pytest.raises(ObjectVerificationError, match="verification failed"):
        transport.resolve(stored.location, "00000000", len(data))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO required")
def test_filesystem_resolve_rejects_fifo_without_blocking(tmp_path, managed_transport):
    transport = managed_transport(FilesystemCanonicalTransport(tmp_path / "objects"))
    fifo = tmp_path / "objects" / "root.fifo"
    os.mkfifo(fifo)
    location = DeltaLocation(filesystem=FilesystemLocation(path=str(fifo)))
    errors = []

    def resolve():
        try:
            transport.resolve(location, "00000000", 1024)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=resolve, daemon=True)
    thread.start()
    thread.join(timeout=0.25)
    completed_without_writer = not thread.is_alive()
    if not completed_without_writer:
        writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.close(writer)
        thread.join(timeout=1)
    assert completed_without_writer, "filesystem FIFO blocked before fstat"
    assert len(errors) == 1
    assert isinstance(errors[0], ObjectVerificationError)


def test_filesystem_concurrent_identical_publish_has_one_immutable_result(
    tmp_path, managed_transport
):
    transports = [
        managed_transport(FilesystemCanonicalTransport(tmp_path / "objects"))
        for _ in range(8)
    ]
    data = b"same bytes"
    checksum = crc32c_hex(data)
    results = []
    errors = []

    def publish(transport):
        try:
            results.append(transport.publish("same/key", data, checksum))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=publish, args=(transport,)) for transport in transports
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 8
    assert len(set(results)) == 1
    assert list((tmp_path / "objects").rglob("*.partial")) == []
    for transport in transports:
        transport.close()


class _S3Error(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}


class _Body(io.BytesIO):
    closed_by_client = False

    def __init__(self, data):
        super().__init__(data)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)

    def close(self):
        self.closed_by_client = True
        super().close()


class _FakeS3:
    def __init__(self) -> None:
        self.objects = {}
        self.calls = []
        self.corrupt_reads = False
        self.bad_head = False
        self.omit_head_version = False
        self.omit_get_version = False
        self.bodies = []

    def put_object(self, **kwargs):
        self.calls.append(("put", kwargs))
        identity = (kwargs["Bucket"], kwargs["Key"])
        if identity in self.objects:
            raise _S3Error("PreconditionFailed")
        version = "version-1"
        self.objects[identity] = (
            bytes(kwargs["Body"]),
            version,
            kwargs["ChecksumCRC32C"],
        )
        return {"VersionId": version}

    def head_object(self, **kwargs):
        self.calls.append(("head", kwargs))
        data, version, checksum = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        response = {
            "ContentLength": len(data) + int(self.bad_head),
            "ChecksumCRC32C": checksum,
            "VersionId": version,
        }
        if self.omit_head_version:
            response.pop("VersionId")
        return response

    def get_object(self, **kwargs):
        self.calls.append(("get", kwargs))
        data, version, _checksum = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        if kwargs.get("VersionId") not in (None, version):
            raise _S3Error("NoSuchVersion")
        body = _Body(data + (b"corrupt" if self.corrupt_reads else b""))
        self.bodies.append(body)
        response = {
            "Body": body,
            "ContentLength": len(body.getvalue()),
            "VersionId": version,
        }
        if self.omit_get_version:
            response.pop("VersionId")
        return response


def test_s3_transport_uses_conditional_create_version_pinning_and_readback_verification(
    managed_transport,
):
    client = _FakeS3()
    transport = managed_transport(
        S3CanonicalTransport("bucket", "prefix", client=client)
    )
    data = b"s3 canonical object"
    checksum = crc32c_hex(data)

    stored = transport.publish("revision/root.json", data, checksum)
    repeated = transport.publish("revision/root.json", data, checksum)

    assert repeated == stored
    assert stored.location.s3.object_version == "version-1"
    put = next(call for operation, call in client.calls if operation == "put")
    assert put["IfNoneMatch"] == "*"
    assert put["ChecksumAlgorithm"] == "CRC32C"
    assert put["ChecksumCRC32C"] == base64.b64encode(
        int(checksum, 16).to_bytes(4, "big")
    ).decode("ascii")
    assert all(
        call.get("VersionId") == "version-1"
        for operation, call in client.calls
        if operation in {"head", "get"}
    )
    assert all(body.read_sizes == [len(data) + 1] for body in client.bodies)


def test_s3_publish_promotes_a_version_discovered_during_verification(
    managed_transport,
):
    class PutOmitsVersionS3(_FakeS3):
        def put_object(self, **kwargs):
            response = super().put_object(**kwargs)
            response.pop("VersionId")
            return response

    client = PutOmitsVersionS3()
    transport = managed_transport(S3CanonicalTransport("bucket", client=client))

    stored = transport.publish("root.json", b"payload", crc32c_hex(b"payload"))

    assert stored.location.s3.object_version == "version-1"
    assert all(
        call.get("VersionId") == "version-1"
        for operation, call in client.calls
        if operation == "get"
    )


def test_s3_owned_client_has_bounded_request_timeouts(monkeypatch, managed_transport):
    captured = {}

    class Client:
        def close(self):
            captured["closed"] = True

    def client(service, **kwargs):
        captured["service"] = service
        captured.update(kwargs)
        return Client()

    monkeypatch.setattr(boto3, "client", client)
    managed_transport(S3CanonicalTransport("bucket", request_timeout_seconds=2.5))

    config = captured["config"]
    assert captured["service"] == "s3"
    assert config.connect_timeout == 2.5
    assert config.read_timeout == 2.5
    assert config.retries == {
        "mode": "standard",
        "total_max_attempts": 3,
    }
    assert config.tcp_keepalive is True


def test_s3_resolves_manifest_location_with_verified_bounded_size(managed_transport):
    client = _FakeS3()
    transport = managed_transport(
        S3CanonicalTransport("bucket", "prefix", client=client)
    )
    data = b"manifest-addressed root"
    stored = transport.publish("revision/root.json", data, crc32c_hex(data))
    manifest_location = DeltaLocation(
        s3=S3Location(
            bucket="bucket",
            key=stored.location.s3.key,
        )
    )
    client.calls.clear()
    client.bodies.clear()

    resolved = transport.resolve(manifest_location, stored.checksum, len(data))

    assert resolved == stored
    assert [operation for operation, _call in client.calls] == ["head", "head", "get"]
    assert client.bodies[0].read_sizes == [len(data) + 1]

    client.calls.clear()
    client.bodies.clear()
    with pytest.raises(ObjectVerificationError, match="maximum_size"):
        transport.resolve(manifest_location, stored.checksum, len(data) - 1)
    assert [operation for operation, _call in client.calls] == ["head"]
    assert client.bodies == []

    client.calls.clear()
    with pytest.raises(ObjectVerificationError, match="checksum"):
        transport.resolve(manifest_location, "00000000", len(data))
    assert [operation for operation, _call in client.calls] == ["head"]


def test_fresh_transport_instances_verify_identical_retries_and_reject_forged_keys(
    tmp_path, managed_transport
):
    data = b"same process-independent bytes"
    first_fs = managed_transport(FilesystemCanonicalTransport(tmp_path / "objects"))
    second_fs = managed_transport(FilesystemCanonicalTransport(tmp_path / "objects"))
    stored = first_fs.publish("revision/root.json", data, crc32c_hex(data))
    assert second_fs.publish("revision/root.json", data, crc32c_hex(data)) == stored

    client = _FakeS3()
    first_s3 = managed_transport(
        S3CanonicalTransport("bucket", "prefix", client=client)
    )
    second_s3 = managed_transport(
        S3CanonicalTransport("bucket", "prefix", client=client)
    )
    stored = first_s3.publish("revision/root.json", data, crc32c_hex(data))
    assert second_s3.publish("revision/root.json", data, crc32c_hex(data)) == stored

    forged = StoredObject(
        DeltaLocation(s3=S3Location(bucket="bucket", key="prefix/a//root.json")),
        crc32c_hex(data),
        len(data),
    )
    with pytest.raises(ObjectVerificationError, match="canonical"):
        second_s3.fetch(forged)


def test_filesystem_transport_rejects_symlinked_parent_before_writing_outside_root(
    tmp_path, managed_transport
):
    root = tmp_path / "objects"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "revision").symlink_to(outside, target_is_directory=True)
    transport = managed_transport(FilesystemCanonicalTransport(root))
    data = b"must stay confined"

    with pytest.raises(ObjectVerificationError, match="outside the configured root"):
        transport.publish("revision/root.json", data, crc32c_hex(data))

    assert not (outside / "root.json").exists()

    replacement = b"replacement"
    with pytest.raises(ObjectVerificationError, match="outside the configured root"):
        transport.publish("revision/root.json", replacement, crc32c_hex(replacement))


def test_filesystem_transport_never_follows_an_existing_destination_symlink(
    tmp_path, managed_transport
):
    root = tmp_path / "objects"
    outside = tmp_path / "outside.bin"
    root.mkdir()
    (root / "revision").mkdir()
    outside.write_bytes(b"outside")
    (root / "revision" / "root.json").symlink_to(outside)
    transport = managed_transport(FilesystemCanonicalTransport(root))

    with pytest.raises(ObjectVerificationError, match="symlink"):
        transport.publish("revision/root.json", b"outside", crc32c_hex(b"outside"))

    assert outside.read_bytes() == b"outside"


class _ConflictOnceS3(_FakeS3):
    def __init__(self) -> None:
        super().__init__()
        self.conflicts = 1

    def put_object(self, **kwargs):
        if self.conflicts:
            self.conflicts -= 1
            self.calls.append(("put", kwargs))
            raise _S3Error("ConditionalRequestConflict")
        return super().put_object(**kwargs)


def test_s3_conditional_conflict_retries_the_create_when_no_winner_is_readable(
    managed_transport,
):
    client = _ConflictOnceS3()
    transport = managed_transport(S3CanonicalTransport("bucket", client=client))

    stored = transport.publish("root.json", b"payload", crc32c_hex(b"payload"))

    assert stored.location.s3.object_version == "version-1"
    assert len([call for operation, call in client.calls if operation == "put"]) == 2


def test_s3_rejects_a_malformed_object_version_before_return(managed_transport):
    class MalformedVersionS3(_FakeS3):
        def put_object(self, **kwargs):
            response = super().put_object(**kwargs)
            identity = (kwargs["Bucket"], kwargs["Key"])
            data, _version, checksum = self.objects[identity]
            self.objects[identity] = (data, 7, checksum)
            response["VersionId"] = 7
            return response

    transport = managed_transport(
        S3CanonicalTransport("bucket", client=MalformedVersionS3())
    )
    with pytest.raises(ObjectVerificationError, match="version"):
        transport.publish("root.json", b"payload", crc32c_hex(b"payload"))


@pytest.mark.parametrize("missing_from", ["head", "get"])
def test_s3_pinned_object_requires_exact_version_proof(missing_from, managed_transport):
    client = _FakeS3()
    transport = managed_transport(S3CanonicalTransport("bucket", client=client))
    stored = transport.publish("root.json", b"payload", crc32c_hex(b"payload"))
    if missing_from == "head":
        client.omit_head_version = True
    else:
        client.omit_get_version = True

    with pytest.raises(ObjectVerificationError, match="version"):
        transport.verify(stored)


def test_s3_transport_fails_before_return_on_unreadable_or_unverified_objects(
    managed_transport,
):
    client = _FakeS3()
    client.bad_head = True
    transport = managed_transport(S3CanonicalTransport("bucket", client=client))

    with pytest.raises(ObjectVerificationError, match="metadata"):
        transport.publish("bad-head", b"payload", crc32c_hex(b"payload"))

    client = _FakeS3()
    transport = managed_transport(S3CanonicalTransport("bucket", client=client))
    stored = transport.publish("corrupt", b"payload", crc32c_hex(b"payload"))
    client.corrupt_reads = True
    with pytest.raises(ObjectVerificationError, match="verification failed"):
        transport.fetch(stored)
