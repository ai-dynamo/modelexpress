# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-owned cache identity, placement, and publication contracts."""

import hashlib
import json
import tarfile
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from modelexpress import p2p_pb2
from modelexpress.metadata import artifact_lifecycle
from modelexpress.metadata.artifact_transfer import (
    ArtifactCacheRoot,
    triton_cache_artifact_transfer,
)
from modelexpress.metadata.framework_cache import (
    FrameworkCacheArtifact,
    framework_cache_entries,
)
from modelexpress.metadata.source_id import compute_mx_source_id


def _cache(tmp_path, **overrides):
    values = dict(
        name="startup_plan",
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        backend_framework_version="0.29.1",
        compatibility_key="startup-fingerprint",
        schema="1",
        root=tmp_path / "source",
    )
    values.update(overrides)
    return FrameworkCacheArtifact(**values)


def _entry(tmp_path, **overrides):
    return _cache(tmp_path, **overrides).entry(
        "deepseek-ai/DeepSeek-V3", tmp_path / "bundle"
    )


def test_framework_cache_requires_explicit_compatible_identity_and_placement(tmp_path):
    transfer, identity = _entry(tmp_path)

    assert transfer.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_FRAMEWORK_CACHE
    assert identity.mx_source_type == transfer.mx_source_type
    assert identity.model_name == "deepseek-ai/DeepSeek-V3"
    assert identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_VLLM
    assert identity.backend_framework_version == "0.29.1"
    assert identity.compile_config_digest == hashlib.sha256(
        json.dumps({"compatibility_key": "startup-fingerprint"}).encode()
    ).hexdigest()
    assert dict(identity.extra_parameters) == {
        "framework_cache_name": "startup_plan",
        "framework_cache_schema": "1",
    }
    assert transfer.roots == (
        ArtifactCacheRoot("primary", tmp_path / "source", tmp_path / "source"),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"name": ""},
        {"name": "../startup_plan"},
        {"name": "/startup_plan"},
        {"name": "StartupPlan"},
        {"backend_framework": p2p_pb2.BACKEND_FRAMEWORK_UNKNOWN},
        {"backend_framework": 99},
        {"backend_framework_version": ""},
        {"compatibility_key": ""},
        {"schema": ""},
    ],
)
def test_framework_cache_rejects_underspecified_or_unsafe_entries(tmp_path, changes):
    with pytest.raises(ValueError):
        _entry(tmp_path, **changes)


@pytest.mark.parametrize(
    "framework",
    [p2p_pb2.BACKEND_FRAMEWORK_SGLANG, p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM],
)
def test_framework_cache_accepts_other_engines_without_new_source_types(tmp_path, framework):
    _, identity = _entry(tmp_path, backend_framework=framework, name="engine_plan")
    assert identity.backend_framework == framework
    assert identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_FRAMEWORK_CACHE


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "execution_plan"},
        {"backend_framework": p2p_pb2.BACKEND_FRAMEWORK_SGLANG},
        {"backend_framework_version": "0.30.0"},
        {"compatibility_key": "different-rank-or-config"},
        {"schema": "2"},
    ],
)
def test_framework_cache_compatibility_dimensions_partition_discovery(tmp_path, changes):
    _, original = _entry(tmp_path)
    _, different = _entry(tmp_path, **changes)
    assert compute_mx_source_id(original) != compute_mx_source_id(different)


@pytest.mark.parametrize("keys", [("AbCd", "aBcD"), (" key", "key ")])
def test_opaque_compatibility_keys_preserve_case_and_whitespace(tmp_path, keys):
    first, first_identity = _entry(tmp_path, compatibility_key=keys[0])
    second, second_identity = _entry(tmp_path, compatibility_key=keys[1])

    assert compute_mx_source_id(first_identity) != compute_mx_source_id(second_identity)
    assert first.bundle_root != second.bundle_root


@pytest.mark.parametrize("operation", ["prepare_source", "target_file_paths"])
def test_framework_cache_name_cannot_interfere_with_legacy_bundle(tmp_path, operation):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.bin").write_bytes(b"compiled kernel")
    framework, _ = _entry(tmp_path, name="triton_cache")
    legacy = triton_cache_artifact_transfer(
        source, tmp_path / "legacy-target", tmp_path / "bundle" / "triton_cache"
    )

    legacy_bundle = legacy.prepare_source()
    legacy_bytes = legacy_bundle.tar_path.read_bytes()
    framework.prepare_source()
    result = getattr(legacy, operation)()

    assert result is not None
    assert legacy_bundle.tar_path.read_bytes() == legacy_bytes
    assert not framework.bundle_root.is_relative_to(legacy.bundle_root)


def test_framework_cache_relocates_into_receiver_owned_root(tmp_path):
    source = tmp_path / "pod-a" / "cache" / "startup_plan"
    target = tmp_path / "pod-b" / "different-root" / "startup_plan"
    source.mkdir(parents=True)
    plan = {
        "schema": 1,
        "fingerprint": "899373b27ccb0598",
        "kv_cache_memory_bytes": 168148837315,
        "free_memory_baseline": 284845211648,
    }
    filename = "startup_plan_899373b27ccb0598.json"
    (source / filename).write_text(json.dumps(plan))
    cache = _cache(tmp_path)
    sender, sender_identity = replace(cache, root=source).entry(
        "deepseek-ai/DeepSeek-V3", tmp_path / "sender-bundle"
    )
    receiver, receiver_identity = replace(cache, root=target).entry(
        "deepseek-ai/DeepSeek-V3", tmp_path / "receiver-bundle"
    )

    bundle = sender.prepare_source()
    staged = receiver.target_file_paths()
    assert len(staged) == len(bundle.manifest.files) == 1
    staged[0].write_bytes(bundle.tar_path.read_bytes())
    received_file = p2p_pb2.ArtifactManifestFile()
    received_file.CopyFrom(bundle.manifest.files[0])
    received_file.path = str(staged[0])
    header = p2p_pb2.GetArtifactManifestHeaderResponse(
        artifact_id=bundle.artifact_id,
        mx_source_type=sender.mx_source_type,
        files=[received_file],
        total_size=received_file.size,
    )
    receiver.install(header)

    assert compute_mx_source_id(sender_identity) == compute_mx_source_id(receiver_identity)
    assert json.loads((target / filename).read_text()) == plan
    assert sorted(path.name for path in target.iterdir()) == [filename]
    assert json.loads((source / filename).read_text()) == plan


def test_same_name_different_keys_keep_prepared_bundles_independent(tmp_path):
    first_root = tmp_path / "cache-first"
    second_root = tmp_path / "cache-second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "plan.json").write_text('{"kv_cache_memory_bytes": 1024}')
    (second_root / "plan.json").write_text('{"kv_cache_memory_bytes": 2048}')
    first, _ = _entry(tmp_path, root=first_root)
    second, _ = _entry(
        tmp_path,
        compatibility_key="different-native-fingerprint",
        root=second_root,
    )

    first_bundle = first.prepare_source()
    prepared_bytes = first_bundle.tar_path.read_bytes()
    second_bundle = second.prepare_source()

    assert first_bundle.tar_path.read_bytes() == prepared_bytes
    assert first_bundle.tar_path != second_bundle.tar_path
    for bundle, expected in ((first_bundle, 1024), (second_bundle, 2048)):
        with tarfile.open(bundle.tar_path) as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            assert len(members) == 1
            with archive.extractfile(members[0]) as payload:
                assert json.load(payload)["kv_cache_memory_bytes"] == expected


def test_unknown_framework_enum_roundtrips_through_legacy_protobuf_and_hash(tmp_path):
    descriptor = descriptor_pb2.FileDescriptorProto.FromString(
        p2p_pb2.DESCRIPTOR.serialized_pb
    )
    enum = next(item for item in descriptor.enum_type if item.name == "MxSourceType")
    retained = [item for item in enum.value if item.number != 10]
    del enum.value[:]
    enum.value.extend(retained)
    legacy_file = descriptor_pool.DescriptorPool().AddSerializedFile(
        descriptor.SerializeToString()
    )
    legacy_identity_cls = message_factory.GetMessageClass(
        legacy_file.message_types_by_name["SourceIdentity"]
    )
    _, identity = _entry(tmp_path)
    # Pin the digest so the golden ID stays shared with the Rust contract test.
    identity.compile_config_digest = hashlib.sha256(b"startup-fingerprint").hexdigest()
    identity.mx_version = "0.8.0"
    identity.tensor_parallel_size = 8
    identity.pipeline_parallel_size = 1
    identity.dtype = "bfloat16"

    legacy = legacy_identity_cls.FromString(identity.SerializeToString())
    current = p2p_pb2.SourceIdentity.FromString(legacy.SerializeToString())

    assert legacy.mx_source_type == current.mx_source_type == 10
    assert compute_mx_source_id(identity) == "7d4cf89c29780927"
    assert compute_mx_source_id(legacy) == compute_mx_source_id(identity)
    assert current == identity


def _publication_context():
    return SimpleNamespace(
        device_id=0,
        global_rank=0,
        worker_rank=0,
        worker_id="test-worker",
        node_rank=0,
        mx_client=object(),
        nixl_manager=None,
    )


def _fake_transport(monkeypatch):
    transport = MagicMock()
    transport.publish_requires_install_state = False
    transport.should_publish.return_value = True
    monkeypatch.setattr(artifact_lifecycle, "_artifact_backend", lambda: "mooncake")
    monkeypatch.setattr(
        artifact_lifecycle, "_create_artifact_transport", lambda *_args, **_kwargs: transport
    )
    return transport


@pytest.mark.parametrize(
    "second_changes",
    [{"name": "execution_plan"}, {"compatibility_key": "second-rank"}],
)
def test_distinct_framework_caches_do_not_stop_each_others_publishers(
    monkeypatch, tmp_path, second_changes
):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    _fake_transport(monkeypatch)
    monkeypatch.setattr(
        artifact_lifecycle,
        "mark_publish_scheduled",
        lambda ctx, transfer, identity: tmp_path / compute_mx_source_id(identity),
    )
    publisher_factory = MagicMock(side_effect=lambda **_kwargs: MagicMock())
    monkeypatch.setattr(artifact_lifecycle, "PublisherThread", publisher_factory)
    entries = [_entry(tmp_path), _entry(tmp_path, **second_changes)]
    scheduled = {}
    artifact_lifecycle.schedule_artifact_publish(
        _publication_context(),
        lambda: entries,
        engine_label="vLLM",
        ready_fn_factory=lambda _roots: lambda: True,
        artifact_publish_fn=MagicMock(),
        scheduled_publishers=scheduled,
    )

    assert set(scheduled) == {
        (0, compute_mx_source_id(identity)) for _, identity in entries
    }
    assert publisher_factory.call_count == 2
    for publisher in scheduled.values():
        publisher.start.assert_called_once()
        publisher.stop.assert_not_called()


@pytest.mark.parametrize(
    "second_changes",
    [{"name": "execution_plan"}, {"compatibility_key": "second-rank"}],
)
def test_distinct_framework_caches_keep_their_publication_handles(
    monkeypatch, tmp_path, second_changes
):
    transport = _fake_transport(monkeypatch)
    transport.publish.side_effect = lambda *_args, **_kwargs: MagicMock()
    source = tmp_path / "source"
    source.mkdir()
    (source / "plan.json").write_text('{"kv_cache_memory_bytes": 1024}')
    entries = [_entry(tmp_path), _entry(tmp_path, **second_changes)]
    published = {}
    handles = []
    for transfer, identity in entries:
        handles.append(
            artifact_lifecycle.publish_artifact(
                _publication_context(),
                transfer,
                identity,
                engine_label="vLLM",
                accelerator="cuda",
                published_sources=published,
            )
        )

    assert set(published) == {
        (0, compute_mx_source_id(identity)) for _, identity in entries
    }
    for handle in handles:
        handle.stop.assert_not_called()


def test_legacy_publications_keep_source_type_registry_key(monkeypatch, tmp_path):
    transport = _fake_transport(monkeypatch)
    transport.publish.return_value = MagicMock()
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.bin").write_bytes(b"kernel")
    framework_transfer, identity = _entry(tmp_path)
    transfer = replace(
        framework_transfer,
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
    )
    identity.mx_source_type = transfer.mx_source_type
    previous = MagicMock()
    key = (0, p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE)
    published = {key: previous}

    current = artifact_lifecycle.publish_artifact(
        _publication_context(),
        transfer,
        identity,
        engine_label="vLLM",
        accelerator="cuda",
        published_sources=published,
    )

    previous.stop.assert_called_once()
    assert published == {key: current}


def _install(transfer, bundle) -> None:
    (staged,) = transfer.target_file_paths()
    staged.write_bytes(bundle.tar_path.read_bytes())
    received_file = p2p_pb2.ArtifactManifestFile()
    received_file.CopyFrom(bundle.manifest.files[0])
    received_file.path = str(staged)
    transfer.install(
        p2p_pb2.GetArtifactManifestHeaderResponse(
            artifact_id=bundle.artifact_id,
            mx_source_type=transfer.mx_source_type,
            files=[received_file],
            total_size=received_file.size,
        )
    )


def test_include_publishes_only_selected_files(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    for name in ("selected.json", "other.json", "nested/selected.json"):
        (source / name).write_text(name)
    seen = []

    def include(path):
        seen.append(path)
        return path.endswith("selected.json")

    transfer, _ = _entry(tmp_path, include=include)
    bundle = transfer.prepare_source()

    with tarfile.open(bundle.tar_path) as archive:
        files = sorted(member.name for member in archive.getmembers() if member.isfile())
    assert files == ["./nested/selected.json", "./selected.json"]
    assert sorted(seen) == ["nested/selected.json", "other.json", "selected.json"]


def test_include_without_selected_files_does_not_publish(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "other.json").write_text("{}")
    transfer, _ = _entry(tmp_path, include=lambda path: False)

    with pytest.raises(LookupError, match="selected no files"):
        transfer.prepare_source()


def test_install_keeps_unrelated_files_in_the_framework_root(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "plan_a.json").write_text("fetched")
    (target / "plan_a.json").write_text("stale")
    (target / "plan_b.json").write_text("local")
    sender, _ = _entry(tmp_path, include=lambda path: path == "plan_a.json")
    receiver, _ = replace(_cache(tmp_path), root=target).entry(
        "deepseek-ai/DeepSeek-V3", tmp_path / "receiver-bundle"
    )

    _install(receiver, sender.prepare_source())

    assert (target / "plan_a.json").read_text() == "fetched"
    assert (target / "plan_b.json").read_text() == "local"


def test_operator_compile_config_digest_partitions_framework_caches(monkeypatch, tmp_path):
    _, default = _entry(tmp_path)
    monkeypatch.setenv("MX_ARTIFACT_COMPILE_CONFIG_DIGEST", "cluster-a")
    _, cluster_a = _entry(tmp_path)
    monkeypatch.setenv("MX_ARTIFACT_COMPILE_CONFIG_DIGEST", "cluster-b")
    _, cluster_b = _entry(tmp_path)

    ids = {compute_mx_source_id(identity) for identity in (default, cluster_a, cluster_b)}
    assert len(ids) == 3


def test_framework_cache_entries_follow_the_adapter_registration(tmp_path):
    caches = (_cache(tmp_path), _cache(tmp_path, name="execution_plan"))
    ctx = SimpleNamespace(
        adapter=SimpleNamespace(framework_cache_artifacts=caches),
        identity=p2p_pb2.SourceIdentity(model_name="deepseek-ai/DeepSeek-V3"),
    )

    entries = framework_cache_entries(ctx, tmp_path / "bundle")

    assert [transfer.name for transfer, _ in entries] == ["startup_plan", "execution_plan"]
    assert framework_cache_entries(SimpleNamespace(adapter=None), tmp_path) == []
