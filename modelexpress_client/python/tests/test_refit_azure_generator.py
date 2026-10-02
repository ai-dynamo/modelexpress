# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
pytest.importorskip('azure.storage.blob')
pytest.importorskip('azure.identity')
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from azure.storage.blob import BlobServiceClient
import requests
import numpy as np
import safetensors.numpy
from modelexpress_rl import (
    ObjectStorageGeneratorConfig,
    ObjectStorageSource,
    ObjectStorageType,
    WeightPayloadFormat,
    WeightSource,
    WeightVersion,
    WeightVersionState,
)
from modelexpress_rl.inference.adapter import GeneratorEngineContext
from modelexpress_rl.inference.checkpoint_store import (
    CheckpointState,
    LocalCheckpointStore,
)
import modelexpress_rl.inference.engines as engines_module
from modelexpress_rl.inference.methods import CanonicalDeltaUpdateMethod
from modelexpress_rl.inference.plan import (
    EngineCapabilities,
    ObjectStorageUpdateSource,
    PreparedCheckpointArtifact,
)
from modelexpress_rl.inference.receiver import _LocalCheckpoint
from modelexpress_rl.inference.runtime import (
    EngineRuntime,
    initialize_generator_runtime,
)
from modelexpress_rl.inference.version_chain import resolve_replay_chain
from modelexpress_rl.utils import checksum_factory, compress_delta, compute_delta

from tests.test_refit_azure_reader import sdk


@pytest.fixture
def azure_control():
    version = _azure_version()
    versions = {version.version_id: version}
    leases = []

    def start_lease(_version_id):
        lease = SimpleNamespace(close=Mock())
        leases.append(lease)
        return lease

    return SimpleNamespace(
        version=version,
        versions=versions,
        fetch_ready_version=Mock(side_effect=versions.__getitem__),
        start_lease=Mock(side_effect=start_lease),
        leases=leases,
    )


@pytest.fixture
def checkpoint_store(tmp_path):
    return LocalCheckpointStore(root=tmp_path / "cache", model_name="test/model")


@pytest.fixture
def azure_runtime(sdk, azure_control, checkpoint_store, monkeypatch, tmp_path, request):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    startup_seed = getattr(request, "param", "external")
    initial_version = (
        "base-a" if startup_seed == "external" else azure_control.version.version_id
    )
    seed = tmp_path / "seed" if startup_seed == "external" else None
    if startup_seed != "missing":
        seed_path = seed or checkpoint_store.full_path(initial_version)
        seed_path.mkdir(parents=True)
        safetensors.numpy.save_file(
            {
                "weight": np.array([1.0, 2.0], dtype=np.float32),
                "bias": np.array([0.0, 0.0], dtype=np.float32),
            },
            seed_path / "model.safetensors",
        )
    installer = Mock()
    installer.capabilities = EngineCapabilities(
        artifact_types=frozenset({PreparedCheckpointArtifact})
    )
    monkeypatch.setattr(
        engines_module,
        "_create_engine_runtime",
        lambda _context: EngineRuntime(model_name="test/model", installer=installer),
    )
    runtime = initialize_generator_runtime(
        engine_context=GeneratorEngineContext(),
        worker_id="generator-0",
        server_url="mx:8000",
        object_storage=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.AZURE,
            initial_base_version_id=initial_version,
            seed_checkpoint_path=seed,
            refit_checkpoint_dir=tmp_path / "cache",
        ),
        source_order=(WeightSource.OBJECT_STORAGE,),
        max_transfer_attempts=1,
        rpc_timeout_seconds=30,
        service=Mock(),
        start_lease=azure_control.start_lease,
        resolve_replay_chain=lambda version_id, from_full_root: resolve_replay_chain(
            target_version_id=version_id,
            fetch_ready_version=azure_control.fetch_ready_version,
            max_chain_length=16,
            stop_before_version_id=None if from_full_root else initial_version,
        ),
    )
    try:
        yield runtime
    finally:
        runtime.close()
        sdk.client.close.assert_called_once()
        sdk.credential.close.assert_called_once()


def _azure_version(payload_format=WeightPayloadFormat.FULL_HF_CHECKPOINT):
    return WeightVersion(
        version_id="full-a",
        model_name="test/model",
        payload_format=payload_format,
        base_version_id=(
            "base-a" if payload_format is WeightPayloadFormat.XOR_DELTA else None
        ),
        object_storage=ObjectStorageSource(
            storage_type=ObjectStorageType.AZURE,
            uri="az://models/V2/model.safetensors.index.json",
        ),
        expected_source_slots=(),
        layout_signature="",
        state=WeightVersionState.READY,
        created_at_unix_ms=1,
    )


@pytest.fixture
def full_checkpoint(sdk):
    tensors = {
        "weight": np.array([3.0, 4.0], dtype=np.float32),
        "bias": np.array([5.0, 6.0], dtype=np.float32),
    }
    weight_map = {
        "weight": "model-00001-of-00002.safetensors",
        "bias": "model-00002-of-00002.safetensors",
    }
    index_key = ("models", "V2/model.safetensors.index.json")
    sdk.client.objects = {
        index_key: json.dumps(
            {
                "metadata": {"checksum_format": "adler32"},
                "weight_map": weight_map,
            }
        ).encode(),
    }
    expected_calls = [("get", index_key)]
    for name, shard in weight_map.items():
        checksum = checksum_factory("adler32")
        checksum.update(tensors[name].view(np.uint8))
        shard_key = ("models", f"V2/{shard}")
        sdk.client.objects[shard_key] = safetensors.numpy.save(
            {name: tensors[name]}, metadata={name: checksum.hexdigest()}
        )
        expected_calls.extend([("size", shard_key), ("get", shard_key)])
    return SimpleNamespace(
        tensors=tensors,
        weight_map=weight_map,
        index_key=index_key,
        expected_calls=expected_calls,
    )


@pytest.fixture
def azure_delta_chain(full_checkpoint, azure_control, sdk):
    base = azure_control.version
    tensors = full_checkpoint.tensors
    expected_calls = []
    for number in (1, 2):
        version_id = f"delta-{number}"
        target = {name: value + number for name, value in tensors.items()}
        encoded, checksums = {}, {}
        for name, value in target.items():
            raw = value.view(np.uint8)
            delta, _ = compute_delta(raw, tensors[name].view(np.uint8))
            encoded[name] = compress_delta(delta)
            checksum = checksum_factory("adler32")
            checksum.update(raw)
            checksums[name] = checksum.hexdigest()
        index_key = ("models", f"{version_id}/model.safetensors.index.json")
        shard_key = ("models", f"{version_id}/delta.safetensors")
        sdk.client.objects[index_key] = json.dumps({
            "metadata": {
                "version": version_id, "base_version": base.version_id,
                "delta_encoding": "xor", "compression_format": "zstd",
                "checksum_format": "adler32",
            },
            "weight_map": dict.fromkeys(target, "delta.safetensors"),
        }).encode()
        sdk.client.objects[shard_key] = safetensors.numpy.save(
            encoded, metadata=checksums
        )
        base = replace(
            base, version_id=version_id, base_version_id=base.version_id,
            payload_format=WeightPayloadFormat.XOR_DELTA,
            object_storage=ObjectStorageSource(
                ObjectStorageType.AZURE, f"az://models/{index_key[1]}"
            ),
        )
        azure_control.versions[version_id] = base
        tensors = target
        expected_calls.extend([("get", index_key), ("get", shard_key)])
    return SimpleNamespace(
        target=base, tensors=tensors, expected_calls=expected_calls,
        index_key=index_key, shard_key=shard_key,
    )


def _assert_checkpoint_tensors(path, expected):
    actual = {}
    for shard in Path(path).glob("*.safetensors"):
        actual.update(safetensors.numpy.load_file(shard))
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value)


def test_generator_replays_and_reuses_azure_delta_chain(
    azure_runtime, azure_control, azure_delta_chain, checkpoint_store, sdk
):
    session = azure_runtime.session
    root = session.stage(azure_control.version)
    session.apply(root)
    session.release(root)
    sdk.client.calls.clear()
    chain = azure_delta_chain
    update = session.stage(chain.target)
    try:
        assert checkpoint_store.active_version() == "full-a"
        _assert_checkpoint_tensors(update.prepared.checkpoint.path, chain.tensors)
        assert Counter(sdk.client.calls) == Counter(chain.expected_calls)
        session.apply(update)
        assert checkpoint_store.active_version() == chain.target.version_id
    finally:
        session.release(update)
    sdk.client.calls.clear()
    reused = session.stage(chain.target)
    try:
        _assert_checkpoint_tensors(reused.prepared.checkpoint.path, chain.tensors)
        assert sdk.client.calls == []
    finally:
        session.release(reused)


@pytest.mark.parametrize("azure_runtime", ["missing", "unrecorded"], indirect=True)
def test_generator_bootstraps_azure_replay_without_a_recorded_seed(
    azure_runtime, azure_control, full_checkpoint, azure_delta_chain,
    checkpoint_store, sdk
):
    session = azure_runtime.session
    method = azure_runtime.methods[0]
    installer = azure_runtime.engine.installer
    assert method.requires_full_root
    assert checkpoint_store.state() is None
    assert not checkpoint_store.active_path.exists()
    assert sdk.client.calls == []

    sdk.client.fail_get_once = (
        "models", f"V2/{full_checkpoint.weight_map['weight']}"
    )
    with pytest.raises(RuntimeError, match="full HF checkpoint download failed"):
        session.stage(azure_delta_chain.target)
    assert method.requires_full_root
    assert checkpoint_store.state() is None
    assert not checkpoint_store.active_path.exists()
    installer.install.assert_not_called()
    azure_control.leases[-1].close.assert_called_once()

    update = session.stage(azure_delta_chain.target)
    try:
        assert not method.requires_full_root
        _assert_checkpoint_tensors(
            checkpoint_store.full_path(azure_control.version.version_id),
            full_checkpoint.tensors,
        )
        _assert_checkpoint_tensors(
            update.prepared.checkpoint.path, azure_delta_chain.tensors
        )
        assert not checkpoint_store.active_path.exists()
        installer.install.assert_not_called()
        session.apply(update)
        installer.install.assert_called_once_with(update.prepared)
        assert checkpoint_store.active_version() == azure_delta_chain.target.version_id
        for lease in azure_control.leases:
            lease.close.assert_called_once()
    finally:
        session.release(update)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing-shard", "BlobNotFound"),
        ("interrupted", "injected Blob download interruption"),
        ("checksum", "checksum"),
        ("layout", "layout format mismatch"),
    ],
)
def test_failed_azure_delta_chain_preserves_active_version(
    azure_runtime, azure_control, azure_delta_chain, checkpoint_store, sdk,
    failure, message
):
    chain = azure_delta_chain
    objects = dict(sdk.client.objects)
    versions = dict(azure_control.versions)
    if failure == "missing-shard":
        del sdk.client.objects[chain.shard_key]
    elif failure == "interrupted":
        sdk.client.fail_get_once = chain.shard_key
    elif failure == "checksum":
        shard = safetensors.numpy.load(sdk.client.objects[chain.shard_key])
        sdk.client.objects[chain.shard_key] = safetensors.numpy.save(
            shard, metadata=dict.fromkeys(shard, "invalid-checksum")
        )
    else:
        for version_id in ("delta-1", "delta-2"):
            azure_control.versions[version_id] = replace(
                versions[version_id], layout_signature=version_id
            )
    with pytest.raises(RuntimeError, match=message):
        azure_runtime.session.stage(chain.target)
    azure_runtime.engine.installer.install.assert_not_called()
    assert checkpoint_store.active_version() == "base-a"
    assert checkpoint_store.state().status is CheckpointState.READY
    assert checkpoint_store.state().version == "base-a"
    for lease in azure_control.leases:
        lease.close.assert_called_once()
    if failure not in {"missing-shard", "interrupted"}:
        return
    sdk.client.objects = objects
    azure_control.versions.update(versions)
    update = azure_runtime.session.stage(chain.target)
    try:
        _assert_checkpoint_tensors(update.prepared.checkpoint.path, chain.tensors)
        azure_runtime.session.apply(update)
        assert checkpoint_store.active_version() == chain.target.version_id
    finally:
        azure_runtime.session.release(update)


def _assert_ready_checkpoint(store, update, full_checkpoint):
    checkpoint = update.prepared.checkpoint
    assert checkpoint.target_version == update.plan.version.version_id
    assert checkpoint.path == store.full_path(checkpoint.target_version)
    state = store.state()
    assert state is not None
    assert state.status is CheckpointState.READY
    assert state.version == checkpoint.target_version
    assert {path.name for path in checkpoint.path.glob("*.safetensors")} == set(
        full_checkpoint.weight_map.values()
    )
    for name, shard in full_checkpoint.weight_map.items():
        tensors = safetensors.numpy.load_file(checkpoint.path / shard)
        assert set(tensors) == {name}
        np.testing.assert_array_equal(tensors[name], full_checkpoint.tensors[name])


def test_generator_stages_and_applies_full_checkpoint_through_azure_reader(
    azure_runtime, azure_control, checkpoint_store, full_checkpoint, sdk
):
    version = azure_control.version
    session = azure_runtime.session
    installer = azure_runtime.engine.installer
    expected_path = checkpoint_store.full_path(version.version_id)
    installations = []

    def install(prepared):
        checkpoint = prepared.checkpoint
        assert checkpoint.path == expected_path
        assert checkpoint.target_version == version.version_id
        assert checkpoint_store.active_version() == "base-a"
        installations.append((checkpoint.path, checkpoint.target_version))
        return "installed"

    installer.install.side_effect = install
    assert checkpoint_store.active_version() == "base-a"
    assert sdk.client.calls == []

    update = session.stage(version)
    try:
        assert update.plan.source.kind is WeightSource.OBJECT_STORAGE
        assert update.plan.source.storage == version.object_storage
        _assert_ready_checkpoint(checkpoint_store, update, full_checkpoint)
        assert checkpoint_store.active_version() == "base-a"
        installer.install.assert_not_called()
        azure_control.fetch_ready_version.assert_called_once_with(version.version_id)
        azure_control.start_lease.assert_called_once_with(version.version_id)
        update.lease.close.assert_not_called()
        assert Counter(sdk.client.calls) == Counter(full_checkpoint.expected_calls)

        assert session.apply(update) == "installed"

        installer.install.assert_called_once_with(update.prepared)
        assert installations == [(expected_path, version.version_id)]
        assert update.applied
        assert checkpoint_store.active_version() == version.version_id
        update.lease.close.assert_called_once()
    finally:
        session.release(update)

    assert update.released
    azure_runtime.close()
    sdk.client.close.assert_called_once()
    sdk.credential.close.assert_called_once()


def test_generator_stages_azure_full_checkpoint_without_checksums(
    azure_runtime, azure_control, checkpoint_store, full_checkpoint, sdk
):
    index = json.loads(sdk.client.objects[full_checkpoint.index_key])
    del index["metadata"]["checksum_format"]
    sdk.client.objects[full_checkpoint.index_key] = json.dumps(index).encode()
    for name, shard in full_checkpoint.weight_map.items():
        sdk.client.objects[("models", f"V2/{shard}")] = safetensors.numpy.save(
            {name: full_checkpoint.tensors[name]}
        )

    session = azure_runtime.session
    update = session.stage(azure_control.version)
    try:
        assert update.plan.source.storage == azure_control.version.object_storage
        _assert_ready_checkpoint(checkpoint_store, update, full_checkpoint)
        assert Counter(sdk.client.calls) == Counter(full_checkpoint.expected_calls)
        assert checkpoint_store.active_version() == "base-a"
        azure_runtime.engine.installer.install.assert_not_called()
    finally:
        session.release(update)


def _assert_stage_fails(runtime, control, store, message):
    with pytest.raises(RuntimeError, match=message) as error:
        runtime.session.stage(control.version)
    runtime.engine.installer.install.assert_not_called()
    assert store.active_version() == "base-a"
    control.leases[-1].close.assert_called_once()
    return error.value


@pytest.mark.parametrize("missing", ["index", "shard"])
def test_generator_rejects_missing_azure_checkpoint_object(
    azure_runtime, azure_control, checkpoint_store, full_checkpoint, sdk, missing
):
    key = (
        full_checkpoint.index_key
        if missing == "index"
        else ("models", f"V2/{full_checkpoint.weight_map['weight']}")
    )
    del sdk.client.objects[key]
    message = (
        "replay validation failed.*BlobNotFound"
        if missing == "index"
        else "replay failed.*BlobNotFound"
    )
    _assert_stage_fails(azure_runtime, azure_control, checkpoint_store, message)
    assert ("get" if missing == "index" else "size", key) in sdk.client.calls


def test_generator_rejects_azure_checkpoint_checksum_mismatch(
    azure_runtime, azure_control, checkpoint_store, full_checkpoint, sdk
):
    key = ("models", f"V2/{full_checkpoint.weight_map['weight']}")
    shard = bytearray(sdk.client.objects[key])
    shard[-1] ^= 1
    sdk.client.objects[key] = bytes(shard)

    _assert_stage_fails(
        azure_runtime, azure_control, checkpoint_store,
        "full HF checkpoint checksum differs for 'weight'",
    )
    assert not checkpoint_store.full_path(azure_control.version.version_id).exists()


def test_generator_rejects_azure_checkpoint_authorization_failure(
    azure_runtime, azure_control, checkpoint_store, full_checkpoint, sdk
):
    sdk.client.error = HttpResponseError("AuthorizationPermissionMismatch")
    error = _assert_stage_fails(
        azure_runtime, azure_control, checkpoint_store,
        "replay validation failed.*AuthorizationPermissionMismatch",
    )
    assert error.__cause__ is sdk.client.error
    assert sdk.client.calls == [("get", full_checkpoint.index_key)]


def test_generator_propagates_token_failure_from_real_blob_pipeline(
    sdk, azure_control, checkpoint_store, monkeypatch, request
):
    failure = ClientAuthenticationError("injected token acquisition failure")
    sdk.credential = Mock(spec=["get_token", "close"])
    sdk.credential.get_token.side_effect = failure
    sdk.credential_factory.return_value = sdk.credential
    http = Mock(side_effect=AssertionError("HTTP must not be reached"))
    monkeypatch.setattr(requests.Session, "request", http)

    def create(**kwargs):
        kwargs["retry_total"] = 0
        sdk.client = BlobServiceClient(**kwargs)
        monkeypatch.setattr(sdk.client, "close", Mock(wraps=sdk.client.close))
        return sdk.client

    sdk.factory.side_effect = create
    # Configure the SDK boundary before the existing runtime fixture constructs it.
    runtime = request.getfixturevalue("azure_runtime")
    sdk.credential.get_token.assert_not_called()
    try:
        error = _assert_stage_fails(
            runtime, azure_control, checkpoint_store,
            "replay validation failed.*injected token acquisition failure",
        )
        assert error.__cause__ is failure
        sdk.credential.get_token.assert_called_once()
        http.assert_not_called()
        assert not checkpoint_store.full_path("full-a").exists()
    finally:
        runtime.close()
    sdk.client.close.assert_called_once()
    sdk.credential.close.assert_called_once()


def test_generator_recovers_and_retries_interrupted_azure_checkpoint(
    azure_runtime, azure_control, checkpoint_store, full_checkpoint, sdk, monkeypatch
):
    monkeypatch.setenv("MX_REFIT_DOWNLOAD_WORKERS", "1")
    first_key = ("models", f"V2/{full_checkpoint.weight_map['weight']}")
    second_key = ("models", f"V2/{full_checkpoint.weight_map['bias']}")
    sdk.client.fail_get_once = second_key

    error = _assert_stage_fails(
        azure_runtime, azure_control, checkpoint_store,
        "full HF checkpoint download failed for 'model-00002-of-00002.safetensors'",
    )
    assert "injected Blob download interruption" in str(error.__cause__.__cause__)
    assert [key for operation, key in sdk.client.calls if operation == "get"] == [
        full_checkpoint.index_key, first_key, second_key,
    ]
    assert sdk.client.fail_get_once is None
    assert checkpoint_store.state().status is CheckpointState.READY
    assert checkpoint_store.state().version == "base-a"
    assert not checkpoint_store.full_path("full-a").with_suffix(".tmp").exists()

    sdk.client.calls.clear()
    session = azure_runtime.session
    update = session.stage(azure_control.version)
    try:
        _assert_ready_checkpoint(checkpoint_store, update, full_checkpoint)
        assert Counter(sdk.client.calls) == Counter(full_checkpoint.expected_calls)
        assert len(azure_control.leases) == 2
        assert update.lease is not azure_control.leases[0]
        update.lease.close.assert_not_called()
        azure_runtime.engine.installer.install.assert_not_called()

        def install(prepared):
            assert prepared is update.prepared
            assert checkpoint_store.active_version() == "base-a"
            return "installed"

        azure_runtime.engine.installer.install.side_effect = install
        assert session.apply(update) == "installed"
        azure_runtime.engine.installer.install.assert_called_once_with(update.prepared)
        assert checkpoint_store.active_version() == azure_control.version.version_id
        update.lease.close.assert_called_once()
    finally:
        session.release(update)


def test_generator_reuses_released_azure_checkpoint_without_shard_downloads(
    azure_runtime, azure_control, checkpoint_store, full_checkpoint, sdk
):
    session = azure_runtime.session
    first = session.stage(azure_control.version)
    try:
        _assert_ready_checkpoint(checkpoint_store, first, full_checkpoint)
    finally:
        session.release(first)
    first.lease.close.assert_called_once()
    sdk.client.calls.clear()

    reused = session.stage(azure_control.version)
    try:
        assert reused is not first
        assert reused.prepared is not first.prepared
        assert reused.prepared.checkpoint.path == first.prepared.checkpoint.path
        _assert_ready_checkpoint(checkpoint_store, reused, full_checkpoint)
        assert sdk.client.calls in ([], [("get", full_checkpoint.index_key)])
        azure_runtime.engine.installer.install.assert_not_called()
        assert checkpoint_store.active_version() == "base-a"
    finally:
        session.release(reused)
    reused.lease.close.assert_called_once()


def test_generator_rejects_changed_azure_source_for_cached_version(
    azure_runtime, azure_control, checkpoint_store, full_checkpoint, sdk
):
    session = azure_runtime.session
    first = session.stage(azure_control.version)
    session.release(first)
    for (container, blob), data in list(sdk.client.objects.items()):
        sdk.client.objects[(container, blob.replace("V2/", "alternate/", 1))] = data
    alternate = replace(
        azure_control.version,
        object_storage=ObjectStorageSource(
            storage_type=ObjectStorageType.AZURE,
            uri="az://models/alternate/model.safetensors.index.json",
        ),
    )
    azure_control.version = alternate
    azure_control.versions[alternate.version_id] = alternate

    _assert_stage_fails(
        azure_runtime, azure_control, checkpoint_store,
        "prepared checkpoint has different source identity",
    )


def test_generator_keeps_previous_active_version_when_azure_installer_fails(
    azure_runtime, azure_control, checkpoint_store, full_checkpoint
):
    session = azure_runtime.session
    update = session.stage(azure_control.version)
    failure = RuntimeError("recording installer failed")

    def install(prepared):
        assert prepared is update.prepared
        assert checkpoint_store.active_version() == "base-a"
        raise failure

    azure_runtime.engine.installer.install.side_effect = install
    try:
        _assert_ready_checkpoint(checkpoint_store, update, full_checkpoint)
        with pytest.raises(RuntimeError, match="recording installer failed") as error:
            session.apply(update)
        assert error.value is failure
        assert not update.applied
        assert checkpoint_store.state().status is CheckpointState.READY
        assert checkpoint_store.state().version == "full-a"
        assert checkpoint_store.active_version() == "base-a"
        azure_runtime.engine.installer.install.assert_called_once_with(update.prepared)
        update.lease.close.assert_called_once()
    finally:
        session.release(update)
    assert update.released


def test_azure_full_tensor_fails_before_reads_or_preparation(
    azure_runtime, sdk, monkeypatch
):
    payload_format = WeightPayloadFormat.FULL_TENSOR
    version = _azure_version(payload_format)
    assert list(azure_runtime.session._planner.plans(version)) == []

    method = azure_runtime.methods[0]
    prepare = Mock(side_effect=AssertionError("checkpoint preparation must not start"))
    monkeypatch.setattr(method._checkpoint, "prepare_chain", prepare)
    source = ObjectStorageUpdateSource(
        storage=version.object_storage, payload_format=payload_format
    )
    with pytest.raises(
        ValueError, match="unsupported canonical object-storage payload"
    ):
        method.prepare(version=version, source=source)
    full = _azure_version()
    with pytest.raises(
        ValueError, match="unsupported canonical object-storage payload"
    ):
        method.prepare_chain(
            (
                (
                    full,
                    ObjectStorageUpdateSource(
                        storage=full.object_storage, payload_format=full.payload_format
                    ),
                ),
                (version, source),
            )
        )
    prepare.assert_not_called()
    assert sdk.client.calls == []


@pytest.mark.parametrize("field", ["endpoint_url", "region_name"])
@pytest.mark.parametrize("value", ["", "secret-value"])
def test_azure_generator_rejects_s3_settings_before_sdk_construction(
    sdk, tmp_path, field, value
):
    with pytest.raises(ValueError, match=field) as error:
        CanonicalDeltaUpdateMethod(
            model_name="test/model",
            config=ObjectStorageGeneratorConfig(
                storage_type=ObjectStorageType.AZURE,
                initial_base_version_id="base-a",
                seed_checkpoint_path=tmp_path / "seed",
                refit_checkpoint_dir=tmp_path / "cache",
                **{field: value},
            ),
        )
    assert "secret-value" not in str(error.value)
    sdk.factory.assert_not_called()
    sdk.factory.from_connection_string.assert_not_called()
    sdk.credential_factory.assert_not_called()


def test_canonical_method_closes_azure_reader_when_initialization_fails(
    sdk, monkeypatch, tmp_path
):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    monkeypatch.setattr(
        _LocalCheckpoint, "initialize", Mock(side_effect=OSError("seed unavailable"))
    )
    with pytest.raises(OSError, match="seed unavailable"):
        CanonicalDeltaUpdateMethod(
            model_name="test/model",
            config=ObjectStorageGeneratorConfig(
                storage_type=ObjectStorageType.AZURE,
                initial_base_version_id="base-a",
                seed_checkpoint_path=tmp_path / "seed",
                refit_checkpoint_dir=tmp_path / "cache",
            ),
        )
    sdk.client.close.assert_called_once()
    sdk.credential.close.assert_called_once()
