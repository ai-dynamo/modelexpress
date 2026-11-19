# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from modelexpress import p2p_pb2
from modelexpress.adapter import EngineAdapter, StrategyRecoveryError
from modelexpress.load_strategy.context import LoadContext
from modelexpress_rl import (
    ModelExpressControlClient,
    ObjectStorageSource,
    ObjectStorageType,
)
from modelexpress_rl.inference.checkpoint_store import (
    CheckpointState,
    LocalCheckpointStore,
)
from modelexpress_rl.inference.load_strategy import (
    DesiredVersionObjectStorageStrategy,
    RLLoadStrategyChain,
)

from tests.test_refit_azure_reader import sdk
from tests.test_refit_azure_generator import (
    _assert_checkpoint_tensors,
    azure_control,
    azure_delta_chain,
    checkpoint_store,
    full_checkpoint,
)


def _context():
    return LoadContext(
        model_config=SimpleNamespace(model="test/model"),
        load_config=SimpleNamespace(),
        target_device=torch.device("cpu"),
        identity=p2p_pb2.SourceIdentity(
            model_name="test/model",
            revision="launch-version",
        ),
        global_rank=0,
        worker_rank=0,
        local_rank=0,
        device_id=0,
        worker_id="generator-0",
        mx_client=Mock(),
        adapter=EngineAdapter(),
        mx_server_url="mx:8000",
    )


@pytest.fixture
def azure_cold_start(sdk, azure_control, monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", azure_control.version.version_id)
    monkeypatch.setenv("MX_REFIT_CHECKPOINT_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MX_GENERATOR_SOURCE_ORDER", "OBJECT_STORAGE")
    monkeypatch.setattr(
        ModelExpressControlClient, "connect",
        Mock(return_value=nullcontext(SimpleNamespace(
            get_weight_version=azure_control.fetch_ready_version,
        ))),
    )
    monkeypatch.setattr(
        DesiredVersionObjectStorageStrategy,
        "supports_explicit_uri", lambda _self, _ctx: True,
    )

    def load(result, _ctx, _uri):
        result.publishable = False
        return result

    load_uri = Mock(side_effect=load)
    monkeypatch.setattr(DesiredVersionObjectStorageStrategy, "load_uri", load_uri)
    yield SimpleNamespace(ctx=_context(), model=nn.Linear(2, 2), load_uri=load_uri)
    assert sdk.client.close.call_count == sdk.factory.call_count
    assert sdk.credential.close.call_count == sdk.credential_factory.call_count


@pytest.mark.parametrize("target", ["full", "delta"])
def test_cold_start_restores_exact_azure_version(
    azure_cold_start, azure_control, azure_delta_chain, full_checkpoint,
    checkpoint_store, sdk, monkeypatch, target,
):
    cold = azure_cold_start
    version = azure_control.version if target == "full" else azure_delta_chain.target
    tensors = full_checkpoint.tensors if target == "full" else azure_delta_chain.tensors
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", version.version_id)

    def load(result, ctx, uri):
        assert Path(uri) == checkpoint_store.checkpoint_path(version.version_id)
        _assert_checkpoint_tensors(uri, tensors)
        assert ctx.identity.revision == "launch-version"
        assert not checkpoint_store.active_path.exists()
        assert checkpoint_store.state().status is CheckpointState.READY
        assert checkpoint_store.state().version == version.version_id
        result.publishable = False
        return result

    cold.load_uri.side_effect = load
    assert RLLoadStrategyChain.run(cold.model, cold.ctx) is cold.model
    cold.load_uri.assert_called_once()
    assert cold.ctx.identity.revision == version.version_id
    assert checkpoint_store.active_version() == version.version_id
    expected = full_checkpoint.expected_calls
    if target == "delta":
        expected = expected + azure_delta_chain.expected_calls
    assert Counter(sdk.client.calls) == Counter(expected)


@pytest.mark.parametrize("cache", ["valid", "damaged-shard", "different-source"])
def test_cold_start_reuses_or_refetches_azure_cache_in_new_context(
    azure_cold_start, azure_control, checkpoint_store, full_checkpoint,
    sdk, monkeypatch, cache,
):
    cold = azure_cold_start
    version = azure_control.version
    path = checkpoint_store.full_path(version.version_id)
    activate = Mock(wraps=LocalCheckpointStore.activate)
    monkeypatch.setattr(
        LocalCheckpointStore, "activate", lambda store, uid: activate(store, uid)
    )
    RLLoadStrategyChain.run(cold.model, cold.ctx)
    assert cold.ctx.identity.revision == version.version_id
    activate.assert_called_once()

    if cache == "damaged-shard":
        (path / full_checkpoint.weight_map["weight"]).write_bytes(b"damaged")
    elif cache == "different-source":
        checkpoint_store.record_artifact(
            path, source={"uri": "az://models/other/model.safetensors.index.json"}
        )

    sdk.client.calls.clear()
    cold.load_uri.reset_mock()
    replacement = _context()

    def load(result, ctx, uri):
        assert ctx is replacement
        assert ctx.identity.revision == "launch-version"
        assert Path(uri) == path
        activate.assert_called_once()
        checkpoint_store.verify_artifact_source(path, {"uri": version.object_storage.uri})
        _assert_checkpoint_tensors(uri, full_checkpoint.tensors)
        result.publishable = False
        return result

    cold.load_uri.side_effect = load
    model = nn.Linear(2, 2)
    assert RLLoadStrategyChain.run(model, replacement) is model
    cold.load_uri.assert_called_once()
    assert replacement.identity.revision == version.version_id
    assert activate.call_count == 2
    assert checkpoint_store.active_version() == version.version_id
    assert checkpoint_store.state().status is CheckpointState.READY
    expected = [] if cache == "valid" else full_checkpoint.expected_calls
    assert Counter(sdk.client.calls) == Counter(expected)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing-index", "BlobNotFound"),
        ("checksum", "full HF checkpoint checksum differs for 'weight'"),
    ],
)
def test_cold_start_rejects_bad_azure_checkpoint_before_engine_load(
    azure_cold_start, checkpoint_store, full_checkpoint, sdk, damage, message,
):
    if damage == "missing-index":
        del sdk.client.objects[full_checkpoint.index_key]
    else:
        key = ("models", f"V2/{full_checkpoint.weight_map['weight']}")
        original = sdk.client.objects[key]
        sdk.client.objects[key] = original[:-1] + bytes([original[-1] ^ 1])
    cold = azure_cold_start
    with pytest.raises(
        StrategyRecoveryError, match="failed to prepare the object-storage checkpoint",
    ) as error:
        RLLoadStrategyChain.run(cold.model, cold.ctx)
    assert message in str(error.value.__cause__)
    cold.load_uri.assert_not_called()
    assert cold.ctx.identity.revision == "launch-version"
    assert not checkpoint_store.active_path.exists()


@pytest.mark.parametrize("failure", ["missing-version", "mixed-providers"])
def test_cold_start_rejects_invalid_azure_chain_before_storage_reads(
    azure_cold_start, azure_control, azure_delta_chain, checkpoint_store,
    sdk, monkeypatch, failure,
):
    if failure == "missing-version":
        message = "desired revision was not found"
        azure_control.fetch_ready_version.side_effect = RuntimeError(message)
    else:
        message = "replay chain spans multiple object-storage providers"
        root = azure_control.version
        azure_control.versions[root.version_id] = replace(
            root,
            object_storage=ObjectStorageSource(
                ObjectStorageType.S3, "s3://models/V2/model.safetensors.index.json"
            ),
        )
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", azure_delta_chain.target.version_id)
    cold = azure_cold_start
    with pytest.raises(
        StrategyRecoveryError, match="failed to resolve the object-storage replay chain",
    ) as error:
        RLLoadStrategyChain.run(cold.model, cold.ctx)
    assert message in str(error.value.__cause__)
    cold.load_uri.assert_not_called()
    sdk.factory.assert_not_called()
    assert not sdk.client.calls
    assert cold.ctx.identity.revision == "launch-version"
    assert not checkpoint_store.active_path.exists()


def test_cold_start_failed_azure_engine_load_does_not_activate_or_report_target(
    azure_cold_start, azure_control, checkpoint_store, full_checkpoint,
):
    cold = azure_cold_start
    failure = RuntimeError("cold-start engine load failed")
    cold.load_uri.side_effect = failure
    with pytest.raises(
        StrategyRecoveryError, match="failed to load the prepared object-storage checkpoint",
    ) as error:
        RLLoadStrategyChain.run(cold.model, cold.ctx)
    assert error.value.__cause__ is failure
    cold.load_uri.assert_called_once()
    assert cold.ctx.identity.revision == "launch-version"
    assert checkpoint_store.state().status is CheckpointState.READY
    assert checkpoint_store.state().version == azure_control.version.version_id
    assert not checkpoint_store.active_path.exists()
