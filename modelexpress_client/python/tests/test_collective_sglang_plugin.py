# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the worker-local SGLang collective plugin."""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from modelexpress_rl.collective import envs
from modelexpress_rl.collective.integrations import sglang_plugin
from modelexpress_rl.collective.integrations import sglang as sglang_integration
from modelexpress_rl.collective.integrations.miles import CollectiveTopology
from modelexpress_rl.collective.integrations.wire import (
    CollectiveControl,
    DestinationManifestEntry,
    encode_control,
)
from modelexpress_rl.collective.types import (
    MeshSpec,
    ParamPlan,
    Placement,
    ReshardPlan,
)


def _plan() -> ReshardPlan:
    return ReshardPlan(
        bulk=[
            ParamPlan(
                name="model.layers.0.weight",
                global_shape=(12, 4),
                dtype="bfloat16",
                partition_id=0,
                src_mesh=MeshSpec((1,)),
                src_placements=(Placement.shard(0),),
                dst_mesh=MeshSpec((6,), rank_offset=1),
                dst_placements=(Placement.shard(0),),
            )
        ],
        source_partition_count=2,
    )


def _bulk_plan(entries: list[tuple[str, tuple[int, ...], str]]) -> ReshardPlan:
    """Multi-entry bulk plan; dim 0 of every shape must be divisible by 6."""
    return ReshardPlan(
        bulk=[
            ParamPlan(
                name=name,
                global_shape=shape,
                dtype=dtype,
                partition_id=0,
                src_mesh=MeshSpec((1,)),
                src_placements=(Placement.shard(0),),
                dst_mesh=MeshSpec((6,), rank_offset=1),
                dst_placements=(Placement.shard(0),),
            )
            for name, shape, dtype in entries
        ],
        source_partition_count=2,
    )


def _captured_prepare(monkeypatch, plan: ReshardPlan):
    """Run _prepare with a mocked session and return its captured kwargs."""
    captured = {}

    class Session:
        def prepare(self):
            pass

        def close(self):
            pass

    def create_session(**kwargs):
        captured.update(kwargs)
        return Session()

    channels, _rendezvous = _patch_runtime(monkeypatch, create_session)
    manager = _slotted_manager()
    result = sglang_plugin._prepare(
        manager,
        CollectiveControl(
            action="prepare",
            plan=plan,
            topology=_topology(),
            generator_slot_offset=0,
            endpoint="mx:50051",
        ),
    )
    return result, captured, channels, manager


def _topology(*, slot_prefix: str = "") -> CollectiveTopology:
    return CollectiveTopology(
        model_name="qwen",
        trainer_slots=tuple(f"{slot_prefix}trainer-{rank}" for rank in range(2)),
        generator_slots=tuple(f"{slot_prefix}generator-{rank}" for rank in range(6)),
        source_partition_count=2,
        m2n_abi_version="abi-1",
    )


@dataclass(slots=True)
class _SlottedManager:
    tp_worker: object
    _weight_update_in_progress: bool = True


def _slotted_manager() -> _SlottedManager:
    runner = SimpleNamespace(
        model=SimpleNamespace(load_weights=lambda _weights: None),
        device=torch.device("cpu"),
    )
    return _SlottedManager(
        tp_worker=SimpleNamespace(
            ps=SimpleNamespace(
                tp_rank=1,
                tp_size=6,
                moe_ep_rank=0,
                moe_ep_size=1,
                dp_rank=0,
                dp_size=1,
                pp_rank=0,
                pp_size=1,
            ),
            server_args=SimpleNamespace(
                enable_eplb=False,
                elastic_ep_backend=None,
                ep_num_redundant_experts=0,
                init_expert_location="trivial",
            ),
            model_runner=runner,
        )
    )


def _control_request(action: str, **kwargs):
    return SimpleNamespace(
        group_name=encode_control(CollectiveControl(action=action, **kwargs))
    )


def _patch_runtime(monkeypatch, create_session):
    channels = []
    rendezvous = []

    class Channel:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Rendezvous:
        def __init__(self, channel):
            self.channel = channel
            self.closed = False

        def close(self):
            self.closed = True

    def make_channel(_endpoint):
        channel = Channel()
        channels.append(channel)
        return channel

    def make_rendezvous(channel):
        value = Rendezvous(channel)
        rendezvous.append(value)
        return value

    monkeypatch.setattr(
        sglang_plugin.SglangGeneratorSession,
        "create",
        create_session,
    )
    monkeypatch.setattr(sglang_plugin, "CollectiveRendezvous", make_rendezvous)
    monkeypatch.setattr(sglang_plugin.grpc, "insecure_channel", make_channel)
    monkeypatch.setattr(sglang_plugin.auth, "with_auth", lambda channel: channel)
    monkeypatch.setattr(
        sglang_plugin,
        "_output",
        lambda success, message: SimpleNamespace(success=success, message=message),
    )
    monkeypatch.setattr(
        sglang_integration,
        "_tensor_signature",
        lambda _name, tensor, **_kwargs: SimpleNamespace(
            address=tensor.data_ptr(),
            shape=tuple(tensor.shape),
            dtype="bfloat16",
            device=str(tensor.device),
        ),
    )
    monkeypatch.setattr(
        sglang_integration,
        "_check_stable",
        lambda *_args, **_kwargs: None,
    )
    return channels, rendezvous


def _patch_refit_api(
    monkeypatch,
    *,
    parameter_name="model.layers.0.weight",
    recipe="dense_down",
):
    events = []

    @dataclass(frozen=True)
    class BlockFp8Spec:
        quant_method: str
        activation_scheme: str
        weight_block_size: tuple[int, int]
        weight_dtype: torch.dtype
        scale_dtype: torch.dtype
        scale_format: str

    @dataclass(frozen=True)
    class CanonicalRefitTensor:
        name: str
        local_shape: tuple[int, ...]
        dtype: torch.dtype
        family: str
        pair_id: str | None = None
        tensor_role: str | None = None
        quantization: BlockFp8Spec | None = None

    @dataclass(frozen=True)
    class SglangRefitTopology:
        tp_rank: int
        tp_size: int
        moe_ep_rank: int
        moe_ep_size: int
        moe_tp_rank: int
        moe_tp_size: int
        dp_rank: int
        dp_size: int
        pp_rank: int
        pp_size: int
        static_expert_placement: bool

    class Adapter:
        def __init__(self, descriptors):
            self.descriptors = tuple(descriptors)
            self.destinations = {}

        def prepare_round(self):
            events.append("prepare")
            self.destinations = {}
            for descriptor in self.descriptors:
                live = torch.zeros(descriptor.local_shape, dtype=descriptor.dtype)
                receive = torch.empty(descriptor.local_shape, dtype=descriptor.dtype)

                def install(
                    *,
                    source=receive,
                    target=live,
                    name=descriptor.name,
                ):
                    events.append(("install", name))
                    target.copy_(source)

                metadata = SimpleNamespace(
                    canonical_name=descriptor.name,
                    parameter_name=parameter_name,
                    recipe=recipe,
                    family=descriptor.family,
                    tensor_role=descriptor.tensor_role,
                    quantization=descriptor.quantization,
                    receive_shape=tuple(receive.shape),
                    receive_dtype=receive.dtype,
                    live_shape=tuple(live.shape),
                    live_dtype=live.dtype,
                    receive_aliases_live=False,
                    requires_install=True,
                )
                self.destinations[descriptor.name] = SimpleNamespace(
                    live=live,
                    receive=receive,
                    install=install,
                    metadata=metadata,
                )

        def start_round(self):
            events.append("start")

        def finish_round(self):
            events.append("finish")

        def abort_round(self):
            events.append("abort")

    adapters = []

    def bind(_model, descriptors, *, topology, device):
        events.append(("bind", topology, device))
        adapter = Adapter(descriptors)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(
        sglang_plugin,
        "_sglang_refit_api",
        lambda: (
            BlockFp8Spec,
            CanonicalRefitTensor,
            SglangRefitTopology,
            bind,
            lambda: SimpleNamespace(moe_tp_rank=1, moe_tp_size=6),
        ),
    )
    return events, adapters


def test_endpoint_rejects_unsupported_secure_schemes():
    for scheme in ("https", "grpcs"):
        with pytest.raises(ValueError, match="secure ModelExpress endpoints"):
            sglang_plugin._endpoint(f"{scheme}://mx.example:50051")


def test_prepare_uses_auth_wrapped_channel(monkeypatch):
    class Session:
        def prepare(self):
            pass

        def close(self):
            pass

    channels, rendezvous = _patch_runtime(monkeypatch, lambda **_kwargs: Session())
    captured = {}

    class AuthChannel:
        def close(self):
            pass

    auth_channel = AuthChannel()
    monkeypatch.setattr(
        sglang_plugin.auth,
        "with_auth",
        lambda channel: captured.setdefault("raw_channel", channel) and auth_channel,
    )
    manager = _slotted_manager()

    result = sglang_plugin._prepare(
        manager,
        CollectiveControl(
            action="prepare",
            plan=_plan(),
            topology=_topology(),
            generator_slot_offset=0,
            endpoint="mx:50051",
        ),
    )

    assert result.success
    assert captured["raw_channel"] is channels[0]
    assert rendezvous[0].channel is auth_channel
    sglang_plugin._close(manager)


def test_prepare_uses_the_declared_run_scoped_slot_with_sparse_gpu_offsets(
    monkeypatch,
):
    captured = {}

    class Session:
        def prepare(self):
            pass

        def close(self):
            pass

    def create_session(**kwargs):
        captured.update(kwargs)
        return Session()

    _patch_runtime(monkeypatch, create_session)
    topology = CollectiveTopology(
        model_name="qwen",
        trainer_slots=("run-7:trainer-0", "run-7:trainer-1"),
        generator_slots=(
            "run-7:generator-0",
            "run-7:generator-1",
            "run-7:generator-4",
            "run-7:generator-5",
        ),
        source_partition_count=2,
        m2n_abi_version="abi-1",
    )

    result = sglang_plugin._prepare(
        _slotted_manager(),
        CollectiveControl(
            action="prepare",
            plan=_plan(),
            topology=topology,
            generator_slot_offset=2,
            endpoint="mx:50051",
            semantic_manifest_version="native-records-v1",
            semantic_manifest_digest="a" * 64,
        ),
    )

    assert result.success
    assert captured["slot_id"] == "run-7:generator-5"
    assert captured["index_in_role"] == 3
    assert captured["semantic_manifest_version"] == "native-records-v1"
    assert captured["semantic_manifest_digest"] == "a" * 64


def test_pr3304_prepare_rejects_destination_drift_before_ready(
    monkeypatch,
):
    events, _adapters = _patch_refit_api(
        monkeypatch,
        parameter_name="model.layers.0.mlp.gate_up_proj.weight",
        recipe="dense_gate",
    )
    session_creations = []
    _patch_runtime(
        monkeypatch,
        lambda **kwargs: session_creations.append(kwargs),
    )
    plan = ReshardPlan(
        bulk=[
            ParamPlan(
                name="model.layers.0.mlp.gate_proj.weight",
                global_shape=(12, 4),
                dtype="bfloat16",
                partition_id=0,
                src_mesh=MeshSpec((1,)),
                src_placements=(Placement.shard(0),),
                dst_mesh=MeshSpec((6,), rank_offset=1),
                dst_placements=(Placement.shard(0),),
            )
        ],
        source_partition_count=2,
    )
    control = CollectiveControl(
        action="prepare",
        plan=plan,
        topology=_topology(),
        generator_slot_offset=0,
        endpoint="mx:50051",
        semantic_manifest_version="miles-nccl-m2n-manifest-v1",
        semantic_manifest_digest="a" * 64,
        destination_manifest=(
            DestinationManifestEntry(
                name="model.layers.0.mlp.gate_proj.weight",
                dtype="bfloat16",
                local_shape=(2, 4),
                parameter="model.layers.0.mlp.WRONG.weight",
                recipe="dense_gate",
            ),
        ),
    )

    with pytest.raises(ValueError, match="resolved destination semantics differ"):
        sglang_plugin._prepare(_slotted_manager(), control)

    assert events[-1] == "abort"
    assert session_creations == []


def test_pr3304_prepare_executes_resolved_destination_and_reuses_bindings(
    monkeypatch,
):
    events, adapters = _patch_refit_api(monkeypatch)
    captured = {}

    class Session:
        def prepare(self):
            captured["prepared"] = True

        def close(self):
            captured["closed"] = True

    def create_session(**kwargs):
        captured.update(kwargs)
        return Session()

    _patch_runtime(monkeypatch, create_session)
    manager = _slotted_manager()
    model_loads = []
    manager.tp_worker.model_runner.model.load_weights = lambda weights: (
        model_loads.extend(weights)
    )
    control = CollectiveControl(
        action="prepare",
        plan=_plan(),
        topology=_topology(),
        generator_slot_offset=0,
        endpoint="mx:50051",
        semantic_manifest_version="miles-nccl-m2n-manifest-v1",
        semantic_manifest_digest="a" * 64,
        destination_manifest=(
            DestinationManifestEntry(
                name="model.layers.0.weight",
                dtype="bfloat16",
                local_shape=(2, 4),
                parameter="model.layers.0.weight",
                recipe="dense_down",
            ),
        ),
    )

    result = sglang_plugin._prepare(manager, control)
    loader = captured["loader"]
    destination = adapters[0].destinations["model.layers.0.weight"]
    receive_address = destination.receive.data_ptr()

    for version, value in (("1", 3), ("2", 7)):
        loader.start_new_round(version)
        loader.local_params()["model.layers.0.weight"].base.fill_(value)
        loader.install(0)
        loader.finish()
        assert torch.all(destination.live == value)

    sglang_plugin._close(manager)

    assert result.success
    assert captured["prepared"]
    assert captured["closed"]
    assert destination.receive.data_ptr() == receive_address
    assert model_loads == []
    assert events[1:] == [
        "prepare",
        ("install", "model.layers.0.weight"),
        "finish",
        "start",
        ("install", "model.layers.0.weight"),
        "finish",
    ]


def test_non_marker_requests_pass_through_unchanged():
    request = SimpleNamespace(group_name="stock-nccl-group")
    manager = object()
    called = []

    def original(received_manager, received_request):
        called.append((received_manager, received_request))
        return "stock-result"

    result = sglang_plugin.around_update_weights_from_distributed(
        original,
        manager,
        request,
    )

    assert result == "stock-result"
    assert called == [(manager, request)]


def test_prepare_builds_canonical_staging_and_installs_through_model_load_weights(
    monkeypatch,
):
    monkeypatch.setenv("MX_MILES_VERIFY_TENSOR_EQUALITY", "1")
    loaded = []

    class Model:
        def load_weights(self, weights):
            loaded.extend(weights)

    runner = SimpleNamespace(model=Model(), device=torch.device("cpu"))
    manager = SimpleNamespace(
        tp_worker=SimpleNamespace(
            ps=SimpleNamespace(tp_rank=1),
            model_runner=runner,
        ),
        _weight_update_in_progress=True,
    )
    captured = {}

    class Session:
        def prepare(self):
            captured["prepared"] = True

        def close(self):
            pass

    def create_session(**kwargs):
        captured.update(kwargs)
        return Session()

    monkeypatch.setattr(
        sglang_plugin.SglangGeneratorSession,
        "create",
        create_session,
    )
    monkeypatch.setattr(
        sglang_plugin,
        "CollectiveRendezvous",
        lambda _channel: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        sglang_plugin.grpc,
        "insecure_channel",
        lambda _endpoint: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(sglang_plugin.auth, "with_auth", lambda channel: channel)
    monkeypatch.setattr(
        sglang_plugin,
        "_output",
        lambda success, message: SimpleNamespace(success=success, message=message),
    )
    monkeypatch.setattr(
        sglang_integration,
        "_tensor_signature",
        lambda _name, tensor, **_kwargs: SimpleNamespace(
            address=tensor.data_ptr(),
            shape=tuple(tensor.shape),
            dtype="bfloat16",
            device=str(tensor.device),
        ),
    )
    monkeypatch.setattr(
        sglang_integration,
        "_check_stable",
        lambda *_args, **_kwargs: None,
    )
    request = SimpleNamespace(
        group_name=encode_control(
            CollectiveControl(
                action="prepare",
                plan=_plan(),
                topology=_topology(),
                generator_slot_offset=0,
                endpoint="mx:50051",
            )
        )
    )

    response = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        request,
    )
    loader = captured["loader"]
    monkeypatch.setattr(
        sglang_integration,
        "_exact_tensor_sha256",
        lambda _tensor: "01" * 32,
    )
    loader.expect_round(
        version="1",
        operation_id="operation-1",
        tensor_digests=(("model.layers.0.weight", "01" * 32),),
    )
    loader.start_new_round("1")
    loader.local_params()["model.layers.0.weight"].base.fill_(7)
    loader.install(0)
    closed = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request("close"),
    )

    assert response.success
    assert closed.success
    assert captured["prepared"]
    assert captured["slot_id"] == "generator-1"
    assert captured["index_in_role"] == 1
    assert loaded[0][0] == "model.layers.0.weight"
    assert tuple(loaded[0][1].shape) == (2, 4)
    assert torch.all(loaded[0][1] == 7)


def test_slotted_managers_have_isolated_prepare_run_and_close_state(monkeypatch):
    sessions = []
    worker_ids = []

    class Session:
        def __init__(self):
            self.prepared = False
            self.rounds = []
            self.closed = False

        def prepare(self):
            self.prepared = True

        def run_round(self, *, version, operation_id, tensor_digests):
            self.rounds.append((version, operation_id, tensor_digests))

        def close(self):
            self.closed = True

    def create_session(**kwargs):
        session = Session()
        sessions.append(session)
        worker_ids.append(kwargs["worker_id"])
        return session

    channels, _rendezvous = _patch_runtime(monkeypatch, create_session)
    manager = _slotted_manager()
    other_manager = _slotted_manager()

    prepared = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request(
            "prepare",
            plan=_plan(),
            topology=_topology(),
            generator_slot_offset=0,
            endpoint="mx:50051",
        ),
    )
    other_prepared = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        other_manager,
        _control_request(
            "prepare",
            plan=_plan(),
            topology=_topology(),
            generator_slot_offset=0,
            endpoint="mx:50051",
        ),
    )
    other_run = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        other_manager,
        _control_request(
            "run_round",
            version="version-other",
            operation_id="operation-other",
        ),
    )
    completed = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request(
            "run_round",
            version="version-1",
            operation_id="operation-1",
        ),
    )
    closed = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request("close"),
    )
    other_closed = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        other_manager,
        _control_request("close"),
    )
    after_close = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request(
            "run_round",
            version="version-2",
            operation_id="operation-2",
        ),
    )

    assert prepared.success
    assert other_prepared.success
    assert other_run.success
    assert completed.success
    assert sessions[0].prepared
    assert sessions[0].rounds == [
        (
            "version-1",
            "operation-1",
            None,
        )
    ]
    assert sessions[1].rounds == [
        (
            "version-other",
            "operation-other",
            None,
        )
    ]
    assert closed.success
    assert sessions[0].closed
    assert channels[0].closed
    assert other_closed.success
    assert sessions[1].closed
    assert channels[1].closed
    assert worker_ids[0].startswith("sglang-generator-1-")
    assert worker_ids[1].startswith("sglang-generator-1-")
    assert worker_ids[0] != worker_ids[1]
    assert all(len(worker_id.rsplit("-", 1)[1]) == 32 for worker_id in worker_ids)
    assert not after_close.success
    assert "not prepared" in after_close.message


def test_failed_prepare_cleans_resources_and_allows_retry(monkeypatch):
    sessions = []

    class Session:
        def __init__(self, error=None):
            self.error = error
            self.closed = False

        def prepare(self):
            if self.error is not None:
                raise self.error

        def close(self):
            self.closed = True

    def create_session(**_kwargs):
        error = RuntimeError("prepare failed") if not sessions else None
        session = Session(error)
        sessions.append(session)
        return session

    channels, rendezvous = _patch_runtime(monkeypatch, create_session)
    manager = _slotted_manager()
    request = _control_request(
        "prepare",
        plan=_plan(),
        topology=_topology(),
        generator_slot_offset=0,
        endpoint="mx:50051",
    )

    failed = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        request,
    )
    retried = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        request,
    )
    closed = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request("close"),
    )

    assert not failed.success
    assert "prepare failed" in failed.message
    assert sessions[0].closed
    assert rendezvous[0].closed
    assert channels[0].closed
    assert retried.success
    assert closed.success
    assert sessions[1].closed
    assert channels[1].closed


def test_round_rejects_tensor_verification_setting_mismatch_and_cleans_state(
    monkeypatch,
):
    sessions = []

    class Session:
        def __init__(self):
            self.closed = False

        def prepare(self):
            pass

        def close(self):
            self.closed = True

    def create_session(**_kwargs):
        session = Session()
        sessions.append(session)
        return session

    channels, rendezvous = _patch_runtime(monkeypatch, create_session)
    manager = _slotted_manager()
    prepared = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request(
            "prepare",
            plan=_plan(),
            topology=_topology(),
            generator_slot_offset=0,
            endpoint="mx:50051",
        ),
    )
    monkeypatch.setenv("MX_MILES_VERIFY_TENSOR_EQUALITY", "1")

    mismatch = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request(
            "run_round",
            version="version-1",
            operation_id="operation-1",
        ),
    )
    after_failure = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request(
            "run_round",
            version="version-2",
            operation_id="operation-2",
        ),
    )

    assert prepared.success
    assert not mismatch.success
    assert "disagree on MX_MILES_VERIFY_TENSOR_EQUALITY" in mismatch.message
    assert sessions[0].closed
    assert rendezvous[0].closed
    assert channels[0].closed
    assert not after_failure.success
    assert "not prepared" in after_failure.message


def test_state_publication_failure_cleans_all_prepared_resources(monkeypatch):
    sessions = []

    class Session:
        def __init__(self):
            self.closed = False

        def prepare(self):
            pass

        def close(self):
            self.closed = True

    def create_session(**_kwargs):
        session = Session()
        sessions.append(session)
        return session

    publish_state = sglang_plugin._publish_state

    def fail_publication(*args, **kwargs):
        publish_state(*args, **kwargs)
        raise RuntimeError("state publication failed")

    channels, rendezvous = _patch_runtime(monkeypatch, create_session)
    monkeypatch.setattr(
        sglang_plugin,
        "_publish_state",
        fail_publication,
    )
    manager = _slotted_manager()

    response = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request(
            "prepare",
            plan=_plan(),
            topology=_topology(),
            generator_slot_offset=0,
            endpoint="mx:50051",
        ),
    )
    after_failure = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request(
            "run_round",
            version="version-1",
            operation_id="operation-1",
        ),
    )

    assert not response.success
    assert "state publication failed" in response.message
    assert sessions[0].closed
    assert rendezvous[0].closed
    assert channels[0].closed
    assert not after_failure.success
    assert "not prepared" in after_failure.message


def test_marked_request_requires_the_existing_weight_update_session(monkeypatch):
    monkeypatch.setattr(
        sglang_plugin,
        "_output",
        lambda success, message: SimpleNamespace(success=success, message=message),
    )
    manager = SimpleNamespace(_weight_update_in_progress=False)
    request = SimpleNamespace(
        group_name=encode_control(
            CollectiveControl(
                action="run_round",
                version="1",
                operation_id="operation-1",
            )
        )
    )

    response = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        request,
    )

    assert not response.success
    assert "open begin_weight_update session" in response.message


def test_prepare_succeeds_after_weight_update_session_opens(monkeypatch):
    class Session:
        def prepare(self):
            pass

        def close(self):
            pass

    _patch_runtime(monkeypatch, lambda **_kwargs: Session())
    manager = _slotted_manager()
    manager._weight_update_in_progress = False
    request = _control_request(
        "prepare",
        plan=_plan(),
        topology=_topology(),
        generator_slot_offset=0,
        endpoint="mx:50051",
    )

    before_session = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        request,
    )
    manager._weight_update_in_progress = True
    during_session = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        request,
    )
    sglang_plugin._close(manager)

    assert not before_session.success
    assert "open begin_weight_update session" in before_session.message
    assert during_session.success


def test_close_is_allowed_after_the_weight_update_session_ends(monkeypatch):
    class Session:
        def __init__(self):
            self.closed = False

        def prepare(self):
            pass

        def close(self):
            self.closed = True

    sessions = []

    def create_session(**_kwargs):
        session = Session()
        sessions.append(session)
        return session

    channels, rendezvous = _patch_runtime(monkeypatch, create_session)
    manager = _slotted_manager()
    prepared = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request(
            "prepare",
            plan=_plan(),
            topology=_topology(),
            generator_slot_offset=0,
            endpoint="mx:50051",
        ),
    )
    manager._weight_update_in_progress = False

    closed = sglang_plugin.around_update_weights_from_distributed(
        lambda *_args: None,
        manager,
        _control_request("close"),
    )

    assert prepared.success
    assert closed.success
    assert sessions[0].closed
    assert rendezvous[0].closed
    assert channels[0].closed


def test_prepare_group_bytes_zero_keeps_one_group_per_bulk_entry(monkeypatch):
    monkeypatch.setenv("MX_NCCL_REFIT_GROUP_BYTES", "0")
    plan = _bulk_plan(
        [
            ("model.layers.0.weight", (6, 4), "bfloat16"),
            ("model.layers.1.weight", (12, 4), "bfloat16"),
            ("model.layers.2.weight", (6, 2), "bfloat16"),
        ]
    )

    result, captured, _channels, manager = _captured_prepare(monkeypatch, plan)

    assert result.success
    assert captured["layer_groups"] == tuple((entry.name,) for entry in plan.bulk)
    assert captured["layer_groups"] == (
        ("model.layers.0.weight",),
        ("model.layers.1.weight",),
        ("model.layers.2.weight",),
    )
    sglang_plugin._close(manager)


def test_prepare_group_bytes_packs_consecutive_entries_within_target(monkeypatch):
    # bf16 sizes: (6, 4) -> 48 B, (12, 4) -> 96 B, (6, 2) -> 24 B.
    monkeypatch.setenv("MX_NCCL_REFIT_GROUP_BYTES", "96")
    plan = _bulk_plan(
        [
            ("a", (6, 4), "bfloat16"),
            ("b", (6, 4), "bfloat16"),
            ("c", (12, 4), "bfloat16"),
            ("d", (6, 2), "bfloat16"),
            ("e", (6, 2), "bfloat16"),
        ]
    )

    result, captured, _channels, manager = _captured_prepare(monkeypatch, plan)

    assert result.success
    groups = captured["layer_groups"]
    assert groups == (("a", "b"), ("c",), ("d", "e"))
    # Plan order is preserved and every entry is covered exactly once.
    assert [name for group in groups for name in group] == [
        entry.name for entry in plan.bulk
    ]
    # No group exceeds the byte target (this plan has no oversize entry).
    bytes_by_name = {
        entry.name: math.prod(entry.global_shape) * 2 for entry in plan.bulk
    }
    for group in groups:
        assert sum(bytes_by_name[name] for name in group) <= 96
    sglang_plugin._close(manager)


def test_prepare_group_bytes_gives_an_oversize_entry_its_own_group(monkeypatch):
    # bf16 sizes: (6, 2) -> 24 B, (12, 8) -> 192 B, oversize for a 48 B target.
    monkeypatch.setenv("MX_NCCL_REFIT_GROUP_BYTES", "48")
    plan = _bulk_plan(
        [
            ("a", (6, 2), "bfloat16"),
            ("big", (12, 8), "bfloat16"),
            ("c", (6, 2), "bfloat16"),
        ]
    )

    result, captured, _channels, manager = _captured_prepare(monkeypatch, plan)

    assert result.success
    assert captured["layer_groups"] == (("a",), ("big",), ("c",))
    sglang_plugin._close(manager)


def test_prepare_group_bytes_rejects_unknown_dtype_before_connecting(monkeypatch):
    monkeypatch.setenv("MX_NCCL_REFIT_GROUP_BYTES", "1024")
    plan = _bulk_plan([("bad", (6, 4), "complex64")])
    captured = {}

    class Session:
        def prepare(self):
            pass

        def close(self):
            pass

    def create_session(**kwargs):
        captured.update(kwargs)
        return Session()

    channels, rendezvous = _patch_runtime(monkeypatch, create_session)

    with pytest.raises(ValueError, match=r"bad.*unknown dtype.*complex64"):
        sglang_plugin._prepare(
            _slotted_manager(),
            CollectiveControl(
                action="prepare",
                plan=plan,
                topology=_topology(),
                generator_slot_offset=0,
                endpoint="mx:50051",
            ),
        )

    # The failure happens before any channel, rendezvous or session exists.
    assert channels == []
    assert rendezvous == []
    assert captured == {}


def test_entry_bytes_sizes_the_dtypes_plans_carry():
    cases = [
        ("bfloat16", 2),
        ("float16", 2),
        ("float32", 4),
        ("float8_e4m3fn", 1),
        ("int32", 4),
        ("int64", 8),
        ("uint8", 1),
    ]
    for dtype, size in cases:
        entry = ParamPlan(
            name=f"p-{dtype}",
            global_shape=(6, 4),
            dtype=dtype,
            partition_id=0,
            src_mesh=MeshSpec((1,)),
            src_placements=(Placement.shard(0),),
            dst_mesh=MeshSpec((6,), rank_offset=1),
            dst_placements=(Placement.shard(0),),
        )
        assert sglang_plugin._entry_bytes(entry) == 24 * size


def test_plan_layer_groups_is_deterministic_and_covers_the_plan(monkeypatch):
    monkeypatch.setenv("MX_NCCL_REFIT_GROUP_BYTES", "96")
    plan = _bulk_plan(
        [
            ("a", (6, 4), "bfloat16"),
            ("b", (6, 4), "bfloat16"),
            ("c", (12, 4), "bfloat16"),
            ("d", (6, 2), "bfloat16"),
            ("e", (6, 2), "bfloat16"),
        ]
    )

    first = sglang_plugin._plan_layer_groups(
        list(plan.bulk), envs.MX_NCCL_REFIT_GROUP_BYTES
    )
    second = sglang_plugin._plan_layer_groups(
        list(plan.bulk), envs.MX_NCCL_REFIT_GROUP_BYTES
    )

    assert first == second
    assert [name for group in first for name in group] == [
        entry.name for entry in plan.bulk
    ]
    assert len({name for group in first for name in group}) == len(plan.bulk)
