# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-authority PR 3304 manifest integration."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from modelexpress_rl import refit_collective_pb2 as collective_pb
from modelexpress_rl.collective import RefitClientGenerator, RefitClientTrainer
from modelexpress_rl.collective.integrations.miles_pr3304 import (
    MilesPr3304ProtocolCore,
    SEMANTIC_MANIFEST_VERSION,
    _pr3304_hf_exclusions,
    canonical_pr3304_manifest_digest,
    project_pr3304_manifest,
)
from modelexpress_rl.collective.integrations.miles_topology import (
    MilesTrainerTopology,
)


def _topology(world_rank: int, pp_rank: int, ep_rank: int) -> MilesTrainerTopology:
    return MilesTrainerTopology(
        world_rank=world_rank,
        pp_rank=pp_rank,
        pp_size=2,
        tp_rank=0,
        tp_size=1,
        cp_rank=0,
        cp_size=1,
        dense_dp_rank=ep_rank,
        dense_dp_size=2,
        ep_rank=ep_rank,
        ep_size=2,
        etp_rank=0,
        etp_size=1,
        expert_dp_rank=0,
        expert_dp_size=1,
        independent_dp_rank=0,
        independent_dp_size=1,
    )


def _entry(
    *,
    name: str,
    pp_rank: int,
    family: str,
    source_ranks: list[int],
    source_names: dict[str, list[str]],
    source_recipe: str,
    pair_id: str | None = None,
    tensor_role: str | None = None,
) -> dict:
    entry = {
        "name": name,
        "family": family,
        "pp_rank": pp_rank,
        "dtype": "bfloat16",
        "global_shape": [8, 4],
        "source": {
            "mesh": [source_ranks],
            "placements": [
                {"type": "replicate"},
                {"type": "shard", "dim": 0},
            ],
            "local_shape": [8 // len(source_ranks), 4],
            "names_by_rank": source_names,
            "recipe": source_recipe,
        },
        "destination": {
            "mesh": [[4, 5]],
            "placements": [
                {"type": "replicate"},
                {"type": "shard", "dim": 0},
            ],
            "local_shape": [4, 4],
            "parameter": name,
            "recipe": "destination",
        },
    }
    if pair_id is not None:
        entry["pair_id"] = pair_id
        entry["tensor_role"] = tensor_role
    return entry


def _manifest() -> dict:
    entries = [
        _entry(
            name="model.layers.0.mlp.gate_proj.weight",
            pp_rank=0,
            family="dense",
            source_ranks=[0],
            source_names={"0": ["native.pp0.fc1"]},
            source_recipe="dense_fc1_0",
        ),
        _entry(
            name="model.layers.0.mlp.experts.down_proj.weight",
            pp_rank=0,
            family="routed_expert",
            source_ranks=[0, 1],
            source_names={
                "0": ["native.pp0.expert0.fc2"],
                "1": ["native.pp0.expert1.fc2"],
            },
            source_recipe="expert_fc2",
        ),
        _entry(
            name="model.layers.1.mlp.gate_proj.weight",
            pp_rank=1,
            family="dense",
            source_ranks=[2],
            source_names={"2": ["native.pp1.fc1"]},
            source_recipe="dense_fc1_0",
        ),
        _entry(
            name="model.layers.1.mlp.experts.down_proj.weight",
            pp_rank=1,
            family="routed_expert",
            source_ranks=[2, 3],
            source_names={
                "2": ["native.pp1.expert0.fc2"],
                "3": ["native.pp1.expert1.fc2"],
            },
            source_recipe="expert_fc2",
        ),
    ]
    manifest = {
        "schema_version": 1,
        "source_world_ranks": [30, 10, 40, 20],
        "trainer_world_to_comm_rank": {
            "30": 0,
            "10": 1,
            "40": 2,
            "20": 3,
        },
        "communicator_world_size": 6,
        "routed_update_units": [
            ["native.pp0.fc1"],
            ["native.pp0.expert0.fc2", "native.pp0.expert1.fc2"],
            ["native.pp1.fc1"],
            ["native.pp1.expert0.fc2", "native.pp1.expert1.fc2"],
        ],
        "entries": entries,
    }
    manifest["manifest_hash"] = canonical_pr3304_manifest_digest(manifest)
    return manifest


def _topologies() -> list[MilesTrainerTopology]:
    return [
        _topology(30, 0, 0),
        _topology(10, 0, 1),
        _topology(40, 1, 0),
        _topology(20, 1, 1),
    ]


def _update_units() -> list[tuple[str, ...]]:
    return [
        ("native.pp0.fc1",),
        ("native.pp0.expert0.fc2", "native.pp0.expert1.fc2"),
        ("native.pp1.fc1",),
        ("native.pp1.expert0.fc2", "native.pp1.expert1.fc2"),
        ("native.final_norm",),
    ]


def test_projection_uses_exact_hashed_manifest_and_lane_local_ranks() -> None:
    projection = project_pr3304_manifest(
        _manifest(),
        _topologies(),
        update_units=_update_units(),
        pp_wave_size=2,
    )

    assert projection.semantic_manifest_version == SEMANTIC_MANIFEST_VERSION
    assert projection.semantic_manifest_digest == _manifest()["manifest_hash"]
    assert projection.topology_plan.trainer_lanes == ((30, 10), (40, 20))
    assert projection.topology_plan.residual_update_units == (("native.final_norm",),)

    entries = {entry.name: entry for entry in projection.topology_plan.plan.bulk}
    assert entries["model.layers.0.mlp.gate_proj.weight"].src_mesh.ranks() == [0]
    assert entries["model.layers.0.mlp.experts.down_proj.weight"].src_mesh.ranks() == [
        0,
        1,
    ]
    assert entries["model.layers.1.mlp.gate_proj.weight"].src_mesh.ranks() == [0]
    assert entries["model.layers.0.mlp.gate_proj.weight"].dst_mesh.ranks() == [
        2,
        3,
    ]

    routes = {route.canonical_name: route for route in projection.topology_plan.routes}
    assert routes[
        "model.layers.0.mlp.experts.down_proj.weight"
    ].source_names_by_world == (
        (30, ("native.pp0.expert0.fc2",)),
        (10, ("native.pp0.expert1.fc2",)),
    )
    destinations = {entry.name: entry for entry in projection.destination_manifest}
    assert (
        destinations["model.layers.0.mlp.gate_proj.weight"].parameter
        == "model.layers.0.mlp.gate_proj.weight"
    )
    assert destinations["model.layers.0.mlp.gate_proj.weight"].recipe == "destination"


def test_projection_rejects_any_mutation_not_covered_by_manifest_hash() -> None:
    manifest = _manifest()
    manifest["entries"][0]["source"]["recipe"] = "dense_fc1_1"

    with pytest.raises(ValueError, match="manifest hash validation failed"):
        project_pr3304_manifest(
            manifest,
            _topologies(),
            update_units=_update_units(),
            pp_wave_size=2,
        )


def test_projection_rejects_hashed_source_local_shape_drift() -> None:
    manifest = _manifest()
    manifest["entries"][0]["source"]["local_shape"] = [4, 4]
    manifest["manifest_hash"] = canonical_pr3304_manifest_digest(manifest)

    with pytest.raises(ValueError, match="source local shape"):
        project_pr3304_manifest(
            manifest,
            _topologies(),
            update_units=_update_units(),
            pp_wave_size=2,
        )


def test_hf_exclusions_match_pr3304_expert_expansion() -> None:
    exclusions = _pr3304_hf_exclusions(_manifest())

    assert "model.layers.0.mlp.gate_proj.weight" in exclusions
    assert "model.layers.0.mlp.experts.down_proj.weight" not in exclusions
    assert {
        f"model.layers.0.mlp.experts.{expert}.down_proj.weight" for expert in range(8)
    }.issubset(exclusions)


def test_recipe_and_atomic_metadata_change_admission_identity() -> None:
    left = _manifest()
    right = deepcopy(left)
    right["entries"][0]["source"]["recipe"] = "dense_fc1_1"
    right["routed_update_units"][0] = ["native.pp0.fc1", "native.extra"]
    right["manifest_hash"] = canonical_pr3304_manifest_digest(right)

    left_projection = project_pr3304_manifest(
        left,
        _topologies(),
        update_units=_update_units(),
        pp_wave_size=2,
    )
    with pytest.raises(ValueError, match="unknown MILES update units"):
        project_pr3304_manifest(
            right,
            _topologies(),
            update_units=_update_units(),
            pp_wave_size=2,
        )

    right["routed_update_units"][0] = ["native.pp0.fc1"]
    right["manifest_hash"] = canonical_pr3304_manifest_digest(right)
    right_projection = project_pr3304_manifest(
        right,
        _topologies(),
        update_units=_update_units(),
        pp_wave_size=2,
    )
    assert (
        left_projection.semantic_manifest_digest
        != right_projection.semantic_manifest_digest
    )


class _Engine:
    def __init__(self, plan):
        self.plan = plan

    def capture(self):
        return self.plan

    def parameter_names(self):
        return self.plan.parameter_names()


class _AdmissionBroker:
    def __init__(self, expected_slots: tuple[str, ...]):
        self.expected_slots = expected_slots
        self.digests = {}

    def join(self, slot: str, digest: str) -> str:
        self.digests[slot] = digest
        if set(self.digests) != set(self.expected_slots):
            return "FORMING"
        return "READY" if len(set(self.digests.values())) == 1 else "FORMING"


@pytest.mark.parametrize("divergence", ["recipe", "atomic-unit"])
def test_cross_participant_semantic_divergence_never_reaches_ready(
    divergence: str,
) -> None:
    update_units = _update_units()
    if divergence == "atomic-unit":
        update_units = [
            *update_units,
            ("native.pp0.expert0.fc2",),
            ("native.pp0.expert1.fc2",),
        ]
    left = project_pr3304_manifest(
        _manifest(),
        _topologies(),
        update_units=update_units,
        pp_wave_size=2,
    )
    changed = _manifest()
    if divergence == "recipe":
        changed["entries"][0]["source"]["recipe"] = "dense_fc1_1"
    else:
        changed["routed_update_units"][1:2] = [
            ["native.pp0.expert0.fc2"],
            ["native.pp0.expert1.fc2"],
        ]
    changed["manifest_hash"] = canonical_pr3304_manifest_digest(changed)
    right = project_pr3304_manifest(
        changed,
        _topologies(),
        update_units=update_units,
        pp_wave_size=2,
    )

    trainer = RefitClientTrainer(
        rendezvous=SimpleNamespace(),
        model_name="model",
        trainer_slots=["trainer"],
        generator_slots=["generator"],
        source_partition_count=2,
        slot_id="trainer",
        worker_id="trainer-worker",
        index_in_role=0,
        semantic_manifest_version=left.semantic_manifest_version,
        semantic_manifest_digest=left.semantic_manifest_digest,
    )
    generator = RefitClientGenerator(
        rendezvous=SimpleNamespace(),
        model_name="model",
        trainer_slots=["trainer"],
        generator_slots=["generator"],
        source_partition_count=2,
        slot_id="generator",
        worker_id="generator-worker",
        index_in_role=0,
        semantic_manifest_version=right.semantic_manifest_version,
        semantic_manifest_digest=right.semantic_manifest_digest,
    )
    trainer._capture(_Engine(left.topology_plan.plan), None)
    generator._capture(_Engine(right.topology_plan.plan), None)

    broker = _AdmissionBroker(("trainer", "generator"))
    assert broker.join("trainer", trainer._digest) == "FORMING"
    assert broker.join("generator", generator._digest) == "FORMING"


def test_in_session_prepare_precedes_refresh_and_receiver_launch(monkeypatch) -> None:
    events = []
    core = MilesPr3304ProtocolCore(SimpleNamespace())
    core._generator_endpoint = "mx:50051"
    core._pending_version = "7"
    core._publisher = SimpleNamespace(
        refresh=lambda weights: events.append(("refresh", weights))
    )
    core._session = SimpleNamespace(
        membership=SimpleNamespace(group_id="group"),
        prepare=lambda: events.append("trainer-prepare"),
        run_round=lambda **kwargs: events.append(("trainer", kwargs)),
    )
    core._coordinator = SimpleNamespace(
        create=lambda *args, **kwargs: SimpleNamespace(operation_id="operation"),
        delete=lambda operation_id: events.append(("delete", operation_id)),
    )
    core._generator_futures = lambda action, **kwargs: (
        events.append(("receiver", action, kwargs)) or []
    )
    core._wait_futures = lambda futures: None
    core._wait_terminal = lambda operation_id: (
        events.append(("terminal", operation_id))
        or SimpleNamespace(
            state=0,
            failure_message="",
        )
    )
    core._require_complete = lambda operation_id, transfer: None
    monkeypatch.setattr(core, "_collect_errors", lambda error: [error] if error else [])
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        dist,
        "broadcast_object_list",
        lambda values, src, group: None,
    )
    monkeypatch.setattr(
        "modelexpress_rl.collective.integrations.miles_pr3304._gloo_group",
        lambda: object(),
    )

    weights = {"native": object()}
    core.before_base_weights(weights)

    assert events[0] == ("receiver", "prepare", {"endpoint": "mx:50051"})
    assert events[1] == "trainer-prepare"
    assert events[2] == ("refresh", weights)
    assert events[3][0:2] == ("receiver", "run_round")
    assert events[4][0] == "trainer"


def test_receiver_launch_failure_deletes_operation_before_trainer(monkeypatch) -> None:
    events = []
    core = MilesPr3304ProtocolCore(SimpleNamespace())
    core._generators_prepared = True
    core._pending_version = "8"
    core._publisher = SimpleNamespace(refresh=lambda weights: events.append("refresh"))

    def report_failure(*, operation_id, error):
        events.append(("report", operation_id, str(error)))

    core._session = SimpleNamespace(
        membership=SimpleNamespace(group_id="group"),
        run_round=lambda **kwargs: events.append("trainer"),
        report_failure=report_failure,
    )
    core._coordinator = SimpleNamespace(
        create=lambda *args, **kwargs: SimpleNamespace(operation_id="operation"),
        delete=lambda operation_id: events.append(("delete", operation_id)),
    )

    def fail_launch(*args, **kwargs):
        events.append("receiver")
        raise RuntimeError("receiver launch failed")

    core._generator_futures = fail_launch
    core._wait_futures = lambda futures: events.append(("settle", futures))
    core._wait_terminal = lambda operation_id: (
        events.append(("terminal", operation_id))
        or SimpleNamespace(
            state=0,
            failure_message="",
        )
    )
    core._require_complete = lambda operation_id, transfer: None
    monkeypatch.setattr(core, "_collect_errors", lambda error: [error] if error else [])
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        dist,
        "broadcast_object_list",
        lambda values, src, group: None,
    )
    monkeypatch.setattr(
        "modelexpress_rl.collective.integrations.miles_pr3304._gloo_group",
        lambda: object(),
    )

    with pytest.raises(RuntimeError, match="receiver launch failed"):
        core.before_base_weights({"native": object()})

    assert events == [
        "refresh",
        "receiver",
        ("report", "operation", "RuntimeError('receiver launch failed')"),
        ("settle", []),
        ("terminal", "operation"),
        ("delete", "operation"),
    ]
    assert core._pending_version is None


def test_receiver_launch_failure_keeps_nonterminal_operation(monkeypatch) -> None:
    events = []
    core = MilesPr3304ProtocolCore(SimpleNamespace())
    core._generators_prepared = True
    core._pending_version = "9"
    core._publisher = SimpleNamespace(refresh=lambda weights: None)
    core._session = SimpleNamespace(
        membership=SimpleNamespace(group_id="group"),
        report_failure=lambda **kwargs: events.append("report"),
    )
    core._coordinator = SimpleNamespace(
        create=lambda *args, **kwargs: SimpleNamespace(operation_id="operation"),
        delete=lambda operation_id: pytest.fail("deleted nonterminal operation"),
    )
    core._generator_futures = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("receiver launch failed")
    )
    core._wait_futures = lambda futures: events.append("settle")
    core._wait_terminal = lambda operation_id: (_ for _ in ()).throw(
        TimeoutError("still running")
    )
    monkeypatch.setattr(core, "_collect_errors", lambda error: [error] if error else [])
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        dist,
        "broadcast_object_list",
        lambda values, src, group: None,
    )
    monkeypatch.setattr(
        "modelexpress_rl.collective.integrations.miles_pr3304._gloo_group",
        lambda: object(),
    )

    with pytest.raises(RuntimeError, match="terminal wait"):
        core.before_base_weights({"native": object()})

    assert events == ["report", "settle"]


@pytest.mark.parametrize(
    ("terminal_state", "failure_message"),
    (
        (
            collective_pb.COLLECTIVE_TRANSFER_STATE_FAILED,
            "receiver failed",
        ),
        (
            collective_pb.COLLECTIVE_TRANSFER_STATE_ABORTED,
            "receiver aborted",
        ),
    ),
)
def test_unsuccessful_terminal_operation_is_deleted_after_observation(
    monkeypatch,
    terminal_state,
    failure_message,
) -> None:
    events = []
    core = MilesPr3304ProtocolCore(SimpleNamespace())
    core._generators_prepared = True
    core._pending_version = "10"
    core._publisher = SimpleNamespace(refresh=lambda weights: None)
    core._session = SimpleNamespace(
        membership=SimpleNamespace(group_id="group"),
        run_round=lambda **kwargs: events.append("trainer"),
    )
    core._coordinator = SimpleNamespace(
        create=lambda *args, **kwargs: SimpleNamespace(operation_id="operation"),
        delete=lambda operation_id: events.append(("delete", operation_id)),
    )
    core._generator_futures = lambda *args, **kwargs: []
    core._wait_futures = lambda futures: events.append("settle")
    core._wait_terminal = lambda operation_id: (
        events.append(("terminal", operation_id))
        or SimpleNamespace(
            state=terminal_state,
            failure_message=failure_message,
        )
    )
    monkeypatch.setattr(core, "_collect_errors", lambda error: [error] if error else [])
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        dist,
        "broadcast_object_list",
        lambda values, src, group: None,
    )
    monkeypatch.setattr(
        "modelexpress_rl.collective.integrations.miles_pr3304._gloo_group",
        lambda: object(),
    )

    with pytest.raises(RuntimeError, match=failure_message):
        core.before_base_weights({"native": object()})

    assert events == [
        "trainer",
        "settle",
        ("terminal", "operation"),
        ("delete", "operation"),
    ]


def test_residual_failure_is_deferred_until_iterator_drain() -> None:
    core = MilesPr3304ProtocolCore(SimpleNamespace())
    marker = object()
    first_bucket = [("first", marker)]
    second_bucket = [("second", object())]
    sends = []

    def fail_send(bucket):
        sends.append(list(bucket))
        raise RuntimeError("broadcast failed")

    core._send_residual_bucket(first_bucket, fail_send)
    core._send_residual_bucket(
        second_bucket,
        lambda bucket: pytest.fail("send retried after deferred failure"),
    )

    assert sends == [[("first", marker)]]
    assert first_bucket == []
    assert second_bucket == []
    assert "broadcast failed" in core._deferred_update_error

    core._collect_errors = lambda error: [error]
    with pytest.raises(RuntimeError, match="residual update failed"):
        core._finish_residual_stream()


def test_run_engine_session_propagates_rank_zero_failure_to_peers() -> None:
    core = MilesPr3304ProtocolCore(SimpleNamespace())
    observed = []
    core._collect_errors = lambda error: observed.append(error) or [error]

    def fail_driver(operation):
        operation()
        raise RuntimeError("pause failed")

    with pytest.raises(RuntimeError, match="pause failed"):
        core._run_engine_session_collectively(lambda: None, fail_driver)

    assert observed == ["rollout session: RuntimeError: pause failed"]


def test_generator_prepare_is_deferred_until_weight_update_session(
    monkeypatch,
) -> None:
    events = []
    session_open = False
    core = MilesPr3304ProtocolCore(SimpleNamespace())
    core._generator_endpoint = "mx:50051"
    core._session = SimpleNamespace(prepare=lambda: events.append("trainer-prepare"))

    def generator_futures(action, **kwargs):
        assert session_open
        events.append((action, kwargs))
        return ["future"]

    core._generator_futures = generator_futures
    core._wait_futures = lambda futures: events.append(("wait", futures))
    core._collect_errors = lambda error: [error] if error else []
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    # connect_modelexpress establishes these resources while the SGLang
    # begin_weight_update session is still closed.
    assert not core._generators_prepared
    assert events == []

    session_open = True
    core._prepare_modelexpress_sessions()
    core._prepare_modelexpress_sessions()

    assert events == [
        ("prepare", {"endpoint": "mx:50051"}),
        "trainer-prepare",
        ("wait", ["future"]),
    ]
    assert core._generators_prepared


def test_speculative_updates_require_target_only_selector() -> None:
    core = MilesPr3304ProtocolCore(
        SimpleNamespace(sglang_speculative_algorithm="EAGLE")
    )

    with pytest.raises(ValueError, match="target-only"):
        core._validate_selector("all")

    core._validate_selector("target")
