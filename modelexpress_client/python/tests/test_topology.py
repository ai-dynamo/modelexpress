# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
from modelexpress.engines.sglang.adapter import (
    _get_sglang_topology,
    _get_sglang_worker_rank,
)
from modelexpress.engines.trtllm.adapter import _get_trtllm_topology
from modelexpress.engines.vllm.adapter import _get_vllm_topology
from modelexpress.topology import (
    ParallelTopology,
    build_topology,
    flat_shard_rank,
)


def _install_sglang(monkeypatch, **accessors):
    sglang_mod = ModuleType("sglang")
    srt_mod = ModuleType("sglang.srt")
    distributed_mod = ModuleType("sglang.srt.distributed")
    for name, value in accessors.items():
        setattr(distributed_mod, name, value)
    srt_mod.distributed = distributed_mod
    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed_mod)
    return distributed_mod


def _install_vllm_parallel_state(monkeypatch, **accessors):
    parallel_state = ModuleType("vllm.distributed.parallel_state")
    for name, value in accessors.items():
        setattr(parallel_state, name, value)
    distributed_mod = sys.modules["vllm.distributed"]
    monkeypatch.setattr(distributed_mod, "parallel_state", parallel_state, raising=False)
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.parallel_state", parallel_state
    )
    return parallel_state


@pytest.mark.parametrize(
    "tp_rank,tp_size,pp_rank,pp_size",
    [(0, 1, 0, 1), (1, 4, 0, 1), (0, 4, 2, 3), (3, 4, 2, 3), (1, 2, 1, 2)],
)
def test_flat_shard_rank_matches_legacy_sglang_formula(
    tp_rank, tp_size, pp_rank, pp_size
):
    topology = ParallelTopology(
        tp_rank=tp_rank, tp_size=tp_size, pp_rank=pp_rank, pp_size=pp_size
    )
    assert flat_shard_rank(topology) == pp_rank * tp_size + tp_rank


def test_flat_shard_rank_ignores_data_and_expert_axes():
    base = ParallelTopology(tp_rank=1, tp_size=4, pp_rank=2, pp_size=3)
    widened = ParallelTopology(
        dp_rank=3,
        dp_size=8,
        tp_rank=1,
        tp_size=4,
        pp_rank=2,
        pp_size=3,
        ep_rank=5,
        ep_size=6,
    )
    assert flat_shard_rank(widened) == flat_shard_rank(base) == 9


@pytest.mark.parametrize("unknown", ["tp_rank", "pp_rank"])
def test_flat_shard_rank_rejects_unknown_contributing_rank(unknown):
    axes = {"tp_rank": 1, "tp_size": 4, "pp_rank": 2, "pp_size": 3}
    axes[unknown] = None
    with pytest.raises(ValueError, match=unknown):
        flat_shard_rank(ParallelTopology(**axes))


def test_flat_shard_rank_tolerates_unknown_expert_rank():
    topology = ParallelTopology(
        tp_rank=1, tp_size=4, pp_rank=2, pp_size=3, ep_rank=None, ep_size=8
    )
    assert flat_shard_rank(topology) == 9


@pytest.mark.parametrize("axis", ["dp", "tp", "pp", "ep"])
def test_rank_at_or_above_size_is_rejected(axis):
    with pytest.raises(ValueError, match=f"{axis}_rank"):
        ParallelTopology(**{f"{axis}_rank": 4, f"{axis}_size": 4})


@pytest.mark.parametrize("axis", ["dp", "tp", "pp", "ep"])
def test_negative_rank_is_rejected(axis):
    with pytest.raises(ValueError, match=f"{axis}_rank"):
        ParallelTopology(**{f"{axis}_rank": -1, f"{axis}_size": 4})


@pytest.mark.parametrize("axis", ["dp", "tp", "pp", "ep"])
def test_size_below_one_is_rejected(axis):
    with pytest.raises(ValueError, match=f"{axis}_size"):
        ParallelTopology(**{f"{axis}_rank": 0, f"{axis}_size": 0})


def test_unknown_rank_is_never_promoted_to_zero():
    topology = ParallelTopology(tp_rank=None, tp_size=1)
    assert topology.tp_rank is None


def test_default_topology_is_the_single_worker_case():
    topology = ParallelTopology()
    assert flat_shard_rank(topology) == 0
    assert (topology.dp_size, topology.tp_size) == (1, 1)
    assert (topology.pp_size, topology.ep_size) == (1, 1)


def test_build_topology_drops_only_the_inconsistent_rank():
    topology = build_topology(
        tp_rank=9, tp_size=2, pp_rank=1, pp_size=4, dp_rank=0, dp_size=1
    )
    assert topology.tp_rank is None
    assert topology.tp_size == 2
    assert topology.pp_rank == 1
    assert topology.dp_rank == 0


def test_build_topology_warns_about_every_value_it_drops(caplog):
    with caplog.at_level(logging.WARNING, logger="modelexpress.topology"):
        topology = build_topology(tp_rank=9, tp_size=2, pp_rank=1, pp_size=4)

    assert topology.tp_rank is None
    assert "tp_rank 9" in caplog.text
    assert "tp_size 2" in caplog.text
    # Only the axis that disagreed with itself is reported.
    assert len(caplog.records) == 1
    assert "pp_rank" not in caplog.text


def test_build_topology_passes_a_consistent_placement_through():
    axes = dict(tp_rank=1, tp_size=2, pp_rank=3, pp_size=4)
    assert build_topology(**axes) == ParallelTopology(**axes)


def test_sglang_topology_reads_the_distributed_accessors(monkeypatch):
    _install_sglang(
        monkeypatch,
        get_tensor_model_parallel_rank=lambda: 1,
        get_tensor_model_parallel_world_size=lambda: 4,
        get_pipeline_model_parallel_rank=lambda: 2,
        get_pipeline_model_parallel_world_size=lambda: 3,
        get_moe_expert_parallel_rank=lambda: 5,
        get_moe_expert_parallel_world_size=lambda: 8,
    )

    topology = _get_sglang_topology()

    assert (topology.tp_rank, topology.tp_size) == (1, 4)
    assert (topology.pp_rank, topology.pp_size) == (2, 3)
    assert (topology.ep_rank, topology.ep_size) == (5, 8)


def test_sglang_topology_degrades_without_an_expert_rank_accessor(monkeypatch):
    # Builds that predate get_moe_expert_parallel_rank report the expert world
    # size only. The axis stays unknown there instead of collapsing every
    # worker onto expert rank 0.
    _install_sglang(
        monkeypatch,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_pipeline_model_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_world_size=lambda: 1,
        get_moe_expert_parallel_world_size=lambda: 8,
    )

    topology = _get_sglang_topology()

    assert topology.ep_rank is None
    assert topology.ep_size == 8


def test_sglang_worker_rank_is_the_derivation_of_its_topology(monkeypatch):
    _install_sglang(
        monkeypatch,
        get_tensor_model_parallel_rank=lambda: 1,
        get_tensor_model_parallel_world_size=lambda: 4,
        get_pipeline_model_parallel_rank=lambda: 2,
        get_pipeline_model_parallel_world_size=lambda: 3,
    )

    load_config = SimpleNamespace(tp_rank=99)
    assert _get_sglang_worker_rank(load_config) == 9
    assert _get_sglang_worker_rank(load_config) == flat_shard_rank(
        _get_sglang_topology()
    )


def test_sglang_worker_rank_ignores_a_missing_pipeline_world_size(monkeypatch):
    # pp_size does not enter the formula, so an engine exposing a pipeline
    # rank without a pipeline world size still publishes the same key rather
    # than falling back.
    _install_sglang(
        monkeypatch,
        get_tensor_model_parallel_rank=lambda: 1,
        get_tensor_model_parallel_world_size=lambda: 4,
        get_pipeline_model_parallel_rank=lambda: 2,
    )

    assert _get_sglang_topology().pp_size is None
    assert _get_sglang_worker_rank(SimpleNamespace(tp_rank=99)) == 9


def test_unknown_size_is_accepted():
    topology = ParallelTopology(pp_rank=7, pp_size=None)
    assert (topology.pp_rank, topology.pp_size) == (7, None)


def test_sglang_worker_rank_falls_back_when_accessors_are_unavailable(monkeypatch):
    _install_sglang(monkeypatch)

    assert _get_sglang_worker_rank(SimpleNamespace(tp_rank=7)) == 7


def test_sglang_topology_degrades_when_the_engine_is_not_initialized(monkeypatch):
    _install_sglang(monkeypatch)

    topology = _get_sglang_topology()

    assert topology.tp_rank is None
    assert topology.pp_rank is None
    # An axis nobody could read is unknown, not an axis of extent 1.
    assert (topology.tp_size, topology.pp_size) == (None, None)


def test_vllm_topology_combines_parallel_config_and_live_groups(monkeypatch):
    _install_vllm_parallel_state(
        monkeypatch,
        get_tensor_model_parallel_rank=lambda: 1,
        get_tensor_model_parallel_world_size=lambda: 2,
        get_pp_group=lambda: SimpleNamespace(rank_in_group=1, world_size=2),
        get_ep_group=lambda: SimpleNamespace(rank_in_group=3, world_size=4),
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=7,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
            data_parallel_rank=1,
        )
    )

    topology = _get_vllm_topology(vllm_config)

    assert (topology.dp_rank, topology.dp_size) == (1, 2)
    assert (topology.tp_rank, topology.tp_size) == (1, 2)
    assert (topology.pp_rank, topology.pp_size) == (1, 2)
    assert (topology.ep_rank, topology.ep_size) == (3, 4)


def test_vllm_worker_rank_and_shard_derivation_disagree_under_data_parallel(
    monkeypatch,
):
    # The divergence is deliberate: get_worker_rank() returns the world rank
    # so data-parallel replicas are not paired under expert parallelism.
    _install_vllm_parallel_state(
        monkeypatch,
        get_tensor_model_parallel_rank=lambda: 1,
        get_tensor_model_parallel_world_size=lambda: 2,
        get_pp_group=lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=3,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=2,
            data_parallel_rank=1,
        )
    )

    topology = _get_vllm_topology(vllm_config)

    assert flat_shard_rank(topology) == 1
    assert int(vllm_config.parallel_config.rank) == 3


def test_vllm_topology_degrades_when_groups_are_not_initialized(monkeypatch):
    def _raise():
        raise RuntimeError("distributed environment is not initialized")

    _install_vllm_parallel_state(
        monkeypatch,
        get_tensor_model_parallel_rank=_raise,
        get_tensor_model_parallel_world_size=_raise,
        get_pp_group=_raise,
        get_ep_group=_raise,
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=0, tensor_parallel_size=4, pipeline_parallel_size=2
        )
    )

    topology = _get_vllm_topology(vllm_config)

    assert topology.tp_rank is None
    assert topology.pp_rank is None
    # Sizes survive the outage because ParallelConfig carries them; the
    # axes it does not carry stay unknown.
    assert (topology.tp_size, topology.pp_size) == (4, 2)
    assert (topology.dp_size, topology.ep_size) == (None, None)


def test_vllm_topology_survives_a_config_without_a_parallel_section(monkeypatch):
    def _raise():
        raise RuntimeError("distributed environment is not initialized")

    _install_vllm_parallel_state(
        monkeypatch,
        get_tensor_model_parallel_rank=_raise,
        get_tensor_model_parallel_world_size=_raise,
        get_pp_group=_raise,
        get_ep_group=_raise,
    )

    topology = _get_vllm_topology(SimpleNamespace())

    assert (topology.tp_rank, topology.tp_size) == (None, None)
    assert (topology.pp_rank, topology.pp_size) == (None, None)


def test_trtllm_topology_reads_the_mapping_axes():
    mapping = SimpleNamespace(
        rank=5,
        tp_rank=1,
        tp_size=4,
        pp_rank=1,
        pp_size=2,
        moe_ep_rank=1,
        moe_ep_size=4,
    )

    topology = _get_trtllm_topology(mapping)

    assert (topology.tp_rank, topology.tp_size) == (1, 4)
    assert (topology.pp_rank, topology.pp_size) == (1, 2)
    assert (topology.ep_rank, topology.ep_size) == (1, 4)
    # Mapping.dp_size mirrors tp_size under attention-DP rather than adding an
    # independent axis, so the data axis is reported as absent.
    assert (topology.dp_rank, topology.dp_size) == (0, 1)


def test_trtllm_topology_normalizes_the_no_moe_sentinel():
    mapping = SimpleNamespace(
        rank=1, tp_rank=1, tp_size=2, pp_rank=0, pp_size=1, moe_ep_size=-1
    )

    topology = _get_trtllm_topology(mapping)

    assert (topology.ep_rank, topology.ep_size) == (0, 1)


def test_trtllm_topology_degrades_on_a_mapping_without_axis_fields():
    topology = _get_trtllm_topology(SimpleNamespace(rank=5, local_rank=1))

    assert topology.tp_rank is None
    assert topology.pp_rank is None
    assert (topology.tp_size, topology.pp_size) == (None, None)


def test_trtllm_topology_drops_a_rank_a_property_cannot_compute():
    class _Mapping:
        tp_size = 2
        pp_rank = 0
        pp_size = 1
        moe_ep_size = 4

        @property
        def tp_rank(self):
            raise ZeroDivisionError("group sizes not built yet")

        @property
        def moe_ep_rank(self):
            raise ZeroDivisionError("group sizes not built yet")

    topology = _get_trtllm_topology(_Mapping())

    assert topology.tp_rank is None
    assert topology.ep_rank is None
    assert topology.ep_size == 4
