# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the refit coverage record and the optional coverage gate.

Why this file exists: on 2026-07-27 a Topology B benchmark row was published as
beating Topology A on every axis while refitting 51% of the model. Two pipeline
stages published pipeline-LOCAL layer names, they collided in the name-keyed shard
table, first-writer-wins kept one stage's 24 layers and the other 24 were never
requested. Every existing check passed:

  * the digest gate compared bytes that arrived against the publisher's digest for
    the same name, and every byte that arrived was correct;
  * `params_installed` counted what the plan covered, not what the engine holds;
  * `useful_bytes_per_rank` was *derived* analysis-side from an assumed sharding,
    so a wire volume that was half of what it should be looked like a win.

Bytes that are never requested are never checked. The coverage record is the only
thing that can see that, because it is the only thing that compares against the
engine's own parameter footprint. These tests pin that behaviour.
"""

from __future__ import annotations

import json
import types

import pytest
import torch

from modelexpress.refit.reshard import receiver as receiver_mod
from modelexpress.refit.reshard.receiver import ReshardReceiver
from modelexpress.refit.reshard.types import IncompleteRefit, UnsupportedReshard


class _Plan:
    """Only the accessors `_log_coverage` reads."""

    def __init__(
        self,
        *,
        planned=1_000,
        extra=100,
        descriptors=10,
        savings=5,
        full_pulls=(),
        unbounded=(),
        converts=(),
        fallback=(),
    ):
        self._planned = planned
        self._extra = extra
        self._descriptors = descriptors
        self._savings = savings
        self.full_pulls = list(full_pulls)
        self.unbounded_sources = list(unbounded)
        self.converts = list(converts)
        self.fallback = list(fallback)

    def bytes_planned(self):
        return self._planned

    def extra_wire_bytes(self):
        return self._extra

    def descriptor_count(self):
        return self._descriptors

    def descriptor_savings(self):
        return self._savings


def _capture(copies=3, unsupported=()):
    return types.SimpleNamespace(
        copies=list(range(copies)), unsupported=list(unsupported)
    )


def _layout(names, shape=(4, 8), dtype=torch.bfloat16):
    return {n: (shape, dtype) for n in names}


def _emit(caplog, *, param_layout, all_params, capture=None, plan=None, rank=0):
    """Call the method unbound against a stub self, and return the parsed record.

    `_log_coverage` touches only `self._global_rank`, so constructing a real
    receiver (MxClient, NIXL agent, CUDA pool) would add setup that cannot run in
    a unit test and would test none of this logic.
    """
    stub = types.SimpleNamespace(_global_rank=rank)
    caplog.clear()
    # INFO, because a clean refit reports at INFO and only an incomplete one is
    # escalated; capturing at WARNING here would see the record for some inputs
    # and not others.
    with caplog.at_level("INFO"):
        ReshardReceiver._log_coverage(
            stub,
            capture or _capture(),
            param_layout,
            all_params,
            plan or _Plan(),
        )
    for rec in caplog.records:
        if "MX_REFIT_COVERAGE" in rec.getMessage():
            return json.loads(rec.getMessage().split("MX_REFIT_COVERAGE ", 1)[1])
    raise AssertionError("no MX_REFIT_COVERAGE record was emitted")


@pytest.fixture(autouse=True)
def _gate_off(monkeypatch):
    """Default the gate off, as it ships, whatever the ambient environment is."""
    monkeypatch.delenv("MX_RESHARD_REQUIRE_FULL_COVERAGE", raising=False)
    monkeypatch.delenv("MX_RESHARD_COVERAGE_FLOOR", raising=False)


def test_a_complete_refit_reports_full_coverage(caplog):
    names = [f"layers.{i}.weight" for i in range(4)]
    rec = _emit(caplog, param_layout=_layout(names), all_params=names)
    assert rec["coverage_pct"] == 100.0
    assert rec["params_installed"] == 4
    assert rec["engine_params"] == 4
    assert rec["params_never_written"] == 0
    assert rec["params_never_written_sample"] == []
    assert rec["dest_bytes"] == rec["engine_bytes"]


def test_the_bug_8_shape_half_the_model_never_written(caplog):
    """The regression this record was built to catch.

    48 engine layers, only the first 24 requested, which is what a PP2 name
    collision produces. Coverage must read ~50%, not 100%.
    """
    engine = [f"layers.{i}.weight" for i in range(48)]
    installed = [f"layers.{i}.weight" for i in range(24)]
    rec = _emit(caplog, param_layout=_layout(engine), all_params=installed)
    assert rec["coverage_pct"] == 50.0
    assert rec["params_installed"] == 24
    assert rec["engine_params"] == 48
    assert rec["params_never_written"] == 24
    # The sample must name the missing layers, which is what turns "the number is
    # wrong" into "layers 24-47 were never refit".
    assert rec["params_never_written_sample"][0].startswith("layers.2")
    assert len(rec["params_never_written_sample"]) == 10


def test_coverage_is_measured_in_bytes_not_parameter_count(caplog):
    """Half the params can be far from half the bytes.

    The pre-existing `params_installed` count could not have caught Bug 8's byte
    shortfall even in principle, because a refit that covers most parameters can
    still miss most bytes. Coverage is a byte ratio.
    """
    param_layout = {
        "big": ((1024, 1024), torch.bfloat16),  # 2 MiB
        "small": ((1, 1), torch.bfloat16),  # 2 B
    }
    rec = _emit(caplog, param_layout=param_layout, all_params=["small"])
    assert rec["params_installed"] == 1
    assert rec["engine_params"] == 2
    # 1 of 2 params, but essentially none of the bytes.
    assert rec["coverage_pct"] < 0.001
    assert rec["dest_bytes"] == 2


def test_byte_accounting_respects_dtype_width(caplog):
    """fp8 is one byte per element, bf16 two, fp32 four."""
    for dtype, width in (
        (torch.float8_e4m3fn, 1),
        (torch.bfloat16, 2),
        (torch.float32, 4),
    ):
        names = ["w"]
        rec = _emit(
            caplog,
            param_layout=_layout(names, shape=(4, 8), dtype=dtype),
            all_params=names,
        )
        assert rec["engine_bytes"] == 32 * width, dtype
        assert rec["dest_bytes"] == 32 * width, dtype


def test_unsupported_params_are_surfaced_and_sampled(caplog):
    """A param the planner cannot serve is silently absent from the wire.

    The correctness gate cannot see it, so a non-zero count here is the signal.
    """
    names = ["a", "b"]
    cap = _capture(copies=2, unsupported=[f"cannot reshard {i}" for i in range(15)])
    rec = _emit(caplog, param_layout=_layout(names), all_params=names, capture=cap)
    assert rec["unsupported"] == 15
    assert len(rec["unsupported_sample"]) == 10
    assert all(len(s) <= 120 for s in rec["unsupported_sample"])


def test_plan_economics_ride_along_on_the_same_record(caplog):
    """One record carries coverage and byte economics together.

    They have to be read together: 40 GB of wire is efficient against 30 GB of
    useful bytes and catastrophic against 15 GB.
    """
    names = ["a"]
    plan = _Plan(
        planned=40_000,
        extra=10_000,
        descriptors=19_011,
        savings=12_675_024,
        full_pulls=range(6_192),
        unbounded=(),
        converts=(),
        fallback=(),
    )
    rec = _emit(caplog, param_layout=_layout(names), all_params=names, plan=plan)
    assert rec["planned_wire_bytes"] == 40_000
    assert rec["extra_wire_bytes"] == 10_000
    assert rec["descriptors"] == 19_011
    assert rec["descriptor_savings"] == 12_675_024
    assert rec["full_pull_sources"] == 6_192
    assert rec["unbounded_sources"] == 0
    assert rec["converts"] == 0
    assert rec["fallback"] == 0


def _levels_of(caplog):
    return [
        r.levelname for r in caplog.records if "MX_REFIT_COVERAGE" in r.getMessage()
    ]


def test_the_record_is_machine_readable(caplog):
    """One record per refit, parseable, carrying the schema and the rank.

    The economics were computed and logged before this record existed, but as
    prose, which is why useful bytes were derived rather than measured.
    """
    names = ["a"]
    stub = types.SimpleNamespace(_global_rank=7)
    caplog.clear()
    with caplog.at_level("INFO"):
        ReshardReceiver._log_coverage(stub, _capture(), _layout(names), names, _Plan())
    hits = [r for r in caplog.records if "MX_REFIT_COVERAGE" in r.getMessage()]
    assert len(hits) == 1
    rec = json.loads(hits[0].getMessage().split("MX_REFIT_COVERAGE ", 1)[1])
    assert rec["schema"] == "refit-coverage-v1"
    assert rec["rank"] == 7


def test_a_clean_refit_reports_at_info(caplog):
    """It happens on every refit of a healthy run. At WARNING it would teach
    operators that MX warnings are routine."""
    names = ["a", "b"]
    _emit(caplog, param_layout=_layout(names), all_params=names)

    assert _levels_of(caplog) == ["INFO"]


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"all_params": ["a"]}, id="a param was never written"),
        pytest.param(
            {"all_params": ["a", "b"], "capture": _capture(unsupported=["a: op"])},
            id="a source was unsupported",
        ),
        pytest.param(
            {"all_params": ["a", "b"], "plan": _Plan(fallback=["a"])},
            id="a source fell back",
        ),
    ],
)
def test_a_refit_with_something_to_look_at_is_escalated(caplog, kwargs):
    """The severity tracks the content, so a WARNING here always means something."""
    layout = _layout(["a", "b"])
    _emit(caplog, param_layout=layout, **kwargs)

    assert _levels_of(caplog) == ["WARNING"]


def test_gate_off_by_default_lets_a_partial_refit_through(caplog):
    """Partial and subset refit are intended features."""
    engine = [f"w{i}" for i in range(10)]
    rec = _emit(caplog, param_layout=_layout(engine), all_params=engine[:1])
    assert rec["coverage_pct"] == 10.0  # reported, not raised


def test_gate_on_raises_below_the_floor_and_names_what_is_missing(caplog, monkeypatch):
    monkeypatch.setenv("MX_RESHARD_REQUIRE_FULL_COVERAGE", "1")
    engine = [f"layers.{i}.weight" for i in range(48)]
    with pytest.raises(IncompleteRefit) as ei:
        _emit(
            caplog,
            param_layout=_layout(engine),
            all_params=engine[:24],
        )
    msg = str(ei.value)
    assert "50.00%" in msg
    assert "24 of 48" in msg
    # The message must say why no other check would have caught it, because the
    # first instinct on seeing it is to trust the passing digest gate.
    assert "never requested are never checked" in msg
    assert "MX_RESHARD_REQUIRE_FULL_COVERAGE=0" in msg


def test_an_incomplete_refit_is_not_an_unsupported_reshard(caplog, monkeypatch):
    """`transfer_plan` catches `UnsupportedReshard` per source to drop that source
    and carry on. A refit that skipped part of the model must not be reachable by
    a handler written for that meaning, while still being a `RuntimeError` for
    callers that catch only the builtin."""
    monkeypatch.setenv("MX_RESHARD_REQUIRE_FULL_COVERAGE", "1")
    engine = [f"w{i}" for i in range(10)]

    with pytest.raises(IncompleteRefit) as ei:
        _emit(caplog, param_layout=_layout(engine), all_params=engine[:1])

    assert not isinstance(ei.value, UnsupportedReshard)
    assert isinstance(ei.value, RuntimeError)


def test_gate_on_passes_at_the_floor_so_stray_buffers_do_not_fail_a_good_run(
    caplog, monkeypatch
):
    """Not 1.0 on purpose.

    A few engine params are legitimately not refit material - rotary `inv_freq`
    and similar non-float buffers that surface as params in some models - and
    failing a complete refit over a few kilobytes would make the gate unusable.
    """
    monkeypatch.setenv("MX_RESHARD_REQUIRE_FULL_COVERAGE", "1")
    monkeypatch.setenv("MX_RESHARD_COVERAGE_FLOOR", "0.995")
    param_layout = {f"w{i}": ((1000,), torch.bfloat16) for i in range(1000)}
    param_layout["inv_freq"] = ((1,), torch.bfloat16)  # 2 B of 2,000,002
    installed = [f"w{i}" for i in range(1000)]
    rec = _emit(caplog, param_layout=param_layout, all_params=installed)
    assert rec["coverage_pct"] > 99.99
    assert rec["params_never_written"] == 1


def test_an_engine_with_no_parameters_does_not_divide_by_zero(caplog):
    rec = _emit(caplog, param_layout={}, all_params=[])
    assert rec["coverage_pct"] == 0.0
    assert rec["engine_bytes"] == 0


@pytest.mark.parametrize("floor", ["-0.1", "1.5"])
def test_a_floor_outside_zero_to_one_is_rejected(caplog, monkeypatch, floor):
    """A negative floor passes every refit, silently disabling the gate the caller
    just asked for; above 1.0 rejects even a complete one. Neither is honored."""
    monkeypatch.setenv("MX_RESHARD_REQUIRE_FULL_COVERAGE", "1")
    monkeypatch.setenv("MX_RESHARD_COVERAGE_FLOOR", floor)
    names = ["a", "b"]

    with pytest.raises(ValueError, match=r"fraction in \[0.0, 1.0\]"):
        _emit(caplog, param_layout=_layout(names), all_params=names[:1])


def test_the_floor_is_read_live_so_the_gate_is_configurable_per_run(
    caplog, monkeypatch
):
    monkeypatch.setenv("MX_RESHARD_REQUIRE_FULL_COVERAGE", "1")
    monkeypatch.setenv("MX_RESHARD_COVERAGE_FLOOR", "0.4")
    engine = [f"w{i}" for i in range(10)]

    # 50% coverage clears a 0.4 floor and fails a 0.6 one.
    assert _emit(caplog, param_layout=_layout(engine), all_params=engine[:5])
    monkeypatch.setenv("MX_RESHARD_COVERAGE_FLOOR", "0.6")
    with pytest.raises(RuntimeError):
        _emit(caplog, param_layout=_layout(engine), all_params=engine[:5])


def test_coverage_is_recorded_before_an_unsupported_plan_is_rejected(
    caplog, monkeypatch
):
    """An unsupported plan is the case where the hole most needs naming.

    `_prepare` fails closed on `plan.fallback`, so recording coverage after that
    raise would mean the runs with a known coverage hole are exactly the runs with
    no coverage record.
    """
    engine = [f"layers.{i}.weight" for i in range(4)]
    source = types.SimpleNamespace(dtype="torch.bfloat16", global_shape=(4, 8))
    monkeypatch.setattr(
        receiver_mod,
        "gather_sources",
        lambda *_args, **_kwargs: ({"layers.0.weight": source}, {}, {}, {}),
    )
    monkeypatch.setattr(
        receiver_mod,
        "plan_transfer",
        lambda *_args, **_kwargs: _Plan(fallback=["layers.3.weight"]),
    )
    stub = types.SimpleNamespace(
        _global_rank=0,
        _num_trainer_sources=1,
        _mx_client=object(),
        _model_name="model",
        _manager=types.SimpleNamespace(fetch_remote_and_wait=lambda *a, **k: None),
        _log_coverage=lambda *args: ReshardReceiver._log_coverage(stub, *args),
        _capture=lambda _manifest: (
            types.SimpleNamespace(
                copies=[types.SimpleNamespace(param_name="layers.0.weight")],
                unsupported=[],
            ),
            _layout(engine),
        ),
    )

    caplog.clear()
    with caplog.at_level("WARNING"):
        with pytest.raises(UnsupportedReshard):
            ReshardReceiver._prepare(stub, timeout=1.0)

    records = [r for r in caplog.records if "MX_REFIT_COVERAGE" in r.getMessage()]
    assert len(records) == 1
    rec = json.loads(records[0].getMessage().split("MX_REFIT_COVERAGE ", 1)[1])
    assert rec["fallback"] == 1
    assert rec["params_never_written"] == 3
