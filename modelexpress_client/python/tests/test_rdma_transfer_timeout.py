# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-source RDMA receive budget must be configurable, without regressing.

The budget was hard-coded at 300s on this path while MX_TRANSFER_TIMEOUT already
existed for other transfer paths. When a source is wedged the transfer yields
neither a completion nor an error status, so the timeout is the only thing that ends
the wait, and with MAX_SOURCE_RETRIES candidates a target can burn three full
budgets before falling back to disk.

The subtlety these tests pin: MX_TRANSFER_TIMEOUT *defaults to 900*. Wiring it
naively would have tripled this path's budget for everyone who never set it, making
the reported stall three times worse. So it is honoured only when explicitly set.

Run: pytest tests/test_rdma_transfer_timeout.py
"""

import pytest

from modelexpress.load_strategy import rdma_strategy
from modelexpress.load_strategy.rdma_strategy import (
    DEFAULT_RDMA_TRANSFER_TIMEOUT_S,
    MAX_SOURCE_RETRIES,
    _transfer_timeout_seconds,
)


def test_unset_preserves_the_previous_hard_coded_budget(monkeypatch):
    """Unset must not change behaviour for anyone relying on the old constant."""
    monkeypatch.delenv("MX_TRANSFER_TIMEOUT", raising=False)
    assert DEFAULT_RDMA_TRANSFER_TIMEOUT_S == 300.0
    assert _transfer_timeout_seconds() == 300.0


def test_the_900_default_of_the_env_var_is_not_inherited(monkeypatch):
    """The regression this fix must not introduce.

    ``envs.MX_TRANSFER_TIMEOUT`` reports 900 when unset. If this path adopted that,
    a wedged source would stall a target for 900s instead of 300s, and up to 45
    minutes across three candidates.
    """
    monkeypatch.delenv("MX_TRANSFER_TIMEOUT", raising=False)
    assert rdma_strategy.envs.MX_TRANSFER_TIMEOUT == 900
    assert _transfer_timeout_seconds() == 300.0


def test_an_operator_can_shorten_the_budget(monkeypatch):
    """The point of the fix: a 1.2 GB model should not cost 300s to give up on."""
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", "30")
    assert _transfer_timeout_seconds() == 30.0


def test_an_operator_can_lengthen_the_budget(monkeypatch):
    """Large models legitimately need longer than the old constant."""
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", "1800")
    assert _transfer_timeout_seconds() == 1800.0


def test_an_explicit_900_is_honoured(monkeypatch):
    """Declining the default is not the same as forbidding the value."""
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", "900")
    assert _transfer_timeout_seconds() == 900.0


def test_zero_falls_back_rather_than_meaning_unbounded(monkeypatch):
    """receive_from_source treats None as wait-forever, which is the failure mode
    being bounded; a misconfigured 0 must not resurrect it."""
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", "0")
    assert _transfer_timeout_seconds() == 300.0


def test_a_negative_value_falls_back(monkeypatch):
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", "-1")
    assert _transfer_timeout_seconds() == 300.0


def test_a_non_numeric_value_falls_back(monkeypatch):
    """A bad env var should degrade to the documented default, not crash a load."""
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", "not-a-number")
    assert _transfer_timeout_seconds() == 300.0


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_a_non_finite_value_falls_back(monkeypatch, value):
    """A non-finite budget would make the transfer wait loop unbounded."""
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", value)
    assert _transfer_timeout_seconds() == 300.0


def test_a_fractional_value_is_accepted(monkeypatch):
    """The budget is a float; sub-second values are legitimate in tests and rigs."""
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", "0.5")
    assert _transfer_timeout_seconds() == 0.5


def test_the_budget_is_per_candidate_not_per_load(monkeypatch):
    """Documents the multiplier that made the reported 300s so expensive."""
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", "60")
    assert _transfer_timeout_seconds() * MAX_SOURCE_RETRIES == 180.0
