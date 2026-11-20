# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import grpc
import pytest

from modelexpress_rl import RefitSourceUnavailable
from modelexpress_rl.inference.session import WeightUpdateSession

VERSION = SimpleNamespace(version_id="version-a")


def _session(plannable: bool) -> WeightUpdateSession:
    session = WeightUpdateSession.__new__(WeightUpdateSession)
    session._max_transfer_attempts = 3
    session._plan = lambda version, source: object() if plannable else None
    return session


@pytest.mark.parametrize(
    ("candidates", "plannable", "transient"),
    [
        ([], True, True),  # nothing serves the version yet
        ([SimpleNamespace(kind="peer")], False, False),  # a source exists but no method can install it
    ],
)
def test_only_an_absent_source_is_reported_as_unavailable(candidates, plannable, transient):
    resolver = SimpleNamespace(supports=lambda version: True, candidates=lambda version: candidates, kind="peer")

    with pytest.raises(RuntimeError, match="no usable refit source") as raised:
        _session(plannable)._prepare_candidates(VERSION, resolver=resolver)

    assert isinstance(raised.value, RefitSourceUnavailable) is transient


class _RefusedSource(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode) -> None:
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


@pytest.mark.parametrize(
    ("code", "transient"),
    [
        (grpc.StatusCode.RESOURCE_EXHAUSTED, True),  # every source is at its reader cap
        (grpc.StatusCode.UNAVAILABLE, True),  # the source is draining
        (grpc.StatusCode.NOT_FOUND, True),  # the version was retired
        (grpc.StatusCode.INTERNAL, False),
    ],
)
def test_only_a_refusing_source_is_reported_as_unavailable(code, transient):
    session = WeightUpdateSession.__new__(WeightUpdateSession)
    session._resolvers = (SimpleNamespace(kind="peer"),)
    error = _RefusedSource(code)

    def refuse(version, *, resolver):
        raise error

    session._stage_source = refuse

    with pytest.raises((RefitSourceUnavailable, _RefusedSource)) as raised:
        session.stage(VERSION)

    assert (isinstance(raised.value, RefitSourceUnavailable), raised.value.__cause__ is error) == (
        transient,
        transient,
    )
