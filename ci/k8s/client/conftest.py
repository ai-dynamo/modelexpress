# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--namespace", default=os.environ.get("NAMESPACE", ""))
    parser.addoption("--model", default=os.environ.get("MX_CI_MODEL", "Qwen/Qwen2.5-0.5B"))
    parser.addoption("--source-port", default=int(os.environ.get("SOURCE_PORT", "8001")), type=int)
    parser.addoption("--worker-port", default=int(os.environ.get("WORKER_PORT", "8002")), type=int)
    parser.addoption(
        "--p2p-marker",
        default=os.environ.get("P2P_MARKER", "RDMA transfer complete"),
        help="String that must appear in target pod logs to confirm P2P transfer ran.",
    )
    parser.addoption(
        "--tp-size",
        default=int(os.environ.get("TP_SIZE", "1")),
        type=int,
        help="Tensor-parallel size — the per-rank transfer test expects this many distinct ranks.",
    )


@pytest.fixture(scope="session")
def namespace(request: pytest.FixtureRequest) -> str:
    ns = request.config.getoption("--namespace")
    assert ns, "Pass --namespace or set NAMESPACE env var"
    return ns


@pytest.fixture(scope="session")
def model(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def source_port(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--source-port")


@pytest.fixture(scope="session")
def worker_port(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--worker-port")


@pytest.fixture(scope="session")
def p2p_marker(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--p2p-marker")


@pytest.fixture(scope="session")
def tp_size(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--tp-size")
