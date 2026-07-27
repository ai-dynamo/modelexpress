# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import pytest


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (>= 1), got {ivalue}")
    return ivalue


def _nonnegative_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {ivalue}")
    return ivalue


def _nonnegative_float(value: str) -> float:
    fvalue = float(value)
    if fvalue < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative number, got {fvalue}")
    return fvalue


def _port(value: str) -> int:
    ivalue = int(value)
    if not 1 <= ivalue <= 65535:
        raise argparse.ArgumentTypeError(f"must be a valid TCP port (1-65535), got {ivalue}")
    return ivalue


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--namespace", default=os.environ.get("NAMESPACE", ""))
    parser.addoption("--model", default=os.environ.get("MX_CI_MODEL", "Qwen/Qwen2.5-0.5B"))
    parser.addoption("--source-port", default=_port(os.environ.get("SOURCE_PORT", "8001")), type=_port)
    parser.addoption("--worker-port", default=_port(os.environ.get("WORKER_PORT", "8002")), type=_port)
    parser.addoption(
        "--p2p-marker",
        default=os.environ.get("P2P_MARKER", "RDMA transfer complete"),
        help="String that must appear in target pod logs to confirm P2P transfer ran.",
    )
    parser.addoption(
        "--tp-size",
        default=_positive_int(os.environ.get("TP_SIZE", "1")),
        type=_positive_int,
        help="Tensor-parallel size — the per-rank transfer test expects this many distinct ranks.",
    )
    parser.addoption(
        "--dp-size",
        default=_positive_int(os.environ.get("DP_SIZE", "1")),
        type=_positive_int,
        help=(
            "Data-parallel size — DP replicas are interchangeable full copies "
            "at the same worker_rank, so the per-rank transfer test accepts a "
            "distinct-agent count anywhere in [tp_size, tp_size * dp_size] "
            "instead of exactly tp_size. Defaults to 1 (pure TP / single copy)."
        ),
    )
    parser.addoption(
        "--transport",
        default=os.environ.get("TRANSPORT", "nixl"),
        help=(
            "P2P transport under test. 'nixl' (default) enables the per-rank "
            "NIXL agent assertion (add_remote_agent log lines). For "
            "'transfer_engine' (Mooncake) that assertion is skipped — the "
            "Mooncake path does not register NIXL agents."
        ),
    )
    parser.addoption(
        "--require-artifact-transfer",
        action="store_true",
        default=os.environ.get("REQUIRE_ARTIFACT_TRANSFER", "").lower() in {"1", "true", "yes"},
        help="Assert that an artifact source was published and installed by the target.",
    )
    parser.addoption(
        "--expected-artifact-sources",
        default=_nonnegative_int(os.environ.get("EXPECTED_ARTIFACT_SOURCES", "0")),
        type=_nonnegative_int,
        help="Minimum ready non-weight artifact ModelMetadata CRs expected at pytest time.",
    )
    parser.addoption(
        "--expected-artifact-source-types",
        default=os.environ.get("EXPECTED_ARTIFACT_SOURCE_TYPES", ""),
        help=(
            "Comma-separated artifact sourceType values expected to be ready "
            "and installed, for example 'torch_compile_cache,triton_cache'."
        ),
    )
    parser.addoption(
        "--dgd-name",
        default=os.environ.get("DGD_NAME", "mx-dynamo-vllm"),
        help=(
            "metadata.name of the DGD under test. Drives the Frontend Service name "
            "(<dgd-name>-frontend) and the operator's auto-applied "
            "nvidia.com/dynamo-graph-deployment-name label that the worker pod "
            "selector uses. Must match what the action / manifest applied."
        ),
    )
    parser.addoption(
        "--expected-cr-count",
        default=_positive_int(os.environ.get("EXPECTED_CR_COUNT", "2")),
        type=_positive_int,
        help=(
            "Final ModelMetadata CR count expected at pytest time. "
            "Aggregated: 2 (2 VllmWorker replicas after scale-up). "
            "Disaggregated: 3 (1 prefill + 2 decode replicas after scale-up)."
        ),
    )
    parser.addoption(
        "--heartbeat-timeout-secs",
        default=_positive_int(os.environ.get("HEARTBEAT_TIMEOUT_SECS", "30")),
        type=_positive_int,
        help=(
            "Server-side reaper heartbeat threshold (MX_HEARTBEAT_TIMEOUT_SECS). "
            "Must match what the deployed mx-server is configured with — the "
            "run-mx-stale-metadata-test action threads the same value to both "
            "the manifest and pytest to keep them aligned."
        ),
    )
    parser.addoption(
        "--reaper-scan-interval-secs",
        default=_positive_int(os.environ.get("REAPER_SCAN_INTERVAL_SECS", "10")),
        type=_positive_int,
        help=(
            "Server-side reaper scan interval (MX_REAPER_SCAN_INTERVAL_SECS). "
            "Same alignment requirement as --heartbeat-timeout-secs."
        ),
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


@pytest.fixture(scope="session")
def dp_size(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--dp-size")


@pytest.fixture(scope="session")
def transport(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--transport")


@pytest.fixture(scope="session")
def require_artifact_transfer(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--require-artifact-transfer")


@pytest.fixture(scope="session")
def expected_artifact_sources(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--expected-artifact-sources")


@pytest.fixture(scope="session")
def expected_artifact_source_types(request: pytest.FixtureRequest) -> set[str]:
    raw = request.config.getoption("--expected-artifact-source-types")
    return {item.strip() for item in raw.split(",") if item.strip()}


@pytest.fixture(scope="session")
def vram_measurements() -> dict:
    path = os.environ.get("MX_VRAM_MEASUREMENTS_PATH")
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        measurements = json.load(f)
    assert isinstance(measurements, dict), f"Invalid VRAM measurements in {path}: {measurements!r}"
    return measurements


def _vram_mib(measurements: dict, side: str, metric: str) -> int | None:
    value = measurements.get(side, {}).get(metric)
    if value is None:
        return None
    return _nonnegative_int(str(value))


@pytest.fixture(scope="session")
def source_peak_vram_mib(vram_measurements: dict) -> int | None:
    return _vram_mib(vram_measurements, "source", "peak_mib")


@pytest.fixture(scope="session")
def target_peak_vram_mib(vram_measurements: dict) -> int | None:
    return _vram_mib(vram_measurements, "target", "peak_mib")


@pytest.fixture(scope="session")
def source_final_vram_mib(vram_measurements: dict) -> int | None:
    return _vram_mib(vram_measurements, "source", "final_mib")


@pytest.fixture(scope="session")
def target_final_vram_mib(vram_measurements: dict) -> int | None:
    return _vram_mib(vram_measurements, "target", "final_mib")


@pytest.fixture(scope="session")
def vram_tolerance_percent(vram_measurements: dict) -> float:
    return _nonnegative_float(str(vram_measurements.get("tolerance_percent", 0.5)))


@pytest.fixture(scope="session")
def dgd_name(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--dgd-name")


@pytest.fixture(scope="session")
def expected_cr_count(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--expected-cr-count")


@pytest.fixture(scope="session")
def heartbeat_timeout_secs(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--heartbeat-timeout-secs")


@pytest.fixture(scope="session")
def reaper_scan_interval_secs(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--reaper-scan-interval-secs")
