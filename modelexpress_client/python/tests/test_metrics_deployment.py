# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment-artifact regression tests for the metrics pipeline.

D1 and D5 are not code defects — the code was fine and the deployment could
never exercise it. Both failed the same way: the scrape target reported
``up == 0`` fleet-wide, indistinguishable from a crashed pod, with nothing in
any log to say why.

* **D1** ``prometheus.io/port`` was pinned to ``"8001"``, the same value as
  ``service.port`` and ``MODEL_EXPRESS_SERVER_PORT``. That is the tonic gRPC
  listener. tonic speaks HTTP/2 only; Prometheus issues an HTTP/1.1
  ``GET /metrics``, so the scrape could never complete.
* **D5** ``prometheus-client`` is an optional extra and no container image
  installed it. Every image either used ``pip install .`` (core deps only,
  extras excluded) or ``pip install --no-deps .`` (no dependencies at all), so
  the collector caught the ``ImportError`` and disabled itself.

These are cheap file-level assertions on purpose. The failure mode they guard
is a one-character edit in a values file or a Dockerfile that no unit test
touching Python code would ever notice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HELM = _REPO_ROOT / "helm"

#: Images that layer the ModelExpress Python client onto an engine base image.
#: Each one runs the client in a GPU pod, so each one needs prometheus-client
#: for MX_METRICS_ENABLED=1 to do anything at all.
_CLIENT_IMAGE_DOCKERFILES = [
    "ci/k8s/client/vllm/Dockerfile",
    "ci/k8s/client/vllm/dynamo/Dockerfile",
    "ci/k8s/client/sgl/Dockerfile",
    "ci/k8s/client/sgl/Dockerfile.mooncake",
    "examples/p2p_transfer_k8s/client/vllm/Dockerfile",
    "examples/p2p_transfer_k8s/client/vllm/aws_efa/Dockerfile",
    "examples/p2p_transfer_k8s/client/sglang/Dockerfile",
    "examples/p2p_transfer_k8s/client/trtllm/Dockerfile",
    "examples/model_streamer_k8s/client/sglang/Dockerfile",
    "examples/dynamo_p2p_transfer_k8s/Dockerfile",
]

_VALUES_FILES = [
    "values.yaml",
    "values-production.yaml",
    "values-development.yaml",
    "values-local-storage.yaml",
    "test-values.yaml",
]


def _read(relative: str) -> str:
    path = _REPO_ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} is not present in this checkout")
    return path.read_text()


def _install_lines(content: str) -> list[str]:
    """Lines that actually run an install, with comments removed.

    Matching a bare substring over the whole file is worthless here, because
    every one of these Dockerfiles carries a comment explaining *why* it installs
    prometheus-client. That comment alone satisfied the old assertion, so
    deleting the real install line left both D5 tests green.
    """
    lines = []
    for raw in content.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.search(r"\b(uv\s+)?pip\s+install\b", stripped) or re.search(
            r"python3?\s+-m\s+pip\s+install\b", stripped
        ):
            lines.append(stripped)
    return lines


@pytest.mark.parametrize("dockerfile", _CLIENT_IMAGE_DOCKERFILES)
def test_client_images_install_prometheus_client(dockerfile):
    """D5: the collector needs the dependency to exist in the image.

    Two acceptable spellings: the ``[metrics]`` extra where pip resolves
    dependencies, and an explicit ``prometheus-client`` install where the image
    deliberately uses ``--no-deps`` to protect the engine's CUDA/NIXL/Torch
    stack. Asserted against install lines with comments stripped — the comments
    in these files mention prometheus-client by name, so a whole-file substring
    search passes even with the install deleted.
    """
    installs = _install_lines(_read(dockerfile))
    assert installs, f"{dockerfile} has no recognizable install line"
    satisfied = [
        line
        for line in installs
        if ".[metrics]" in line or "prometheus-client" in line
    ]
    assert satisfied, (
        f"{dockerfile} installs the ModelExpress client but no install line "
        f"brings in prometheus-client. MX_METRICS_ENABLED=1 would silently "
        f"disable itself in this image and the pod would report up == 0.\n"
        f"install lines seen: {installs}"
    )


@pytest.mark.parametrize("dockerfile", _CLIENT_IMAGE_DOCKERFILES)
def test_no_deps_images_install_prometheus_client_explicitly(dockerfile):
    """``--no-deps`` excludes extras, so the extra alone is not enough there."""
    installs = _install_lines(_read(dockerfile))
    # The client install itself: `... --no-deps .` (a bare dot as the target).
    no_deps_client = [
        line
        for line in installs
        if "--no-deps" in line and re.search(r"\s\.\s*(&&|;|\\)?\s*$", line)
    ]
    if not no_deps_client:
        pytest.skip(f"{dockerfile} does not use --no-deps for the client install")
    explicit = [line for line in installs if "prometheus-client" in line]
    assert explicit, (
        f"{dockerfile} installs the client with --no-deps, which excludes the "
        f"[metrics] extra; prometheus-client must be installed by its own "
        f"install line.\ninstall lines seen: {installs}"
    )


def test_helm_scrape_annotation_is_not_hardcoded_to_the_grpc_port():
    """D1: the annotation must not be hand-written in any values file.

    It is generated in ``deployment.yaml`` from ``.Values.metrics.port`` so it
    cannot drift back onto the gRPC listener. A literal ``prometheus.io/port``
    in a values file is how it got pinned to 8001 in the first place.
    """
    for name in _VALUES_FILES:
        content = _read(f"helm/{name}")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "prometheus.io/port" not in stripped, (
                f"helm/{name} hand-writes prometheus.io/port. Set "
                f"metrics.port instead; deployment.yaml generates the "
                f"annotation from it."
            )


def test_metrics_port_default_differs_from_the_grpc_port():
    """The two ports must never converge: one is HTTP/1.1, the other HTTP/2."""
    values = _read("helm/values.yaml")
    service_port = re.search(r"^service:\n(?:.*\n)*?\s+port:\s*(\d+)", values, re.M)
    metrics_port = re.search(r"^metrics:\n(?:.*\n)*?\s+port:\s*(\d+)", values, re.M)
    assert service_port and metrics_port, values
    assert service_port.group(1) != metrics_port.group(1)


def test_deployment_publishes_the_metrics_port_and_env():
    """The listener needs a containerPort and the clap-only env override.

    Keyed on the emission form, not the bare identifier: the chart mentions
    ``MODEL_EXPRESS_SERVER_METRICS_PORT`` three times — in the extraEnv collision
    check, in an explanatory comment, and in the actual emission — so a
    substring test stayed green with the whole emission block deleted.
    """
    deployment = _read("helm/templates/deployment.yaml")
    assert "name: metrics" in deployment
    assert "- name: MODEL_EXPRESS_SERVER_METRICS_PORT" in deployment
    # The generated annotation must reference the metrics port, never
    # service.port.
    assert 'prometheus.io/port" (.Values.metrics.port' in deployment


def test_rust_metrics_port_env_is_wired_through_clap_not_the_config_loader():
    """The config-loader trap, asserted where it would silently regress.

    ``Environment::with_prefix("MODEL_EXPRESS").separator("_")`` maps
    ``MODEL_EXPRESS_SERVER_METRICS_PORT`` to the key path ``server.metrics.port``,
    which matches no field; serde drops it without a warning. Only the clap
    ``#[arg(env = ...)]`` override actually reaches the field.
    """
    config_rs = _read("modelexpress_server/src/config.rs")
    assert "pub metrics_port: Option<u16>" in config_rs
    # The attribute form, not the bare identifier: the name also appears in the
    # comment warning against removing this very attribute, so a substring test
    # passed even with `#[arg(long, env = ...)]` reduced to `#[arg(long)]`.
    assert (
        "env = modelexpress_common::envs::MODEL_EXPRESS_SERVER_METRICS_PORT" in config_rs
    )
    # The serde default is mandatory: without it, every existing
    # model-express.yaml fails to parse and load_layered_config silently falls
    # back to a full default configuration.
    assert 'default = "default_metrics_port"' in config_rs
    # And the stored type must stay a plain u16. NonZeroU16 rejects the
    # documented `0` during deserialization, which under load_layered_config's
    # error-swallowing fallback means the entire config file is discarded —
    # gRPC port, cache directory, eviction policy and auth settings included.
    assert "pub metrics_port: u16," in config_rs
