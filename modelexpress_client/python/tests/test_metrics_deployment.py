# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment-artifact regression tests for the metrics pipeline.

D1 was not a code defect — the code was fine and the deployment could never
exercise it. ``prometheus.io/port`` was pinned to ``"8001"``, the same value as
``service.port`` and ``MODEL_EXPRESS_SERVER_PORT``. That is the tonic gRPC
listener; tonic speaks HTTP/2 only, and Prometheus issues an HTTP/1.1
``GET /metrics``, so the scrape could never complete and the target reported
``up == 0`` fleet-wide, indistinguishable from a crashed pod.

These are cheap file-level assertions on purpose. The failure mode they guard is
a one-character edit in a values file that no unit test touching Python code
would ever notice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HELM = _REPO_ROOT / "helm"

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


@pytest.mark.parametrize("name", _VALUES_FILES)
def test_metrics_port_differs_from_the_grpc_port(name):
    """The two ports must never converge: one is HTTP/1.1, the other HTTP/2.

    Checked in **every** values file that sets both, not just `values.yaml`.
    `values-production.yaml` overrides `metrics.port`, so a check against the
    defaults alone would pass while production scraped the gRPC listener — which
    is exactly the shape of the original defect.
    """
    values = _read(f"helm/{name}")
    service_port = re.search(r"^service:\n(?:.*\n)*?\s+port:\s*(\d+)", values, re.M)
    metrics_port = re.search(r"^metrics:\n(?:.*\n)*?\s+port:\s*(\d+)", values, re.M)
    if not metrics_port:
        pytest.skip(f"helm/{name} does not set metrics.port; it inherits values.yaml")
    assert service_port, f"helm/{name} sets metrics.port but no service.port: {values}"
    assert service_port.group(1) != metrics_port.group(1), (
        f"helm/{name} points metrics.port at the gRPC port ({service_port.group(1)}). "
        f"tonic serves HTTP/2 only, so every scrape of it fails and the target "
        f"reports up == 0."
    )


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


def test_metrics_port_is_the_single_source_of_truth():
    """Four things must move together, and none may be skippable.

    ``metrics.port`` drives where the server listens (the env var), where
    Prometheus scrapes (the annotation), what the pod advertises (the
    containerPort) and what a Service-based scrape targets. Any change that lets
    one of them move alone reproduces the original issue one layer up — a server
    listening on a port nothing scrapes, with no error anywhere.

    This is a regression guard for a real one: an earlier revision *skipped* the
    generated env var when the user set it directly, leaving the other three on
    ``metrics.port``. An override of ``9500`` then had the server on 9500 while
    Kubernetes advertised 9401. The chart now rejects that override instead.

    Rendering all the combinations is stronger and lives in the reviewer notes;
    this is the part that can run without helm installed.
    """
    deployment = _read("helm/templates/deployment.yaml")
    service = _read("helm/templates/service.yaml")

    # Every consumer derives from the same value.
    for consumer, text, needle in [
        ("scrape annotation", deployment, 'prometheus.io/port" (.Values.metrics.port'),
        ("containerPort", deployment, "containerPort: {{ .Values.metrics.port"),
        ("env var", deployment, "{{ .Values.metrics.port | default 9401 | quote }}"),
        ("service port", service, "port: {{ .Values.metrics.port"),
    ]:
        assert needle in text, f"the {consumer} no longer derives from .Values.metrics.port"

    # And an env-only override is rejected, not silently honoured. Without this
    # the env var moves alone and the other three keep pointing elsewhere.
    assert "fail " in deployment and "MODEL_EXPRESS_SERVER_METRICS_PORT" in deployment, (
        "the chart must reject MODEL_EXPRESS_SERVER_METRICS_PORT set via .Values.env "
        "or .Values.extraEnv, because it moves only where the server listens"
    )
    emission = deployment.index("- name: MODEL_EXPRESS_SERVER_METRICS_PORT")
    guard = deployment.rindex("{{- fail ", 0, emission)
    between = deployment[guard:emission]
    assert "{{- if" not in between.split("{{- end }}")[-1], (
        "the env emission is inside a conditional again; it must be unconditional "
        "so it can never disagree with the annotation and containerPort"
    )
