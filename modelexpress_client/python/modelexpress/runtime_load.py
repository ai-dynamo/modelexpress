# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-runtime serving load as a ``source_load`` signal (vLLM / SGLang).

The MX client runs inside the inference worker's pod, so it can scrape the
worker's own Prometheus ``/metrics`` and turn a serving-load proxy into a
``[0, 1]`` ``source_load``. This complements the physical NIC-utilization
provider (`nic_metrics.py`): the runtime signal is *predictive* -- a full KV
cache or a deep prefill queue foretells RDMA the NIC counter has not seen yet --
and it works on fabrics the sysfs counters cannot read.

It reads whichever known load metric is present, so the same provider covers
vLLM and SGLang without configuration beyond the endpoint URL
(``MX_P2P_RUNTIME_METRICS_URL``). Everything is best-effort: an unreachable
endpoint or an unrecognized exposition yields ``0.0``, so the blend falls back
to the NIC signal (and ultimately to ``rendezvous_hash``).
"""

from __future__ import annotations

import logging
import re
import urllib.request
from typing import Callable, Optional

logger = logging.getLogger("modelexpress.runtime_load")

# Metrics already expressed as a [0, 1] utilization ratio, in priority order.
# vLLM: fraction of KV-cache blocks in use (renamed kv_cache_usage_perc in
# v0.17; gpu_cache_usage_perc on older builds). SGLang: token/KV usage fraction.
# dynamo_component_gpu_cache_usage_percent is Dynamo's runtime-agnostic re-export
# of the same GPU KV-cache usage (0.0-1.0); kept last so engine-specific gauges
# still win on vLLM/SGLang and this only resolves a signal where they are absent
# (e.g. TRT-LLM behind a Dynamo runtime).
_RATIO_METRICS = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "sglang:token_usage",
    "sglang:cache_usage",
    "dynamo_component_gpu_cache_usage_percent",
)


def _scrape_gauge(text: str, name: str) -> Optional[float]:
    """Return the max value across series of Prometheus gauge ``name`` (or None).

    Matches ``name`` and ``name{labels...}`` lines; skips ``# HELP``/``# TYPE``.
    Taking the max over label series is the conservative choice (busiest shard).
    """
    pat = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9eE.+-]+)\s*$")
    best: Optional[float] = None
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = pat.match(line.strip())
        if m:
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            best = v if best is None else max(best, v)
    return best


class RuntimeLoadProvider:
    """Provides ``source_load`` in ``[0, 1]`` from a runtime's /metrics endpoint."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 1.0,
        _fetch: Optional[Callable[[str, float], Optional[str]]] = None,
    ) -> None:
        self._url = url
        self._timeout = timeout
        self._fetch = _fetch or _http_get

    def sample(self) -> float:
        text = self._fetch(self._url, self._timeout)
        if not text:
            return 0.0
        for name in _RATIO_METRICS:
            v = _scrape_gauge(text, name)
            if v is not None:
                return min(1.0, max(0.0, v))
        return 0.0


def _http_get(url: str, timeout: float) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", "replace")
    except Exception as e:  # pragma: no cover - network best-effort
        logger.debug("runtime /metrics scrape failed (%s): %s", url, e)
        return None
