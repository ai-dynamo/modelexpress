# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vLLM/SGLang runtime source-load provider."""

from __future__ import annotations

from modelexpress.runtime_load import RuntimeLoadProvider, _scrape_gauge

_VLLM = """
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="Qwen"} 0.62
vllm:num_requests_running{model_name="Qwen"} 12.0
"""

_VLLM_017 = """
# vLLM >= 0.17 renamed the KV-cache gauge.
vllm:kv_cache_usage_perc{model="Qwen"} 0.73
"""

_SGLANG = """
sglang:token_usage 0.41
"""

_MULTI = """
vllm:gpu_cache_usage_perc{gpu="0"} 0.3
vllm:gpu_cache_usage_perc{gpu="1"} 0.8
"""


def test_scrape_gauge_with_labels():
    assert _scrape_gauge(_VLLM, "vllm:gpu_cache_usage_perc") == 0.62


def test_scrape_gauge_takes_max_across_series():
    assert _scrape_gauge(_MULTI, "vllm:gpu_cache_usage_perc") == 0.8


def test_scrape_gauge_missing_returns_none():
    assert _scrape_gauge(_VLLM, "does_not_exist") is None


def test_scrape_gauge_skips_comments():
    assert _scrape_gauge("# TYPE x gauge\n# HELP x foo", "x") is None


def test_provider_reads_vllm_kv_usage():
    p = RuntimeLoadProvider("http://x/metrics", _fetch=lambda u, t: _VLLM)
    assert abs(p.sample() - 0.62) < 1e-9


def test_provider_reads_vllm_017_kv_cache_usage():
    # vLLM >= 0.17 exposes vllm:kv_cache_usage_perc instead of gpu_cache_usage_perc.
    p = RuntimeLoadProvider("http://x/metrics", _fetch=lambda u, t: _VLLM_017)
    assert abs(p.sample() - 0.73) < 1e-9


def test_provider_reads_sglang_when_vllm_absent():
    p = RuntimeLoadProvider("http://x/metrics", _fetch=lambda u, t: _SGLANG)
    assert abs(p.sample() - 0.41) < 1e-9


def test_provider_unreachable_returns_zero():
    p = RuntimeLoadProvider("http://x/metrics", _fetch=lambda u, t: None)
    assert p.sample() == 0.0


def test_provider_no_known_metric_returns_zero():
    p = RuntimeLoadProvider("http://x/metrics", _fetch=lambda u, t: "other_metric 5\n")
    assert p.sample() == 0.0


def test_provider_clamps():
    p = RuntimeLoadProvider(
        "http://x/metrics", _fetch=lambda u, t: "vllm:gpu_cache_usage_perc 1.5\n"
    )
    assert p.sample() == 1.0


def test_make_source_load_provider_blends_when_url_set(monkeypatch):
    # With the runtime URL set, the provider is the max of NIC and runtime; both
    # degrade to 0 (no IB device, runtime fetch returns nothing), so it must not
    # crash and must return a valid [0,1] float. The runtime fetch seam is patched
    # so the test does no real socket I/O.
    from modelexpress import runtime_load
    from modelexpress.nic_metrics import make_source_load_provider

    monkeypatch.setenv("MX_P2P_RUNTIME_METRICS_URL", "http://runtime.invalid/metrics")
    monkeypatch.setattr(runtime_load, "_http_get", lambda url, timeout: None)
    provider = make_source_load_provider(0)
    val = provider()
    assert isinstance(val, float) and 0.0 <= val <= 1.0
