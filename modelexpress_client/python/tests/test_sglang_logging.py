# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import logging

import modelexpress


def test_sglang_subprocess_logs_include_mx_rdma_evidence(monkeypatch):
    engine = logging.getLogger("sglang")
    mx = logging.getLogger("modelexpress")
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    monkeypatch.setattr(engine, "handlers", [handler])
    monkeypatch.setattr(engine, "level", logging.INFO)
    monkeypatch.setattr(mx, "handlers", [])
    monkeypatch.setattr(mx, "level", logging.NOTSET)
    monkeypatch.setattr(mx, "propagate", False)
    monkeypatch.delenv("MODEL_EXPRESS_LOG_LEVEL", raising=False)

    modelexpress.configure_sglang_logging()
    modelexpress.configure_sglang_logging()
    logging.getLogger("modelexpress.nixl_transfer").info("Using pre-loaded remote agent mx-auto-worker0-source")
    assert output.getvalue().count("mx-auto-worker0-source") == 1
