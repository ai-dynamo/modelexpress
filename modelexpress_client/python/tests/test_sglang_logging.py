# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import logging

import pytest

import modelexpress


@pytest.mark.parametrize("handler_namespace", ["sglang", ""])
@pytest.mark.parametrize("propagate", [False, True])
def test_sglang_subprocess_logs_include_mx_rdma_evidence(monkeypatch, handler_namespace, propagate):
    engine = logging.getLogger("sglang")
    mx = logging.getLogger("modelexpress")
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [handler])
    monkeypatch.setattr(root, "level", logging.INFO)
    monkeypatch.setattr(engine, "handlers", [handler] if handler_namespace else [])
    monkeypatch.setattr(engine, "level", logging.INFO if handler_namespace else logging.NOTSET)
    monkeypatch.setattr(mx, "handlers", [])
    monkeypatch.setattr(mx, "level", logging.NOTSET)
    monkeypatch.setattr(mx, "propagate", propagate)
    monkeypatch.delenv("MODEL_EXPRESS_LOG_LEVEL", raising=False)

    modelexpress.configure_sglang_logging()
    modelexpress.configure_sglang_logging()
    logging.getLogger("modelexpress.nixl_transfer").info("Using pre-loaded remote agent mx-auto-worker0-source")
    assert output.getvalue().count("mx-auto-worker0-source") == 1
