# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import modelexpress_rl.inference.bootstrap as bootstrap_module
from modelexpress_rl import ModelExpressGeneratorBootstrap


def test_generator_bootstrap_initializes_and_transfers_ownership(monkeypatch):
    created = []

    def create_transfer(**kwargs):
        transfer = object()
        created.append((kwargs, transfer))
        return transfer

    monkeypatch.setattr(bootstrap_module, "_NixlStagedTransfer", create_transfer)

    bootstrap = ModelExpressGeneratorBootstrap(device_id=3)

    assert created[0][0] == {
        "agent_name": f"mx-refit-{bootstrap.worker_id}",
        "device_id": 3,
        "device": bootstrap_module.torch.device("cuda", 3),
        "listen_port": None,
    }
    assert bootstrap.claim(device_id=3) is created[0][1]
    with pytest.raises(RuntimeError, match="already claimed"):
        bootstrap.claim(device_id=3)


def test_generator_bootstrap_rejects_mismatched_engine_device(monkeypatch):
    monkeypatch.setattr(
        bootstrap_module, "_NixlStagedTransfer", lambda **_kwargs: object()
    )
    bootstrap = ModelExpressGeneratorBootstrap(device_id=2)

    with pytest.raises(ValueError, match="does not match"):
        bootstrap.claim(device_id=1)


def test_generator_bootstrap_closes_an_unclaimed_transfer_once(monkeypatch):
    class Transfer:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    transfer = Transfer()
    monkeypatch.setattr(
        bootstrap_module, "_NixlStagedTransfer", lambda **_kwargs: transfer
    )
    bootstrap = ModelExpressGeneratorBootstrap(device_id=2)

    bootstrap.close()
    bootstrap.close()

    assert transfer.close_calls == 1
    with pytest.raises(RuntimeError, match="was closed"):
        bootstrap.claim(device_id=2)


def test_generator_bootstrap_does_not_close_a_claimed_transfer(monkeypatch):
    class Transfer:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    transfer = Transfer()
    monkeypatch.setattr(
        bootstrap_module, "_NixlStagedTransfer", lambda **_kwargs: transfer
    )
    bootstrap = ModelExpressGeneratorBootstrap(device_id=2)

    assert bootstrap.claim(device_id=2) is transfer
    bootstrap.close()

    assert transfer.close_calls == 0
    transfer.close()
    assert transfer.close_calls == 1
