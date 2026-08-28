# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

import modelexpress_rl
from modelexpress_rl.train.resources import _TrainerResources


@pytest.mark.parametrize(
    ("rank", "agent_name"),
    [
        ("7", "modelexpress-trainer-7-01234567"),
        (None, "modelexpress-trainer-01234567"),
    ],
)
def test_resources_initialize_transport_and_manifest_service(
    monkeypatch, rank, agent_name
):
    monkeypatch.setenv("MX_WORKER_HOST", "trainer.test")
    monkeypatch.setenv("MX_METADATA_PORT", "18000")
    monkeypatch.setenv("MX_WORKER_GRPC_PORT", "19000")
    if rank is None:
        monkeypatch.delenv("RANK", raising=False)
    else:
        monkeypatch.setenv("RANK", rank)
    manager = MagicMock()
    server = MagicMock()
    server.add_insecure_port.return_value = 19002

    with (
        patch(
            "modelexpress_rl.train.resources.uuid.uuid4",
            return_value=MagicMock(hex="0123456789abcdef"),
        ),
        patch(
            "modelexpress.nixl_transfer.NixlTransferManager",
            return_value=manager,
        ) as manager_type,
        patch(
            "modelexpress_rl.train.resources.grpc.server",
            return_value=server,
        ),
    ):
        resources = _TrainerResources.initialize(device_id=2)

    manager_type.assert_called_once_with(
        agent_name=agent_name,
        device_id=2,
        listen_port=18002,
    )
    manager.initialize.assert_called_once_with()
    server.add_insecure_port.assert_called_once_with("[::]:19002")
    server.start.assert_called_once_with()
    assert resources.worker_endpoint == "trainer.test:19002"

    resources.close()


def test_resources_own_only_private_transport_resources():
    manager = MagicMock()
    manifest_service = MagicMock()
    server = MagicMock()
    resources = _TrainerResources(
        manager=manager,
        manifest_service=manifest_service,
        server=server,
        worker_endpoint="trainer.test:19000",
    )

    assert resources.manager is manager
    assert resources.manifest_service is manifest_service
    assert resources.worker_endpoint == "trainer.test:19000"
    resources.close()
    resources.close()

    server.stop.assert_called_once_with(grace=None)
    server.stop.return_value.wait.assert_called_once_with()
    manager.shutdown.assert_called_once_with()


def test_resources_are_not_part_of_public_api():
    assert not hasattr(modelexpress_rl, "TrainerResources")
