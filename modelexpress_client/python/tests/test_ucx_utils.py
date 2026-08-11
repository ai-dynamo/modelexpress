# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from modelexpress import ucx_utils


def test_nic_access_requires_exposed_uverbs_device(monkeypatch):
    monkeypatch.setattr(
        ucx_utils.os,
        "listdir",
        lambda path: ["uverbs4"] if path.endswith("infiniband_verbs") else [],
    )
    monkeypatch.setattr(
        ucx_utils.os.path,
        "exists",
        lambda path: path == "/dev/infiniband/uverbs4",
    )

    assert ucx_utils._nic_has_accessible_verbs_device("mlx5_4")


def test_nic_access_rejects_host_only_uverbs_device(monkeypatch):
    monkeypatch.setattr(
        ucx_utils.os,
        "listdir",
        lambda path: ["uverbs0"] if path.endswith("infiniband_verbs") else [],
    )
    monkeypatch.setattr(ucx_utils.os.path, "exists", lambda path: False)

    assert not ucx_utils._nic_has_accessible_verbs_device("mlx5_0")
