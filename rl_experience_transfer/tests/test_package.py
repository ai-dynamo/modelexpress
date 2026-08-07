# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import rlxfer


@pytest.mark.unit
def test_package_imports() -> None:
    assert rlxfer.__version__ == "0.1.0"
    public_contracts = {
        "AuthenticatedExperienceSerializer",
        "BufferManager",
        "ConsumerContract",
        "DeliveryStateStore",
        "DeliveryReceipt",
        "ExperienceAdapter",
        "ExperienceSerializer",
        "ExperienceTransport",
        "FallbackTransport",
        "SchemaMigrationRegistry",
        "SqliteDeliveryState",
        "TraceContext",
        "TransferPlan",
        "TransportRegistry",
    }
    assert public_contracts <= set(rlxfer.__all__)
    for name in public_contracts:
        assert getattr(rlxfer, name).__name__ == name
