# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the benchmark scenario shared by preparation, execution, and reporting."""


def load(config):
    name = config.get("scenario", "delta")
    if name != "delta":
        raise ValueError(f"Unknown benchmark scenario: {name}")
    import delta_scenario

    return delta_scenario
