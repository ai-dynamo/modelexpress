# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the resolved per-run configuration alongside the mounted packages."""

import json
from pathlib import Path


def load_config():
    return json.loads((Path(__file__).parents[1] / "config.json").read_text())
