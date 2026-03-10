# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allow running as ``python -m modelexpress.gms``."""

import sys

from .main import main

sys.exit(main())
