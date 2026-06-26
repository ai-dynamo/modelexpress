# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-specific MX integrations.

Currently:

- :mod:`modelexpress.integrations.verl_checkpoint_engine` —
  the rank-to-rank verl checkpoint engine that publishes via
  :class:`RankLocalPublisher` and pulls via :class:`MxRefitReceiver`
  + the reshard planner.

Other framework adapters (NemoRL v2 at :mod:`modelexpress.nemo_rl_v2`,
PrimeRL's ``mx_v2`` broadcast in the prime-rl tree) live alongside their
respective frameworks. This subpackage is the home for new MX-side
framework adapters going forward.
"""
