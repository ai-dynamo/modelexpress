# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark harnesses for ModelExpress v2 + rank-to-rank refit paths.

Entry points:

- ``bench_elastic_scaling.py`` — exercises :class:`MxWeightTransferEngine`
  against a representative v2 publisher/receiver pair; covers the
  ``MxV2TrainingPublisher`` + ``MxV2RefitReceiver`` surface.
- ``bench_verl_rank_to_rank.py`` — exercises :class:`RankLocalPublisher`
  + ``plan_coverage`` + :meth:`MxRefitReceiver.receive_segment` end to
  end against a live NIXL transport; checksum-verifies bytes landed
  and reports per-source byte distribution + Gbps + savings vs the v1
  ``receive_weights`` baseline.

See README.md alongside this file for invocation examples and
reproducibility notes.
"""
