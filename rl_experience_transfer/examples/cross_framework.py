# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Show actionable rejection of unsafe cross-framework experience."""

from rlxfer.compatibility import CompatibilityRequirements, check_compatibility
from rlxfer.model import ExperienceBatch, ExperienceMetadata

batch = ExperienceBatch(
    metadata=ExperienceMetadata(
        "slime-rollout",
        "slime",
        "0.3.1",
        algorithm="grpo",
        tokenizer_id="tokenizer-a",
        model_id="model-a",
    ),
    payload={"reward": 1.0},
)
report = check_compatibility(
    batch,
    CompatibilityRequirements(
        consumer_framework="prime_rl",
        consumer_framework_version="0.5.0",
        algorithm="ppo",
        tokenizer_id="tokenizer-b",
        model_id="model-a",
    ),
)
assert not report.compatible
for issue in report.issues:
    print(issue)
