# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate vLLM startup on serving-version reconciliation."""

from __future__ import annotations

import argparse

from .... import envs
from .control import VllmControlClient


def reconcile_serving_version(
    *,
    address: str,
    desired_version_uid: str | None,
    timeout_seconds: float = 3.0,
) -> str:
    """Record the desired version after EngineCore startup, then verify it."""
    with VllmControlClient(
        address,
        timeout_seconds=timeout_seconds,
    ) as client:
        if desired_version_uid is not None:
            # vLLM exposes Control only after its workers finish loading. The
            # desired-version loader fails startup unless every rank loads this UID.
            client.update_weight_version(desired_version_uid)
        observed_version_uid = client.get_weight_version()
    if desired_version_uid is not None and observed_version_uid != desired_version_uid:
        raise RuntimeError(
            "vLLM serving-version reconciliation failed: "
            f"desired={desired_version_uid!r}, observed={observed_version_uid!r}"
        )
    return observed_version_uid


def main() -> None:
    """Reconcile the configured version against a running vLLM server."""
    parser = argparse.ArgumentParser(
        description="Reconcile vLLM's reported serving weight version",
    )
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    args = parser.parse_args()
    observed = reconcile_serving_version(
        address=args.address,
        desired_version_uid=envs.MX_REFIT_DESIRED_VERSION_UID,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"vLLM serving version: {observed}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["reconcile_serving_version"]
