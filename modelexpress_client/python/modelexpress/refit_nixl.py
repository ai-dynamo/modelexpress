# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL compatibility adapter for the refit POC."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import time
from typing import Any

from .resharding import SegmentPlan


class NixlAdapter:
    """Small wrapper around NIXL API variants used by the POC."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agent, self.backends, self.config_errors = _make_nixl_agent(agent_name)

    def register_tensor(self, tensor) -> Any:
        return _nixl_register_tensor(self.agent, tensor, self.backends)

    def metadata_bytes(self) -> bytes:
        return _nixl_metadata_bytes(self.agent.get_agent_metadata())

    def add_remote_agent(self, metadata: bytes) -> str:
        return self.agent.add_remote_agent(metadata)

    def read_descriptor_group(
        self,
        *,
        remote_agent_name: str,
        remote_descs: list[tuple[int, int, int]],
        local_descs: list[tuple[int, int, int]],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        prep_start = time.perf_counter()
        remote_side = _nixl_prep_xfer(
            self.agent,
            remote_agent_name,
            remote_descs,
            self.backends,
        )
        local_side = _nixl_prep_xfer(self.agent, "", local_descs, self.backends)
        prep_duration_ms = (time.perf_counter() - prep_start) * 1000

        indices = list(range(len(remote_descs)))
        handle = _nixl_make_read(
            self.agent,
            local_side,
            indices,
            remote_side,
            self.backends,
        )
        read_duration_ms, backend, telemetry = _nixl_wait_for_read(
            self.agent,
            handle,
            timeout_seconds,
        )
        return {
            "prep_duration_ms": prep_duration_ms,
            "read_duration_ms": read_duration_ms,
            "backend": backend,
            "telemetry": telemetry,
        }


def read_segment_groups(
    *,
    adapter: NixlAdapter,
    target,
    sources_by_id: dict[str, dict[str, Any]],
    remote_agent_names: dict[str, str],
    plans: list[SegmentPlan],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    torch = _import_torch()
    grouped: dict[str, list[SegmentPlan]] = {}
    for plan in plans:
        grouped.setdefault(plan.source_id, []).append(plan)

    reads: list[dict[str, Any]] = []
    for source_id, source_plans in sorted(grouped.items()):
        source = sources_by_id[source_id]
        remote_descs = [
            (
                int(source["addr"]) + plan.source_byte_offset,
                plan.bytes,
                int(source["device_id"]),
            )
            for plan in source_plans
        ]
        local_descs = [
            (
                int(target.data_ptr()) + plan.target_byte_offset,
                plan.bytes,
                int(target.get_device()),
            )
            for plan in source_plans
        ]
        read = adapter.read_descriptor_group(
            remote_agent_name=remote_agent_names[source_id],
            remote_descs=remote_descs,
            local_descs=local_descs,
            timeout_seconds=timeout_seconds,
        )
        torch.cuda.synchronize(target.device)
        reads.append(
            {
                "source_id": source_id,
                "source_rank": source["rank"],
                "segment_count": len(source_plans),
                "bytes": sum(plan.bytes for plan in source_plans),
                "prep_duration_ms": read["prep_duration_ms"],
                "read_duration_ms": read["read_duration_ms"],
                "backend": read["backend"],
                "telemetry": (
                    str(read["telemetry"])
                    if read["telemetry"] is not None
                    else None
                ),
                "segments": [plan.to_dict() for plan in source_plans],
            }
        )

    return reads


def _import_nixl():
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError:
        from nixl import nixl_agent, nixl_agent_config

    return nixl_agent, nixl_agent_config


def _import_torch():
    import torch

    return torch


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def select_cuda_device(local_rank: int) -> tuple[int, bool]:
    torch = _import_torch()
    device_count = torch.cuda.device_count()
    if local_rank < device_count:
        return local_rank, False
    if _env_truthy("MX_REFIT_ALLOW_GPU_REUSE") and device_count > 0:
        return local_rank % device_count, True
    raise RuntimeError(
        f"local rank {local_rank} requires CUDA device {local_rank}, "
        f"but only {device_count} devices are visible. Set "
        "MX_REFIT_ALLOW_GPU_REUSE=1 for the capacity-constrained POC fallback."
    )


def _nixl_backends() -> list[str]:
    return [os.environ.get("MX_NIXL_BACKEND", "UCX").strip().upper()]


def apply_nixl_ucx_pin(device_id: int) -> None:
    if "UCX" not in _nixl_backends():
        return
    if not os.environ.get("MX_RDMA_NIC_PIN"):
        return

    try:
        spec = importlib.util.spec_from_file_location(
            "mx_ucx_utils_direct",
            Path(__file__).with_name("ucx_utils.py"),
        )
        if spec is not None and spec.loader is not None:
            ucx_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ucx_utils)
            ucx_utils.apply_nic_pin_for_device(device_id)
            return
    except Exception as exc:
        print(f"NIXL NIC pin direct import failed: {exc}", flush=True)

    try:
        from modelexpress import ucx_utils

        ucx_utils.apply_nic_pin_for_device(device_id)
    except Exception as exc:
        print(f"NIXL NIC pin package import failed: {exc}", flush=True)


def _make_nixl_agent(agent_name: str):
    nixl_agent, nixl_agent_config = _import_nixl()
    backends = _nixl_backends()
    config = None
    config_errors: list[str] = []

    for build_config in (
        lambda: nixl_agent_config(backends=backends),
        lambda: nixl_agent_config(True, True, 0, backends=backends),
        lambda: nixl_agent_config(True, True, 0),
    ):
        try:
            config = build_config()
            break
        except TypeError as exc:
            config_errors.append(str(exc))

    try:
        return nixl_agent(agent_name, config), backends, config_errors
    except TypeError:
        return nixl_agent(agent_name), backends, config_errors


def _nixl_register_tensor(agent, tensor, backends: list[str]) -> Any:
    for register in (
        lambda: agent.register_memory([tensor], backends=backends),
        lambda: agent.register_memory(tensor, backends=backends),
        lambda: agent.register_memory([tensor]),
        lambda: agent.register_memory(tensor),
    ):
        try:
            return register()
        except TypeError:
            continue
    raise RuntimeError("NIXL register_memory did not accept tensor registration")


def _nixl_metadata_bytes(metadata: Any) -> bytes:
    if isinstance(metadata, bytes):
        return metadata
    if isinstance(metadata, bytearray):
        return bytes(metadata)
    if isinstance(metadata, str):
        return metadata.encode("utf-8")
    return bytes(metadata)


def _nixl_prep_xfer(
    agent,
    agent_name: str,
    descs: list[tuple[int, int, int]],
    backends: list[str],
):
    for prep in (
        lambda: agent.prep_xfer_dlist(
            agent_name=agent_name,
            xfer_list=descs,
            mem_type="cuda",
            backends=backends,
        ),
        lambda: agent.prep_xfer_dlist(agent_name, descs, "cuda", backends),
        lambda: agent.prep_xfer_dlist(agent_name, descs),
    ):
        try:
            return prep()
        except TypeError:
            continue
    raise RuntimeError("NIXL prep_xfer_dlist did not accept CUDA descriptors")


def _nixl_make_read(
    agent,
    local_side,
    indices: list[int],
    remote_side,
    backends: list[str],
):
    for make_read in (
        lambda: agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=local_side,
            local_indices=indices,
            remote_xfer_side=remote_side,
            remote_indices=indices,
            backends=backends,
        ),
        lambda: agent.make_prepped_xfer(
            "READ",
            local_side,
            indices,
            remote_side,
            indices,
            b"",
            backends,
        ),
        lambda: agent.make_prepped_xfer(
            "READ",
            local_side,
            indices,
            remote_side,
            indices,
        ),
    ):
        try:
            return make_read()
        except TypeError:
            continue
    raise RuntimeError("NIXL make_prepped_xfer did not accept READ descriptors")


def _nixl_release_handle(agent, handle) -> None:
    release = getattr(agent, "release_xfer_handle", None)
    if release is not None:
        release(handle)


def _nixl_status_text(status: Any) -> str:
    text = str(status)
    return text.rsplit(".", 1)[-1].upper()


def _nixl_wait_for_read(
    agent,
    handle,
    timeout_seconds: float,
) -> tuple[float, str | None, Any]:
    start = time.perf_counter()
    transfer_state = _nixl_status_text(agent.transfer(handle))
    if transfer_state in {"ERR", "ERROR", "FAIL", "FAILED"}:
        _nixl_release_handle(agent, handle)
        raise RuntimeError(f"NIXL transfer failed to post: {transfer_state}")

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= timeout_seconds:
            _nixl_release_handle(agent, handle)
            raise TimeoutError("NIXL READ timed out")
        status = _nixl_status_text(agent.check_xfer_state(handle))
        if status in {"DONE", "SUCCESS"}:
            backend = None
            telemetry = None
            query_backend = getattr(agent, "query_xfer_backend", None)
            if query_backend is not None:
                try:
                    backend = str(query_backend(handle))
                except Exception:
                    backend = None
            get_telemetry = getattr(agent, "get_xfer_telemetry", None)
            if get_telemetry is not None:
                try:
                    telemetry = get_telemetry(handle)
                except Exception:
                    telemetry = None
            _nixl_release_handle(agent, handle)
            return elapsed * 1000, backend, telemetry
        if status in {"ERR", "ERROR", "FAIL", "FAILED"}:
            _nixl_release_handle(agent, handle)
            raise RuntimeError(f"NIXL READ failed with status {status}")
        time.sleep(0.001)
