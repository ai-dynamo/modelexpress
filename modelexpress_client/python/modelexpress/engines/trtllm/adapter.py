# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM implementation of the ModelExpress engine adapter contract."""

from __future__ import annotations

import json
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from ... import p2p_pb2
from ...accelerators import accelerator_backend_for
from ...adapter import EngineAdapter
from ...load_strategy.context import LoadContext, LoadResult
from ...metadata.client_factory import create_metadata_client

_SOURCE_IDENTITY_KEY = "trtllm_source_identity"
_WEIGHT_LAYOUT_KEY = "trtllm_weight_layout"
_TRANSFORM_PROTOCOL_KEY = "trtllm_transform_protocol_version"
_POST_TRANSFORM_LAYOUT = "post_transform"
_TRANSFORM_PROTOCOL_VERSION = "1"
# TRT-LLM 1.3 PyTorch model aliases introduced by runtime model wiring. Keep
# these aligned with the qualified TRT-LLM release used by the integration.
_RUNTIME_ALIAS_COMPONENTS = frozenset({"next_attn", "next_layer_layernorm"})


def _mx_version() -> str:
    try:
        return version("modelexpress")
    except PackageNotFoundError:
        return "0.0.0"


def _normalize_model_identity(model_name: str) -> str:
    """Normalize TRT-LLM's configured model name for source discovery."""
    if not model_name:
        return "unknown"

    looks_like_path = (
        model_name.startswith(("/", "./", "../", "~")) or model_name.count("/") > 1
    )
    if not looks_like_path:
        return model_name

    path = Path(model_name).expanduser()
    if "snapshots" in path.parts:
        for ancestor in path.parents:
            if ancestor.name.startswith("models--"):
                return ancestor.name[len("models--") :].replace("--", "/")
    return path.name or "unknown"


def _canonical_named_parameters(model: Any) -> list[tuple[str, torch.Tensor]]:
    """Return one stable non-runtime-alias name for each parameter storage."""
    canonical: list[tuple[str, torch.Tensor]] = []
    canonical_storages: set[tuple[str, int | None, int]] = set()
    runtime_aliases: dict[tuple[str, int | None, int], list[str]] = {}

    for name, parameter in model.named_parameters(remove_duplicate=False):
        storage = (
            parameter.device.type,
            parameter.device.index,
            parameter.data.data_ptr(),
        )
        if _RUNTIME_ALIAS_COMPONENTS.intersection(name.split(".")):
            runtime_aliases.setdefault(storage, []).append(name)
            continue
        if storage in canonical_storages:
            continue
        canonical_storages.add(storage)
        canonical.append((name, parameter))

    alias_only = set(runtime_aliases).difference(canonical_storages)
    if alias_only:
        examples = [runtime_aliases[key][0] for key in list(alias_only)[:3]]
        raise RuntimeError(
            "TensorRT-LLM runtime aliases have no canonical parameter path: "
            f"{len(alias_only)} storages; examples: {examples}"
        )
    return canonical


def build_mx_identity(
    source_identity: Any,
    transform_protocol_version: int = int(_TRANSFORM_PROTOCOL_VERSION),
) -> p2p_pb2.SourceIdentity:
    """Convert TRT-LLM's authoritative SourceIdentity into an MX identity."""
    to_dict = getattr(source_identity, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("TensorRT-LLM SourceIdentity must provide to_dict()")

    payload = dict(to_dict())
    model_name = _normalize_model_identity(
        str(payload.pop("model_name", None) or "unknown")
    )
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return p2p_pb2.SourceIdentity(
        mx_version=_mx_version(),
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
        tensor_parallel_size=int(getattr(source_identity, "tp_size", 1)),
        pipeline_parallel_size=int(getattr(source_identity, "pp_size", 1)),
        expert_parallel_size=max(1, int(getattr(source_identity, "ep_size", 1))),
        dtype=str(getattr(source_identity, "dtype", None) or "").replace("torch.", ""),
        extra_parameters={
            _SOURCE_IDENTITY_KEY: serialized,
            _WEIGHT_LAYOUT_KEY: _POST_TRANSFORM_LAYOUT,
            _TRANSFORM_PROTOCOL_KEY: str(transform_protocol_version),
        },
    )


class TrtllmAdapter(EngineAdapter):
    """Expose TRT-LLM lifecycle operations to shared MX load strategies."""

    def __init__(
        self,
        *,
        checkpoint_loader: Any,
        checkpoint_dir: str,
        native_loader_kwargs: dict[str, Any],
        mapping: Any,
        source_identity: Any,
        prepare_post_transform_receiver: Any,
        transform_protocol_version: int,
    ) -> None:
        self.checkpoint_loader = checkpoint_loader
        self.checkpoint_dir = checkpoint_dir
        self.native_loader_kwargs = native_loader_kwargs
        self.mapping = mapping
        self.transform_protocol_version = transform_protocol_version
        self.identity = build_mx_identity(
            source_identity,
            transform_protocol_version=transform_protocol_version,
        )
        self._prepare_post_transform_receiver = prepare_post_transform_receiver
        self.native_loaded = False
        self.rdma_loaded = False
        self.rdma_transform_protocol_version: int | None = None
        self.current_model: torch.nn.Module | None = None
        self.target_device = torch.device("cuda", self.get_device_id())
        self.accelerator_backend = accelerator_backend_for(self.target_device)

    def build_identity(self) -> p2p_pb2.SourceIdentity:
        return self.identity

    def get_worker_rank(self) -> int:
        return int(getattr(self.mapping, "rank", 0))

    def get_global_rank(self) -> int:
        return int(getattr(self.mapping, "rank", 0))

    def get_device_id(self) -> int:
        local_rank = getattr(self.mapping, "local_rank", None)
        if local_rank is not None:
            return int(local_rank)
        return int(torch.cuda.current_device())

    def get_target_device(self) -> torch.device:
        return self.target_device

    def is_cuda_alike(self) -> bool:
        return True

    def requires_exact_tensor_catalog(self) -> bool:
        return True

    def discover_tensors(self, result: LoadResult) -> dict[str, torch.Tensor]:
        if result.model is None:
            raise RuntimeError("TensorRT-LLM tensor discovery requires result.model")

        return {
            name: parameter.data
            for name, parameter in _canonical_named_parameters(result.model)
            if parameter.device == self.target_device
        }

    def load_via_native(self, result: LoadResult) -> LoadResult:
        from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import (
            HfCheckpointLoader,
        )

        self.native_loaded = True
        weights = HfCheckpointLoader.load_weights(
            self.checkpoint_loader,
            self.checkpoint_dir,
            mapping=self.mapping,
            **self.native_loader_kwargs,
        )
        result.value = weights
        result.publishable = False
        self.current_model = result.model
        return result

    def reinit_for_retry(self, result: LoadResult) -> LoadResult:
        # TRT-LLM's outer ModelLoader retains the model reference, so this
        # adapter cannot replace it here. The Llama profile requires an exact
        # canonical parameter catalog: another RDMA attempt or the native HF
        # mapping pipeline overwrites that complete catalog before the model is
        # used. Keep this hook only while that fail-closed invariant holds.
        if not self.requires_exact_tensor_catalog():
            raise RuntimeError(
                "TensorRT-LLM retry without model reconstruction requires an "
                "exact tensor catalog"
            )
        result.publishable = False
        return result

    def before_rdma_receive(self, result: LoadResult) -> LoadResult:
        if result.model is None:
            raise RuntimeError("TensorRT-LLM RDMA preparation requires a model")
        self._prepare_post_transform_receiver(result.model)
        result.publishable = False
        self.current_model = result.model
        return result

    def after_rdma_receive(self, result: LoadResult) -> LoadResult:
        self.rdma_loaded = True
        self.rdma_transform_protocol_version = self.transform_protocol_version
        result.publishable = False
        self.current_model = result.model
        return result


def build_trtllm_load_context(
    *,
    model_config: Any,
    load_config: Any,
    checkpoint_loader: Any,
    checkpoint_dir: str,
    native_loader_kwargs: dict[str, Any],
    mapping: Any,
    source_identity: Any,
    prepare_post_transform_receiver: Any,
    transform_protocol_version: int,
    p2p_enabled: bool,
    mx_server_url: str | None = None,
) -> LoadContext:
    """Build a strategy context from TRT-LLM-owned state and callbacks."""
    adapter = TrtllmAdapter(
        checkpoint_loader=checkpoint_loader,
        checkpoint_dir=checkpoint_dir,
        native_loader_kwargs=native_loader_kwargs,
        mapping=mapping,
        source_identity=source_identity,
        prepare_post_transform_receiver=prepare_post_transform_receiver,
        transform_protocol_version=transform_protocol_version,
    )
    worker_rank = adapter.get_worker_rank()
    return LoadContext(
        engine="trtllm",
        model_config=model_config,
        load_config=load_config,
        target_device=adapter.get_target_device(),
        global_rank=adapter.get_global_rank(),
        worker_rank=worker_rank,
        device_id=adapter.get_device_id(),
        identity=adapter.build_identity(),
        mx_client=create_metadata_client(
            worker_rank=worker_rank,
            server_url=mx_server_url,
        ),
        worker_id=uuid.uuid4().hex[:8],
        mx_server_url=mx_server_url,
        node_rank=int(getattr(mapping, "node_rank", 0)),
        adapter=adapter,
        accelerator_backend=adapter.accelerator_backend,
        p2p_enabled=p2p_enabled,
    )
