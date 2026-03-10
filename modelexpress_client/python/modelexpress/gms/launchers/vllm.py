# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM engine launcher for multi-GPU MX GMS.

Spawns tp_size * ep_size worker processes via torch.multiprocessing.spawn.
Each worker initializes vLLM distributed state, loads its weight shard
(from pluggable weight sources), and calls shared mx_hooks for
GMS + NIXL + MX Server operations.
"""

from __future__ import annotations

import copy
import logging
import signal
import socket
from typing import Any

import torch
import torch.multiprocessing as mp

from ..config import MxConfig, GmsConfig, GmsMode, WeightSourceType

logger = logging.getLogger(__name__)


def run(config: GmsConfig) -> None:
    """Entry point called by CLI dispatcher."""
    mx_config = config.to_mx_config()

    if config.mode == GmsMode.SOURCE:
        _run_source(config, mx_config)
    else:
        _run_target(config, mx_config)


def _run_source(config: GmsConfig, mx_config: MxConfig) -> None:
    """Launch source worker(s).

    Each worker loads its weight shard, registers with NIXL, then blocks on
    signal.pause(). This function only returns when all workers are killed
    (e.g. SIGTERM from kubelet).

    For multi-GPU: mp.spawn creates child processes. join=True makes the main
    process wait until all children exit. Each child blocks in signal.pause()
    after loading, so the main process also blocks -- keeping the entire
    process tree alive.

    For single-GPU: the worker runs in the main process directly (no spawn
    overhead). signal.pause() blocks the main process.
    """
    nprocs = config.total_workers
    port = _find_free_port()
    if nprocs == 1:
        _source_worker(0, config, mx_config, port)
    else:
        mp.spawn(
            _source_worker,
            nprocs=nprocs,
            args=(config, mx_config, port),
            join=True,
        )


def _run_target(config: GmsConfig, mx_config: MxConfig) -> None:
    """Launch target worker(s).

    Same lifecycle as _run_source -- see its docstring for details.
    """
    nprocs = config.total_workers
    port = _find_free_port()
    if nprocs == 1:
        _target_worker(0, config, mx_config, port)
    else:
        mp.spawn(
            _target_worker,
            nprocs=nprocs,
            args=(config, mx_config, port),
            join=True,
        )


# ---------------------------------------------------------------------------
# Worker functions (run in spawned child processes)
# ---------------------------------------------------------------------------


def _source_worker(
    rank: int,
    config: GmsConfig,
    mx_config: MxConfig,
    master_port: int,
) -> None:
    """Source worker: load weights -> GMS -> NIXL -> MX Server."""
    from torch.cuda.memory import use_mem_pool

    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    from ..mx_hooks import (
        source_commit_gms,
        source_connect_gms,
        source_finalize,
        source_register_nixl,
    )

    torch.cuda.set_device(rank)

    _init_vllm_distributed(rank, config.total_workers, config.tp_size, master_port)

    # Connect to GMS and get memory pool. All tensor allocations during
    # model init, weight loading, and post-processing must go through this
    # pool so that register_module_tensors() can find them.
    gms_client, pool = source_connect_gms(device_id=rank)

    vllm_config, model_config, _load_config = _build_vllm_configs(config)
    target_device = torch.device("cuda", rank)
    weights_iter = _get_weights_iterator(config, model_config)

    with use_mem_pool(pool, device=target_device):
        model = _load_model(vllm_config, model_config, weights_iter, target_device)

        # Register raw tensors with NIXL for RDMA (before post-processing).
        # Targets receive raw weights and run post-processing locally.
        source_register_nixl(
            device_id=rank, rank=rank, model=model, mx_config=mx_config,
        )

        # Post-processing (FP8, MLA absorption) must be inside use_mem_pool
        # so derived tensors are also in GMS-managed memory.
        process_weights_after_loading(model, model_config, target_device)
        torch.cuda.empty_cache()

    # Commit post-processed state to GMS so the engine reads the final weights.
    source_commit_gms(gms_client, device_id=rank, model=model)

    source_finalize(rank=rank, mx_config=mx_config)
    logger.info("Worker %d: loading complete. Waiting for termination signal.", rank)

    # Block until SIGTERM/SIGINT. The worker must stay alive so that:
    # - NIXL agent remains registered (rkeys valid for RDMA reads by targets)
    # - GPU memory holding loaded weight tensors is not freed
    # - GMS shared memory mappings remain valid
    # RDMA reads are one-sided: the target NIC reads directly from this
    # worker's registered GPU memory without any CPU involvement here.
    signal.pause()


def _target_worker(
    rank: int,
    config: GmsConfig,
    mx_config: MxConfig,
    master_port: int,
) -> None:
    """Target worker: allocate -> RDMA receive -> post-process -> GMS commit."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    from ..mx_hooks import target_allocate, target_commit, target_receive

    torch.cuda.set_device(rank)

    # 1. vLLM distributed init
    _init_vllm_distributed(rank, config.total_workers, config.tp_size, master_port)

    # 2. Create model skeleton with dummy weights
    vllm_config, model_config, load_config = _build_vllm_configs(config)
    target_device = torch.device("cuda", rank)
    model = _create_dummy_model(vllm_config, model_config, load_config, target_device)

    # 3. Shared hooks: allocate + receive
    gms_client, nixl_mgr = target_allocate(
        device_id=rank, rank=rank, model=model, mx_config=mx_config
    )
    target_receive(rank=rank, nixl_mgr=nixl_mgr, mx_config=mx_config)

    # 4. vLLM post-processing
    process_weights_after_loading(model, model_config, target_device)

    # 5. Shared hook: commit to GMS
    target_commit(
        device_id=rank,
        rank=rank,
        model=model,
        gms_client=gms_client,
        mx_config=mx_config,
    )

    logger.info("Worker %d: loading complete. Waiting for termination signal.", rank)
    # Block until SIGTERM/SIGINT. Same rationale as _source_worker: keep
    # GPU memory, NIXL agents, and GMS mappings alive for serving.
    signal.pause()


# ---------------------------------------------------------------------------
# vLLM-specific helpers
# ---------------------------------------------------------------------------


def _init_vllm_distributed(
    rank: int,
    world_size: int,
    tp_size: int,
    master_port: int,
) -> None:
    """Initialize vLLM's distributed environment for this worker."""
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{master_port}",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
    )


def _build_vllm_configs(config: GmsConfig) -> tuple[Any, Any, Any]:
    """Build vLLM configuration objects from GMS config."""
    from vllm.engine.arg_utils import AsyncEngineArgs

    engine_args = AsyncEngineArgs(
        model=config.model,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        revision=config.revision,
        max_model_len=config.max_model_len,
        tensor_parallel_size=config.tp_size,
        enable_expert_parallel=config.enable_expert_parallel,
        disable_log_stats=True,
        enable_prefix_caching=False,
    )
    vllm_config = engine_args.create_engine_config()
    return vllm_config, vllm_config.model_config, vllm_config.load_config


def _get_weights_iterator(config: GmsConfig, model_config: Any):
    """Select and return weight iterator based on configured source.

    Returns None for disk (engine's built-in loader will be used).
    """
    if config.weight_source == WeightSourceType.GDS:
        from ..weight_sources.gds import get_weights_iterator

        return get_weights_iterator(
            model_path=config.model, model_config=model_config
        )
    elif config.weight_source == WeightSourceType.S3:
        from ..weight_sources.s3 import get_weights_iterator

        return get_weights_iterator(
            bucket=config.s3_bucket,
            prefix=config.s3_prefix,
            model_config=model_config,
        )
    else:
        from ..weight_sources.disk import get_weights_iterator

        return get_weights_iterator()


def _load_model(
    vllm_config: Any,
    model_config: Any,
    weights_iter,
    target_device: torch.device,
) -> torch.nn.Module:
    """Load model with sharded weights from the given source."""
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            model = initialize_model(
                vllm_config=vllm_config, model_config=model_config
            )

        if weights_iter is not None:
            # Custom weight source: feed iterator into model's load_weights.
            # The model handles TP/EP sharding internally.
            model.load_weights(weights_iter)
        else:
            # Default: engine's built-in disk loader
            modified_config = copy.copy(vllm_config.load_config)
            try:
                modified_config.load_format = "auto"
            except AttributeError:
                object.__setattr__(modified_config, "load_format", "auto")

            from vllm.model_executor.model_loader.default_loader import (
                DefaultModelLoader,
            )

            loader = DefaultModelLoader(modified_config)
            loader.load_weights(model, model_config)

    return model


def _create_dummy_model(
    vllm_config: Any,
    model_config: Any,
    load_config: Any,
    target_device: torch.device,
) -> torch.nn.Module:
    """Create model with dummy weights (for target before RDMA receive)."""
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    modified_config = copy.copy(load_config)
    try:
        modified_config.load_format = "dummy"
    except AttributeError:
        object.__setattr__(modified_config, "load_format", "dummy")

    loader = DummyModelLoader(modified_config)

    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            model = initialize_model(
                vllm_config=vllm_config, model_config=model_config
            )
        loader.load_weights(model, model_config)

    return model


def _find_free_port() -> int:
    """Find a free port for torch.distributed rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
