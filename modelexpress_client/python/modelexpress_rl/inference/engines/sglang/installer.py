# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang checkpoint installer."""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from ...plan import (
    EngineCapabilities,
    EngineInstaller,
    PreparedArtifact,
    PreparedCheckpointArtifact,
)
from ...receiver import PreparedCheckpoint, ReceiverInstallError


class _SglangInstaller(EngineInstaller):
    def __init__(self, model_runner) -> None:
        self._model_runner = model_runner

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            artifact_types=frozenset({PreparedCheckpointArtifact})
        )

    def install(self, prepared: PreparedArtifact) -> dict[str, float]:
        if not isinstance(prepared, PreparedCheckpointArtifact):
            raise TypeError("SGLang requires a prepared checkpoint")
        checkpoint = prepared.checkpoint
        if not isinstance(checkpoint, PreparedCheckpoint):
            raise TypeError("checkpoint preparation has an invalid value")
        started = time.perf_counter()
        self._install_checkpoint(checkpoint)
        return {"perf/mx_receive_install_time": time.perf_counter() - started}

    def _install_checkpoint(self, prepared: PreparedCheckpoint) -> None:
        runner = self._model_runner
        try:
            from sglang.srt.configs.load_config import LoadConfig, LoadFormat
            from sglang.srt.model_loader.loader import (
                DefaultModelLoader,
                get_model_loader,
            )

            loader = get_model_loader(
                LoadConfig(
                    load_format=LoadFormat.SAFETENSORS,
                    download_dir=runner.server_args.download_dir,
                    model_loader_extra_config=(
                        runner.server_args.model_loader_extra_config
                    ),
                ),
                runner.model_config,
            )
            if not isinstance(loader, DefaultModelLoader):
                raise TypeError("ModelExpress requires DefaultModelLoader")
            weights = loader._get_weights_iterator(
                SimpleNamespace(
                    model_or_path=str(prepared.path),
                    revision=None,
                    prefix="",
                    fall_back_to_pt=False,
                    model_config=runner.model_config,
                )
            )
        except Exception as error:
            raise ReceiverInstallError(str(error)) from error

        try:
            from sglang.srt.model_loader.utils import set_default_torch_dtype

            with set_default_torch_dtype(runner.model_config.dtype):
                loader.load_weights_and_postprocess(
                    runner.model,
                    weights,
                    torch.device(runner.device),
                )
            device = torch.get_device_module(runner.device)
            if hasattr(device, "synchronize"):
                device.synchronize()
        except Exception as error:
            raise ReceiverInstallError(str(error)) from error


__all__: list[str] = []
