# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang checkpoint installer."""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch
from modelexpress.refit.timing import refit_span

from ...plan import (
    EngineCapabilities,
    EngineInstaller,
    PreparedArtifact,
    PreparedCheckpointArtifact,
    PreparedEngineTensors,
)
from ...receiver import PreparedCheckpoint, ReceiverInstallError


class _SglangInstaller(EngineInstaller):
    def __init__(self, model_runner, *, layout=None) -> None:
        self._model_runner = model_runner
        self._layout = layout
        self._poisoned = False

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            artifact_types=frozenset(
                {PreparedCheckpointArtifact}
                | ({PreparedEngineTensors} if self._layout is not None else set())
            )
        )

    def install(self, prepared: PreparedArtifact) -> dict[str, float]:
        if self._poisoned:
            raise ReceiverInstallError(
                "SGLang installer is poisoned; restart the worker"
            )
        if isinstance(prepared, PreparedEngineTensors) and self._layout is not None:
            started = time.perf_counter()
            with refit_span("installation"):
                self._install_tensors(prepared.staged.tensors)
            return {"perf/mx_receive_install_time": time.perf_counter() - started}
        if not isinstance(prepared, PreparedCheckpointArtifact):
            raise TypeError("SGLang requires a prepared checkpoint")
        checkpoint = prepared.checkpoint
        if not isinstance(checkpoint, PreparedCheckpoint):
            raise TypeError("checkpoint preparation has an invalid value")
        started = time.perf_counter()
        self._install_checkpoint(checkpoint)
        return {"perf/mx_receive_install_time": time.perf_counter() - started}

    @torch.no_grad()
    def _install_tensors(self, tensors):
        self._layout.validate_storage()
        params = dict(self._model_runner.model.named_parameters())
        if set(tensors) != set(params):
            raise ReceiverInstallError(
                "SGLang install requires every destination parameter"
            )
        for name, param in params.items():
            incoming = tensors[name]
            if (
                incoming.shape != param.shape
                or incoming.dtype != param.dtype
                or incoming.device != param.device
            ):
                raise ReceiverInstallError(f"SGLang install layout mismatch for {name}")
        try:
            for name, param in params.items():
                param.copy_(tensors[name])
            device = torch.get_device_module(self._model_runner.device)
            if hasattr(device, "synchronize"):
                device.synchronize()
        except BaseException:
            self._poisoned = True
            raise

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
