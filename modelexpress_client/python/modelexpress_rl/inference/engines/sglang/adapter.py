# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang installation adapter for canonical S3 refit."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from modelexpress_rl.inference.receiver import (
    CanonicalS3GeneratorAdapter,
    ObjectStorageGeneratorConfig,
    PreparedCheckpoint,
    ReceiverInstallError,
)

from .context import SglangGeneratorContext


class SGLangGeneratorAdapter(CanonicalS3GeneratorAdapter):
    """Reload a prepared canonical checkpoint into one SGLang model runner."""

    def __init__(
        self,
        *,
        context: SglangGeneratorContext,
        config: ObjectStorageGeneratorConfig,
    ) -> None:
        self.model_runner = context.model_runner
        super().__init__(
            model_name=self.model_runner.model_config.model_path,
            config=config,
        )

    def install_prepared_checkpoint(self, prepared: PreparedCheckpoint) -> None:
        try:
            from sglang.srt.configs.load_config import LoadConfig, LoadFormat
            from sglang.srt.model_loader.loader import (
                DefaultModelLoader,
                get_model_loader,
            )

            loader = get_model_loader(
                LoadConfig(
                    load_format=LoadFormat.SAFETENSORS,
                    download_dir=self.model_runner.server_args.download_dir,
                    model_loader_extra_config=(
                        self.model_runner.server_args.model_loader_extra_config
                    ),
                ),
                self.model_runner.model_config,
            )
            if not isinstance(loader, DefaultModelLoader):
                raise TypeError("ModelExpress requires DefaultModelLoader")
            weights = loader._get_weights_iterator(
                SimpleNamespace(
                    model_or_path=str(prepared.path),
                    revision=None,
                    prefix="",
                    fall_back_to_pt=False,
                    model_config=self.model_runner.model_config,
                )
            )
        except Exception as error:
            raise ReceiverInstallError(str(error)) from error

        try:
            from sglang.srt.model_loader.utils import set_default_torch_dtype

            with set_default_torch_dtype(self.model_runner.model_config.dtype):
                loader.load_weights_and_postprocess(
                    self.model_runner.model,
                    weights,
                    torch.device(self.model_runner.device),
                )
            device = torch.get_device_module(self.model_runner.device)
            if hasattr(device, "synchronize"):
                device.synchronize()
        except Exception as error:
            raise ReceiverInstallError(str(error)) from error


__all__ = [
    "SGLangGeneratorAdapter",
]
