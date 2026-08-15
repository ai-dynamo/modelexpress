# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang delta receiver: the two engine hooks against a SGLang model runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from modelexpress.refit.receiver import (
    ModelExpressWeightReceiver,
    PreparedRevision,
    ReceiverConfig,
    ReceiverInstallError,
)


class SglangWeightReceiver(ModelExpressWeightReceiver):
    """Delta receiver driven directly by SGLang's model runner."""

    def __init__(self, config: ReceiverConfig, receiver_id: str, model_runner) -> None:
        # Set before super().__init__, which calls _launch_checkpoint.
        self.model_runner = model_runner
        super().__init__(config, receiver_id)

    def _launch_checkpoint(self) -> Path:
        checkpoint, _, _ = self.model_runner.loader._prepare_weights(
            self.model_runner.model_config.model_path,
            self.model_runner.model_config.revision,
            False,
        )
        return Path(checkpoint)

    def install_prepared_checkpoint(self, prepared: PreparedRevision) -> None:
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
            source = SimpleNamespace(
                model_or_path=str(prepared.path),
                revision=None,
                prefix="",
                fall_back_to_pt=False,
                model_config=self.model_runner.model_config,
            )
            weights = loader._get_weights_iterator(source)
        except Exception as error:
            raise ReceiverInstallError(str(error), False) from error

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
            raise ReceiverInstallError(str(error), True) from error
