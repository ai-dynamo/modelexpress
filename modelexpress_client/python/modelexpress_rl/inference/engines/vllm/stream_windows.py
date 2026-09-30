# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream a full checkpoint in decoder-layer windows.

vLLM layerwise reload keeps a layer's incoming tensors until the whole layer has
arrived. ModelStreamer's distributed partition spreads each layer across ranks,
so one whole-checkpoint stream leaves most layers partially loaded at once.
Streaming a few layers per request bounds that retention while still reading
every byte once.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence

import torch

logger = logging.getLogger(__name__)
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def layer_windows(names: Iterable[str], layers_per_window: int) -> list[list[str]]:
    """Group tensors by ``layers.<N>`` into ascending windows of N // size."""
    if layers_per_window < 0:
        raise ValueError("layers_per_window must be non-negative")
    names = sorted(names)
    if layers_per_window == 0:
        return [names]
    groups: dict[int, list[str]] = defaultdict(list)
    for name in names:
        match = _LAYER.search(name)
        groups[int(match.group(1)) // layers_per_window if match else -1].append(name)
    return [groups[key] for key in sorted(groups)]


def windowed_weights(
    shard_uris: Sequence[str],
    windows: Sequence[Sequence[str]],
    *,
    is_distributed: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield checkpoint tensors, issuing one ranged stream request per window."""
    from runai_model_streamer import DistributedStreamer, FileChunks
    from runai_model_streamer.safetensors_streamer import safetensors_pytorch
    from vllm.platforms import current_platform

    device = (
        f"cuda:{current_platform.current_device()}"
        if is_distributed and current_platform.is_cuda_alike()
        else "cpu"
    )
    paths = list(shard_uris)
    with DistributedStreamer() as streamer:
        located = {}
        prepared = safetensors_pytorch.prepare_request(streamer, paths, None)
        for index, (offset, tensors, sizes) in enumerate(prepared):
            ranges = FileChunks.contiguous(index, paths[index], offset, sizes)
            for chunk, metadata in enumerate(tensors):
                located[metadata.name] = (
                    index,
                    ranges.offsets[chunk],
                    ranges.sizes[chunk],
                    metadata,
                )
        requested = [name for window in windows for name in window]
        if len(requested) != len(set(requested)) or set(requested) != set(located):
            raise ValueError("stream windows must cover each checkpoint tensor once")

        for number, window in enumerate(windows, 1):
            started = time.perf_counter()
            by_file: dict[int, list] = defaultdict(list)
            for name in window:
                by_file[located[name][0]].append(located[name])
            requests, lookup = [], {}
            for index in sorted(by_file):
                entries = sorted(by_file[index], key=lambda entry: entry[1])
                requests.append(
                    FileChunks(
                        index,
                        paths[index],
                        [entry[1] for entry in entries],
                        [entry[2] for entry in entries],
                    )
                )
                lookup[index] = [entry[3] for entry in entries]
            streamer.stream_files(requests, None, device, is_distributed)
            for index, chunk, buffer in streamer.get_chunks():
                metadata = lookup[index][chunk]
                tensor = safetensors_pytorch.create_torch_tensor(buffer, metadata)
                yield metadata.name, tensor.clone()
            logger.info(
                "Streamed refit window %d/%d tensors=%d bytes=%d seconds=%.3f",
                number,
                len(windows),
                len(window),
                sum(sum(request.sizes) for request in requests),
                time.perf_counter() - started,
            )
