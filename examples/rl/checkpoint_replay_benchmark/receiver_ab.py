# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Receiver-only Qwen30B loader versus captured replay qualification and timings.

All source tensors are replayed; this isolates mapping, not unchanged-tensor skipping.
The caller measures serving pause; this worker reports synchronized install time.
"""

import hashlib
import time


def worker_call(worker, action, checkpoint="/work/model"):
    import torch
    from modelexpress_rl.inference.engines.vllm.installer import _VllmInstaller
    from modelexpress_rl.utils import index_checkpoint_tensors

    model = worker.model_runner.model
    device = worker.device
    if not hasattr(worker, "winter_installer"):
        worker.winter_installer = _VllmInstaller(
            model=model,
            vllm_config=worker.vllm_config,
            model_config=worker.vllm_config.model_config,
            device=device,
        )
    installer = worker.winter_installer
    rank = torch.distributed.get_rank()
    if action == "hash":
        digest = hashlib.sha256()
        count = 0
        pointers = []
        for name, tensor in sorted((*model.named_parameters(), *model.named_buffers())):
            if tensor.is_meta:
                raise RuntimeError("meta runtime tensor: " + name)
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            value = (
                tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            )
            digest.update(memoryview(value))
            count += tensor.numel() * tensor.element_size()
            pointers.append(
                (name, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
            )
        return {
            "rank": rank,
            "sha256": digest.hexdigest(),
            "bytes": count,
            "pointers": pointers,
        }
    if action == "capture":
        _, _, metadata = index_checkpoint_tensors(checkpoint)
        dtypes = {"BF16": torch.bfloat16, "F32": torch.float32}
        manifest = [
            (name, dtypes[info["dtype"]], tuple(info["shape"]))
            for name, info in metadata.items()
        ]
        torch.cuda.synchronize()
        start = time.perf_counter()
        capture, _ = installer.capture(manifest)
        torch.cuda.synchronize()
        if (
            capture.unsupported
            or capture.unattributed
            or installer.source_install_plan is None
        ):
            raise RuntimeError(
                f"Incomplete mapping: unsupported={capture.unsupported}, unattributed={capture.unattributed}"
            )
        worker.winter_capture = capture
        return {
            "rank": rank,
            "capture_s": time.perf_counter() - start,
            "copies": len(capture.copies),
            "mapped_sources": len({c.src_name for c in capture.copies}),
            "schema_sources": len(manifest),
        }
    torch.distributed.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    if action == "ordinary":
        installer.install_checkpoint(checkpoint)
    elif action == "mapped":
        installer.install_checkpoint_replay(checkpoint)
    else:
        raise ValueError(action)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    torch.distributed.barrier()
    return {
        "rank": rank,
        "arm": action,
        "install_s": elapsed,
        "checkpoint": checkpoint,
        "mapping_used": action == "mapped",
        "gpu_peak_allocated": torch.cuda.max_memory_allocated(),
    }


class WinterWorkerExtension:
    def winter_call(self, action, checkpoint="/work/model"):
        return worker_call(self, action, checkpoint)
