<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress Benchmarks

These results show how source selection, NIXL registration, and cache-artifact reuse affect different portions of model startup. Keep the measurement boundaries separate:

- **Model loading time** measures the weight-loading path.
- **NIXL registration time** measures GPU-memory registration after load.
- **API ready time** measures total wall-clock startup from process start until the inference API is ready.

The cold-start and artifact-transfer runs used DeepSeek-V4-Pro with vLLM 0.23.0, TP=8, and `--enable-flashinfer-autotune` on an 8×B200 GPU node with NVIDIA ConnectX-7 NICs. The registration comparison used the same model and TP configuration. Results are specific to this benchmark environment and will vary with model, storage, network, software versions, and runtime configuration.

## Cold-Start Loading Paths

![DeepSeek-V4-Pro cold-start loading benchmark comparing Hugging Face, S3 ModelStreamer, local storage, and P2P RDMA](../benchmark-cold-start-loading.png)

This comparison measures model loading time only.

| Loading path | Model loading time | Speedup vs. cold Hugging Face pull |
|--------------|-------------------:|-----------------------------------:|
| Cold pull from Hugging Face | 8m 53s | 1× |
| ModelStreamer from S3 | 3m 16s | 2.7× |
| High-throughput local storage, cold page cache | 1m 10s | 7.6× |
| P2P GPU-to-GPU over NIXL/RDMA | 11s | 48× |

The P2P result starts from weights that are already loaded, post-processed, and registered on a compatible serving replica. ModelStreamer overlaps concurrent object-store reads with GPU placement through a bounded CPU staging buffer. The local-storage result uses the default storage-loading path with a cold page cache.

GDS is not shown in this comparison. With TP > 1, the current GDS loader reads full checkpoint tensors on every rank before the inference engine slices out each shard, so aggregate I/O grows with the TP degree. See [GDS Reads Full Checkpoint Tensors Under TP](ARCHITECTURE.md#gds-reads-full-checkpoint-tensors-under-tp).

## NIXL Memory Registration

![DeepSeek-V4-Pro NIXL registration benchmark comparing per-tensor, pool, and VMM arena registration](../benchmark-nixl-registration.png)

This comparison measures average NIXL registration time for DeepSeek-V4-Pro with TP=8.

| Registration strategy | Registration time | Speedup |
|-----------------------|------------------:|--------:|
| Per tensor (default) | 8.16s | 1× |
| Pool registration (`MX_POOL_REG=1`) | 1.14s | 7.1× |
| VMM arena (`MX_VMM_ARENA=1`) | 0.79s | 10.3× |

Pool registration discovers and registers each underlying CUDA allocation once instead of registering every tensor. VMM arena registration places load-time allocations in one CUDA virtual-memory arena and registers the used range once. These are alternative strategies; enable only one. See [VMM Arena: Single-MR Registration](DEPLOYMENT.md#vmm-arena-single-mr-registration).

## Weight and Kernel-Artifact Transfer

![DeepSeek-V4-Pro startup benchmark comparing storage loading, P2P weights, and P2P weights with kernel artifacts](../benchmark-artifact-transfer.png)

This comparison measures total wall-clock startup from process start until the API is ready.

| Startup path | API ready time | Speedup |
|--------------|---------------:|--------:|
| Cold start from VAST, no P2P source | 8m 1s | 1× |
| P2P RDMA weights only | 7m | 1.1× |
| P2P RDMA weights and kernel artifacts | 1m 44s | 4.6× |

P2P weight transfer alone removes most weight-loading time, but the new replica still pays for kernel compilation, autotuning, and CUDA graph capture. The artifact-enabled run reused compatible Triton, DeepGEMM, TileLang, CuTe DSL, and FlashInfer caches. ModelExpress transfers these file-backed artifacts between registered host-memory buffers, verifies them, and installs them into the target engine's filesystem cache; they are not loaded into GPU memory.

Artifact reuse is compatibility-scoped. The model, software stack, accelerator, and artifact-specific source identity must match. See [P2P Metadata Exchange and Artifact Transfer](DEPLOYMENT.md#p2p-metadata-exchange).
