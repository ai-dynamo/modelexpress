<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Using ModelExpress with SGLang

ModelExpress can serve as the remote-instance weight loader for SGLang,
streaming weights GPU-to-GPU over RDMA between SGLang processes instead
of loading from disk on every replica. The integration is contributed by
upstream [sgl-project/sglang#23105](https://github.com/sgl-project/sglang/pull/23105),
which adds the `--modelexpress-config` flag and supports two transports:

- **Mooncake TransferEngine** — SGLang's existing remote-instance path.
- **NIXL** — selected with `transport: "nixl"`.

## 1. Install SGLang

Install SGLang either way:

- **Pull the official image** — use a recent `lmsysorg/sglang` tag from
  Docker Hub. PR #23105 was merged on `main`, so any image built from
  `main` after the merge contains it.
- **Build from `main`** — follow SGLang's official install guide at
  [docs.sglang.ai/start/install](https://docs.sglang.ai/start/install.html).

Either way, confirm the flag is present before running:

## 2. Install the ModelExpress Python client

```bash
pip install "modelexpress @ git+https://github.com/ai-dynamo/modelexpress.git#subdirectory=modelexpress_client/python"
```

## 3. Start a ModelExpress server

ModelExpress server should be reachable at
`modelexpress-server:8001`. See [`DEPLOYMENT.md`](DEPLOYMENT.md) for how
to start one (Docker, Helm, or Kubernetes).

## 4. Launch SGLang

### NIXL transport

**Seed:**

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 --tp 8 --port 30000 \
  --load-format auto \
  --modelexpress-config '{"url": "modelexpress-server:8001", "source": true, "transport": "nixl"}'
```

**Client:**

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 --tp 8 --port 30001 \
  --load-format remote_instance \
  --remote-instance-weight-loader-backend modelexpress \
  --modelexpress-config '{"url": "modelexpress-server:8001", "transport": "nixl"}'
```


## See also

- Upstream PR: [sgl-project/sglang#23105](https://github.com/sgl-project/sglang/pull/23105).
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — running the ModelExpress server, NIXL/UCX tuning, performance reference.
