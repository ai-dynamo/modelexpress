# Standalone Runtime Configuration

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Worker setup when you run vLLM, SGLang, or TensorRT-LLM directly, for example
as a plain Kubernetes Deployment. For workers managed by NVIDIA Dynamo, see
`references/dynamo.md`. Authoritative sources:
`docs/integrations/runtimes/`, `docs/SGLANG.md`, and the manifests under
`examples/p2p_transfer_k8s/client/`.

## Common To Every Runtime

- Every replica runs the same command. With no ready source, a replica loads
  from storage and publishes itself; later replicas find it and receive
  weights over P2P.
- Give the pod RDMA devices and the `IPC_LOCK` capability. The examples request
  `rdma/ib` and add `IPC_LOCK` under `securityContext.capabilities`.
- The examples set `MX_RDMA_NIC_PIN=auto` for per-rank NIC pinning, a UCX
  workaround.
- Fabric: the default NIXL backend (UCX) targets InfiniBand and RoCE, and can
  fall back to TCP without warning. On AWS EFA, set `MX_NIXL_BACKEND=LIBFABRIC`
  and request `vpc.amazonaws.com/efa` instead. See
  `examples/p2p_transfer_k8s/client/vllm/aws_efa/README.md`.
- The model path must resolve to the configuration and tokenizer files on every
  replica.

## vLLM

vLLM 0.23.0 and newer recognize the load format natively; the image still needs
the MX Python client. On older vLLM, the ModelExpress plugin must load: leave
`VLLM_PLUGINS` unset, or include `modelexpress` in it, since it is an
allowlist.

```bash
export MX_SERVER_ADDRESS=<service>:8001
vllm serve <model> --load-format modelexpress
```

| Starting point | Manifest |
|---|---|
| Single node | `examples/p2p_transfer_k8s/client/vllm/vllm-single-node.yaml` |
| Multi-node | `examples/p2p_transfer_k8s/client/vllm/vllm-multi-node.yaml` |
| AWS EFA | `examples/p2p_transfer_k8s/client/vllm/aws_efa/vllm-aws-efa.yaml` |

The example Dockerfile (`examples/p2p_transfer_k8s/client/vllm/Dockerfile`)
installs the client on a vLLM image.

## SGLang

SGLang delegates its `remote_instance` loader to MX. The image needs SGLang's
upstream ModelExpress delegation hook and the MX client, installed with
`--no-deps` so the base image keeps its own dependency stack. `docs/SGLANG.md`
names a known-good image.

```bash
export MX_SERVER_ADDRESS=<service>:8001
python -m sglang.launch_server --model-path <model> --tp <n> \
  --load-format remote_instance \
  --remote-instance-weight-loader-backend modelexpress \
  --modelexpress-config '{"transport":"nixl"}'
```

`--modelexpress-config` takes two keys:

- `transport`: `nixl`, or `transfer_engine` for Mooncake TransferEngine. The
  `transfer_engine` transport needs a Mooncake-enabled image and does not log
  `RDMA transfer complete`.
- `url`: overrides the server address for this process. Prefer
  `MX_SERVER_ADDRESS` so the address stays in the environment.

Keep `--model-path` on the model identity or a local path. An object-storage
URI there bypasses MX; use `MX_MODEL_URI` for ModelStreamer instead.

| Starting point | Manifest |
|---|---|
| NIXL | `examples/p2p_transfer_k8s/client/sglang/sglang-single-node-p2p.yaml` |
| Mooncake TransferEngine | `examples/p2p_transfer_k8s/client/sglang/sglang-single-node-transfer-engine.yaml` |

## TensorRT-LLM

TensorRT-LLM uses its native MX checkpoint format, configured in the file
passed to `trtllm-serve --config`:

```yaml
checkpoint_format: MX
mx_config:
  server_url: <service>:8001
```

The checked-in example (`examples/p2p_transfer_k8s/client/trtllm/`) is beta: it
covers `LlamaForCausalLM` at TP=4 with the PyTorch backend and pins the
TensorRT-LLM image. Only P2P and the native loader are eligible; the other MX
strategies are not wired into this adapter. Qualify the image, client, and
fabric together before relying on it.
