# RDMA Rail Selection For Refit

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Use this reference for slow or inconsistent network refits, NIC assignment,
and RDMA provisioning. Identify the actual data path first: NIXL/UCX, NCCL,
or S3. NCCL settings do not select NIXL's NICs, and UCX settings do not select
NCCL's. RDMA tuning does not accelerate an S3-only download; generator peer
reads may still use NIXL. A rail probe is not qualification of cross-node M2N.

## Inventory Inside Both Worker Environments

Collect the following on sender **and** receiver, inside the actual container
or scheduler step, before changing configuration:

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv
nvidia-smi topo -m
ls -l /dev/infiniband
ibv_devices
ibv_devinfo
rdma link show
ucx_info -d
```

Record rank, process-local CUDA index, GPU PCI address, NIC PCI/NUMA location,
HCA port, link state/rate, and the device nodes actually exposed. Missing tools
are an inventory gap, not evidence that the fabric is absent. Correlate
`/sys/class/infiniband/<nic>/device/infiniband_verbs/` with `/dev/infiniband/`:
host sysfs can list NICs the container cannot use. Device-node presence still
needs a successful verbs open/transport test to establish access.

Check allocated NIC resources as well as GPUs. Idle GPUs with all suitable NICs
held by another workload are not RDMA-ready capacity. Co-schedule GPUs and
usable NICs; an environment variable cannot grant a missing device. Confirm peer
reachability on the intended fabric rather than assuming identical `mlx5_*`
names on two hosts denote matching physical rails.

## NIXL / UCX: Select Per Worker Before Agent Creation

Use the existing ModelExpress selector when workers share a consistent view of
host topology and NIC allocation:

```bash
export MX_RDMA_NIC_PIN=auto
```

`NixlTransferManager` applies it before constructing the NIXL agent. The probe:

- considers NICs with exposed verbs devices and a readable port-1 rate, excluding
  bonded interfaces; it keeps the fastest discovered rate tier by default;
- builds a deterministic assignment from host PCI GPU identities, not only each
  process's `CUDA_VISIBLE_DEVICES` mask;
- prioritizes fewer assignments per NIC, then PCIe proximity, then NUMA locality;
  it reuses NICs when GPUs outnumber eligible NICs;
- sets `UCX_NET_DEVICES` for this process and logs the resolved mapping.

Set `MX_RDMA_NIC_PIN_MIN_RATE_GBPS` only when a known mixed-speed compute fabric
requires changing the rate filter. A matching rate is not proof of connectivity
or GPUDirect support. The probe uses port 1; validate unusual multiport layouts
explicitly.

For externally assigned per-worker maps, leave `MX_RDMA_NIC_PIN` unset/off and
set `UCX_NET_DEVICES` before the worker starts. For example, **only after
verifying this HCA/port belongs to that worker**:

```bash
export MX_RDMA_NIC_PIN=off
export UCX_NET_DEVICES=mlx5_2:1
```

An enabled MX selector can overwrite an existing `UCX_NET_DEVICES`. Do not set
both expecting the manual value to win. The alternate MX explicit-list form,
`MX_RDMA_NIC_PIN=mlx5_2:1,mlx5_3:1`, indexes by the supplied **process-local
`device_id`**, not global rank. If every isolated worker uses device 0, the same
list sends them all to its first entry. Use a worker-specific `UCX_NET_DEVICES`
map in that case. A bare single NIC value without a comma is interpreted as
auto-probe mode by the current resolver, not a one-entry explicit override.

Inspect startup warnings. If PCI visibility exposes only one GPU per container,
or host enumeration falls back to visible devices, auto selection cannot reliably
spread peer ranks. Workers with different visible NIC sets may also compute
different assignments. Supply an explicit map based on allocation/topology and
verify it across all workers; do not globally hardcode device numbers from a
previous cluster.

## NCCL: Separate Fabric And Runtime Qualification

For an already-qualified NCCL network configuration, `NCCL_IB_HCA` filters RDMA
HCAs. Exact matching avoids selecting `mlx5_10` when requesting `mlx5_1`:

```bash
# Example only: substitute the allocated, reachable HCA/port for this worker.
export NCCL_IB_HCA='=mlx5_2:1'
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
```

Set this before communicator initialization on every participant. Check the
loaded NCCL version and network plugin; a list of allowed HCAs does not force
traffic onto all of them. `NCCL_SOCKET_IFNAME` selects IP interfaces and is not
an RDMA rail assignment. See the [NCCL environment reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca)
for the installed version's supported controls.

Use transport logs and counters to identify the selected physical HCA/port and
memory path. For the built-in IB transport, `/GDRDMA` in channel logs is useful
GPU-direct evidence; log spelling differs by plugin/version. Its absence alone
is not a diagnosis. A deliberately host-staged source is DRAM, so do not require
GPU-direct source access for that NIXL path. Do not force GDR distance, GID or
cross-NIC settings to hide a provisioning problem.

The bundled M2N example's cross-node support limits still apply. Historical
native transport probes on other builds do not establish that the current
ModelExpress refit integration works across nodes.

## Distinguish Convergence From Multi-Rail Scaling

Multiple workers independently choosing the same locally attractive NIC can
collapse aggregate throughput. First compare one reader with all readers
concurrently, then test a verified per-worker spread across allocated NICs.
Locality matters, but choosing the closest NIC for every rank can be worse than
using distinct rails. Conversely, a rail across a NUMA boundary can lose GDR on
some systems. Measure the actual path; neither "always prefer same socket" nor
"cross-socket is free" is a general rule.

A past NCCL test allowed two HCAs but used only one for all fast traffic; its
apparent multi-rail result was corrected after checking channel logs. Another
NIXL workload improved by spreading readers across NICs. These are different
mechanisms, not evidence that adding HCA names always increases bandwidth.

For a bounded comparison on allocated resources:

1. Keep bytes, geometry, dtype, rank placement and source/destination memory
   kinds fixed. Record both endpoints' effective selection and transport logs.
2. Measure per-port counter deltas during the same interval, accounting for
   counter units, shared traffic and resets. Allowed device count, link rate,
   or a printed assignment is not proof of actual multi-rail traffic.
3. Separate communicator/registration cold start from warm transfers. Use
   multiple independent launches: a tight distribution inside one communicator
   can hide a bad route selected at startup.
4. Report per-rank and aggregate throughput with their own byte counts and
   intervals. Planned READ bytes can include replication and are not necessarily
   unique model bytes or measured NIC traffic. Never sum overlapping rank spans.
5. Re-run live-engine refit correctness after a mapping change. Report synthetic
   transport, live refit, and full framework pause/update/resume separately.

For NIXL diagnosis, use a short `UCX_LOG_LEVEL=DEBUG` capture and the RL timing
settings in `docs/rl/CONFIGURATION.md`. `MX_RESHARD_MIN_GBPS` is a diagnostic
threshold, not proof that a link achieved its rated speed. Revert diagnostic
verbosity for clean timing runs. If provisioning or the remote endpoint is
inaccessible, report that gap rather than claiming pinning succeeded.

## Source And Validation Pointers

- `docs/DEPLOYMENT.md`, "NIC Pinning (UCX Workaround)": deployment guidance.
- `modelexpress_client/python/modelexpress/ucx_utils.py`: discovery, assignment, explicit-list parsing and precedence.
- `modelexpress_client/python/modelexpress/nixl_transfer.py`: application before agent creation.
- `modelexpress_client/python/tests/test_ucx_utils.py`: visibility, masked devices and assignment tests; these do not establish physical bandwidth.
- [fsdp-cpu-staging.md](fsdp-cpu-staging.md): source DRAM, snapshot lifetime and memory accounting.
