# gRPC Streaming Benchmark Harness

Measures the throughput and per-chunk latency of the ModelExpress
`StreamModelFiles` gRPC path with disk I/O removed on both ends. The server
emits `FileChunk`s from a single pre-filled in-memory source buffer; the
client consumes the stream into byte counters and a latency histogram with
no filesystem writes.

The wire format, mpsc-backed streaming pattern, and chunk allocation shape
all match the production server in `modelexpress_server/src/services.rs`,
so a number from this harness reflects the production gRPC path's ceiling
when disk is not the bottleneck.

## What is measured

- **Throughput** in bytes/sec and gibibits/sec, end-to-end across the RPC.
- **Time-to-first-byte** from RPC start to first chunk received.
- **Per-chunk receive latency** distribution (min/mean/max/p50/p90/p99/p999).
- **Chunks/sec** and chunk count (sanity check).

A `--strict` flag enables the production client validation invariants
(per-chunk offset, total_size, sequential-write enforcement) so the
difference between strict and non-strict runs reveals validation cost.

## What is not measured

- TLS overhead (production runs plaintext).
- Disk read/write on either side (deliberately stripped).
- RDMA / NIXL P2P path (that uses a different transport entirely).
- HuggingFace / remote model fetch.

## Topologies

Two pod placements:

- **same-node**: client `podAffinity` to server's hostname. Pod-to-pod
  traffic stays on the node's CNI veth pair. Useful as the userland
  ceiling before any physical fabric is involved.
- **cross-node**: client `podAntiAffinity` from server's hostname.
  Production-shaped: traffic crosses the actual cluster fabric.

The cross-node fabric is whatever the CNI bound for the GPU worker nodes,
which is not necessarily the highest-rate NIC on the box. The harness runs
`ip route get` and `ethtool` from the client pod so the result is sized
against a known interface, and an `iperf3` baseline is captured before
each MX sweep so the MX number is interpretable as "headroom on TCP."

## Running

```bash
# Make sure your kubeconfig is set up for the target cluster

# Build and push the bench image (image tag must match the manifest)
docker build -f Dockerfile.bench -t <your-registry>/modelexpress-bench:bench-$(date +%s) .
docker push <your-registry>/modelexpress-bench:bench-$(date +%s)
# update the image tag in examples/bench/grpc-server.yaml and grpc-client-*.yaml,
# or override via kubectl set image after apply

# Run a sweep
NS=<your-namespace> \
TOTAL_BYTES=8G WARMUP_BYTES=512M \
CHUNK_SIZES="32K 256K 1M 4M 16M" \
./examples/bench/run.sh both --keep
```

Results land at `/tmp/bench-grpc-{same-node,cross-node}.jsonl` inside
the run.sh invocation environment. Each line is one JSON object with the
fields above, suitable for `jq` or pandas.

Pass `STRICT=true` to enable per-chunk validation in the client.

## Files

- `grpc-server.yaml` - bench server Deployment + headless Service. Two
  containers in the pod: `bench-server` (the gRPC source) and
  `iperf3-server` (the calibration baseline).
- `grpc-client-same-node.yaml` - client Pod with podAffinity to server.
- `grpc-client-cross-node.yaml` - client Pod with podAntiAffinity to server.
- `run.sh` - orchestration: apply, wait, fabric-discover, iperf3, sweep, optionally tear down.
