<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prometheus metrics

ModelExpress exposes Prometheus metrics from two places: the Rust server, on by
default, and the Python client, opt-in. This page covers how to scrape both,
what the pipeline guarantees, and the two operational choices it leaves to the
deployment.

This documents what ships today: the exposition path and `mx_build_info`, the
family that later ones join against. Request/backend coverage, the load and
transfer timing tiers, and the dashboard surface come later and are not here yet.

---

## Quick start

### Server

Metrics are on by default on port **9401**, separate from the gRPC port.

```bash
curl -s http://<server>:9401/metrics | head
```

With the Helm chart, nothing to do — `metrics.enabled` defaults to `true` and
the chart generates the scrape annotations:

```yaml
metrics:
  enabled: true
  port: 9401
  podAnnotations: true   # emit prometheus.io/{scrape,port,path}
  service: false         # publish the port on the Service as well
```

To turn the listener off, `--set metrics.enabled=false`, or set
`MODEL_EXPRESS_SERVER_METRICS_PORT=0`.

### Client

Opt-in, and it needs three environment variables in the **pod manifest** — not
in code:

```yaml
env:
  - {name: MX_METRICS_ENABLED, value: "1"}
  - {name: PROMETHEUS_MULTIPROC_DIR, value: /tmp/mx-metrics}
  - {name: MX_METRICS_PORT, value: "9402"}
ports:
  - {name: mx-metrics, containerPort: 9402}
```

That is it. Every rank calls `enable_metrics()` from its engine loader; one of
them binds the port and serves the merged union of all of them.

No volume is required. The ranks are processes inside one container, so an
ordinary container path is already shared between them, and a container restart
recreates it empty — which is what a stale `.db` file would otherwise survive.
Mount a memory-backed `emptyDir` at the same path if you want tmpfs and a size
cap; then clear it at the container entrypoint with
`python -m modelexpress.metrics --reset`, because an `emptyDir` *does* persist
across container restarts.

A worked TP=8 example is
[`examples/p2p_transfer_k8s/client/vllm/vllm-single-node-metrics.yaml`](../examples/p2p_transfer_k8s/client/vllm/vllm-single-node-metrics.yaml)
— a metrics-enabled sibling of the stock single-node manifest, so the diff
between the two is exactly what metrics cost you.

Without the directory nothing breaks — the client falls back to
one-rank-per-endpoint and says so in a warning — but only one of your ranks is
then represented, which for TP=8 means losing seven eighths of the pod.

---

## Why one endpoint per pod

A ModelExpress pod runs **one worker process per GPU**, and N is a customer
variable: this repository's own configurations span TP=1 through TP=8. Anything
that assumes one metrics-producing process per scrape target loses N-1 ranks.

So the client uses `prometheus_client` **multiprocess mode**. Each rank mmaps
per-PID files into a pod-local directory; one process binds the port and its
`MultiProcessCollector` serves the union of every rank, including ranks that have
already exited. Losing the bind is the *normal* case, not a failure, and mmap
writes are durable at increment time, so no exit hook is in the path — an
OOM-killed rank's counters still show up.

`benchmarks/metrics_pipeline.py` reconstructs the previous mechanism and measures
what it lost, if you want the numbers.

---

## The rules that are not style

Each of these has one failure mode, and it is silent in every case.

**`PROMETHEUS_MULTIPROC_DIR` goes in the pod manifest, never in Python.**
`prometheus_client.values.get_value_class()` latches at module import. An
in-process assignment lands after `import vllm` and produces zero `.db` files
with no error. `enable()` detects the latched-wrong state and logs an error
naming the cause, rather than serving an empty endpoint that returns 200.

**Never wipe the directory from worker code.** Ranks start staggered, so a late
rank's wipe unlinks an early rank's already-mmapped files; that rank keeps
writing to an unlinked inode and disappears from every later scrape. Wipe at the
container entrypoint only, with `python -m modelexpress.metrics --reset` — and
only if you mounted a volume, since an `emptyDir` survives container restarts
within a pod while a plain container path does not.

**`mx_build_info` is a Gauge set to 1, never an `Info`.** Under multiprocess
mode an `Info` writes no file, exposes nothing, and raises nothing. It would
pass its own health check while being invisible and silently emptying every
`group_left` join.

**Gauges need an explicit `multiprocess_mode`, and `livesum` is not safe here.**
The default, `all`, appends a `pid` label and is unbounded. `livesum` and
`liveall` depend on `mark_process_dead`, which only runs on a clean exit, so an
in-flight value wedges forever when a rank is OOM-killed. In-flight quantities
belong in `_started_total` / `_finished_total` counter pairs differenced in
PromQL, which self-heals.

**Push and scrape are mutually exclusive**, enforced in code. Running both
double-counts every series.

---

## Two choices the deployment has to make

### 1. Scrape cost versus counter monotonicity

`MultiProcessCollector` re-reads every mmap file in Python, holding the GIL,
inside the process driving the engine scheduler. Files for dead PIDs are never
reclaimed, so a pod that recycles workers accumulates them and scrape cost grows.

Reaping them bounds the cost but makes merged counters *decrease*, which
Prometheus reads as a reset and `rate()` then mis-accounts.

**ModelExpress does not reap.** Counters stay monotonic; bound pod lifetime
instead. On an idle H200 node the curve runs ~0.1 ms at one file set to ~2.4 ms at
256 — inside Prometheus's 10 s timeout, but measured without an engine competing
for the GIL. If you recycle workers aggressively, measure your own pod:

```bash
python benchmarks/metrics/metrics_pipeline.py --only scrape-cost --max-file-sets 512
```

### 2. Sharing the directory with the engine's exporter

vLLM sets and manages `PROMETHEUS_MULTIPROC_DIR` itself, and the value class is
process-global, so a genuinely separate directory is not achievable from inside
the engine process.

**The decision is: share the directory, take a distinct port, never wipe it from
code.** ModelExpress families are namespaced `mx_*` and cannot collide with the
engine's. A scrape of the ModelExpress port will also carry the engine's series;
that is a superset, not a conflict. If you would rather scrape one port, point
both at the engine's.

---

## Families

`mx_build_info` is the join target for everything process-constant. Both halves
of the deployment export it, distinguished by `component`.

### Server

| Family | Type | Labels |
| --- | --- | --- |
| `mx_build_info` | Gauge (= 1) | `component="server"`, `version`, `backend`, `scheme` |

### Client

| Family | Type | Labels |
| --- | --- | --- |
| `mx_build_info` | Gauge (= 1) | `component="client"`, `version`, `scheme` |
| `mx_p2p_source_selections_total` | Counter | `policy`, `scheme` |
| `mx_p2p_source_attempts_total` | Counter | `policy`, `scheme`, `result` |
| `mx_p2p_metadata_lookup_failures_total` | Counter | `policy`, `scheme` |
| `mx_p2p_list_sources_total` | Counter | `policy`, `scheme`, `result` |
| `mx_p2p_candidates` | Histogram | `policy`, `scheme`, `stage` |
| `mx_p2p_source_selection_seconds` | Histogram | `policy`, `scheme` |
| `mx_p2p_transfer_seconds` | Histogram | `policy`, `scheme`, `outcome` |

Two changes to the client families are breaking for existing dashboards:

- **`source_worker_id` is off by default** on `mx_p2p_source_selections_total`.
  It is `uuid4().hex[:8]`, minted fresh per process, so its label domain grows
  with process count over time rather than with cluster size — unbounded series
  growth on any long-lived Prometheus watching a fleet with pod churn. Set
  `MX_METRICS_SOURCE_ID_LABEL=1` to restore it for a benchmark run; see
  [Selection skew](#selection-skew-and-the-source_worker_id-label).
- **`mx_p2p_list_sources_total` is new**, with `result` in
  `{ok, empty, error}`. Two things were previously indistinguishable: a
  metadata-backend outage and a healthy cluster that has published no peers.
  Both presented as an absence. Relatedly, `mx_p2p_candidates{stage="listed"}`
  can now observe **zero** — the selection funnel returned before recording
  anything when no instances came back, so the one bucket that separates "no
  peers published" from "peers listed but all filtered out" was unreachable.

### Selection skew and the `source_worker_id` label

*Is one source peer being picked disproportionately?* is the central question
when comparing selection policies, and the obvious way to answer it does not
survive a production fleet — the id is a per-process uuid, so its label domain
grows with process count over time.

**On a fleet, leave it off.** Note that a dashboard built as `sum by
(source_worker_id) (...)` does not error once the label is gone: PromQL groups
every sample under `source_worker_id=""`, so N lines become one and the panel
reads as perfectly balanced. Group by `policy` instead. Pinned matchers return no
data, and were already unreliable — the id was reminted on every process start.

**On a benchmark run, set `MX_METRICS_SOURCE_ID_LABEL=1`**, point it at a
Prometheus you are willing to throw away, and keep the run short enough that
process churn does not outgrow it. Then:

```promql
sum by (source_worker_id) (mx_p2p_source_selections_total)
  / ignoring(source_worker_id) group_left sum(mx_p2p_source_selections_total)
```

The call site always passes the peer id and the collector drops it unless the
variable is set, so the cardinality decision lives in one place. The label set is
latched when the family is built, so flipping the variable mid-process does
nothing — `prometheus_client` fixes a family's labels at construction.

For skew on a long-lived fleet the bounded form is a client-side gauge, which is
follow-on work. The same data is in the structured logs regardless:
`rdma_strategy` emits `source_worker_id=` on every source attempt.

### Useful queries

```promql
# Is the metadata backend down, or has nobody published weights?
sum by (result) (rate(mx_p2p_list_sources_total[5m]))

# Compare two benchmark runs. `scheme` is already a label on every client
# family, so this needs no join.
sum by (scheme) (rate(mx_p2p_transfer_seconds_sum[5m]))
  / sum by (scheme) (rate(mx_p2p_transfer_seconds_count[5m]))

# Attach a constant that is only on build_info (version), not on the family itself.
mx_p2p_list_sources_total
  * on (instance) group_left(version) mx_build_info{component="client"}

# Version skew across the fleet.
count by (version, component) (mx_build_info)

# Pods that intended to export metrics but are not being scraped.
up{job="modelexpress"} == 0
```

`scheme` is on `mx_build_info` **and** on every client `mx_p2p_*` family, so a
`group_left(scheme)` join against `mx_build_info` copies a value the left side
already has. Use `scheme` directly. On the server it is on `mx_build_info` only.
Consolidating the client families onto the join is a follow-on taxonomy change.

---

## Environment variables

### Server

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_EXPRESS_SERVER_METRICS_PORT` | `9401` | `/metrics` port. `0` disables the listener. |
| `MX_METRICS_SCHEME` | `""` | Benchmark run label on `mx_build_info`. |

`MODEL_EXPRESS_SERVER_METRICS_PORT` is read **only** through the clap
`--metrics-port` override. The layered config loader builds its environment
source as `Environment::with_prefix("MODEL_EXPRESS").separator("_")`, so this
name resolves to the key path `server.metrics.port`, matches no field, and is
dropped by serde without a warning. In a config file the field is
`server.metrics_port`.

### Client

| Variable | Default | Meaning |
| --- | --- | --- |
| `MX_METRICS_ENABLED` | `0` | Master switch. |
| `PROMETHEUS_MULTIPROC_DIR` | unset | Shared per-pod directory. Required for multi-rank pods. |
| `MX_METRICS_PORT` | unset | `/metrics` port. One rank binds; it serves them all. |
| `MX_METRICS_PUSHGATEWAY` | unset | Batch-pod escape hatch. Mutually exclusive with `MX_METRICS_PORT`. |
| `MX_METRICS_SCHEME` | `""` | Benchmark run label. Carried on `mx_build_info` and on every `mx_p2p_*` family — see the family table above. |
| `MX_METRICS_BIND_RETRY_SECS` | `15` | How often a rank that lost the bind re-attempts it, so endpoint ownership migrates when the winner exits. |
| `MX_METRICS_SOURCE_ID_LABEL` | `0` | Restore the per-peer `source_worker_id` label. **Benchmark runs only** — the id is a per-process uuid, so its label domain grows with process count over time. See [Selection skew](#selection-skew-and-the-source_worker_id-label). |

The client needs `prometheus-client`, which is the `metrics` extra:

```bash
pip install "modelexpress[metrics]"
```

In practice the engine images already provide it — vLLM, SGLang and TensorRT-LLM
all depend on it for their own metrics — so the extra matters only for an image
built without one of those. If it is missing the collector catches the
`ImportError` and disables itself, so the symptom is `up == 0` with nothing else
to go on.

---

## Diagnosing an endpoint that returns nothing

| Symptom | Likely cause |
| --- | --- |
| `up == 0` on server pods | Scrape is aimed at the gRPC port. tonic is HTTP/2 only; use `metrics.port`. |
| `up == 0` on worker pods | `MX_METRICS_ENABLED` is unset, or `prometheus-client` is missing from the image (it is installed explicitly now, but a custom image may not). |
| 200 with no `mx_*` series | `PROMETHEUS_MULTIPROC_DIR` was set after `prometheus_client` was imported. Check the logs for the "latched its single-process value class" error. |
| Only one rank's numbers | `PROMETHEUS_MULTIPROC_DIR` is not set, so there is nothing to merge. |
| A rank vanishes mid-run | Something wiped the directory after ranks started. Only the entrypoint may do that. |
| Scrapes time out | Dead-PID file sets have accumulated. See "Scrape cost" above. |
| `mx_build_info` present, nothing else | Working as intended: the exporter came up and this run recorded no events. |

---

## Verification

```bash
# Client pipeline, per issue.
cd modelexpress_client/python
pytest tests/test_metrics.py tests/test_metrics_deployment.py -v

# Selection funnel and the ListSources outcome counter.
pytest tests/test_source_selection.py -v

# Server registry, build info, and the config wiring.
cargo test --package modelexpress-server metrics

# The listener itself, scraped over real HTTP/1.1. Gated behind the feature, so
# the command above does NOT run it.
cargo test --package modelexpress-server --features integration-tests --test in_process_server

# Before/after numbers.
python benchmarks/metrics/metrics_pipeline.py --ranks 8
```

`tests/test_metrics.py` runs the merged-endpoint check at both TP=2 and TP=8 by
forking real ranks, SIGKILLing them, and asserting the merged total. That is the
exit criterion for this phase: one port exposing merged series from every rank,
surviving a hard kill.
