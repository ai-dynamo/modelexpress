# Mooncake artifact reuse (vLLM and SGLang)

This example installs file-backed JIT/cache artifacts from a shared Mooncake
store on a later run, even when the original inference process is gone. Model
weights still use the engine's normal ModelExpress loading strategy; selecting
the Mooncake **artifact** backend does not select a model-weight transport.

The runtime image needs ModelExpress, the chosen engine, and a Mooncake Python
package exposing `mooncake.store`. For SGLang, use a build that supports
`--modelexpress-config` and `remote_instance`, with NIXL available for the
selected model-weight loader; see [SGLang setup](../../docs/SGLANG.md).
The GPU nodes must be able to reach the same Mooncake store. Supply cluster
credentials through your deployment's secret mechanism, never in this script.

## Configure the store

Choose one configuration method, using your own endpoints:

```bash
# Preferred when the cluster provides a Mooncake JSON config:
export MX_MOONCAKE_CONFIG_PATH=/secure/path/to/mooncake.json

# Alternatively, set ModelExpress-scoped native settings:
# export MX_MC_METADATA_ADDR='etcd-1:2379;etcd-2:2379'
# export MX_MC_MASTER_SERVER='etcd://etcd-1:2379;etcd-2:2379'
# export MX_MC_PROTOCOL=rdma
# export MX_MC_DEVICE_NAME=mlx5_0

# Set once and keep identical for cold and hit runs. Use a fresh namespace
# when testing from an empty shared cache.
export MX_ARTIFACT_MOONCAKE_NAMESPACE="modelexpress/examples/${USER}/my-test/"
export MX_EXAMPLE_MODEL=/path/to/model
export MX_EXAMPLE_TP_SIZE=2
```

The script sets `MX_ARTIFACT_TRANSFER=1`, `MX_ARTIFACT_BACKEND=mooncake`, and
run-specific cache/bundle roots. It defaults `MX_P2P_METADATA` to `0` because
the Mooncake artifact backend does not need a ModelExpress server. If you also
want P2P **weight** loading, configure its metadata/server separately and set
`MX_P2P_METADATA=1`. Backend selection is exclusive: a Mooncake miss or outage
does not fall back to NIXL artifact transfer.

## Cold run, then cache-hit run

Use `vllm` or `sglang` for both runs. Engine-specific flags may be appended
after the engine name (for example `--trust-remote-code`). Launch from an image
with that engine's executable and matching model dependencies:

```bash
export MX_EXAMPLE_RUN_DIR=/tmp/mx-mooncake-cold
export MX_EXAMPLE_PORT=8000
mkdir -p "$MX_EXAMPLE_RUN_DIR"
bash examples/mooncake_artifact_reuse/run.sh vllm \
  --trust-remote-code >"$MX_EXAMPLE_RUN_DIR/server.log" 2>&1
```

In another terminal, wait for `GET /health` to return 200 **and** for
`artifact publish complete` messages in the cold log; publication may finish
after readiness. Stop the first process, then use a different local cache root
and the same Mooncake namespace, model, and compatible software/GPU stack:

```bash
export MX_EXAMPLE_RUN_DIR=/tmp/mx-mooncake-hit
export MX_EXAMPLE_PORT=8001
mkdir -p "$MX_EXAMPLE_RUN_DIR"
bash examples/mooncake_artifact_reuse/run.sh vllm \
  --trust-remote-code >"$MX_EXAMPLE_RUN_DIR/server.log" 2>&1
```

For SGLang, replace `vllm` with `sglang` in both commands. Its ModelExpress
loader is selected through `remote_instance` with NIXL for model weights;
Mooncake remains the separate artifact backend. Do not switch SGLang's
`modelexpress-config` to `transfer_engine` for this artifact example: that
model-weight mode does not install ModelExpress cache artifacts.

Check the hit run's log for `Mooncake artifact fetch complete` and
`artifact install complete`, then verify `/health`. A cold run should show
publication instead. Not every cache type is produced by every model, so a
missing optional artifact is not necessarily a failure. Keep the namespace
stable only for compatible runs; use a new one for independent experiments.

See [configuration](../../docs/CONFIGURATION.md#artifact-transfer) for store
settings and [deployment guidance](../../docs/DEPLOYMENT.md#artifact-transfer-backends)
for the supported cache roots.
