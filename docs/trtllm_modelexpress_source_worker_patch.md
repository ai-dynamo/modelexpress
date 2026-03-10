# TRT-LLM patch: ModelExpress source publish from workers

When using TensorRT-LLM with the **PyTorch backend and TP>1**, the orchestrator process holds a `GenerationExecutorProxy`. The actual torch model lives in the **executor worker processes** (spawned via MPI). The orchestrator cannot access `model_engine.model` — `_get_torch_model()` will always fail.

ModelExpress source publish must therefore run **inside each executor worker** after the engine is set up and weights are loaded.

## How it works

1. **Before `LLM()` init**, the dynamo engine sets `MODEL_EXPRESS_SOURCE=1`, `MODEL_EXPRESS_URL`, and `MODEL_NAME` in the environment.
2. MPI workers inherit these env vars when spawned.
3. Each worker runs `worker_main()` → creates `GenerationExecutorWorker` (which calls `setup_engine()`, loading weights).
4. **After** the worker is created, `worker_main()` checks `MODEL_EXPRESS_SOURCE` and calls `publish_from_worker(worker)`.
5. `publish_from_worker` accesses `worker.engine.model_engine.model` (the real torch model), registers params with NIXL, and publishes this rank's metadata to the MX server.
6. The orchestrator then polls the MX server to confirm all ranks published before marking the source as ready.

## Apply the patch

In `tensorrt_llm/executor/worker.py`, after the worker is created (line ~303, after the `except` block) and before `with worker:`, add:

```python
    # ModelExpress source: publish this rank's model params via NIXL
    if os.environ.get("MODEL_EXPRESS_SOURCE"):
        try:
            from modelexpress.trtllm_live_transfer import publish_from_worker
            publish_from_worker(worker)
        except Exception as e:
            logger.warning("ModelExpress publish_from_worker failed on rank %d: %s", mpi_rank(), e)
```

## Env vars

Set by the orchestrator (dynamo engine) before `LLM()` init:

- `MODEL_EXPRESS_SOURCE=1` — triggers worker-side publish.
- `MODEL_EXPRESS_URL` — MX server address (e.g. `modelexpress-server:8001`).
- `MODEL_NAME` — model name for coordination.

## Sequencing guarantee

- Source workers publish during startup, before the orchestrator's `LLM()` constructor returns.
- The orchestrator then verifies all ranks published by polling the MX server.
- Targets wait for source metadata via `wait_for_ready` — they will not start transfer until the source is fully published.
- No target can race ahead of a source that hasn't published.
