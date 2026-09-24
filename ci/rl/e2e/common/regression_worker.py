"""Experiment adapter for the public ModelExpress refit API; no runtime fixes."""

import copy
import hashlib
import json
import os
import time
import traceback
import uuid
from pathlib import Path

import torch
from config import CONFIG


class RegressionWorker:
    def _record(self, phase, body):
        p = Path("/refit/benchmark")
        p.mkdir(exist_ok=True)
        body.update(rank=self.rank, phase=phase)
        if hasattr(self, "_refit_session"):
            body["refit_session"] = self._refit_session
        self._last_record = body
        dest = p / f"{phase}-rank{self.rank}.json"
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, indent=2))
        tmp.replace(dest)
        print("HOTLOAD_BENCHMARK " + json.dumps(body), flush=True)

    def _hotload_init_impl(self, run_id, source="OBJECT_STORAGE"):
        try:
            from modelexpress.engines.vllm.loader import get_model_loader
            from modelexpress_rl.inference.client import (
                ModelExpressGeneratorClient,
                ModelExpressGeneratorConfig,
            )
            from modelexpress_rl.inference.engines.vllm.context import (
                VllmGeneratorContext,
            )
            from modelexpress_rl.inference.plan import WeightSource
            from modelexpress_rl.inference.receiver import ObjectStorageGeneratorConfig
            from modelexpress_rl.object_storage import ObjectStorageType

            cfg = copy.copy(self.vllm_config)
            cfg.load_config = copy.copy(cfg.load_config)
            cfg.load_config.model_loader_extra_config = {}
            self._hotload_run = run_id
            self._refit_session = uuid.uuid4().hex
            assert cfg.model_config.enforce_eager is True, (
                "PR783 requires eager execution"
            )
            live = get_model_loader(self.local_rank).tensors
            RegressionWorker._record(
                self,
                "capability",
                {
                    "source": source,
                    "quant_config": str(cfg.quant_config),
                    "registered_runtime_tensors_available": bool(live),
                    "runtime_tensor_count": len(live),
                    "empty_runtime_tensor_count": sum(
                        t.numel() == 0 for t in live.values()
                    ),
                    "runtime_tensor_bytes": sum(
                        t.numel() * t.element_size() for t in live.values()
                    ),
                },
            )
            storage = (
                ObjectStorageGeneratorConfig(
                    storage_type=ObjectStorageType.S3,
                    initial_base_version_id=run_id + "-base",
                    seed_checkpoint_path="/models",
                    refit_checkpoint_dir="/refit",
                    refit_checkpoint_max_size_gb=CONFIG["refit_checkpoint_max_size_gb"],
                    endpoint_url=CONFIG["storage"]["endpoint_url"],
                    region_name=CONFIG["storage"]["region"],
                )
                if source == "OBJECT_STORAGE"
                else None
            )
            self._hotload_client = ModelExpressGeneratorClient.initialize(
                ModelExpressGeneratorConfig(
                    engine_context=VllmGeneratorContext(
                        model=self.model_runner.get_model(), vllm_config=cfg
                    ),
                    model_name=os.environ["BENCH_MODEL"],
                    server_url=CONFIG["resource_prefix"] + "-control:8000",
                    initial_serving_version_id=run_id + "-base",
                    object_storage=storage,
                    source_order=(WeightSource[source],),
                )
            )
            self._hotload_source = source
            loader = get_model_loader(self.local_rank)
            rt = loader.tensors
            self._initial_weight_ptrs = {n: t.data_ptr() for n, t in rt.items()}
            free, total = torch.cuda.mem_get_info(self.device)
            RegressionWorker._record(
                self,
                "init",
                {
                    "version": run_id + "-base",
                    "source": source,
                    "quant_config": str(cfg.quant_config),
                    "registered_runtime_tensors_available": bool(live),
                    "runtime_tensor_bytes": sum(
                        t.numel() * t.element_size() for t in rt.values()
                    ),
                    "runtime_tensor_count": len(rt),
                    "empty_runtime_tensor_count": sum(
                        t.numel() == 0 for t in rt.values()
                    ),
                    "gpu_free_bytes": free,
                    "gpu_total_bytes": total,
                },
            )
        except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RegressionWorker._record(
                self,
                "init-failed",
                {"error": repr(e), "traceback": traceback.format_exc()},
            )

    def _hotload_impl(self, weight_path, allocation_tracing=True):
        version = weight_path
        t = time.perf_counter()
        baseline_allocated = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            from modelexpress_rl.version import WeightVersionRef

            staged = self._hotload_client.stage_weight(
                version=WeightVersionRef(version)
            )
            at = time.perf_counter()
            metrics = dict(staged.metrics)
            RegressionWorker._record(
                self,
                version + "-staged",
                {
                    "version": version,
                    "source": self._hotload_source,
                    "stage_seconds": at - t,
                    "metrics": metrics,
                },
            )
            try:
                import diagnostic_observer

                with diagnostic_observer.allocation_trace(
                    self.rank, enabled=allocation_tracing
                ):
                    applied = self._hotload_client.apply_weight(staged)
                torch.cuda.synchronize(self.device)
                from modelexpress.tensor_utils import collect_module_tensors

                current_ptrs = {
                    n: t.data_ptr()
                    for n, t in collect_module_tensors(
                        self.model_runner.get_model()
                    ).items()
                }
                assert current_ptrs == self._initial_weight_ptrs, (
                    "registered weight addresses changed during refit"
                )
                metrics.update(dict(staged.metrics))
                if isinstance(applied, dict):
                    metrics.update(applied)
                RegressionWorker._record(
                    self,
                    version,
                    {
                        "version": version,
                        "source": self._hotload_source,
                        "stage_seconds": at - t,
                        "install_seconds": time.perf_counter() - at,
                        "allocation_tracing": allocation_tracing,
                        "total_seconds": time.perf_counter() - t,
                        "metrics": metrics,
                        "serving_version": self._hotload_client._serving_version_id,
                        "weight_addresses_preserved": True,
                        "baseline_allocated_bytes": baseline_allocated,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                            self.device
                        ),
                        "gpu_free_bytes": torch.cuda.mem_get_info(self.device)[0],
                    },
                )
            finally:
                staged.release()
        except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RegressionWorker._record(
                self,
                version + "-failed",
                {
                    "version": version,
                    "elapsed_seconds": time.perf_counter() - t,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                },
            )

    def hotload_verify(self, phase):
        try:
            from modelexpress.engines.vllm.loader import get_model_loader

            # Hash actual registered GPU tensors, outside the transfer/install timing.
            rt = get_model_loader(self.local_rank).tensors
            results = {}
            for name, t in sorted(rt.items()):
                if t.numel() == 0:
                    continue
                digest = hashlib.sha256()
                flat = t.detach().reshape(-1)
                for offset in range(0, flat.numel(), 8 * 1024**2):
                    digest.update(
                        flat[offset : offset + 8 * 1024**2]
                        .cpu()
                        .contiguous()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    )
                results[name] = {
                    "sha256": digest.hexdigest(),
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                }
            RegressionWorker._record(self, phase, {"tensors": results})
        except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RegressionWorker._record(
                self,
                phase + "-failed",
                {"error": repr(e), "traceback": traceback.format_exc()},
            )

    def _hotload_verify_checkpoint_impl(self, version_id):
        try:
            if version_id == "host-scales":
                rows = []
                for name, module in self.model_runner.get_model().named_modules():
                    for key in ["q", "k", "v"]:
                        tensor = getattr(module, "_" + key + "_scale", None)
                        host = getattr(module, "_" + key + "_scale_float", None)
                        if tensor is not None and host is not None:
                            cpu = getattr(module, "_" + key + "_scale_cpu", None)
                            rows.append(
                                {
                                    "module": name,
                                    "scale": key,
                                    "gpu": float(tensor.item()),
                                    "host": float(host),
                                    "cpu": float(cpu.item())
                                    if cpu is not None
                                    else None,
                                }
                            )
                caches = []
                for name, module in self.model_runner.get_model().named_modules():
                    impl = getattr(module, "impl", None)
                    values = {
                        k: getattr(impl, k, None)
                        for k in ["bmm1_scale", "bmm2_scale", "o_sf_scale"]
                        if hasattr(impl, k)
                    }
                    if values:
                        values = {
                            k: float(v) if v is not None else None
                            for k, v in values.items()
                        }
                        caches.append({"module": name, "values": values})
                RegressionWorker._record(
                    self,
                    "host-scales",
                    {
                        "scales": rows,
                        "flashinfer_caches": caches,
                        "enforce_eager": self.vllm_config.model_config.enforce_eager,
                    },
                )
                return self._last_record
            if version_id.startswith("hashes:"):
                RegressionWorker.hotload_verify(self, version_id.split(":", 1)[1])
                return self._last_record
            from modelexpress.tensor_utils import collect_module_tensors

            actual_ptrs = {
                n: t.data_ptr()
                for n, t in collect_module_tensors(
                    self.model_runner.get_model()
                ).items()
            }
            changed = [
                n
                for n, p in self._initial_weight_ptrs.items()
                if actual_ptrs.get(n) != p
            ]
            extra = sorted(set(actual_ptrs) - set(self._initial_weight_ptrs))
            RegressionWorker._record(
                self,
                "addresses-verified",
                {
                    "version": version_id,
                    "changed": changed,
                    "extra": extra,
                    "verified": not changed and not extra,
                },
            )
            assert not changed and not extra, {"changed": changed, "extra": extra}
            if version_id.startswith("addresses:"):
                return
            from modelexpress_rl.inference.checkpoint_store import LocalCheckpointStore
            from safetensors import safe_open

            if CONFIG["embedding"]:
                model = self.model_runner.get_model()
                matches = [
                    (n, t)
                    for n, t in model.named_parameters()
                    if n.endswith("embed_tokens.weight")
                ]
                assert len(matches) == 1, [n for n, t in matches]
                runtime_name, actual = matches[0]
                checkpoint = LocalCheckpointStore(
                    root="/refit", model_name=os.environ["BENCH_MODEL"]
                ).checkpoint_path(version_id)
                index = json.loads(
                    (checkpoint / "model.safetensors.index.json").read_text()
                )
                name = CONFIG["embedding"]
                with safe_open(
                    str(checkpoint / index["weight_map"][name]), framework="pt"
                ) as sf:
                    raw = sf.get_tensor(name)
                    start = self.rank * actual.shape[0]
                    if self.rank == 0:
                        RegressionWorker._record(
                            self,
                            "checkpoint-source",
                            {
                                "version": version_id,
                                "tensor": name,
                                "sha256": hashlib.sha256(
                                    raw.contiguous().view(torch.uint8).numpy()
                                ).hexdigest(),
                            },
                        )
                    expected = raw[start : start + actual.shape[0]].to(self.device)
                publisher_hashes = {}
                if self.rank == 0:
                    import model_adapter

                    for tensor_name in [name, *model_adapter.extra_tensor_names(index)]:
                        with safe_open(
                            str(checkpoint / index["weight_map"][tensor_name]),
                            framework="pt",
                        ) as sf:
                            value = sf.get_tensor(tensor_name)
                            publisher_hashes[tensor_name] = hashlib.sha256(
                                value.contiguous().view(torch.uint8).numpy()
                            ).hexdigest()
                assert torch.equal(actual, expected), (
                    "GPU embedding differs from reconstructed delta checkpoint"
                )

                def digest(x):
                    return hashlib.sha256(
                        x.detach()
                        .cpu()
                        .contiguous()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    ).hexdigest()

                RegressionWorker._record(
                    self,
                    "checkpoint-verified",
                    {
                        "version": version_id,
                        "tensor": runtime_name,
                        "shape": list(actual.shape),
                        "sha256": digest(actual),
                        "expected_sha256": digest(expected),
                        "full_checkpoint_sha256": publisher_hashes.get(name),
                        "publisher_hashes": publisher_hashes,
                        "verified": True,
                    },
                )
                return self._last_record
        except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RegressionWorker._record(
                self,
                "checkpoint-verified-failed",
                {
                    "version": version_id,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                },
            )

        return self._last_record

    def hotload_init(self, run_id, source="OBJECT_STORAGE"):
        import importlib

        import regression_worker

        importlib.reload(regression_worker).RegressionWorker._hotload_init_impl(
            self, run_id, source
        )
        return self._last_record

    def hotload(self, weight_path, allocation_tracing=True):
        import importlib

        import regression_worker

        importlib.reload(regression_worker).RegressionWorker._hotload_impl(
            self, weight_path, allocation_tracing
        )
        return self._last_record

    def hotload_verify_checkpoint(self, version_id):
        import importlib

        import regression_worker

        return importlib.reload(
            regression_worker
        ).RegressionWorker._hotload_verify_checkpoint_impl(self, version_id)

    def diagnostic_start(self):
        import diagnostic_observer

        self._diagnostic_observer = diagnostic_observer.start(self)
        return {"rank": self.rank, "diagnostic_observer_started": True}

    def derived_verify(self, phase):
        import model_adapter

        try:
            body = model_adapter.verify_derived(self, phase)
            RegressionWorker._record(self, "derived-" + phase, body)
        except Exception as error:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RegressionWorker._record(
                self,
                "derived-" + phase + "-failed",
                {
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
        return self._last_record

    def checkpoint_layout(self, version_id):
        from modelexpress_rl.inference.checkpoint_store import LocalCheckpointStore

        try:
            store = LocalCheckpointStore(root="/refit", model_name=CONFIG["model"])
            checkpoint = store.checkpoint_path(version_id)
            index = json.loads(
                (checkpoint / "model.safetensors.index.json").read_text()
            )
            files = {}
            for name in sorted(set(index["weight_map"].values())):
                stat = (checkpoint / name).stat()
                files[name] = {
                    "inode": stat.st_ino,
                    "device": stat.st_dev,
                    "bytes": stat.st_size,
                }
            assert files, "Empty checkpoint index"
            RegressionWorker._record(
                self,
                "layout-" + version_id,
                {
                    "version": version_id,
                    "path": str(checkpoint),
                    "files": files,
                },
            )
        except Exception as error:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RegressionWorker._record(
                self,
                "layout-failed",
                {
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
        return self._last_record
