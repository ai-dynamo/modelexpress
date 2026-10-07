# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment adapter for the public ModelExpress refit API; no runtime fixes."""

import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
from harness.config import load_config


class RefitWorkerExtension:
    def _record(self, phase, body):
        p = Path("/refit/benchmark")
        p.mkdir(exist_ok=True)
        body.update(rank=self.rank, phase=phase)
        self._last_record = body
        dest = p / f"{phase}-rank{self.rank}.json"
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, indent=2))
        tmp.replace(dest)
        print("HOTLOAD_BENCHMARK " + json.dumps(body), flush=True)

    def hotload_init(self, initial_version_id, source):
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

            config = load_config()
            cfg = copy.copy(self.vllm_config)
            cfg.load_config = copy.copy(cfg.load_config)
            cfg.load_config.model_loader_extra_config = {}
            live = get_model_loader(self.local_rank).tensors
            RefitWorkerExtension._record(
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
                    initial_base_version_id=initial_version_id,
                    seed_checkpoint_path="/models",
                    refit_checkpoint_dir="/refit",
                    refit_checkpoint_max_size_gb=config["refit_checkpoint_max_size_gb"],
                    endpoint_url=config["storage"]["endpoint_url"],
                    region_name=config["storage"]["region"],
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
                    server_url=config["resource_prefix"] + "-control:8000",
                    initial_serving_version_id=initial_version_id,
                    object_storage=storage,
                    source_order=(WeightSource[source],),
                )
            )
            self._hotload_source = source
            loader = get_model_loader(self.local_rank)
            rt = loader.tensors
            self._initial_weight_ptrs = {n: t.data_ptr() for n, t in rt.items()}
            RefitWorkerExtension._record(
                self,
                "init",
                {
                    "version": initial_version_id,
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
                },
            )
        except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RefitWorkerExtension._record(
                self,
                "init-failed",
                {"error": repr(e), "traceback": traceback.format_exc()},
            )

        return self._last_record

    def hotload(self, version_id):
        version = version_id
        t = time.perf_counter()
        try:
            from modelexpress_rl.version import WeightVersionRef

            torch.cuda.synchronize(self.device)
            stage_start = time.perf_counter()
            staged = self._hotload_client.stage_weight(
                version=WeightVersionRef(version)
            )
            torch.cuda.synchronize(self.device)
            stage_seconds = time.perf_counter() - stage_start
            metrics = dict(staged.metrics)
            RefitWorkerExtension._record(
                self,
                version + "-staged",
                {
                    "version": version,
                    "source": self._hotload_source,
                    "stage_seconds": stage_seconds,
                    "metrics": metrics,
                },
            )
            try:
                install_start = time.perf_counter()
                applied = self._hotload_client.apply_weight(staged)
                torch.cuda.synchronize(self.device)
                install_seconds = time.perf_counter() - install_start
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
                RefitWorkerExtension._record(
                    self,
                    version,
                    {
                        "version": version,
                        "source": self._hotload_source,
                        "stage_seconds": stage_seconds,
                        "install_seconds": install_seconds,
                        "total_seconds": stage_seconds + install_seconds,
                        "metrics": metrics,
                        "serving_version": self._hotload_client._serving_version_id,
                        "weight_addresses_preserved": True,
                    },
                )
            finally:
                staged.release()
        except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RefitWorkerExtension._record(
                self,
                version + "-failed",
                {
                    "version": version,
                    "elapsed_seconds": time.perf_counter() - t,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                },
            )

        return self._last_record

    def tensor_hashes(self, phase):
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
            RefitWorkerExtension._record(self, phase, {"tensors": results})
        except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RefitWorkerExtension._record(
                self,
                phase + "-failed",
                {"error": repr(e), "traceback": traceback.format_exc()},
            )

        return self._last_record

    def verify_host_scales(self):
        try:
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
                                "cpu": float(cpu.item()) if cpu is not None else None,
                            }
                        )
            RefitWorkerExtension._record(
                self,
                "host-scales",
                {
                    "scales": rows,
                    "enforce_eager": self.vllm_config.model_config.enforce_eager,
                },
            )
            return self._last_record
        except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RefitWorkerExtension._record(
                self,
                "host-scales-failed",
                {"error": repr(e), "traceback": traceback.format_exc()},
            )
        return self._last_record

    def verify_checkpoint(self, version_id, tensor_name):
        try:
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
            RefitWorkerExtension._record(
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
            from modelexpress_rl.inference.checkpoint_store import LocalCheckpointStore
            from safetensors import safe_open

            checkpoint = LocalCheckpointStore(
                root="/refit", model_name=os.environ["BENCH_MODEL"]
            ).checkpoint_path(version_id)
            index = json.loads(
                (checkpoint / "model.safetensors.index.json").read_text()
            )
            name = tensor_name
            with safe_open(
                str(checkpoint / index["weight_map"][name]), framework="pt"
            ) as sf:
                raw = sf.get_tensor(name)
                checkpoint_sha256 = hashlib.sha256(
                    raw.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
            RefitWorkerExtension._record(
                self,
                "checkpoint-verified",
                {
                    "version": version_id,
                    "checkpoint_tensor": name,
                    "checkpoint_sha256": checkpoint_sha256,
                    "shape": list(raw.shape),
                    "dtype": str(raw.dtype),
                    "verified": True,
                },
            )
            return self._last_record
        except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RefitWorkerExtension._record(
                self,
                "checkpoint-verified-failed",
                {
                    "version": version_id,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                },
            )

        return self._last_record

    def verify_runtime_tensor(self, version_id, tensor_name):
        try:
            from modelexpress_rl.inference.checkpoint_store import LocalCheckpointStore
            from safetensors import safe_open
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                VocabParallelEmbedding,
            )

            module_name, parameter_name = tensor_name.rsplit(".", 1)
            module = self.model_runner.get_model().get_submodule(module_name)
            if (
                not isinstance(module, VocabParallelEmbedding)
                or parameter_name != "weight"
            ):
                raise ValueError(
                    f"unsupported runtime tensor verification: {tensor_name}"
                )
            actual = module.weight
            if getattr(actual, "packed_dim", None) is not None:
                raise ValueError("packed embedding verification is unsupported")
            checkpoint = LocalCheckpointStore(
                root="/refit", model_name=os.environ["BENCH_MODEL"]
            ).checkpoint_path(version_id)
            index = json.loads(
                (checkpoint / "model.safetensors.index.json").read_text()
            )
            with safe_open(
                str(checkpoint / index["weight_map"][tensor_name]), framework="pt"
            ) as sf:
                raw = sf.get_tensor(tensor_name)
            expected = torch.nn.Parameter(
                torch.empty_like(actual, device="cpu"), requires_grad=False
            )
            expected.output_dim = getattr(actual, "output_dim", None)
            module.weight_loader(expected, raw)
            actual_cpu = actual.detach().cpu().contiguous()
            expected_cpu = expected.detach().contiguous()
            RefitWorkerExtension._record(
                self,
                "runtime-tensor-verified",
                {
                    "version": version_id,
                    "checkpoint_tensor": tensor_name,
                    "shape": list(actual.shape),
                    "dtype": str(actual.dtype),
                    "expected_sha256": hashlib.sha256(
                        expected_cpu.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    "actual_sha256": hashlib.sha256(
                        actual_cpu.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    "verified": torch.equal(actual_cpu, expected_cpu),
                },
            )
        except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
            RefitWorkerExtension._record(
                self,
                "runtime-tensor-verified-failed",
                {
                    "version": version_id,
                    "checkpoint_tensor": tensor_name,
                    "verified": False,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                },
            )
        return self._last_record
