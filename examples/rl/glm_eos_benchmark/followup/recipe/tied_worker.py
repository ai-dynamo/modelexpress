"""Named observation RPCs around the candidate PrimeRL MX worker."""

import hashlib
import inspect
import json
from pathlib import Path

from prime_rl.inference.vllm.worker.mx_refit import MXRefitUpdateWorker
from probe_tied_worker import verify_worker


class TiedProbeWorker(MXRefitUpdateWorker):
    def update_weights_from_path(self, weight_dir=None, version_uid=None):
        from modelexpress_rl.inference.engines.vllm import installer as api
        from vllm.model_executor.parameter import BasevLLMParameter

        original = api._materialization_is_local
        counts = {"calls": 0, "local": 0, "native_parameter_local": 0}
        select = api._select_parameter_aliases
        plans = []

        def observe_plan(*args):
            plan = select(*args)
            bad_edges = []
            for edge in plan.edges or ():
                cls = type(edge.parent)
                attrs = object.__getattribute__(edge.parent, "__dict__")
                checks = {
                    "dictionary": api._ordinary_module_dictionary(cls),
                    "plain_attributes": type(attrs) is dict,
                    "lookup": api._standard_module_access(
                        edge.parent, edge.name, set(), set()
                    ),
                }
                if not all(checks.values()):
                    bad_edges.append({"path": edge.path, "class": str(cls), **checks})
            plans.append(
                {
                    "groups": len(plan.groups),
                    "structure_present": plan.structure is not None,
                    "edges_present": plan.edges is not None,
                    "ordinary_lookup": api._ordinary_alias_lookups(plan),
                    "ordinary_writes": api._standard_parameter_writes(plan),
                    "bad_edges": bad_edges[:10],
                    "root_class": str(type(plan.root)),
                    "root_get_submodule": str(type(plan.root).get_submodule),
                }
            )
            return plan

        def observe(layer, info, *args):
            local = original(layer, info, *args)
            counts["calls"] += 1
            counts["local"] += int(local)
            if local and any(
                isinstance(value, BasevLLMParameter)
                for value in layer._parameters.values()
            ):
                counts["native_parameter_local"] += 1
            return local

        api._materialization_is_local = observe
        api._select_parameter_aliases = observe_plan
        try:
            result = super().update_weights_from_path(
                weight_dir=weight_dir, version_uid=version_uid
            )
            assert counts["native_parameter_local"] > 0, {
                "counts": counts,
                "plans": plans,
            }
            self._probe_locality = counts
            return result
        finally:
            api._materialization_is_local = original
            api._select_parameter_aliases = select

    def verify_probe_weights(self, source_path):
        import torch

        source = Path(inspect.getfile(MXRefitUpdateWorker)).resolve()
        assert source.is_relative_to("/candidate"), str(source)
        snapshot = json.loads(Path("/validation/snapshot.json").read_text())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert (
            digest == snapshot["candidate_files"][str(source.relative_to("/candidate"))]
        )
        result = verify_worker(self, source_path)
        pointers = {
            name: (
                id(parameter),
                parameter.data_ptr(),
                tuple(parameter.shape),
                tuple(parameter.stride()),
            )
            for name, parameter in self.model_runner.get_model().named_parameters(
                remove_duplicate=False
            )
        }
        previous = getattr(self, "_probe_destinations", pointers)
        assert pointers == previous, "DIRECT changed live parameter destinations"
        self._probe_destinations = pointers
        result["destinations_preserved"] = True
        host_scales = []
        for module_name, module in self.model_runner.get_model().named_modules():
            for name in ("_k_scale_cpu", "_v_scale_cpu"):
                if not hasattr(module, name):
                    continue
                value = getattr(module, name)
                qualified_name = f"{module_name}.{name}"
                assert isinstance(value, torch.Tensor), qualified_name
                assert value.device.type == "cpu", (
                    f"Native host scale moved to CUDA: {qualified_name}"
                )
                assert value.dtype == torch.float32 and value.numel() == 1, (
                    qualified_name
                )
                assert value.item() == 1.0, qualified_name
                host_scales.append(qualified_name)
        assert len(host_scales) == 4, host_scales
        previous_host_scales = getattr(self, "_probe_host_scales", host_scales)
        assert host_scales == previous_host_scales
        self._probe_host_scales = host_scales
        result["native_attention_host_scales_cpu"] = host_scales
        result["candidate_worker_source"] = {"path": str(source), "sha256": digest}
        if hasattr(self, "_probe_locality"):
            result["python_materialization_admission"] = self._probe_locality
        return result

    def begin_probe_measurement(self):
        import torch

        torch.cuda.synchronize()
        self._probe_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    def end_probe_measurement(self):
        import torch

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return {
            "allocated_before_bytes": self._probe_allocated,
            "peak_allocated_bytes": peak,
            "peak_incremental_bytes": peak - self._probe_allocated,
            "allocated_after_bytes": torch.cuda.memory_allocated(),
            "scope": "Peak includes update, weight verification and subsequent generation.",
        }

    def close_probe_generator(self):
        self._generator.close()
