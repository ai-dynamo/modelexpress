# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded MX-owned Megatron-Bridge and FSDP canonical capture."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import sys
import time
import types
import weakref
from contextlib import nullcontext

import pytest
import torch

import modelexpress.refit.source.fsdp as fsdp_source_module
import modelexpress.refit.source.megatron_bridge as megatron_source_module
from modelexpress.refit.source import (
    CanonicalSourceError,
    CanonicalTensorSpec,
    FsdpHfBucketConfig,
    MegatronBridgeHfBucketConfig,
    for_each_fsdp_hf_bucket,
    for_each_megatron_hf_bucket,
)
from modelexpress.refit.source.base import canonical_tensor_name
from modelexpress.refit.source.canonical import CanonicalFormatIdentity


_DEFAULT_SCHEMA = (
    CanonicalTensorSpec("a.weight", torch.float32, (2,)),
    CanonicalTensorSpec("b.weight", torch.float32, (1,)),
)


def _distributed_deadline():
    return {
        "deadline_monotonic": time.monotonic() + 60.0,
        "abort_collectives": lambda: None,
    }


def _bucket_signature(buckets):
    return [
        [
            (
                name,
                str(tensor.dtype),
                tuple(tensor.shape),
                tensor.contiguous().view(torch.uint8).numpy().tobytes(),
            )
            for name, tensor in bucket
        ]
        for bucket in buckets
    ]


@pytest.mark.parametrize(
    ("wrapped", "canonical"),
    [
        ("_orig_mod.module.weight", "weight"),
        ("module._orig_mod.module.weight", "weight"),
        ("layer.base_layer.base_layer.weight", "layer.weight"),
    ],
)
def test_canonical_tensor_name_normalization_is_idempotent(wrapped, canonical):
    normalized = canonical_tensor_name(wrapped)
    assert normalized == canonical
    assert canonical_tensor_name(normalized) == canonical


class _FakeBridge:
    def __init__(self) -> None:
        self.calls = []
        self.yielded = []

    def get_conversion_tasks(self, _model):
        return [
            types.SimpleNamespace(
                vp_stage=0,
                param_name="a.weight",
                param_weight=torch.ones(2),
            ),
            types.SimpleNamespace(
                vp_stage=0,
                param_name="b.weight",
                param_weight=torch.ones(1),
            ),
        ]

    def export_hf_weights(self, model, **kwargs):
        self.calls.append((model, kwargs))
        [task] = kwargs["conversion_tasks"]
        outputs = {
            "a.weight": (("a.weight", torch.tensor([1.0, 2.0])),),
            "b.weight": (("b.weight", torch.tensor([3.0])),),
            "decoder.weight": (
                ("a.weight", torch.tensor([1.0, 2.0])),
                ("b.weight", torch.tensor([3.0])),
            ),
        }[task.param_name]
        for name, tensor in outputs:
            if kwargs["cpu"]:
                self.yielded.append(name)
            yield name, tensor, f"native.{name}"


class _FakeFsdpModel:
    def state_dict(self):
        return {
            "module.b.weight": torch.tensor([3.0]),
            "module.a.weight": torch.tensor([1.0, 2.0]),
        }


def test_megatron_and_fsdp_emit_identical_bounded_canonical_buckets_on_rank_zero():
    bridge = _FakeBridge()
    megatron_buckets = []
    fsdp_buckets = []

    for_each_megatron_hf_bucket(
        [object()],
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=8,
            rank=lambda: 0,
            model_context=lambda: nullcontext(),
            canonical_schema=_DEFAULT_SCHEMA,
        ),
        megatron_buckets.append,
    )
    for_each_fsdp_hf_bucket(
        _FakeFsdpModel(),
        FsdpHfBucketConfig(
            bucket_bytes=8, rank=lambda: 0, canonical_schema=_DEFAULT_SCHEMA
        ),
        fsdp_buckets.append,
    )

    assert _bucket_signature(megatron_buckets) == _bucket_signature(fsdp_buckets)
    assert [[name for name, _ in bucket] for bucket in megatron_buckets] == [
        ["a.weight"],
        ["b.weight"],
    ]
    assert all(
        sum(tensor.numel() * tensor.element_size() for _, tensor in bucket) <= 8
        for bucket in megatron_buckets
    )
    assert all(
        tensor.device.type == "cpu"
        for bucket in megatron_buckets
        for _, tensor in bucket
    )
    assert all(
        tensor.is_contiguous() for bucket in fsdp_buckets for _, tensor in bucket
    )


def test_only_rank_zero_consumes_buckets_but_every_rank_drains_its_source():
    bridge = _FakeBridge()
    consumed = []

    for_each_megatron_hf_bucket(
        [object()],
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=16,
            rank=lambda: 1,
            model_context=lambda: nullcontext(),
            canonical_schema=_DEFAULT_SCHEMA,
        ),
        consumed.append,
    )

    assert bridge.yielded == ["a.weight", "b.weight"]
    assert consumed == []


class _FakeDistributedTensor:
    def __init__(self, tensor: torch.Tensor, calls: list[str], name: str) -> None:
        self._tensor = tensor
        self._calls = calls
        self._name = name

    def full_tensor(self) -> torch.Tensor:
        self._calls.append(self._name)
        return self._tensor

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def shape(self):
        return self._tensor.shape


def test_fsdp_dtensor_descriptor_includes_exact_mesh_and_structured_placements(
    monkeypatch,
):
    from torch.distributed.tensor import Replicate, Shard

    class FakeDeviceMesh:
        device_type = "cuda"
        mesh = torch.tensor([[0, 2], [1, 3]])
        mesh_dim_names = ("data_parallel", "tensor_parallel")
        ndim = 2
        shape = (2, 2)

    class FakeDTensor:
        device_mesh = FakeDeviceMesh()
        placements = (Shard(0), Replicate())

    monkeypatch.setattr(torch.distributed.tensor, "DTensor", FakeDTensor)

    descriptor = fsdp_source_module._materializer_descriptor(
        FakeDTensor(), "module.weight", FsdpHfBucketConfig()
    )

    assert descriptor == (
        "dtensor",
        "cuda",
        ((0, 2), (1, 3)),
        ("data_parallel", "tensor_parallel"),
        (("shard", 0), ("replicate",)),
    )


def test_fsdp_rejects_dtensor_mesh_outside_metadata_boundary_before_full_tensor(
    monkeypatch,
):
    from torch.distributed.tensor import Shard

    calls = []

    class FakeDeviceMesh:
        device_type = "cuda"
        mesh = torch.tensor([0, 2])
        mesh_dim_names = ("data_parallel",)
        ndim = 1
        shape = (2,)

    class FakeDTensor:
        device_mesh = FakeDeviceMesh()
        placements = (Shard(0),)
        dtype = torch.float32
        shape = (1,)

        def full_tensor(self):
            calls.append("weight")
            return torch.ones(1)

    metadata_group = object()
    monkeypatch.setattr(torch.distributed.tensor, "DTensor", FakeDTensor)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda _group: [0, 1]
    )

    def all_gather_object(outputs, contribution, group=None):
        assert group is metadata_group
        outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(
        CanonicalSourceError, match="DeviceMesh ranks.*metadata boundary"
    ):
        for_each_fsdp_hf_bucket(
            object(),
            FsdpHfBucketConfig(
                bucket_bytes=16,
                rank=lambda: 0,
                state_dict_getter=lambda _model: {"module.weight": FakeDTensor()},
                metadata_group=metadata_group,
                canonical_schema=(CanonicalTensorSpec("weight", torch.float32, (1,)),),
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert calls == []


def test_fsdp_distributed_duck_materializer_requires_explicit_stable_topology(
    monkeypatch,
):
    calls = []
    state = {"module.weight": _FakeDistributedTensor(torch.ones(1), calls, "weight")}
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)

    def all_gather_object(outputs, contribution, group=None):
        del group
        outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(
        CanonicalSourceError, match="explicit stable topology descriptor"
    ):
        for_each_fsdp_hf_bucket(
            object(),
            FsdpHfBucketConfig(
                bucket_bytes=16,
                rank=lambda: 0,
                state_dict_getter=lambda _model: state,
                canonical_schema=(CanonicalTensorSpec("weight", torch.float32, (1,)),),
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert calls == []


def test_fsdp_distributed_accepts_configured_duck_materializer_topology(monkeypatch):
    calls = []
    state = {"module.weight": _FakeDistributedTensor(torch.ones(1), calls, "weight")}
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    def all_gather_object(outputs, contribution, group=None):
        del group
        outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    for_each_fsdp_hf_bucket(
        object(),
        FsdpHfBucketConfig(
            bucket_bytes=16,
            rank=lambda: 0,
            state_dict_getter=lambda _model: state,
            materializer_topology={
                "module.weight": ("test-sharded", ("ranks", (0, 1)))
            },
            canonical_schema=(CanonicalTensorSpec("weight", torch.float32, (1,)),),
            **_distributed_deadline(),
        ),
        lambda _bucket: None,
    )

    assert calls == ["weight"]


def test_fsdp_releases_materialized_outputs_before_the_next_materialization():
    references = []
    calls = []

    class EphemeralDistributedTensor:
        dtype = torch.float32
        shape = (1,)

        def __init__(self, name, value):
            self._name = name
            self._value = value

        def full_tensor(self):
            if references:
                gc.collect()
                assert references[-1]() is None
            tensor = torch.tensor([self._value])
            references.append(weakref.ref(tensor))
            calls.append(self._name)
            return tensor

    for_each_fsdp_hf_bucket(
        object(),
        FsdpHfBucketConfig(
            bucket_bytes=4,
            rank=lambda: 0,
            state_dict_getter=lambda _model: {
                "module.a": EphemeralDistributedTensor("a", 1.0),
                "module.b": EphemeralDistributedTensor("b", 2.0),
            },
            canonical_schema=(
                CanonicalTensorSpec("a", torch.float32, (1,)),
                CanonicalTensorSpec("b", torch.float32, (1,)),
            ),
        ),
        lambda _bucket: None,
    )

    gc.collect()
    assert calls == ["a", "b"]
    assert all(reference() is None for reference in references)


def test_fsdp_materializes_each_distributed_tensor_in_deterministic_order_on_all_ranks():
    calls = []
    state = {
        "module.b": _FakeDistributedTensor(torch.tensor([2]), calls, "b"),
        "module.a": _FakeDistributedTensor(torch.tensor([1]), calls, "a"),
    }
    consumed = []

    for_each_fsdp_hf_bucket(
        object(),
        FsdpHfBucketConfig(
            bucket_bytes=16,
            rank=lambda: 1,
            state_dict_getter=lambda _model: state,
            canonical_schema=(
                CanonicalTensorSpec("a", torch.int64, (1,)),
                CanonicalTensorSpec("b", torch.int64, (1,)),
            ),
        ),
        consumed.append,
    )

    assert calls == ["a", "b"]
    assert consumed == []


def test_callback_failure_stops_before_the_next_megatron_collective_unit():
    bridge = _FakeBridge()

    def fail(_bucket):
        raise OSError("object store unavailable")

    with pytest.raises(CanonicalSourceError, match="object store unavailable"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                bucket_bytes=8,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                canonical_schema=_DEFAULT_SCHEMA,
            ),
            fail,
        )

    assert bridge.yielded == ["a.weight"]


def test_callback_failure_stops_before_the_next_fsdp_tensor_collective():
    calls = []
    state = {
        "module.a": _FakeDistributedTensor(torch.tensor([1]), calls, "a"),
        "module.b": _FakeDistributedTensor(torch.tensor([2]), calls, "b"),
    }

    def fail(_bucket):
        raise OSError("object store unavailable")

    with pytest.raises(CanonicalSourceError, match="object store unavailable"):
        for_each_fsdp_hf_bucket(
            object(),
            FsdpHfBucketConfig(
                bucket_bytes=16,
                rank=lambda: 0,
                state_dict_getter=lambda _model: state,
                canonical_schema=(
                    CanonicalTensorSpec("a", torch.int64, (1,)),
                    CanonicalTensorSpec("b", torch.int64, (1,)),
                ),
            ),
            fail,
        )

    assert calls == ["a"]


def test_source_rejects_duplicate_names_and_single_tensors_over_the_bound():
    class DuplicateBridge:
        def get_conversion_tasks(self, _model):
            return [
                types.SimpleNamespace(
                    vp_stage=0,
                    param_name="weight",
                    param_weight=torch.ones(1),
                )
            ]

        def export_hf_weights(self, *_args, **_kwargs):
            yield "weight", torch.ones(1)
            yield "weight", torch.ones(1)

    with pytest.raises(CanonicalSourceError, match="more than once"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=DuplicateBridge(),
                bucket_bytes=8,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                canonical_schema=(CanonicalTensorSpec("weight", torch.float32, (1,)),),
            ),
            lambda _bucket: None,
        )

    class OversizedModel:
        def state_dict(self):
            return {"weight": torch.ones(3)}

    with pytest.raises(CanonicalSourceError, match="exceeds bucket_bytes"):
        for_each_fsdp_hf_bucket(
            OversizedModel(),
            FsdpHfBucketConfig(
                bucket_bytes=8,
                rank=lambda: 0,
                canonical_schema=(CanonicalTensorSpec("weight", torch.float32, (3,)),),
            ),
            lambda _bucket: None,
        )


@dataclasses.dataclass(frozen=True)
class _ConversionTask:
    vp_stage: int
    param_name: str
    param_weight: torch.Tensor
    mapping: object | None = None


class _TaskBridge(_FakeBridge):
    def get_conversion_tasks(self, _model):
        return [
            _ConversionTask(
                0,
                "decoder.weight",
                torch.ones(3),
                types.SimpleNamespace(
                    is_grouped_export=False,
                    hf_param={"a": "a.weight", "b": "b.weight"},
                ),
            )
        ]


def test_megatron_bridge_substitutes_the_authoritative_model_handle_weights():
    bridge = _TaskBridge()
    replacement = torch.tensor([9.0, 10.0, 11.0])

    for_each_megatron_hf_bucket(
        [object()],
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=16,
            rank=lambda: 0,
            model_context=lambda: nullcontext(),
            weights_getter=lambda: {
                "module.module.vp_stages.0.decoder.weight": replacement
            },
            canonical_schema=_DEFAULT_SCHEMA,
        ),
        lambda _bucket: None,
    )

    tasks = bridge.calls[0][1]["conversion_tasks"]
    assert torch.equal(tasks[0].param_weight, replacement)


@pytest.mark.parametrize(
    ("replacement_factory", "detail"),
    [
        pytest.param(lambda: torch.ones(2), "shape", id="shape"),
        pytest.param(
            lambda: torch.ones(3, dtype=torch.float64),
            "dtype",
            id="dtype",
        ),
        pytest.param(
            lambda: torch.ones(3, device="meta"),
            "device",
            id="device",
        ),
    ],
)
def test_megatron_rejects_incompatible_authoritative_replacement_before_export(
    replacement_factory,
    detail,
):
    bridge = _TaskBridge()

    with pytest.raises(CanonicalSourceError, match=rf"replacement.*{detail}"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                bucket_bytes=32,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                weights_getter=lambda: {
                    "vp_stages.0.decoder.weight": replacement_factory()
                },
                canonical_schema=_DEFAULT_SCHEMA,
            ),
            lambda _bucket: None,
        )

    assert bridge.calls == []


def test_megatron_defers_authoritative_weight_staging_to_each_bounded_unit(
    monkeypatch,
):
    class Bridge(_FakeBridge):
        def get_conversion_tasks(self, _model):
            return [
                _ConversionTask(0, "a.weight", torch.ones(2)),
                _ConversionTask(0, "b.weight", torch.ones(1)),
            ]

    bridge = Bridge()
    staged = []
    replace_task_weight = megatron_source_module._replace_task_weight

    def tracked_replace(task, weight):
        staged.append(task.param_name)
        return replace_task_weight(task, weight)

    monkeypatch.setattr(megatron_source_module, "_replace_task_weight", tracked_replace)
    export_hf_weights = bridge.export_hf_weights

    def checked_export(model, **kwargs):
        [task] = kwargs["conversion_tasks"]
        expected = [task.param_name]
        if task.param_name == "b.weight":
            expected.insert(0, "a.weight")
        assert staged == expected
        yield from export_hf_weights(model, **kwargs)

    bridge.export_hf_weights = checked_export

    for_each_megatron_hf_bucket(
        [object()],
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=8,
            rank=lambda: 0,
            model_context=lambda: nullcontext(),
            weights_getter=lambda: {
                "vp_stages.0.a.weight": torch.ones(2),
                "vp_stages.0.b.weight": torch.ones(1),
            },
            canonical_schema=_DEFAULT_SCHEMA,
        ),
        lambda _bucket: None,
    )

    assert staged == ["a.weight", "b.weight"]


def test_megatron_default_context_applies_and_removes_bridge_compatibility_patch(
    monkeypatch,
):
    calls = []

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace()
            self.share_embeddings_and_output_weights = True

        def _maintain_float32_expert_bias(self):
            calls.append("maintained")

    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    utils = types.ModuleType("megatron.core.utils")
    utils.unwrap_model = lambda model: model
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)
    model = Model()

    for_each_megatron_hf_bucket(
        [model],
        MegatronBridgeHfBucketConfig(
            bridge=_FakeBridge(),
            bucket_bytes=16,
            rank=lambda: 0,
            canonical_schema=_DEFAULT_SCHEMA,
        ),
        lambda _bucket: None,
    )

    assert calls == ["maintained"]
    assert not hasattr(model.config, "share_embeddings_and_output_weights")


def test_megatron_preflight_uses_exactly_one_collective_when_a_peer_fails(monkeypatch):
    gathers = []
    bridge = _FakeBridge()
    get_task_calls = []
    get_conversion_tasks = bridge.get_conversion_tasks

    def tracked_get_conversion_tasks(model):
        get_task_calls.append(model)
        return get_conversion_tasks(model)

    bridge.get_conversion_tasks = tracked_get_conversion_tasks

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)

    def all_gather_object(outputs, contribution, group=None):
        del group
        gathers.append(contribution)
        peer = list(contribution)
        peer[0] = "peer preflight failed"
        outputs[:] = [contribution, tuple(peer)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(CanonicalSourceError, match="peer preflight failed"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                bucket_bytes=16,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                canonical_schema=_DEFAULT_SCHEMA,
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert len(gathers) == 1
    assert get_task_calls == []
    assert bridge.calls == []


def test_megatron_builds_collective_task_plan_before_weight_substitution_error(
    monkeypatch,
):
    class Bridge(_TaskBridge):
        def __init__(self):
            super().__init__()
            self.task_calls = 0

        def get_conversion_tasks(self, model):
            self.task_calls += 1
            return super().get_conversion_tasks(model)

    bridge = Bridge()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)

    def all_gather_object(outputs, contribution, group=None):
        del group
        outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(CanonicalSourceError, match="weight substitution failed"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                bucket_bytes=16,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                weights_getter=lambda: (_ for _ in ()).throw(
                    RuntimeError("weight substitution failed")
                ),
                canonical_schema=_DEFAULT_SCHEMA,
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert bridge.task_calls == 1


def test_fsdp_peer_local_preflight_failure_precedes_state_dict_collection(
    monkeypatch,
):
    state_dict_calls = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)

    def all_gather_object(outputs, contribution, group=None):
        del group
        peer = list(contribution)
        peer[0] = "peer schema preflight failed"
        outputs[:] = [contribution, tuple(peer)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(CanonicalSourceError, match="peer schema preflight failed"):
        for_each_fsdp_hf_bucket(
            object(),
            FsdpHfBucketConfig(
                bucket_bytes=16,
                rank=lambda: 0,
                state_dict_getter=lambda _model: state_dict_calls.append(True) or {},
                canonical_schema=(CanonicalTensorSpec("a", torch.int64, (1,)),),
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert state_dict_calls == []


@pytest.mark.parametrize("source", ["megatron", "fsdp"])
def test_distributed_sources_require_an_absolute_abortable_deadline_before_collectives(
    source, monkeypatch
):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected collective")
        ),
    )

    with pytest.raises(CanonicalSourceError, match="absolute.*deadline"):
        if source == "megatron":
            for_each_megatron_hf_bucket(
                [object()],
                MegatronBridgeHfBucketConfig(
                    bridge=_FakeBridge(),
                    bucket_bytes=16,
                    rank=lambda: 0,
                    model_context=lambda: nullcontext(),
                    canonical_schema=_DEFAULT_SCHEMA,
                ),
                lambda _bucket: None,
            )
        else:
            for_each_fsdp_hf_bucket(
                object(),
                FsdpHfBucketConfig(
                    bucket_bytes=16,
                    rank=lambda: 0,
                    state_dict_getter=lambda _model: {
                        "a.weight": torch.tensor([1.0, 2.0]),
                        "b.weight": torch.tensor([3.0]),
                    },
                    canonical_schema=_DEFAULT_SCHEMA,
                ),
                lambda _bucket: None,
            )


@pytest.mark.parametrize("source", ["megatron", "fsdp"])
def test_expired_source_deadline_invokes_abort_before_materialization(source):
    aborted = []
    common = {
        "deadline_monotonic": time.monotonic() - 1.0,
        "abort_collectives": lambda: aborted.append(True),
    }

    with pytest.raises(CanonicalSourceError, match="deadline expired"):
        if source == "megatron":
            bridge = _FakeBridge()
            for_each_megatron_hf_bucket(
                [object()],
                MegatronBridgeHfBucketConfig(
                    bridge=bridge,
                    bucket_bytes=16,
                    rank=lambda: 0,
                    model_context=lambda: nullcontext(),
                    canonical_schema=_DEFAULT_SCHEMA,
                    **common,
                ),
                lambda _bucket: None,
            )
        else:
            calls = []
            state = {
                "a.weight": _FakeDistributedTensor(
                    torch.tensor([1.0, 2.0]), calls, "a"
                ),
                "b.weight": _FakeDistributedTensor(torch.tensor([3.0]), calls, "b"),
            }
            for_each_fsdp_hf_bucket(
                object(),
                FsdpHfBucketConfig(
                    bucket_bytes=16,
                    rank=lambda: 0,
                    state_dict_getter=lambda _model: state,
                    canonical_schema=_DEFAULT_SCHEMA,
                    **common,
                ),
                lambda _bucket: None,
            )

    assert aborted == [True]


def test_fsdp_rejects_cross_rank_plan_mismatch_before_tensor_collectives(monkeypatch):
    calls = []
    state = {
        "module.a": _FakeDistributedTensor(torch.tensor([1]), calls, "a"),
    }

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)

    def all_gather_object(outputs, contribution, group=None):
        del group
        local_error, representation, local_plan = contribution
        outputs[:] = [
            contribution,
            (
                local_error,
                representation,
                (("different.weight", "torch.int64", (1,), True),),
            ),
        ]
        assert local_plan == (("a", "torch.int64", (1,), True),)

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(CanonicalSourceError, match="capture plan differs"):
        for_each_fsdp_hf_bucket(
            object(),
            FsdpHfBucketConfig(
                bucket_bytes=16,
                rank=lambda: 0,
                state_dict_getter=lambda _model: state,
                canonical_schema=(CanonicalTensorSpec("a", torch.int64, (1,)),),
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert calls == []


def test_fsdp_rejects_oversized_dtensor_from_metadata_before_full_tensor():
    calls = []
    state = {
        "module.large": _FakeDistributedTensor(torch.ones(3), calls, "large"),
    }

    with pytest.raises(CanonicalSourceError, match="exceeds bucket_bytes"):
        for_each_fsdp_hf_bucket(
            object(),
            FsdpHfBucketConfig(
                bucket_bytes=8,
                rank=lambda: 0,
                state_dict_getter=lambda _model: state,
                canonical_schema=(CanonicalTensorSpec("large", torch.float32, (3,)),),
            ),
            lambda _bucket: None,
        )

    assert calls == []


def test_fsdp_applies_weight_bridge_conversion_and_sync_dtype_inside_mx(monkeypatch):
    transformers = types.ModuleType("transformers")
    core_loading = types.ModuleType("transformers.core_model_loading")

    def revert_weight_conversion(_model, state):
        [(name, tensor)] = state.items()
        assert name == "model.layers.0.mlp.experts.gate_up_proj"
        return {
            "b.weight": tensor[1].contiguous(),
            "a.weight": tensor[0].contiguous(),
        }

    core_loading.revert_weight_conversion = revert_weight_conversion
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.core_model_loading", core_loading)

    class Model:
        config = types.SimpleNamespace(model_type="qwen3_moe")
        _fsdp_sync_dtypes = {"model.layers.0.mlp.experts.gate_up_proj": torch.bfloat16}

    source = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    buckets = []
    for_each_fsdp_hf_bucket(
        Model(),
        FsdpHfBucketConfig(
            bucket_bytes=32,
            rank=lambda: 0,
            state_dict_getter=lambda _model: {
                "model.layers.0.mlp.experts.gate_up_proj": source
            },
            canonical_schema=(
                CanonicalTensorSpec("a.weight", torch.bfloat16, (2, 2)),
                CanonicalTensorSpec("b.weight", torch.bfloat16, (2, 2)),
            ),
        ),
        buckets.append,
    )

    assert [name for bucket in buckets for name, _tensor in bucket] == [
        "a.weight",
        "b.weight",
    ]
    assert all(
        tensor.dtype is torch.bfloat16 for bucket in buckets for _name, tensor in bucket
    )


def test_fsdp_weight_bridge_fails_closed_when_transform_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers.core_model_loading", None)

    class Model:
        config = types.SimpleNamespace(model_type="qwen3_moe")

    name = "model.layers.0.mlp.experts.gate_up_proj"
    source = torch.ones((2, 2, 2))
    with pytest.raises(
        CanonicalSourceError, match="WeightBridge conversion is required"
    ):
        for_each_fsdp_hf_bucket(
            Model(),
            FsdpHfBucketConfig(
                bucket_bytes=64,
                rank=lambda: 0,
                state_dict_getter=lambda _model: {name: source},
                canonical_schema=(CanonicalTensorSpec(name, torch.float32, (2, 2, 2)),),
            ),
            lambda _bucket: None,
        )


def test_fsdp_weight_bridge_passthrough_does_not_call_save_conversion(monkeypatch):
    transformers = types.ModuleType("transformers")
    core_loading = types.ModuleType("transformers.core_model_loading")
    core_loading.revert_weight_conversion = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("passthrough tensor must not enter WeightBridge conversion"))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.core_model_loading", core_loading)

    class Model:
        config = types.SimpleNamespace(model_type="qwen3_moe")

    buckets = []
    for_each_fsdp_hf_bucket(
        Model(),
        FsdpHfBucketConfig(
            bucket_bytes=16,
            rank=lambda: 0,
            state_dict_getter=lambda _model: {"model.norm.weight": torch.ones(2)},
            canonical_schema=(
                CanonicalTensorSpec("model.norm.weight", torch.float32, (2,)),
            ),
        ),
        buckets.append,
    )

    assert [name for bucket in buckets for name, _tensor in bucket] == [
        "model.norm.weight"
    ]


def test_fsdp_fails_when_replica_canonical_content_differs(monkeypatch):
    contributions = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    def all_gather_object(outputs, contribution, group=None):
        del group
        contributions.append(contribution)
        if len(contribution) == 3 or contribution[1] is None:
            outputs[:] = [contribution, contribution]
        else:
            error, digest = contribution
            outputs[:] = [contribution, (error, f"{digest}-different")]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(CanonicalSourceError, match="canonical content differs"):
        for_each_fsdp_hf_bucket(
            object(),
            FsdpHfBucketConfig(
                bucket_bytes=16,
                rank=lambda: 0,
                state_dict_getter=lambda _model: {"module.weight": torch.tensor([1.0])},
                canonical_schema=(CanonicalTensorSpec("weight", torch.float32, (1,)),),
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert len(contributions) == 4


def test_megatron_preflight_enumerates_the_global_hf_output_plan(monkeypatch):
    contributions = []
    bridge = _TaskBridge()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    def all_gather_object(outputs, contribution, group=None):
        del group
        contributions.append(contribution)
        outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    for_each_megatron_hf_bucket(
        [object()],
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=16,
            rank=lambda: 0,
            model_context=lambda: nullcontext(),
            canonical_schema=_DEFAULT_SCHEMA,
            **_distributed_deadline(),
        ),
        lambda _bucket: None,
    )

    assert contributions[0] == (
        None,
        (CanonicalFormatIdentity(), ("gloo", (0, 1))),
        (
            ("a.weight", "torch.float32", (2,), True),
            ("b.weight", "torch.float32", (1,), True),
        ),
    )
    assert contributions[1] == (
        None,
        CanonicalFormatIdentity(),
        (
            ("a.weight", "torch.float32", (2,), True),
            ("b.weight", "torch.float32", (1,), True),
        ),
        (
            (
                0,
                "decoder.weight",
                "types.SimpleNamespace",
                None,
                (
                    "mapping",
                    (("a", "a.weight"), ("b", "b.weight")),
                ),
                ("a.weight", "b.weight"),
                None,
            ),
        ),
        (
            (
                ("torch.float32", (3,), "cpu"),
                (
                    ("pipeline", "gloo", (0, 1)),
                    ("tensor", "gloo", (0, 1)),
                ),
            ),
        ),
    )


def test_megatron_schedule_fingerprints_hf_mapping_key_semantics():
    plan = (
        ("a.weight", "torch.float32", (1,), True),
        ("b.weight", "torch.float32", (1,), True),
    )
    sizes = (("a.weight", 4), ("b.weight", 4))

    def schedule(hf_param):
        task = types.SimpleNamespace(
            vp_stage=0,
            param_name="decoder.weight",
            global_param_name="decoder.weight",
            pp_rank=0,
            param_weight=torch.ones(2),
            mapping=types.SimpleNamespace(
                is_grouped_export=False,
                hf_param=hf_param,
            ),
        )
        return megatron_source_module._conversion_units([task], {}, 16, plan, sizes)[1]

    assert schedule({"left": "a.weight", "right": "b.weight"}) != schedule(
        {"left": "b.weight", "right": "a.weight"}
    )


def test_megatron_rejects_owner_metadata_mismatch_before_bridge_export(monkeypatch):
    bridge = _TaskBridge()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    def all_gather_object(outputs, contribution, group=None):
        del group
        if len(contribution) == 5:
            error, identity, plan, schedule, ownership = contribution
            entry = ownership[0]
            if len(entry) == 2 and isinstance(entry[1], tuple):
                peer_entry = (("torch.float64", (3,), "cpu"), entry[1])
            else:
                peer_entry = ("torch.float64", (3,), "cpu")
            outputs[:] = [
                contribution,
                (error, identity, plan, schedule, (peer_entry,)),
            ]
        else:
            outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(CanonicalSourceError, match="owner metadata"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                bucket_bytes=16,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                canonical_schema=_DEFAULT_SCHEMA,
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert bridge.calls == []


def test_megatron_rejects_inconsistent_mapping_group_membership_before_export(
    monkeypatch,
):
    bridge = _TaskBridge()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    def all_gather_object(outputs, contribution, group=None):
        del group
        if len(contribution) == 5:
            error, identity, plan, schedule, ownership = contribution
            entry = ownership[0]
            metadata = entry[0] if len(entry) == 2 else entry
            bad_topology = (
                ("pipeline", "gloo", (0,)),
                ("tensor", "gloo", (0, 1)),
            )
            outputs[:] = [
                contribution,
                (
                    error,
                    identity,
                    plan,
                    schedule,
                    ((metadata, bad_topology),),
                ),
            ]
        else:
            outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(CanonicalSourceError, match="collective topology"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                bucket_bytes=16,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                canonical_schema=_DEFAULT_SCHEMA,
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert bridge.calls == []


def test_megatron_accepts_global_schedule_with_rank_local_owner_placeholders(
    monkeypatch,
):
    class PlaceholderBridge:
        def __init__(self):
            self.calls = []

        def get_conversion_tasks(self, _model):
            return [
                types.SimpleNamespace(
                    vp_stage=None,
                    param_name="a.weight",
                    global_param_name="a.weight",
                    pp_rank=1,
                    param_weight=None,
                ),
                types.SimpleNamespace(
                    vp_stage=0,
                    param_name="b.weight",
                    global_param_name="b.weight",
                    pp_rank=0,
                    param_weight=torch.ones(1),
                ),
            ]

        def export_hf_weights(self, _model, **kwargs):
            [task] = kwargs["conversion_tasks"]
            self.calls.append(task.global_param_name)
            values = {
                "a.weight": torch.tensor([1.0, 2.0]),
                "b.weight": torch.tensor([3.0]),
            }
            yield task.global_param_name, values[task.global_param_name]

    bridge = PlaceholderBridge()
    buckets = []
    gathers = 0
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    def all_gather_object(outputs, contribution, group=None):
        nonlocal gathers
        del group
        gathers += 1
        if gathers == 1:
            assert len(contribution) == 3
            outputs[:] = [contribution, contribution]
        elif gathers == 2:
            assert len(contribution) == 5
            error, identity, canonical_plan, schedule, ownership = contribution
            assert error is None
            topology = (
                ("pipeline", "gloo", (0, 1)),
                ("tensor", "gloo", (0, 1)),
            )
            assert ownership == (
                (None, topology),
                (("torch.float32", (1,), "cpu"), topology),
            )
            outputs[:] = [
                contribution,
                (
                    None,
                    identity,
                    canonical_plan,
                    schedule,
                    (
                        (("torch.float32", (2,), "cuda"), topology),
                        (None, topology),
                    ),
                ),
            ]
        else:
            outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    for_each_megatron_hf_bucket(
        [object()],
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=16,
            rank=lambda: 0,
            model_context=lambda: nullcontext(),
            canonical_schema=_DEFAULT_SCHEMA,
            **_distributed_deadline(),
        ),
        buckets.append,
    )

    assert bridge.calls == ["a.weight", "b.weight"]
    assert [name for bucket in buckets for name, _tensor in bucket] == [
        "a.weight",
        "b.weight",
    ]


def test_megatron_keeps_grouped_bridge_export_tasks_in_one_bounded_unit():
    mapping = types.SimpleNamespace(is_grouped_export=True, group_key="experts.weight")

    class GroupedBridge:
        def __init__(self):
            self.units = []

        def get_conversion_tasks(self, _model):
            return [
                types.SimpleNamespace(
                    vp_stage=0,
                    param_name=f"experts.{expert}.weight",
                    global_param_name=f"experts.{expert}.weight",
                    pp_rank=0,
                    param_weight=torch.tensor([float(expert)]),
                    mapping=mapping,
                )
                for expert in range(2)
            ]

        def export_hf_weights(self, _model, **kwargs):
            tasks = tuple(kwargs["conversion_tasks"])
            self.units.append(tuple(task.global_param_name for task in tasks))
            if len(tasks) == 2:
                yield (
                    "experts.weight",
                    torch.cat(tuple(task.param_weight for task in tasks)),
                )

    bridge = GroupedBridge()
    buckets = []
    for_each_megatron_hf_bucket(
        [object()],
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=8,
            rank=lambda: 0,
            model_context=lambda: nullcontext(),
            canonical_schema=(
                CanonicalTensorSpec("experts.weight", torch.float32, (2,)),
            ),
        ),
        buckets.append,
    )

    assert bridge.units == [("experts.0.weight", "experts.1.weight")]
    assert [name for bucket in buckets for name, _tensor in bucket] == [
        "experts.weight"
    ]


def test_megatron_groups_interleaved_bridge_tasks_by_their_global_group_key():
    mappings = {
        key: types.SimpleNamespace(is_grouped_export=True, group_key=key)
        for key in ("experts.gate", "experts.down")
    }

    class InterleavedGroupedBridge:
        def __init__(self):
            self.units = []

        def get_conversion_tasks(self, _model):
            return [
                types.SimpleNamespace(
                    vp_stage=0,
                    param_name=f"experts.{expert}.{projection}",
                    global_param_name=f"experts.{expert}.{projection}",
                    pp_rank=0,
                    param_weight=torch.tensor([float(expert)]),
                    mapping=mappings[f"experts.{projection}"],
                )
                for expert in range(2)
                for projection in ("down", "gate")
            ]

        def export_hf_weights(self, _model, **kwargs):
            tasks = tuple(kwargs["conversion_tasks"])
            group_key = tasks[0].mapping.group_key
            self.units.append(tuple(task.global_param_name for task in tasks))
            yield group_key, torch.cat(tuple(task.param_weight for task in tasks))

    bridge = InterleavedGroupedBridge()
    buckets = []
    for_each_megatron_hf_bucket(
        [object()],
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=8,
            rank=lambda: 0,
            model_context=lambda: nullcontext(),
            canonical_schema=(
                CanonicalTensorSpec("experts.gate", torch.float32, (2,)),
                CanonicalTensorSpec("experts.down", torch.float32, (2,)),
            ),
        ),
        buckets.append,
    )

    assert bridge.units == [
        ("experts.0.gate", "experts.1.gate"),
        ("experts.0.down", "experts.1.down"),
    ]
    assert [name for bucket in buckets for name, _tensor in bucket] == [
        "experts.gate",
        "experts.down",
    ]


def test_megatron_rejects_multi_output_unit_above_bound_before_bridge_collective():
    mapping = types.SimpleNamespace(
        is_grouped_export=False,
        hf_param={"a": "a.weight", "b": "b.weight"},
    )

    class MultiOutputBridge:
        def __init__(self):
            self.calls = 0

        def get_conversion_tasks(self, _model):
            return [
                types.SimpleNamespace(
                    vp_stage=0,
                    param_name="decoder.weight",
                    global_param_name="decoder.weight",
                    pp_rank=0,
                    param_weight=torch.ones(1),
                    mapping=mapping,
                )
            ]

        def export_hf_weights(self, _model, **_kwargs):
            self.calls += 1
            yield "a.weight", torch.ones(2)
            yield "b.weight", torch.ones(1)

    bridge = MultiOutputBridge()
    with pytest.raises(CanonicalSourceError, match="planned output bytes"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                bucket_bytes=8,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                canonical_schema=_DEFAULT_SCHEMA,
            ),
            lambda _bucket: None,
        )

    assert bridge.calls == 0


def test_megatron_rejects_dynamic_export_hooks_before_bridge_collective():
    mapping = types.SimpleNamespace(
        is_grouped_export=False,
        hf_param="weight",
    )

    class HookedBridge:
        def __init__(self):
            self.calls = 0

        def get_conversion_tasks(self, _model):
            return [
                types.SimpleNamespace(
                    vp_stage=0,
                    param_name="weight",
                    global_param_name="weight",
                    pp_rank=0,
                    param_weight=torch.ones(1),
                    mapping=mapping,
                    export_hook=lambda _name, _tensor: (),
                )
            ]

        def export_hf_weights(self, _model, **_kwargs):
            self.calls += 1
            yield "weight", torch.ones(1)

    bridge = HookedBridge()
    with pytest.raises(CanonicalSourceError, match="export_hook"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                bucket_bytes=8,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                canonical_schema=(CanonicalTensorSpec("weight", torch.float32, (1,)),),
            ),
            lambda _bucket: None,
        )

    assert bridge.calls == 0


def test_megatron_ownerless_mapped_slot_fails_before_export():
    class OwnerlessBridge:
        def __init__(self):
            self.calls = 0

        def get_conversion_tasks(self, _model):
            return [
                types.SimpleNamespace(
                    vp_stage=None,
                    param_name="weight",
                    global_param_name="weight",
                    pp_rank=0,
                    param_weight=None,
                )
            ]

        def export_hf_weights(self, _model, **_kwargs):
            self.calls += 1
            yield "weight", torch.ones(1)

    bridge = OwnerlessBridge()
    with pytest.raises(CanonicalSourceError, match="no owning rank"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                bucket_bytes=8,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                canonical_schema=(CanonicalTensorSpec("weight", torch.float32, (1,)),),
            ),
            lambda _bucket: None,
        )

    assert bridge.calls == 0


def test_megatron_plan_compares_global_hf_dtype_and_shape_before_real_export(
    monkeypatch,
):
    bridge = _TaskBridge()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)

    def all_gather_object(outputs, contribution, group=None):
        del group
        local_error, representation, local_plan = contribution
        assert local_error is None
        assert local_plan == (
            ("a.weight", "torch.float32", (2,), True),
            ("b.weight", "torch.float32", (1,), True),
        )
        outputs[:] = [
            contribution,
            (
                None,
                representation,
                (
                    ("a.weight", "torch.float64", (2,), True),
                    ("b.weight", "torch.float32", (1,), True),
                ),
            ),
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(CanonicalSourceError, match="capture plan differs"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                bucket_bytes=16,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                weights_getter=lambda: {
                    "module.module.vp_stages.0.decoder.weight": torch.ones(3)
                },
                canonical_schema=_DEFAULT_SCHEMA,
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert bridge.calls == []


def test_fsdp_rejects_derived_collective_schedule_mismatch_before_full_tensor(
    monkeypatch,
):
    calls = []
    state = {
        "module.a": _FakeDistributedTensor(torch.tensor([1]), calls, "a"),
    }
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    def all_gather_object(outputs, contribution, group=None):
        del group
        if len(contribution) == 3:
            local_error, representation, canonical_plan = contribution
            if isinstance(representation, tuple):
                identity, execution_plan = representation
                changed = list(execution_plan)
                changed[0] = ("different.source", *changed[0][1:])
                peer_representation = (identity, tuple(changed))
                outputs[:] = [
                    contribution,
                    (local_error, peer_representation, canonical_plan),
                ]
                return
        outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(CanonicalSourceError, match="capture plan differs"):
        for_each_fsdp_hf_bucket(
            object(),
            FsdpHfBucketConfig(
                bucket_bytes=16,
                rank=lambda: 0,
                state_dict_getter=lambda _model: state,
                materializer_topology={"module.a": ("test-sharded", ("ranks", (0, 1)))},
                canonical_schema=(CanonicalTensorSpec("a", torch.int64, (1,)),),
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert calls == []


def test_megatron_routes_peer_only_pp_output_to_the_rank_zero_consumer(monkeypatch):
    class RankZeroStageBridge(_TaskBridge):
        def export_hf_weights(self, model, **kwargs):
            self.calls.append((model, kwargs))
            yield "a.weight", torch.tensor([1.0, 2.0]), "native.a.weight"

    bridge = RankZeroStageBridge()
    buckets = []
    gathers = 0
    events = []
    identity = CanonicalFormatIdentity()
    routing_group = object()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)
    b_tensor = torch.tensor([3.0])
    b_bytes = b_tensor.view(torch.uint8).numpy().tobytes()
    b_record = (
        "b.weight",
        "torch.float32",
        (1,),
        f"sha256:{hashlib.sha256(b_bytes).hexdigest()}",
    )

    def all_gather_object(outputs, contribution, group=None):
        nonlocal gathers
        del group
        gathers += 1
        events.append("route" if contribution == (None, None) else "gather")
        if gathers in {1, 2, 3}:
            outputs[:] = [contribution, contribution]
        elif gathers == 4:
            outputs[:] = [contribution, (None, (b_record,))]
        else:
            outputs[:] = [contribution, contribution]

    def broadcast(tensor, *, src, group=None):
        assert group is routing_group
        assert src == 1
        assert events[-1] == "route"
        events.append("broadcast")
        tensor.copy_(b_tensor)

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(
        torch.distributed,
        "get_backend",
        lambda group=None: "gloo" if group is routing_group else "nccl",
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        lambda _group: [0, 1],
    )

    for_each_megatron_hf_bucket(
        [object()],
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=16,
            rank=lambda: 0,
            model_context=lambda: nullcontext(),
            routing_group=routing_group,
            format_identity=identity,
            canonical_schema=_DEFAULT_SCHEMA,
            **_distributed_deadline(),
        ),
        buckets.append,
    )

    assert gathers == 7
    assert [name for bucket in buckets for name, _tensor in bucket] == [
        "a.weight",
        "b.weight",
    ]


def test_megatron_rejects_non_gloo_peer_routing_before_bridge_export(monkeypatch):
    bridge = _TaskBridge()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda _group=None: "nccl")
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    def all_gather_object(outputs, contribution, group=None):
        del group
        outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(CanonicalSourceError, match="Gloo"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                bucket_bytes=16,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                canonical_schema=_DEFAULT_SCHEMA,
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )

    assert bridge.calls == []


def test_megatron_releases_staged_tasks_and_outputs_before_the_next_unit():
    class LifetimeBridge(_FakeBridge):
        def __init__(self):
            super().__init__()
            self.task_ref = None
            self.output_ref = None

        def get_conversion_tasks(self, _model):
            return [
                _ConversionTask(0, "a.weight", torch.ones(2)),
                _ConversionTask(0, "b.weight", torch.ones(1)),
            ]

        def export_hf_weights(self, model, **kwargs):
            del model
            if self.task_ref is not None:
                gc.collect()
                assert self.task_ref() is None
                assert self.output_ref() is None
            [task] = kwargs["conversion_tasks"]
            output = task.param_weight + 1
            self.task_ref = weakref.ref(task)
            self.output_ref = weakref.ref(output)
            yield task.param_name, output

    bridge = LifetimeBridge()
    for_each_megatron_hf_bucket(
        [object()],
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=8,
            rank=lambda: 0,
            model_context=lambda: nullcontext(),
            weights_getter=lambda: {
                "vp_stages.0.a.weight": torch.ones(2),
                "vp_stages.0.b.weight": torch.ones(1),
            },
            canonical_schema=_DEFAULT_SCHEMA,
        ),
        lambda _bucket: None,
    )


def test_megatron_fails_when_replicated_hf_content_differs(monkeypatch):
    gathers = 0
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    def all_gather_object(outputs, contribution, group=None):
        nonlocal gathers
        del group
        gathers += 1
        if gathers in {1, 2, 3}:
            outputs[:] = [contribution, contribution]
        elif gathers == 4:
            error, records = contribution
            first, *rest = records
            changed = (*first[:3], f"{first[3]}-different")
            outputs[:] = [
                contribution,
                (error, (changed, *rest)),
            ]
        else:
            outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(CanonicalSourceError, match="replica content differs"):
        for_each_megatron_hf_bucket(
            [object()],
            MegatronBridgeHfBucketConfig(
                bridge=_TaskBridge(),
                bucket_bytes=16,
                rank=lambda: 0,
                model_context=lambda: nullcontext(),
                canonical_schema=_DEFAULT_SCHEMA,
                **_distributed_deadline(),
            ),
            lambda _bucket: None,
        )


def test_megatron_removes_vocab_padding_inside_mx_before_bucket_emission():
    class VocabBridge:
        def get_conversion_tasks(self, _model):
            return [
                types.SimpleNamespace(
                    vp_stage=0,
                    param_name="embedding.word_embeddings.weight",
                    param_weight=torch.ones(1),
                    mapping=types.SimpleNamespace(
                        is_grouped_export=False,
                        hf_param="model.embed_tokens.weight",
                    ),
                )
            ]

        def export_hf_weights(self, _model, **_kwargs):
            yield (
                "model.embed_tokens.weight",
                torch.arange(12, dtype=torch.float32).reshape(6, 2),
                "embedding.word_embeddings.weight",
            )

    buckets = []
    for_each_megatron_hf_bucket(
        [object()],
        MegatronBridgeHfBucketConfig(
            bridge=VocabBridge(),
            bucket_bytes=64,
            vocab_size=4,
            rank=lambda: 0,
            model_context=lambda: nullcontext(),
            canonical_schema=(
                CanonicalTensorSpec("model.embed_tokens.weight", torch.float32, (4, 2)),
            ),
        ),
        buckets.append,
    )

    [(name, tensor)] = [item for bucket in buckets for item in bucket]
    assert name == "model.embed_tokens.weight"
    assert tensor.shape == (4, 2)
    assert torch.equal(tensor, torch.arange(8, dtype=torch.float32).reshape(4, 2))


@pytest.mark.parametrize("source", ["megatron", "fsdp"])
def test_sources_require_an_authoritative_hf_schema_before_materialization(source):
    if source == "megatron":
        bridge = _FakeBridge()
        with pytest.raises(CanonicalSourceError, match="canonical HF schema"):
            for_each_megatron_hf_bucket(
                [object()],
                MegatronBridgeHfBucketConfig(
                    bridge=bridge,
                    rank=lambda: 0,
                    model_context=lambda: nullcontext(),
                ),
                lambda _bucket: None,
            )
        assert bridge.calls == []
    else:
        calls = []
        state = {
            "module.weight": _FakeDistributedTensor(torch.ones(1), calls, "weight")
        }
        with pytest.raises(CanonicalSourceError, match="canonical HF schema"):
            for_each_fsdp_hf_bucket(
                object(),
                FsdpHfBucketConfig(
                    rank=lambda: 0,
                    state_dict_getter=lambda _model: state,
                ),
                lambda _bucket: None,
            )
        assert calls == []


@pytest.mark.parametrize("source", ["megatron", "fsdp"])
def test_sources_reject_unsupported_quantization_before_materialization(
    source,
):
    identity = CanonicalFormatIdentity(quantization_profile="fp8-e4m3fn-v1")
    if source == "megatron":
        bridge = _FakeBridge()
        with pytest.raises(CanonicalSourceError, match="quantization profile"):
            for_each_megatron_hf_bucket(
                [object()],
                MegatronBridgeHfBucketConfig(
                    bridge=bridge,
                    rank=lambda: 0,
                    model_context=lambda: nullcontext(),
                    format_identity=identity,
                ),
                lambda _bucket: None,
            )
        assert bridge.calls == []
    else:
        calls = []
        state = {
            "module.weight": _FakeDistributedTensor(torch.ones(1), calls, "weight")
        }
        with pytest.raises(CanonicalSourceError, match="quantization profile"):
            for_each_fsdp_hf_bucket(
                object(),
                FsdpHfBucketConfig(
                    rank=lambda: 0,
                    state_dict_getter=lambda _model: state,
                    format_identity=identity,
                ),
                lambda _bucket: None,
            )
        assert calls == []


def test_atomic_group_bound_is_checked_before_fsdp_tensor_collectives():
    calls = []
    state = {
        "module.a": _FakeDistributedTensor(torch.ones(2), calls, "a"),
        "module.b": _FakeDistributedTensor(torch.ones(1), calls, "b"),
    }
    identity = CanonicalFormatIdentity(atomic_groups=(("a", "b"),))

    with pytest.raises(CanonicalSourceError, match="atomic group.*exceeds"):
        for_each_fsdp_hf_bucket(
            object(),
            FsdpHfBucketConfig(
                bucket_bytes=8,
                rank=lambda: 0,
                state_dict_getter=lambda _model: state,
                format_identity=identity,
                canonical_schema=(
                    CanonicalTensorSpec("a", torch.float32, (2,)),
                    CanonicalTensorSpec("b", torch.float32, (1,)),
                ),
            ),
            lambda _bucket: None,
        )

    assert calls == []


@pytest.mark.parametrize("bucket_bytes", [0, -1])
def test_source_bucket_bound_must_be_positive(bucket_bytes):
    with pytest.raises(ValueError, match="bucket_bytes"):
        MegatronBridgeHfBucketConfig(bucket_bytes=bucket_bytes)
    with pytest.raises(ValueError, match="bucket_bytes"):
        FsdpHfBucketConfig(bucket_bytes=bucket_bytes)
