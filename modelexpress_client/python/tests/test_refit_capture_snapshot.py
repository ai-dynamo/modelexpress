# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import copyreg
import sys
from contextlib import nullcontext
from types import ModuleType

import pytest
import torch
from modelexpress.refit.reshard.types import CaptureResult, RecordedCopy
from modelexpress_rl.inference.engines.vllm import _capture_snapshot as snapshot_module
from modelexpress_rl.inference.engines.vllm import installer as module
from modelexpress_rl.inference.engines.vllm._capture_snapshot import _CaptureSnapshot
from torch import nn


def _result(op_chain=()):
    record = RecordedCopy(
        "source", op_chain, "weight", 0, (2, 2), (2, 1), torch.float32
    )
    return CaptureResult(copies=[record, record]), {"weight": ((2, 2), torch.float32)}


def test_snapshot_returns_fresh_mutable_records_and_preserves_shared_references():
    result = _result()
    snapshot = _CaptureSnapshot.create(result)
    assert type(snapshot) is _CaptureSnapshot
    first, layout = snapshot.clone()
    second, other_layout = snapshot.clone()
    assert type(first) is CaptureResult and type(first.copies[0]) is RecordedCopy
    assert first.copies[0] is first.copies[1]
    assert first.copies[0] is not second.copies[0]
    assert first.copies[0] is not result[0].copies[0]
    first.copies[0].dest_offset = 7
    first.copies.clear()
    first.unsupported.append("changed")
    first.unsupported_reasons["changed"] = "changed"
    layout.clear()
    assert (second, other_layout) == result
    assert snapshot.clone() == result


def test_slice_subtrees_are_copied_with_shared_identity_preserved():
    item = slice(None, 2)
    chain = (("__getitem__", ((item, item),), ()),)
    result = _result(chain)
    result[0].copies.append(copy.copy(result[0].copies[0]))
    snapshot = _CaptureSnapshot.create(result)
    assert type(snapshot) is _CaptureSnapshot
    cloned, _ = snapshot.clone()
    first = cloned.copies[0].op_chain[0][1][0]
    third = cloned.copies[2].op_chain[0][1][0]
    assert first[0] is first[1] is third[0]
    assert first[0] is not item
    assert cloned.copies[0] is cloned.copies[1]
    assert cloned.copies[0] is not cloned.copies[2]
    assert snapshot.clone()[0].copies[0].op_chain[0][1][0][0] is not first[0]


@pytest.mark.parametrize("value", [[2, 2], {"shape": (2, 2)}, slice([1], 2)])
def test_nested_mutable_op_payload_uses_original_deepcopy(value):
    result = _result((("view", (value,), ()),))
    assert _CaptureSnapshot.create(result) is result
    cloned = copy.deepcopy(result)
    assert cloned[0].copies[0] is cloned[0].copies[1]
    assert cloned[0].copies[0].op_chain[0][1][0] is not value


def test_extra_record_state_and_aliased_capture_containers_decline_snapshot():
    result = _result()
    result[0].copies[0].extra = []
    assert _CaptureSnapshot.create(result) is result
    del result[0].copies[0].extra
    result = result[0], result[0].unsupported_reasons
    assert _CaptureSnapshot.create(result) is result


@pytest.mark.parametrize("phase", ["before", "after"])
def test_custom_deepcopy_is_not_bypassed(monkeypatch, phase):
    result = _result()
    snapshot = _CaptureSnapshot.create(result)
    calls = []

    def custom(record, memo):
        calls.append(record)
        changed = RecordedCopy(**vars(record))
        changed.dest_offset = 11
        return changed

    monkeypatch.setattr(RecordedCopy, "__deepcopy__", custom, raising=False)
    if phase == "before":
        assert _CaptureSnapshot.create(result) is result
        cloned = copy.deepcopy(result)
    else:
        cloned = snapshot.clone()
    assert calls == [result[0].copies[0]]
    assert cloned[0].copies[0].dest_offset == 11
    assert cloned[0].copies[0] is cloned[0].copies[1]


def test_copyreg_handler_after_snapshot_uses_whole_result_fallback(monkeypatch):
    result = _result()
    snapshot = _CaptureSnapshot.create(result)
    calls = []

    def reduce(record):
        calls.append(record)
        values = list(vars(record).values())
        values[3] = 13
        return RecordedCopy, tuple(values)

    monkeypatch.setitem(copyreg.dispatch_table, RecordedCopy, reduce)
    cloned = snapshot.clone()
    assert calls == [result[0].copies[0]]
    assert cloned[0].copies[0].dest_offset == 13


def test_rebound_record_class_uses_original_types(monkeypatch):
    result = _result()
    snapshot = _CaptureSnapshot.create(result)

    class Replacement:
        pass

    monkeypatch.setattr(snapshot_module, "RecordedCopy", Replacement)
    assert _CaptureSnapshot.create(result) is result
    cloned = snapshot.clone()
    assert type(cloned[0].copies[0]) is RecordedCopy
    assert cloned == result


def test_changed_slice_copy_protocol_uses_whole_result_fallback(monkeypatch):
    value = slice(1, 2)
    result = _result((("__getitem__", (value,), ()),))
    snapshot = _CaptureSnapshot.create(result)
    calls = []

    def changed(item, memo):
        calls.append(item)
        return slice(7, 8)

    monkeypatch.setitem(copy._deepcopy_dispatch, slice, changed)
    cloned = snapshot.clone()
    assert calls == [value]
    assert cloned[0].copies[0].op_chain[0][1][0] == slice(7, 8)


def test_rebound_copy_dispatch_table_uses_original_fallback(monkeypatch):
    result = _result()
    snapshot = _CaptureSnapshot.create(result)
    calls = []

    def reduce(record):
        calls.append(record)
        values = list(vars(record).values())
        values[3] = 19
        return RecordedCopy, tuple(values)

    monkeypatch.setattr(copy, "dispatch_table", {RecordedCopy: reduce})
    cloned = snapshot.clone()
    assert calls == [result[0].copies[0]]
    assert cloned[0].copies[0].dest_offset == 19


def test_changed_deepcopy_default_uses_original_fallback(monkeypatch):
    result = _result()
    snapshot = _CaptureSnapshot.create(result)
    calls = []

    def changed(value, memo):
        calls.append(value)
        return copy.deepcopy(value, memo)

    monkeypatch.setattr(copy._deepcopy_list, "__defaults__", (changed,))
    cloned = snapshot.clone()
    assert calls == result[0].copies
    assert cloned == result


def test_custom_copyreg_key_equality_is_not_bypassed(monkeypatch):
    result = _result()
    snapshot = _CaptureSnapshot.create(result)
    calls = []

    class Meta(type):
        def __hash__(cls):
            return hash(RecordedCopy)

        def __eq__(cls, other):
            calls.append(other)
            raise RuntimeError("copy registration equality")

    class Custom(metaclass=Meta):
        pass

    table = dict(copyreg.dispatch_table)
    table[Custom] = lambda value: None
    monkeypatch.setattr(copyreg, "dispatch_table", table)
    monkeypatch.setattr(copy, "dispatch_table", table)
    with pytest.raises(RuntimeError, match="copy registration equality"):
        snapshot.clone()
    assert calls == [RecordedCopy]


def test_subclass_and_custom_state_protocol_decline_snapshot(monkeypatch):
    class Custom(RecordedCopy):
        pass

    result = _result()
    result[0].copies[0] = Custom(**vars(result[0].copies[0]))
    assert _CaptureSnapshot.create(result) is result
    result = _result()
    monkeypatch.setattr(
        RecordedCopy, "__getstate__", lambda record: vars(record), raising=False
    )
    assert _CaptureSnapshot.create(result) is result


@pytest.fixture
def capture_installer(monkeypatch):
    names = (
        "vllm",
        "vllm.config",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.reload",
        "vllm.model_executor.model_loader.reload.layerwise",
    )
    modules = {name: ModuleType(name) for name in names}
    modules["vllm.config"].set_current_vllm_config = lambda config: nullcontext()
    layerwise = modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise.LAYERWISE_INFO = {}
    layerwise.initialize_layerwise_reload = lambda model: None
    layerwise._get_original_loader = lambda parameter: None
    layerwise._place_kernel_tensors = lambda layer, info: None
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    weight_utils.default_weight_loader = lambda parameter, weight: parameter.data.copy_(
        weight
    )
    monkeypatch.setitem(sys.modules, weight_utils.__name__, weight_utils)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))
            self.capture_calls = 0

        def load_weights(self, weights):
            self.capture_calls += 1
            for name, weight in weights:
                self.weight.weight_loader(self.weight, weight[:2, :2])

    model = Model()
    installer = module._VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    return installer, model


def test_real_capture_cache_uses_snapshot_for_slicing_and_retains_invalidation(
    capture_installer,
):
    installer, model = capture_installer
    manifest = [("weight", torch.float32, (3, 3))]
    cold, _ = installer.capture(manifest)
    assert type(installer._capture_cache[1]) is _CaptureSnapshot
    assert cold.copies[0].op_chain[0][0] == "__getitem__"
    original = copy.deepcopy(cold)
    cold.copies[0].dest_offset = 55
    first, layout = installer.capture(manifest)
    assert first == original and model.capture_calls == 1
    first.copies[0].op_chain = (("view", ([4],), ()),)
    first.copies.clear()
    layout.clear()
    assert installer.capture(manifest)[0] == original
    assert model.capture_calls == 1
    installer.capture([("weight", torch.float32, (4, 4))])
    assert model.capture_calls == 2
    model.weight = nn.Parameter(torch.zeros(1, 2))
    changed, changed_layout = installer.capture(manifest)
    assert model.capture_calls == 3
    assert changed.copies[0].dest_shape == (1, 2)
    assert changed_layout == {"weight": ((1, 2), torch.float32)}
    assert installer.capture(manifest) == (changed, changed_layout)
    assert model.capture_calls == 3


def test_failed_fresh_capture_discards_snapshot(capture_installer, monkeypatch):
    installer, _ = capture_installer
    installer.capture([("weight", torch.float32, (3, 3))])

    def fail(*args, **kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(module, "capture_weights", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        installer.capture([("weight", torch.float32, (4, 4))])
    assert installer._capture_cache is None


def test_real_capture_cache_preserves_duplicate_record_identity(
    capture_installer, monkeypatch
):
    installer, _ = capture_installer
    capture = module.capture_weights

    def duplicate(*args, **kwargs):
        result = capture(*args, **kwargs)
        result.copies.append(result.copies[0])
        return result

    monkeypatch.setattr(module, "capture_weights", duplicate)
    manifest = [("weight", torch.float32, (3, 3))]
    installer.capture(manifest)
    warm, _ = installer.capture(manifest)
    assert type(installer._capture_cache[1]) is _CaptureSnapshot
    assert warm.copies[0] is warm.copies[1]


def test_real_capture_cache_nested_mutable_payload_remains_isolated(
    capture_installer, monkeypatch
):
    installer, _ = capture_installer
    capture = module.capture_weights

    def mutable(*args, **kwargs):
        result = capture(*args, **kwargs)
        result.copies[0].op_chain = (("view", ([4],), ()),)
        return result

    monkeypatch.setattr(module, "capture_weights", mutable)
    manifest = [("weight", torch.float32, (3, 3))]
    installer.capture(manifest)
    assert type(installer._capture_cache[1]) is tuple
    first, _ = installer.capture(manifest)
    first.copies[0].op_chain[0][1][0][0] = 99
    assert installer.capture(manifest)[0].copies[0].op_chain[0][1][0][0] == 4
