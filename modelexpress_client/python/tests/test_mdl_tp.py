import torch

from modelexpress.engines.vllm.mdl import MdlLoader


class _Model:
    def named_parameters(self):
        return []


def _loader(rank=1):
    loader = MdlLoader(_Model())
    loader._tp_size = 2
    loader._tp_rank = rank
    return loader


def test_tp_local_tensor_axis_zero():
    tensor = torch.arange(24).reshape(6, 4)
    local = _loader(rank=1)._tp_local_tensor(tensor, (3, 4))
    assert torch.equal(local, tensor[3:6])
    assert local.is_contiguous()


def test_tp_local_tensor_axis_one():
    tensor = torch.arange(24).reshape(4, 6)
    local = _loader(rank=1)._tp_local_tensor(tensor, (4, 3))
    assert torch.equal(local, tensor[:, 3:6])
    assert local.is_contiguous()


def test_tp_shape_compatibility_requires_one_sharded_dimension():
    loader = _loader()
    assert loader._tp_shape_compatible((6, 4), (3, 4))
    assert loader._tp_shape_compatible((4, 6), (4, 3))
    assert not loader._tp_shape_compatible((6, 8), (3, 4))
    assert not loader._tp_shape_compatible((5, 4), (3, 4))
