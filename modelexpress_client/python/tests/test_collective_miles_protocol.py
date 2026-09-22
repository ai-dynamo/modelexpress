# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the concrete optional MILES collective protocol."""

import multiprocessing
import sys
from datetime import timedelta
from queue import Empty
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from modelexpress_rl.collective.integrations import miles_protocol
from modelexpress_rl.collective.integrations.miles_protocol import (
    MilesCollectiveProtocolCore,
)

_ROUND_DIGESTS = (("model.weight", "01" * 32),)


class _Group:
    def __init__(self, size=1, rank=0):
        self.size = size
        self.rank = rank


def _parallel_state(pp_size=2):
    return SimpleNamespace(
        pp=_Group(pp_size),
        tp=_Group(),
        ep=_Group(),
        etp=_Group(),
        cp=_Group(),
        intra_dp=_Group(),
        indep_dp=_Group(),
    )


def _placement():
    return SimpleNamespace(gather_pp=False, gather_tp=True, gather_ep=True)


def _args():
    return SimpleNamespace(
        model="qwen",
        modelexpress_server_address="mx:50051",
        modelexpress_m2n_timeout_s=1,
    )


def test_endpoint_rejects_unsupported_secure_schemes():
    for scheme in ("https", "grpcs"):
        args = _args()
        args.modelexpress_server_address = f"{scheme}://mx.example:50051"

        with pytest.raises(ValueError, match="secure ModelExpress endpoints"):
            miles_protocol._endpoint(args)


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_exact_tensor_verification_can_be_enabled_explicitly(monkeypatch, value):
    monkeypatch.setenv("MX_MILES_VERIFY_TENSOR_EQUALITY", value)

    protocol = MilesCollectiveProtocolCore(_args())

    assert protocol._verify_tensor_equality is True


def test_exact_tensor_verification_defaults_off(monkeypatch):
    monkeypatch.delenv("MX_MILES_VERIFY_TENSOR_EQUALITY", raising=False)

    protocol = MilesCollectiveProtocolCore(_args())

    assert protocol._verify_tensor_equality is False


def test_exact_tensor_verification_rejects_ambiguous_values(monkeypatch):
    monkeypatch.setenv("MX_MILES_VERIFY_TENSOR_EQUALITY", "sometimes")

    with pytest.raises(ValueError, match="MX_MILES_VERIFY_TENSOR_EQUALITY"):
        MilesCollectiveProtocolCore(_args())


def test_prepare_sessions_uses_auth_wrapped_channel(monkeypatch):
    _install_fake_miles_async(monkeypatch, [])
    raw_channel = object()
    captured = {}
    sessions = []

    class AuthChannel:
        def close(self):
            pass

    auth_channel = AuthChannel()

    class Session:
        def __init__(self, worker_id):
            self.worker_id = worker_id

        def prepare(self):
            captured["prepared"] = True

        def close(self):
            pass

    def create_session(**kwargs):
        session = Session(kwargs["worker_id"])
        sessions.append(session)
        return session

    monkeypatch.setattr(
        miles_protocol.grpc,
        "insecure_channel",
        lambda endpoint: captured.setdefault("endpoint", endpoint) and raw_channel,
    )
    monkeypatch.setattr(
        miles_protocol.auth,
        "with_auth",
        lambda channel: captured.setdefault("raw_channel", channel) and auth_channel,
    )
    monkeypatch.setattr(
        miles_protocol,
        "CollectiveRendezvous",
        lambda channel: captured.setdefault("rendezvous_channel", channel) and object(),
    )
    monkeypatch.setattr(miles_protocol, "MilesPublisher", lambda **_kwargs: object())
    monkeypatch.setattr(
        miles_protocol.MilesTrainerSession,
        "create",
        create_session,
    )
    monkeypatch.setattr(
        miles_protocol,
        "MilesTransferCoordinator",
        lambda *_args: object(),
    )
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(miles_protocol, "_gloo_group", lambda: object())
    monkeypatch.setattr(
        miles_protocol.dist,
        "all_gather_object",
        lambda output, value, **_kwargs: output.__setitem__(0, value),
    )

    def prepare_protocol():
        protocol = MilesCollectiveProtocolCore(_args())
        protocol.rollout_engines = ()
        protocol._tensors = {"model.weight": torch.ones((1,), dtype=torch.bfloat16)}
        protocol._topology = SimpleNamespace(trainer_slots=("trainer-0",))
        protocol._plan = SimpleNamespace(
            bulk=[SimpleNamespace(name="model.weight", partition_id=0)],
            parameter_names=lambda: ("model.weight",),
        )
        protocol._prepare_sessions()
        return protocol

    first = prepare_protocol()
    second = prepare_protocol()

    assert captured == {
        "endpoint": "mx:50051",
        "raw_channel": raw_channel,
        "rendezvous_channel": auth_channel,
        "prepared": True,
    }
    assert first._channel is auth_channel
    assert second._channel is auth_channel
    assert len(sessions) == 2
    assert sessions[0].worker_id.startswith("miles-trainer-0-")
    assert sessions[1].worker_id.startswith("miles-trainer-0-")
    assert sessions[0].worker_id != sessions[1].worker_id
    assert all(len(session.worker_id.rsplit("-", 1)[1]) == 32 for session in sessions)


def _install_fake_miles_async(monkeypatch, events):
    async_utils = ModuleType("miles.utils.async_utils")

    class Future:
        def __init__(self, coroutine):
            self.coroutine = coroutine

        def result(self):
            try:
                return self.coroutine.send(None)
            except StopIteration as stopped:
                return stopped.value

    async_utils.submit = lambda coroutine: Future(coroutine)

    def wait_futures(futures):
        events.append("wait-generators")
        return [future.result() for future in futures]

    async_utils.wait_futures = wait_futures
    utils = ModuleType("miles.utils")
    utils.async_utils = async_utils
    miles = ModuleType("miles")
    miles.utils = utils
    monkeypatch.setitem(sys.modules, "miles", miles)
    monkeypatch.setitem(sys.modules, "miles.utils", utils)
    monkeypatch.setitem(sys.modules, "miles.utils.async_utils", async_utils)


def _run_gloo_workers(worker, rendezvous_path):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=worker,
            args=(rank, 2, str(rendezvous_path), results),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
    hung = [process for process in processes if process.is_alive()]
    for process in hung:
        process.terminate()
        process.join(timeout=5)
    assert not hung, "MILES protocol workers deadlocked"
    assert [process.exitcode for process in processes] == [0, 0]
    try:
        return sorted(results.get(timeout=2) for _ in processes)
    except Empty:
        pytest.fail("MILES protocol worker exited without reporting a result")


def _begin_sync_failure_worker(rank, world_size, rendezvous_path, results):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    original_empty = torch.empty
    try:
        miles_protocol._gloo_group = lambda: dist.group.WORLD
        state = _parallel_state(pp_size=world_size)
        state.pp.rank = rank
        protocol = MilesCollectiveProtocolCore(_args())
        protocol.connect(
            [object()],
            [1],
            [0],
            state,
            _placement(),
            "target",
        )
        source = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
        if rank == 1:
            failed = False

            def fail_wire_allocation(*args, **kwargs):
                nonlocal failed
                if not failed and args and tuple(args[0]) == (2, 2):
                    failed = True
                    raise torch.OutOfMemoryError("synthetic wire allocation failure")
                return original_empty(*args, **kwargs)

            torch.empty = fail_wire_allocation
        try:
            protocol.begin_sync(
                1,
                lambda *, materialize: iter([[("model.weight", source)]]),
            )
        except (RuntimeError, AssertionError) as error:
            results.put((rank, type(error).__name__, str(error)))
        else:
            results.put((rank, "no-error", ""))
    finally:
        torch.empty = original_empty
        dist.destroy_process_group()


def _round_create_failure_worker(rank, world_size, rendezvous_path, results):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    try:
        miles_protocol._gloo_group = lambda: dist.group.WORLD
        protocol = MilesCollectiveProtocolCore(_args())

        class Session:
            membership = SimpleNamespace(group_id="group-9", epoch=4)

            def run_round(self, *, version, operation_id):
                raise AssertionError("round must not start after create failure")

            def close(self):
                pass

        class Coordinator:
            def create(self, version, *, idempotency_key):
                if rank != 0:
                    raise AssertionError("only rank zero may create the operation")
                raise RuntimeError("synthetic coordinator create failure")

        protocol._session = Session()
        protocol._coordinator = Coordinator()
        protocol._tensor_digests = _ROUND_DIGESTS
        protocol._prepare_sessions = lambda: None
        context = SimpleNamespace(sync_base=True, adapters=(), weight_version=1)
        try:
            protocol.execute_update_round(context)
        except (RuntimeError, AssertionError) as error:
            results.put((rank, type(error).__name__, str(error)))
        else:
            results.put((rank, "no-error", ""))
    finally:
        dist.destroy_process_group()


def _prepare_generator_submission_failure_worker(
    rank, world_size, rendezvous_path, results
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    try:
        miles_protocol._gloo_group = lambda: dist.group.WORLD
        protocol = MilesCollectiveProtocolCore(_args())
        protocol._tensors = {
            f"rank-{rank}.weight": torch.ones((1,), dtype=torch.bfloat16)
        }
        protocol._topology = SimpleNamespace(
            trainer_slots=tuple(f"trainer-{index}" for index in range(world_size))
        )
        protocol._plan = SimpleNamespace(
            bulk=[
                SimpleNamespace(
                    name=f"rank-{rank}.weight",
                    partition_id=rank,
                )
            ],
            parameter_names=lambda: (f"rank-{rank}.weight",),
        )

        class Channel:
            def close(self):
                pass

        entered_prepare = False

        class Session:
            def prepare(self):
                nonlocal entered_prepare
                entered_prepare = True

            def close(self):
                pass

        miles_protocol.auth.with_auth = lambda channel: channel
        miles_protocol.grpc.insecure_channel = lambda _endpoint: Channel()
        miles_protocol.CollectiveRendezvous = lambda _channel: object()
        miles_protocol.MilesPublisher = lambda **_kwargs: object()
        miles_protocol.MilesTrainerSession.create = lambda **_kwargs: Session()
        miles_protocol.MilesTransferCoordinator = lambda *_args: object()

        def generator_futures(action, **_kwargs):
            if rank == 0 and action == "prepare":
                raise RuntimeError("synthetic prepare submit failure")
            return []

        protocol._generator_futures = generator_futures
        try:
            protocol._prepare_sessions()
        except RuntimeError as error:
            results.put((rank, type(error).__name__, str(error), entered_prepare))
        else:
            results.put((rank, "no-error", "", entered_prepare))
    finally:
        dist.destroy_process_group()


def _round_generator_submission_failure_worker(
    rank, world_size, rendezvous_path, results
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    try:
        miles_protocol._gloo_group = lambda: dist.group.WORLD
        protocol = MilesCollectiveProtocolCore(_args())

        entered_round = False

        class Session:
            membership = SimpleNamespace(group_id="group-9", epoch=4)

            def run_round(self, *, version, operation_id):
                nonlocal entered_round
                entered_round = True
                assert version == "1"
                assert operation_id == "operation-7"

            def report_failure(self, *, operation_id, error):
                assert operation_id == "operation-7"
                assert "synthetic round submit failure" in str(error)

            def close(self):
                pass

        class Coordinator:
            def create(self, version, *, idempotency_key):
                assert rank == 0
                assert version == "1"
                assert idempotency_key == "miles-group-9-weight-version-1"
                return SimpleNamespace(operation_id="operation-7")

            def delete(self, operation_id):
                assert operation_id == "operation-7"

        protocol._session = Session()
        protocol._coordinator = Coordinator()
        protocol._tensor_digests = _ROUND_DIGESTS
        protocol._prepare_sessions = lambda: None

        def generator_futures(action, **_kwargs):
            if rank == 0 and action == "run_round":
                raise RuntimeError("synthetic round submit failure")
            return []

        protocol._generator_futures = generator_futures
        context = SimpleNamespace(sync_base=True, adapters=(), weight_version=1)
        try:
            protocol.execute_update_round(context)
        except RuntimeError as error:
            results.put((rank, type(error).__name__, str(error), entered_round))
        else:
            results.put((rank, "no-error", "", entered_round))
    finally:
        dist.destroy_process_group()


def test_lazy_factory_opts_out_of_megatron_forced_pp_gather(monkeypatch):
    protocol_module = ModuleType("miles.backends.training_utils.weight_update.protocol")

    class WeightTransferProtocol:
        def __init__(self, args):
            self.args = args

    protocol_module.WeightTransferProtocol = WeightTransferProtocol
    iterator_module = ModuleType(
        "miles.backends.training_utils.weight_update.hf_weight_iterator"
    )

    class WeightUpdatePlacement:
        def __init__(self, *, gather_pp):
            self.gather_pp = gather_pp

    iterator_module.WeightUpdatePlacement = WeightUpdatePlacement
    module_names = [
        "miles",
        "miles.backends",
        "miles.backends.training_utils",
        "miles.backends.training_utils.weight_update",
    ]
    for module_name in module_names:
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    monkeypatch.setitem(sys.modules, protocol_module.__name__, protocol_module)
    monkeypatch.setitem(sys.modules, iterator_module.__name__, iterator_module)

    protocol = miles_protocol.build_protocol(_args())

    assert protocol.requires_exact_placement is True
    assert protocol.required_placement.gather_pp is False


@pytest.mark.parametrize("generator_count", [1, 6, 64])
def test_begin_sync_storage_does_not_scale_with_generator_count(
    monkeypatch,
    generator_count,
):
    protocol = MilesCollectiveProtocolCore(_args())
    protocol._verify_tensor_equality = True
    protocol.connect(
        [object()],
        [generator_count],
        [0],
        _parallel_state(),
        _placement(),
        "target",
    )
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(miles_protocol, "_gloo_group", lambda: object())

    def gather(output, value, **_kwargs):
        if value is None:
            output[:] = [None, None]
            return
        if isinstance(value, str):
            output[:] = [value, ""]
            return
        if value and len(value[0]) == 2:
            output[:] = [value, [("model.other", "ab" * 32)]]
            return
        output[:] = [
            value,
            [("model.other", (3, 4), "bfloat16", 1)],
        ]

    monkeypatch.setattr(
        miles_protocol.dist,
        "all_gather_object",
        gather,
    )
    source = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)

    protocol.begin_sync(1, lambda *, materialize: iter([[("model.weight", source)]]))

    wire = protocol._tensors["model.weight"]
    assert tuple(wire.shape) == tuple(source.shape)
    assert wire.untyped_storage().nbytes() == source.untyped_storage().nbytes()
    assert protocol._plan.bulk[0].src_mesh.shape == (1,)
    assert protocol._plan.bulk[0].global_shape == tuple(source.shape)
    assert protocol._plan.bulk[0].src_placements == (
        miles_protocol.Placement.replicate(),
    )
    assert protocol._plan.bulk[0].dst_mesh.shape == (generator_count,)
    assert protocol._plan.bulk[0].dst_placements == (
        miles_protocol.Placement.replicate(),
    )
    assert torch.equal(wire, source)
    assert protocol._tensor_digests == (
        ("model.other", "ab" * 32),
        ("model.weight", miles_protocol._exact_tensor_sha256(source)),
    )

    address = wire.data_ptr()
    replacement = torch.full((2, 4), 9, dtype=torch.bfloat16)
    protocol.begin_sync(
        2,
        lambda *, materialize: iter([[("model.weight", replacement)]]),
    )

    assert protocol._tensors["model.weight"].data_ptr() == address
    assert torch.equal(protocol._tensors["model.weight"], replacement)
    assert protocol._tensor_digests == (
        ("model.other", "ab" * 32),
        ("model.weight", miles_protocol._exact_tensor_sha256(replacement)),
    )


@pytest.mark.parametrize(
    ("requested_run_id", "gathered_run_ids"),
    [
        (None, [None, "run-7"]),
        ("run-7", [None, "run-7"]),
        ("run-7", ["run-7", "run-8"]),
    ],
)
def test_mixed_explicit_run_identity_is_rejected_on_every_rank(
    monkeypatch,
    requested_run_id,
    gathered_run_ids,
):
    args = _args()
    if requested_run_id is not None:
        args.modelexpress_m2n_run_id = requested_run_id
    protocol = MilesCollectiveProtocolCore(args)
    protocol.connect(
        [object()],
        [2],
        [0],
        _parallel_state(),
        _placement(),
        "target",
    )
    protocol._tensors = {"model.weight": torch.ones((4, 2), dtype=torch.bfloat16)}
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(miles_protocol, "_gloo_group", lambda: object())
    monkeypatch.setattr(
        miles_protocol.dist,
        "all_gather_object",
        lambda output, _value, **_kwargs: output.__setitem__(
            slice(None), gathered_run_ids
        ),
    )

    with pytest.raises(
        ValueError,
        match="must be set identically on every trainer rank",
    ):
        protocol._build_frozen_contract()


def test_begin_sync_broadcasts_one_rank_allocation_failure(tmp_path):
    results = _run_gloo_workers(
        _begin_sync_failure_worker,
        tmp_path / "begin-sync-rendezvous",
    )

    assert [result[:2] for result in results] == [
        (0, "RuntimeError"),
        (1, "RuntimeError"),
    ]
    assert results[0][2] == results[1][2]
    assert "rank 1" in results[0][2]
    assert "synthetic wire allocation failure" in results[0][2]


def test_begin_sync_skips_exact_hashing_when_verification_is_disabled(
    monkeypatch,
):
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.connect(
        [object()],
        [1],
        [0],
        _parallel_state(pp_size=1),
        _placement(),
        "target",
    )
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(miles_protocol, "_gloo_group", lambda: object())
    monkeypatch.setattr(
        miles_protocol.dist,
        "all_gather_object",
        lambda output, value, **_kwargs: output.__setitem__(0, value),
    )
    monkeypatch.setattr(
        miles_protocol,
        "_exact_tensor_sha256",
        lambda _tensor: pytest.fail("disabled verification hashed a tensor"),
    )
    source = torch.ones((2, 2), dtype=torch.bfloat16)

    protocol.begin_sync(
        1,
        lambda *, materialize: iter([[("model.weight", source)]]),
    )

    assert protocol._tensor_digests is None


def test_begin_sync_preserves_pp_one_validation_error(monkeypatch):
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.connect(
        [object()],
        [1],
        [0],
        _parallel_state(pp_size=1),
        _placement(),
        "target",
    )
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(miles_protocol, "_gloo_group", lambda: object())
    monkeypatch.setattr(
        miles_protocol.dist,
        "all_gather_object",
        lambda output, value, **_kwargs: output.__setitem__(0, value),
    )
    source = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="only BF16 base weights"):
        protocol.begin_sync(
            1,
            lambda *, materialize: iter([[("model.weight", source)]]),
        )


def test_connect_rejects_ambiguous_or_non_full_gather_topologies():
    protocol = MilesCollectiveProtocolCore(_args())

    with pytest.raises(ValueError, match="explicit engine GPU"):
        protocol.connect(
            [object()],
            None,
            None,
            _parallel_state(),
            _placement(),
            "target",
        )
    with pytest.raises(ValueError, match="PP-local"):
        protocol.connect(
            [object()],
            [6],
            [0],
            _parallel_state(),
            SimpleNamespace(gather_pp=True),
            "target",
        )
    with pytest.raises(ValueError, match="one source rank per PP"):
        state = _parallel_state()
        state.tp = _Group(2)
        protocol.connect(
            [object()],
            [6],
            [0],
            state,
            _placement(),
            "target",
        )


def test_rank_one_rejects_tensor_ownership_from_rank_zero(monkeypatch):
    protocol = MilesCollectiveProtocolCore(_args())
    protocol._tensors = {"rank-zero.weight": torch.empty((6, 2))}
    protocol._topology = SimpleNamespace()
    protocol._plan = SimpleNamespace(
        bulk=[
            SimpleNamespace(name="rank-zero.weight", partition_id=0),
            SimpleNamespace(name="rank-one.weight", partition_id=1),
        ]
    )
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 1)

    with pytest.raises(ValueError, match="ownership does not match"):
        protocol._prepare_sessions()


def test_round_uses_one_operation_id_for_all_halves_and_deletes_it(
    monkeypatch,
    caplog,
):
    events = []
    _install_fake_miles_async(monkeypatch, events)
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (object(),)
    protocol._engine_gpu_offsets = (0,)
    protocol._engine_gpu_counts = (1,)
    protocol._generators_prepared = True
    protocol._tensor_digests = _ROUND_DIGESTS
    protocol._verify_tensor_equality = True

    class Session:
        membership = SimpleNamespace(group_id="group-9", epoch=4)

        def run_round(self, *, version, operation_id):
            events.append(("trainer", version, operation_id))

        def close(self):
            events.append("close-trainer")

    class Coordinator:
        def create(self, version, *, idempotency_key):
            events.append(("create", version, idempotency_key))
            return SimpleNamespace(operation_id="operation-7")

        def delete(self, operation_id):
            events.append(("delete", operation_id))

    protocol._session = Session()
    protocol._coordinator = Coordinator()
    monkeypatch.setattr(protocol, "_prepare_sessions", lambda: None)
    monkeypatch.setattr(
        protocol, "_wait_terminal", lambda op: events.append(("poll", op))
    )
    monkeypatch.setattr(miles_protocol, "_gloo_group", lambda: object())

    async def send_generator(kwargs):
        assert kwargs["tensor_digests"] == _ROUND_DIGESTS
        events.append(("generator", "1", "operation-7"))

    monkeypatch.setattr(
        protocol,
        "_generator_futures",
        lambda action, **kwargs: [
            sys.modules["miles.utils.async_utils"].submit(send_generator(kwargs))
        ],
    )
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        miles_protocol.dist,
        "broadcast_object_list",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        miles_protocol.dist,
        "all_gather_object",
        lambda output, value, **_kwargs: output.__setitem__(0, value),
    )
    context = SimpleNamespace(sync_base=True, adapters=(), weight_version=1)

    with caplog.at_level("INFO"):
        protocol.execute_update_round(context)

    assert ("trainer", "1", "operation-7") in events
    assert ("generator", "1", "operation-7") in events
    assert ("create", "1", "miles-group-9-weight-version-1") in events
    assert ("poll", "operation-7") in events
    assert events[-1] == ("delete", "operation-7")
    assert "MILES tensor equality receipt role=trainer" in caplog.text
    assert "version=1" in caplog.text
    assert "operation_id=operation-7" in caplog.text
    assert "tensor_count=1" in caplog.text


def test_round_broadcasts_rank_zero_create_failure(tmp_path):
    results = _run_gloo_workers(
        _round_create_failure_worker,
        tmp_path / "round-create-rendezvous",
    )

    assert [result[:2] for result in results] == [
        (0, "RuntimeError"),
        (1, "RuntimeError"),
    ]
    assert results[0][2] == results[1][2]
    assert "synthetic coordinator create failure" in results[0][2]


def test_prepare_broadcasts_rank_zero_generator_submission_failure(tmp_path):
    results = _run_gloo_workers(
        _prepare_generator_submission_failure_worker,
        tmp_path / "prepare-generator-submit-rendezvous",
    )

    assert [result[:2] for result in results] == [
        (0, "RuntimeError"),
        (1, "RuntimeError"),
    ]
    assert results[0][2] == results[1][2]
    assert "synthetic prepare submit failure" in results[0][2]
    assert [result[3] for result in results] == [False, False]


def test_round_broadcasts_rank_zero_generator_submission_failure(tmp_path):
    results = _run_gloo_workers(
        _round_generator_submission_failure_worker,
        tmp_path / "round-generator-submit-rendezvous",
    )

    assert [result[:2] for result in results] == [
        (0, "RuntimeError"),
        (1, "RuntimeError"),
    ]
    assert results[0][2] == results[1][2]
    assert "synthetic round submit failure" in results[0][2]
    assert [result[3] for result in results] == [False, False]


def test_round_preserves_pp_one_create_error(monkeypatch):
    protocol = MilesCollectiveProtocolCore(_args())

    class Session:
        membership = SimpleNamespace(group_id="group-9", epoch=4)

    class Coordinator:
        def create(self, version, *, idempotency_key):
            raise LookupError("coordinator unavailable")

    protocol._session = Session()
    protocol._coordinator = Coordinator()
    protocol._tensor_digests = _ROUND_DIGESTS
    protocol._verify_tensor_equality = True
    monkeypatch.setattr(protocol, "_prepare_sessions", lambda: None)
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(miles_protocol, "_gloo_group", lambda: object())
    monkeypatch.setattr(
        miles_protocol.dist,
        "broadcast_object_list",
        lambda *_args, **_kwargs: None,
    )
    context = SimpleNamespace(sync_base=True, adapters=(), weight_version=1)

    with pytest.raises(LookupError, match="coordinator unavailable"):
        protocol.execute_update_round(context)


def test_wait_terminal_logs_complete_operation(caplog):
    protocol = MilesCollectiveProtocolCore(_args())
    protocol._coordinator = SimpleNamespace(
        get=lambda operation_id: SimpleNamespace(
            state=miles_protocol.collective_pb.COLLECTIVE_TRANSFER_STATE_COMPLETE,
            version_id="v7",
        )
    )

    with caplog.at_level("INFO"):
        protocol._wait_terminal("operation-7")

    assert (
        "MILES NCCL M2N operation terminal operation_id=operation-7 "
        "version=v7 state=COMPLETE" in caplog.text
    )


def test_wait_terminal_uses_the_shared_transfer_deadline(monkeypatch):
    protocol = MilesCollectiveProtocolCore(_args())
    protocol._coordinator = SimpleNamespace(
        get=lambda _operation_id: SimpleNamespace(
            state=miles_protocol.collective_pb.COLLECTIVE_TRANSFER_STATE_RUNNING,
            version_id="v7",
        )
    )
    clock = iter((10.0, 18.0))
    monkeypatch.setenv("MX_NCCL_REFIT_TRANSFER_TIMEOUT_S", "7")
    monkeypatch.setattr(miles_protocol.time, "monotonic", lambda: next(clock))

    with pytest.raises(TimeoutError, match="within 7s"):
        protocol._wait_terminal("operation-7")


def test_close_reports_generator_failure_and_allows_retry(monkeypatch, caplog):
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (object(),)
    protocol._engine_gpu_offsets = (0,)
    events = []
    attempts = 0

    class Session:
        def close(self):
            events.append("session-close")

    class Channel:
        def close(self):
            events.append("channel-close")

    protocol._session = Session()
    protocol._channel = Channel()

    def generator_futures(action, **_kwargs):
        nonlocal attempts
        assert action == "close"
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic generator close failure")
        return []

    protocol._generator_futures = generator_futures
    monkeypatch.setattr(miles_protocol.dist, "is_available", lambda: True)
    monkeypatch.setattr(miles_protocol.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)

    with pytest.raises(RuntimeError, match="synthetic generator close failure"):
        protocol.close()

    assert protocol._closed is False
    assert protocol._session is None
    assert protocol._channel is None
    assert events == ["session-close", "channel-close"]

    with caplog.at_level("INFO"):
        protocol.close()

    assert attempts == 2
    assert protocol._closed is True
    assert (
        caplog.text.count("MILES NCCL M2N clean shutdown complete trainer_rank=0") == 1
    )

    protocol.close()

    assert (
        caplog.text.count("MILES NCCL M2N clean shutdown complete trainer_rank=0") == 1
    )


def test_round_submit_is_atomic_across_multiple_generator_controls(monkeypatch):
    events = []
    submissions = 0
    async_utils = ModuleType("miles.utils.async_utils")

    class Future:
        def result(self):
            events.append("close-settled")

    def submit(coroutine):
        nonlocal submissions
        submissions += 1
        if submissions == 1:
            coroutine.close()
            raise RuntimeError("synthetic aggregate submit failure")
        events.append("close-submitted")
        coroutine.close()
        return Future()

    async_utils.submit = submit
    async_utils.wait_futures = lambda futures: [future.result() for future in futures]
    utils = ModuleType("miles.utils")
    utils.async_utils = async_utils
    miles = ModuleType("miles")
    miles.utils = utils
    monkeypatch.setitem(sys.modules, "miles", miles)
    monkeypatch.setitem(sys.modules, "miles.utils", utils)
    monkeypatch.setitem(sys.modules, "miles.utils.async_utils", async_utils)

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (object(), object())
    protocol._engine_gpu_offsets = (0, 1)
    protocol._engine_gpu_counts = (1, 1)
    entered_round = False
    control_executions = 0

    async def send_control(_client, _control):
        nonlocal control_executions
        control_executions += 1

    class Session:
        membership = SimpleNamespace(group_id="group-9", epoch=4)

        def run_round(self, *, version, operation_id):
            nonlocal entered_round
            entered_round = True
            assert version == "1"
            assert operation_id == "operation-7"

        def report_failure(self, *, operation_id, error):
            assert operation_id == "operation-7"
            events.append(("operation-failed", str(error)))

        def close(self):
            events.append("session-close")

    class Coordinator:
        def create(self, version, *, idempotency_key):
            assert version == "1"
            assert idempotency_key == "miles-group-9-weight-version-1"
            return SimpleNamespace(operation_id="operation-7")

        def delete(self, operation_id):
            assert operation_id == "operation-7"
            assert any(event[0] == "operation-failed" for event in events)
            events.append("operation-deleted")

    protocol._session = Session()
    protocol._coordinator = Coordinator()
    protocol._tensor_digests = _ROUND_DIGESTS
    protocol._prepare_sessions = lambda: None
    protocol._send_control = send_control
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(miles_protocol.dist, "is_available", lambda: True)
    monkeypatch.setattr(miles_protocol.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(miles_protocol, "_gloo_group", lambda: object())
    monkeypatch.setattr(
        miles_protocol.dist,
        "all_gather_object",
        lambda output, value, **_kwargs: output.__setitem__(0, value),
    )
    monkeypatch.setattr(
        miles_protocol.dist,
        "broadcast_object_list",
        lambda *_args, **_kwargs: None,
    )
    context = SimpleNamespace(sync_base=True, adapters=(), weight_version=1)

    with pytest.raises(RuntimeError, match="synthetic aggregate submit failure"):
        protocol.execute_update_round(context)

    assert entered_round is False
    assert control_executions == 0
    assert submissions == 2
    assert events.index(("operation-failed", "synthetic aggregate submit failure")) < (
        events.index("operation-deleted")
    )
    assert events.index("operation-deleted") < events.index("close-submitted")


def test_generator_fan_out_uses_one_submit_for_multiple_engines(monkeypatch):
    async_utils = ModuleType("miles.utils.async_utils")
    submissions = []
    executions = []

    class Future:
        def __init__(self, coroutine):
            self.coroutine = coroutine

        def result(self):
            return __import__("asyncio").run(self.coroutine)

    def submit(coroutine):
        submissions.append(coroutine)
        return Future(coroutine)

    async_utils.submit = submit
    async_utils.wait_futures = lambda futures: [future.result() for future in futures]
    utils = ModuleType("miles.utils")
    utils.async_utils = async_utils
    miles = ModuleType("miles")
    miles.utils = utils
    monkeypatch.setitem(sys.modules, "miles", miles)
    monkeypatch.setitem(sys.modules, "miles.utils", utils)
    monkeypatch.setitem(sys.modules, "miles.utils.async_utils", async_utils)

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (object(), object())
    protocol._engine_gpu_offsets = (0, 4)
    protocol._engine_gpu_counts = (2, 2)
    protocol._plan = object()
    protocol._topology = object()

    async def send_control(_client, control):
        executions.append(control.generator_slot_offset)

    protocol._send_control = send_control

    futures = protocol._generator_futures("prepare")
    protocol._wait_generator_futures(futures)

    assert len(submissions) == 1
    assert len(futures) == 1
    assert executions == [0, 2]
