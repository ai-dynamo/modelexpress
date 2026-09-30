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
        modelexpress_server_address="mx:50051",
    )


def _entries(sizes):
    mesh = miles_protocol.MeshSpec((1,), rank_offset=0)
    return [
        miles_protocol.ParamPlan(
            name=f"model.{index}",
            global_shape=(size,),
            dtype="bfloat16",
            partition_id=0,
            src_mesh=mesh,
            src_placements=(miles_protocol.Placement.replicate(),),
            dst_mesh=mesh,
            dst_placements=(miles_protocol.Placement.replicate(),),
        )
        for index, size in enumerate(sizes)
    ]


def _install_fake_miles_async(monkeypatch, events, *, submit=None, wait_futures=None):
    async_utils = ModuleType("miles.utils.async_utils")

    class Future:
        def __init__(self, coroutine):
            self.coroutine = coroutine

        def result(self):
            try:
                return self.coroutine.send(None)
            except StopIteration as stopped:
                return stopped.value

    async_utils.submit = (
        submit if submit is not None else (lambda coroutine: Future(coroutine))
    )

    def default_wait_futures(futures):
        events.append("wait-generators")
        return [future.result() for future in futures]

    async_utils.wait_futures = (
        wait_futures if wait_futures is not None else default_wait_futures
    )
    utils = ModuleType("miles.utils")
    utils.async_utils = async_utils
    miles = ModuleType("miles")
    miles.utils = utils
    monkeypatch.setitem(sys.modules, "miles", miles)
    monkeypatch.setitem(sys.modules, "miles.utils", utils)
    monkeypatch.setitem(sys.modules, "miles.utils.async_utils", async_utils)


def _stub_single_rank_collectives(monkeypatch):
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)
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


def _armed_protocol(monkeypatch, publish_groups=1):
    """A connected protocol with one begin_sync round armed and a fake session."""
    args = _args()
    args.modelexpress_m2n_publish_groups = publish_groups
    protocol = MilesCollectiveProtocolCore(args)
    protocol.connect(
        [object()],
        [1],
        [0],
        _parallel_state(pp_size=1),
        _placement(),
        "target",
    )
    _stub_single_rank_collectives(monkeypatch)
    events = []

    class Session:
        membership = SimpleNamespace(group_id="group-9")

        def begin_round(self, *, version):
            events.append(("begin", version))

        def publish_group(self, *, version, layer_group_id):
            events.append(("publish", version, layer_group_id))

        def finish_round(self, *, version):
            events.append(("finish", version))

        def close(self):
            events.append("session-close")

    session = Session()
    monkeypatch.setattr(
        protocol,
        "_prepare_sessions",
        lambda: setattr(protocol, "_session", session),
    )

    def generator_futures(action, **kwargs):
        events.append(("submit", action, kwargs))
        return ["future"]

    monkeypatch.setattr(protocol, "_generator_futures", generator_futures)
    monkeypatch.setattr(
        protocol,
        "_wait_generator_futures",
        lambda futures: events.append(("wait", list(futures))),
    )
    tensors = {
        name: torch.full((2, 2), index + 1, dtype=torch.bfloat16)
        for index, name in enumerate(("model.a", "model.b", "model.c"))
    }
    protocol.begin_sync(1, lambda *, materialize: iter([list(tensors.items())]))
    return protocol, session, events, tensors


def test_endpoint_rejects_unsupported_secure_schemes():
    for scheme in ("https", "grpcs"):
        args = _args()
        args.modelexpress_server_address = f"{scheme}://mx.example:50051"

        with pytest.raises(ValueError, match="secure ModelExpress endpoints"):
            miles_protocol._server_endpoint(args)


def test_validate_args_requires_a_server_address(monkeypatch):
    monkeypatch.delenv("MX_SERVER_ADDRESS", raising=False)
    args = _args()
    args.modelexpress_server_address = None

    with pytest.raises(ValueError, match="--modelexpress-server-address"):
        miles_protocol.build_protocol.validate_args(args)

    miles_protocol.build_protocol.validate_args(_args())


def test_validate_args_accepts_the_env_fallback(monkeypatch):
    monkeypatch.setenv("MX_SERVER_ADDRESS", "mx-env:50051")
    args = _args()
    args.modelexpress_server_address = None

    miles_protocol.build_protocol.validate_args(args)


def test_validate_args_rejects_secure_schemes():
    args = _args()
    args.modelexpress_server_address = "grpcs://mx:50051"

    with pytest.raises(ValueError, match="invalid modelexpress_server_address"):
        miles_protocol.build_protocol.validate_args(args)


@pytest.mark.parametrize("value", [0, -2, "two", True])
def test_validate_args_rejects_non_positive_publish_groups(value):
    args = _args()
    args.modelexpress_m2n_publish_groups = value

    with pytest.raises(ValueError, match="modelexpress_m2n_publish_groups"):
        miles_protocol.build_protocol.validate_args(args)


def test_validate_args_rejects_a_bad_connect_timeout():
    args = _args()
    args.modelexpress_m2n_connect_timeout_s = "never"

    with pytest.raises(ValueError, match="modelexpress_m2n_connect_timeout_s"):
        miles_protocol.build_protocol.validate_args(args)


def test_endpoint_requires_an_address_when_neither_source_is_set(monkeypatch):
    monkeypatch.delenv("MX_SERVER_ADDRESS", raising=False)

    with pytest.raises(ValueError, match="--modelexpress-server-address"):
        miles_protocol._server_endpoint(SimpleNamespace())


def test_check_response_requires_an_explicit_success_field():
    with pytest.raises(RuntimeError, match="no success=true"):
        miles_protocol._check_response({})
    with pytest.raises(RuntimeError, match="quiet failure"):
        miles_protocol._check_response(SimpleNamespace(message="quiet failure"))
    with pytest.raises(RuntimeError, match="boom"):
        miles_protocol._check_response({"success": False, "message": "boom"})
    with pytest.raises(RuntimeError, match="no success=true"):
        miles_protocol._check_response(None)

    assert miles_protocol._check_response({"success": True}) == {"success": True}
    response = SimpleNamespace(success=True, message="ok")
    assert miles_protocol._check_response(response) is response


def test_connect_rejects_a_missing_parallel_state_dimension():
    protocol = MilesCollectiveProtocolCore(_args())
    state = _parallel_state(pp_size=1)
    del state.tp

    with pytest.raises(ValueError, match="one source rank per PP partition"):
        protocol.connect(
            [object()],
            [1],
            [0],
            state,
            _placement(),
            "target",
        )


def test_publish_group_count_falls_back_to_the_env(monkeypatch):
    monkeypatch.setenv("MX_MILES_PUBLISH_GROUPS", "3")

    assert miles_protocol._publish_group_count(SimpleNamespace()) == 3


def test_abi_version_defaults_and_falls_back_to_the_env(monkeypatch):
    monkeypatch.delenv("MX_MILES_ABI_VERSION", raising=False)
    assert (
        miles_protocol._abi_version(SimpleNamespace())
        == "miles-sglang-bf16-replicated-v1"
    )

    monkeypatch.setenv("MX_MILES_ABI_VERSION", "abi-9")
    assert miles_protocol._abi_version(SimpleNamespace()) == "abi-9"


def test_validate_args_rejects_an_empty_abi_version():
    args = _args()
    args.modelexpress_m2n_abi_version = " "

    with pytest.raises(ValueError, match="modelexpress_m2n_abi_version"):
        miles_protocol.build_protocol.validate_args(args)


def test_chunk_publish_groups_balances_bytes_in_canonical_order():
    entries = _entries([2, 2, 2, 6])

    assert miles_protocol._chunk_publish_groups(entries, 1) == (
        ("model.0", "model.1", "model.2", "model.3"),
    )
    assert miles_protocol._chunk_publish_groups(entries, 2) == (
        ("model.0", "model.1", "model.2"),
        ("model.3",),
    )
    assert miles_protocol._chunk_publish_groups(entries, 3) == (
        ("model.0", "model.1"),
        ("model.2",),
        ("model.3",),
    )
    assert miles_protocol._chunk_publish_groups(entries, 64) == (
        ("model.0",),
        ("model.1",),
        ("model.2",),
        ("model.3",),
    )
    with pytest.raises(ValueError, match="positive"):
        miles_protocol._chunk_publish_groups(entries, 0)


def test_speculative_updates_require_target_only_selector():
    args = _args()
    args.sglang_speculative_algorithm = "EAGLE"
    protocol = MilesCollectiveProtocolCore(args)

    with pytest.raises(ValueError, match="frozen draft"):
        protocol._validate_selector("all")

    protocol._validate_selector("target")


def test_selector_is_validated_without_speculative_args():
    protocol = MilesCollectiveProtocolCore(_args())

    protocol._validate_selector("all")
    protocol._validate_selector("target")
    with pytest.raises(ValueError, match="base target model only"):
        protocol._validate_selector("draft")


def test_prepare_sessions_uses_auth_wrapped_channel(monkeypatch):
    _install_fake_miles_async(monkeypatch, [])
    raw_channel = object()
    captured = {}
    sessions = []
    created_groups = []

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
        created_groups.append(kwargs["layer_groups"])
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
    monkeypatch.setattr(miles_protocol, "_await_endpoint_ready", lambda *a, **k: None)
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
        protocol._publish_groups = (("model.weight",),)
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
    assert created_groups == [(("model.weight",),), (("model.weight",),)]
    assert first._channel is auth_channel
    assert second._channel is auth_channel
    assert len(sessions) == 2
    assert sessions[0].worker_id.startswith("miles-trainer-0-")
    assert sessions[1].worker_id.startswith("miles-trainer-0-")
    assert sessions[0].worker_id != sessions[1].worker_id
    assert all(len(session.worker_id.rsplit("-", 1)[1]) == 32 for session in sessions)


def test_prepare_sessions_closes_the_channel_when_the_server_is_unreachable(
    monkeypatch,
):
    protocol = MilesCollectiveProtocolCore(_args())
    protocol._tensors = {"model.weight": torch.ones((1,), dtype=torch.bfloat16)}
    protocol._topology = SimpleNamespace(trainer_slots=("trainer-0",))
    protocol._plan = SimpleNamespace(
        bulk=[SimpleNamespace(name="model.weight", partition_id=0)],
        parameter_names=lambda: ("model.weight",),
    )
    protocol._publish_groups = (("model.weight",),)
    closed = []

    class Channel:
        def close(self):
            closed.append("channel")

    monkeypatch.setattr(
        miles_protocol.grpc, "insecure_channel", lambda _endpoint: Channel()
    )
    monkeypatch.setattr(miles_protocol.auth, "with_auth", lambda channel: channel)
    _stub_single_rank_collectives(monkeypatch)

    def refuse(_channel, *, endpoint, timeout_s):
        raise RuntimeError(f"cannot reach the ModelExpress server at {endpoint!r}")

    monkeypatch.setattr(miles_protocol, "_await_endpoint_ready", refuse)

    with pytest.raises(RuntimeError, match="cannot reach the ModelExpress server"):
        protocol._prepare_sessions()

    assert closed == ["channel"]
    assert protocol._session is None


def test_endpoint_probe_reports_the_address_and_remediation(monkeypatch):
    class ReadyFuture:
        def result(self, timeout=None):
            raise miles_protocol.grpc.FutureTimeoutError()

    monkeypatch.setattr(
        miles_protocol.grpc,
        "channel_ready_future",
        lambda _channel: ReadyFuture(),
    )

    with pytest.raises(RuntimeError, match="MX_SERVER_ADDRESS") as raised:
        miles_protocol._await_endpoint_ready(
            object(), endpoint="mx:50051", timeout_s=0.01
        )

    assert "mx:50051" in str(raised.value)


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
    original_clone = torch.Tensor.clone
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

            def fail_wire_allocation(self, *args, **kwargs):
                nonlocal failed
                if not failed and tuple(self.shape) == (2, 2):
                    failed = True
                    raise torch.OutOfMemoryError("synthetic wire allocation failure")
                return original_clone(self, *args, **kwargs)

            torch.Tensor.clone = fail_wire_allocation
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
        torch.Tensor.clone = original_clone
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
        protocol._publish_groups = ((f"rank-{rank}.weight",),)

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
        miles_protocol._await_endpoint_ready = lambda *a, **k: None
        miles_protocol.CollectiveRendezvous = lambda _channel: object()
        miles_protocol.MilesPublisher = lambda **_kwargs: object()
        miles_protocol.MilesTrainerSession.create = lambda **_kwargs: Session()

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
            membership = SimpleNamespace(group_id="group-9")

            def begin_round(self, *, version):
                nonlocal entered_round
                entered_round = True

            def close(self):
                pass

        protocol._session = Session()
        protocol._prepare_sessions = lambda: None
        protocol._publish_groups = (("model.weight",),)
        protocol._group_of = {"model.weight": 0}
        protocol._local_names = {"model.weight"}
        protocol._round_version = "1"
        protocol._round_seen = set()
        protocol._pending = [{"model.weight"}]
        protocol._next_group = 0
        protocol._round_begun = False
        protocol._round_futures = []

        def generator_futures(action, **_kwargs):
            if rank == 0 and action == "run_round":
                raise RuntimeError("synthetic round submit failure")
            return []

        protocol._generator_futures = generator_futures
        try:
            protocol.send_bucket(
                [("model.weight", torch.ones((1,), dtype=torch.bfloat16))]
            )
        except RuntimeError as error:
            results.put((rank, type(error).__name__, str(error), entered_round))
        else:
            results.put((rank, "no-error", "", entered_round))
    finally:
        dist.destroy_process_group()


def test_lazy_factory_uses_the_bucket_stream_seam(monkeypatch):
    protocol_module = ModuleType("miles.backends.training_utils.weight_update.protocol")

    class WeightTransferProtocol:
        supports_lora = False
        use_weight_update_session = True

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

    assert protocol.required_placement.gather_pp is False
    assert protocol.use_weight_update_session is True
    assert protocol.supports_lora is False
    # The MILES external-protocol loader rejects whole-round protocols.
    assert getattr(protocol, "owns_update_round", False) is False
    assert protocol.is_sender is None
    assert callable(miles_protocol.build_protocol.validate_args)


@pytest.mark.parametrize("generator_count", [1, 6, 64])
def test_begin_sync_storage_does_not_scale_with_generator_count(
    monkeypatch,
    generator_count,
):
    protocol = MilesCollectiveProtocolCore(_args())
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
        if isinstance(value, tuple):
            output[:] = [value, (value[0], "peer-generated")]
            return
        if value is None:
            output[:] = [None, None]
            return
        if isinstance(value, str):
            output[:] = [value, ""]
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
    # Canonical plan order sorts "model.other" before "model.weight".
    assert [entry.name for entry in protocol._plan.bulk] == [
        "model.other",
        "model.weight",
    ]
    entry = protocol._plan.bulk[1]
    assert entry.src_mesh.shape == (1,)
    assert entry.global_shape == tuple(source.shape)
    assert entry.src_placements == (miles_protocol.Placement.replicate(),)
    assert entry.dst_mesh.shape == (generator_count,)
    assert entry.dst_placements == (miles_protocol.Placement.replicate(),)
    assert entry.group_key == "publish-group-0"
    assert protocol._publish_groups == (("model.other", "model.weight"),)
    assert torch.equal(wire, source)
    assert protocol._round_version == "1"

    protocol._disarm_round()
    address = wire.data_ptr()
    replacement = torch.full((2, 4), 9, dtype=torch.bfloat16)
    protocol.begin_sync(
        2,
        lambda *, materialize: iter([[("model.weight", replacement)]]),
    )

    assert protocol._tensors["model.weight"].data_ptr() == address
    assert torch.equal(protocol._tensors["model.weight"], replacement)
    assert protocol._round_version == "2"


@pytest.mark.parametrize(
    ("requested_run_id", "gathered_run_ids"),
    [
        (None, [(None, "g0"), ("run-7", "g1")]),
        ("run-7", [(None, "g0"), ("run-7", "g1")]),
        ("run-7", [("run-7", "g0"), ("run-8", "g1")]),
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
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)
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


def test_connect_marks_every_trainer_rank_a_sender():
    protocol = MilesCollectiveProtocolCore(_args())
    assert protocol.is_sender is None

    protocol.connect(
        [object()],
        [2],
        [0],
        _parallel_state(),
        _placement(),
        "target",
    )

    assert protocol.is_sender is True


def test_connect_with_a_live_session_tears_down_and_reprepares(monkeypatch):
    protocol, session, events, tensors = _armed_protocol(monkeypatch)
    for name in ("model.a", "model.b", "model.c"):
        protocol.send_bucket([(name, tensors[name])])
    protocol.finalize(1)
    assert protocol._session is session

    # miles re-calls connect when the rollout engine set heals: same GPU
    # topology, new engine handles. The stale session must not survive.
    protocol.connect(
        [object()],
        [1],
        [0],
        _parallel_state(pp_size=1),
        _placement(),
        "target",
    )

    assert events.count("session-close") == 1
    assert protocol._session is None

    class Session:
        membership = SimpleNamespace(group_id="group-10")

        def begin_round(self, *, version):
            events.append(("begin", version))

        def publish_group(self, *, version, layer_group_id):
            events.append(("publish", version, layer_group_id))

        def finish_round(self, *, version):
            events.append(("finish", version))

        def close(self):
            events.append("session-2-close")

    healed_session = Session()

    def prepare_again():
        events.append("reprepare")
        protocol._session = healed_session

    protocol._prepare_sessions = prepare_again

    protocol.begin_sync(2, lambda *, materialize: iter([list(tensors.items())]))
    for name in ("model.a", "model.b", "model.c"):
        protocol.send_bucket([(name, tensors[name])])
    protocol.finalize(2)

    assert "reprepare" in events
    assert protocol._session is healed_session
    assert ("begin", "2") in events
    assert ("finish", "2") in events


def test_connect_with_a_changed_engine_topology_fails_closed_at_begin_sync(
    monkeypatch,
):
    protocol, _session, _events, tensors = _armed_protocol(monkeypatch)
    for name in ("model.a", "model.b", "model.c"):
        protocol.send_bucket([(name, tensors[name])])
    protocol.finalize(1)

    protocol.connect(
        [object(), object()],
        [1, 1],
        [0, 1],
        _parallel_state(pp_size=1),
        _placement(),
        "target",
    )

    with pytest.raises(RuntimeError, match="topology changed"):
        protocol.begin_sync(2, lambda *, materialize: iter([list(tensors.items())]))


def test_connect_rejects_a_reconnect_mid_round_and_after_close(monkeypatch):
    protocol, _session, _events, _tensors = _armed_protocol(monkeypatch)

    with pytest.raises(RuntimeError, match="in flight"):
        protocol.connect(
            [object()],
            [1],
            [0],
            _parallel_state(pp_size=1),
            _placement(),
            "target",
        )

    protocol.close()

    with pytest.raises(RuntimeError, match="protocol is closed"):
        protocol.connect(
            [object()],
            [1],
            [0],
            _parallel_state(pp_size=1),
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
    protocol._publish_groups = (("rank-zero.weight", "rank-one.weight"),)
    _stub_single_rank_collectives(monkeypatch)
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 1)

    with pytest.raises(RuntimeError, match="ownership does not match"):
        protocol._prepare_sessions()


def test_bucket_stream_publishes_groups_in_plan_order_and_finishes(monkeypatch):
    protocol, _session, events, tensors = _armed_protocol(monkeypatch, publish_groups=2)

    assert protocol._publish_groups == (("model.a",), ("model.b", "model.c"))
    assert [entry.group_key for entry in protocol._plan.bulk] == [
        "publish-group-0",
        "publish-group-1",
        "publish-group-1",
    ]

    protocol.send_bucket([("model.a", tensors["model.a"])])
    protocol.send_bucket(
        [("model.b", tensors["model.b"]), ("model.c", tensors["model.c"])]
    )
    protocol.finalize(1)

    assert events == [
        (
            "submit",
            "run_round",
            {
                "version": "1",
                "operation_id": "miles-group-9-weight-version-1",
            },
        ),
        ("begin", "1"),
        ("publish", "1", 0),
        ("publish", "1", 1),
        ("finish", "1"),
        ("wait", ["future"]),
    ]
    assert protocol._round_version is None

    protocol.begin_sync(2, lambda *, materialize: iter([list(tensors.items())]))

    assert protocol._round_version == "2"


def test_send_bucket_rejects_unknown_tensors_and_closes_the_protocol(monkeypatch):
    protocol, _session, _events, tensors = _armed_protocol(monkeypatch)

    with pytest.raises(ValueError, match="outside the frozen plan"):
        protocol.send_bucket([("model.unknown", tensors["model.a"])])

    # A bucket stream that diverged from the frozen contract is a mid-round
    # failure: the protocol closes rather than continuing a degraded round.
    assert protocol._closed is True
    with pytest.raises(RuntimeError, match="protocol is closed"):
        protocol.send_bucket([("model.a", tensors["model.a"])])


def test_send_bucket_rejects_repeated_tensors_and_closes_the_protocol(monkeypatch):
    protocol, _session, _events, tensors = _armed_protocol(monkeypatch)
    protocol.send_bucket([("model.a", tensors["model.a"])])

    with pytest.raises(ValueError, match="repeats"):
        protocol.send_bucket([("model.a", tensors["model.a"])])

    assert protocol._closed is True


def test_send_bucket_rejects_tensors_owned_by_another_partition(monkeypatch):
    protocol, _session, _events, tensors = _armed_protocol(monkeypatch)
    # A name the global frozen plan knows (so it passes the unknown check)
    # but that belongs to a different PP partition's lane.
    protocol._group_of["model.foreign"] = 0

    with pytest.raises(ValueError, match="another PP partition"):
        protocol.send_bucket([("model.foreign", tensors["model.a"])])

    assert "model.foreign" not in protocol._round_seen
    assert protocol._closed is True


def test_send_bucket_and_finalize_require_an_armed_round():
    protocol = MilesCollectiveProtocolCore(_args())

    with pytest.raises(RuntimeError, match="begin_sync must arm a round"):
        protocol.send_bucket([("model.a", torch.ones((1,), dtype=torch.bfloat16))])
    with pytest.raises(RuntimeError, match="begin_sync must arm a round"):
        protocol.finalize(1)


def test_begin_sync_rejects_a_new_round_before_finalize(monkeypatch):
    protocol, _session, _events, tensors = _armed_protocol(monkeypatch)

    with pytest.raises(RuntimeError, match="never finalized"):
        protocol.begin_sync(2, lambda *, materialize: iter([list(tensors.items())]))


def test_finalize_must_name_the_armed_version(monkeypatch):
    protocol, _session, _events, _tensors = _armed_protocol(monkeypatch)

    with pytest.raises(RuntimeError, match="round in flight"):
        protocol.finalize(2)


def test_finalize_fails_loudly_when_no_buckets_arrived(monkeypatch):
    protocol, _session, events, _tensors = _armed_protocol(monkeypatch)

    with pytest.raises(RuntimeError, match="no weight buckets"):
        protocol.finalize(1)

    assert ("begin", "1") not in events


def test_finalize_fails_loudly_when_the_bucket_stream_is_incomplete(monkeypatch):
    protocol, _session, events, tensors = _armed_protocol(monkeypatch, publish_groups=2)
    protocol.send_bucket([("model.a", tensors["model.a"])])

    with pytest.raises(
        RuntimeError, match="bucket stream ended before publish group 1"
    ):
        protocol.finalize(1)

    assert protocol._closed is True
    assert ("finish", "1") not in events
    assert protocol._round_version is None


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

    # The first close attempt is terminal for rounds even though teardown
    # itself failed; only the remaining cleanup may be retried.
    assert protocol._closed is True
    assert protocol._close_pending is True
    assert protocol._session is None
    assert protocol._channel is None
    assert events == ["session-close", "channel-close"]

    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(miles_protocol, "_gloo_group", lambda: object())
    monkeypatch.setattr(
        miles_protocol.dist,
        "all_gather_object",
        lambda output, value, **_kwargs: output.__setitem__(0, value),
    )
    with pytest.raises(RuntimeError, match="protocol is closed"):
        protocol.begin_sync(2, lambda *, materialize: iter([]))

    with caplog.at_level("INFO"):
        protocol.close()

    assert attempts == 2
    assert protocol._closed is True
    assert protocol._close_pending is False
    assert caplog.text.count("MILES NCCL M2N teardown complete trainer_rank=0") == 1

    protocol.close()

    assert caplog.text.count("MILES NCCL M2N teardown complete trainer_rank=0") == 1


def test_close_retains_a_resource_whose_teardown_failed_for_retry(monkeypatch):
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (object(),)
    protocol._engine_gpu_offsets = (0,)
    channel_closes = 0

    class Channel:
        def close(self):
            nonlocal channel_closes
            channel_closes += 1
            if channel_closes == 1:
                raise RuntimeError("synthetic channel close failure")

    protocol._channel = Channel()

    def generator_futures(action, **_kwargs):
        assert action == "close"
        return []

    protocol._generator_futures = generator_futures
    monkeypatch.setattr(miles_protocol.dist, "is_available", lambda: True)
    monkeypatch.setattr(miles_protocol.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)

    with pytest.raises(RuntimeError, match="synthetic channel close failure"):
        protocol.close()

    assert protocol._closed is True
    assert protocol._close_pending is True
    # The failed resource is retained, so the retry re-runs its close for
    # real instead of only re-sending the generator fan-out.
    assert protocol._channel is not None

    protocol.close()

    assert channel_closes == 2
    assert protocol._channel is None
    assert protocol._close_pending is False


def test_close_on_a_never_connected_protocol_is_a_silent_no_op(caplog):
    protocol = MilesCollectiveProtocolCore(_args())

    with caplog.at_level("INFO"):
        protocol.close()

    assert protocol._closed is True
    assert protocol._close_pending is False
    assert "teardown complete" not in caplog.text


def _seed_real_fan_out_contract(protocol):
    protocol._plan = miles_protocol.ReshardPlan(
        bulk=_entries([2]),
        source_partition_count=1,
    )
    protocol._topology = miles_protocol.CollectiveTopology(
        model_name="miles-model",
        trainer_slots=("trainer-0",),
        generator_slots=("generator-0", "generator-1"),
        source_partition_count=1,
        m2n_abi_version="miles-sglang-bf16-replicated-v1",
    )


def test_generator_fan_out_uses_one_submit_for_multiple_engines(monkeypatch):
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

    _install_fake_miles_async(
        monkeypatch,
        [],
        submit=submit,
        wait_futures=lambda futures: [future.result() for future in futures],
    )

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (object(), object())
    protocol._engine_gpu_offsets = (0, 4)
    protocol._engine_gpu_counts = (2, 2)
    _seed_real_fan_out_contract(protocol)

    async def send_control(_client, control, **_kwargs):
        executions.append(control.generator_slot_offset)

    protocol._send_control = send_control

    futures = protocol._generator_futures("prepare")
    protocol._wait_generator_futures(futures)

    assert len(submissions) == 1
    assert len(futures) == 1
    assert executions == [0, 2]


def test_generator_fan_out_settles_every_engine_before_raising(monkeypatch):
    executions = []

    class Future:
        def __init__(self, coroutine):
            self.coroutine = coroutine

        def result(self):
            return __import__("asyncio").run(self.coroutine)

    _install_fake_miles_async(
        monkeypatch,
        [],
        submit=lambda coroutine: Future(coroutine),
        wait_futures=lambda futures: [future.result() for future in futures],
    )

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (object(), object())
    protocol._engine_gpu_offsets = (0, 4)
    protocol._engine_gpu_counts = (2, 2)
    _seed_real_fan_out_contract(protocol)

    async def send_control(_client, control, **_kwargs):
        executions.append(control.generator_slot_offset)
        if control.generator_slot_offset == 0:
            raise RuntimeError("engine 0 prepare failed")

    protocol._send_control = send_control

    futures = protocol._generator_futures("prepare")
    with pytest.raises(RuntimeError, match="engine 0 prepare failed"):
        protocol._wait_generator_futures(futures)

    assert executions == [0, 2]


class _DroppedFuture:
    def __init__(self, *, cancellable=True):
        self.cancellable = cancellable
        self.cancelled = False
        self.callbacks = []

    def cancel(self):
        if not self.cancellable:
            return False
        self.cancelled = True
        return True

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def result(self):
        raise RuntimeError("late generator failure")


def test_retire_dropped_futures_observes_futures_that_refuse_to_cancel(caplog):
    future = _DroppedFuture(cancellable=False)

    with caplog.at_level("WARNING"):
        miles_protocol._retire_dropped_futures([future])

    assert future.cancelled is False
    assert len(future.callbacks) == 1
    future.callbacks[0](future)
    assert "failed late" in caplog.text


def test_finalize_retires_dropped_rank_zero_futures_when_the_round_fails(monkeypatch):
    protocol, session, _events, tensors = _armed_protocol(monkeypatch)
    for name in ("model.a", "model.b", "model.c"):
        protocol.send_bucket([(name, tensors[name])])
    dropped = _DroppedFuture()
    protocol._round_futures = [dropped]

    def finish_round(*, version):
        raise RuntimeError("synthetic finish failure")

    session.finish_round = finish_round

    with pytest.raises(RuntimeError, match="round failed"):
        protocol.finalize(1)

    assert dropped.cancelled is True
    assert dropped.callbacks == []


def test_prepare_sessions_retires_dropped_generator_futures_on_failure(monkeypatch):
    _install_fake_miles_async(monkeypatch, [])
    protocol = MilesCollectiveProtocolCore(_args())
    protocol._tensors = {"model.weight": torch.ones((1,), dtype=torch.bfloat16)}
    protocol._topology = SimpleNamespace(trainer_slots=("trainer-0",))
    protocol._plan = SimpleNamespace(
        bulk=[SimpleNamespace(name="model.weight", partition_id=0)],
        parameter_names=lambda: ("model.weight",),
    )
    protocol._publish_groups = (("model.weight",),)
    dropped = _DroppedFuture()

    class Channel:
        def close(self):
            pass

    class Session:
        def prepare(self):
            raise RuntimeError("synthetic session prepare failure")

        def close(self):
            pass

    monkeypatch.setattr(
        miles_protocol.grpc, "insecure_channel", lambda _endpoint: Channel()
    )
    monkeypatch.setattr(miles_protocol.auth, "with_auth", lambda channel: channel)
    monkeypatch.setattr(miles_protocol, "_await_endpoint_ready", lambda *a, **k: None)
    monkeypatch.setattr(
        miles_protocol, "CollectiveRendezvous", lambda channel: object()
    )
    monkeypatch.setattr(miles_protocol, "MilesPublisher", lambda **_kwargs: object())
    monkeypatch.setattr(
        miles_protocol.MilesTrainerSession, "create", lambda **_kwargs: Session()
    )
    _stub_single_rank_collectives(monkeypatch)
    monkeypatch.setattr(
        protocol, "_generator_futures", lambda action, **kwargs: [dropped]
    )

    with pytest.raises(RuntimeError, match="session preparation failed"):
        protocol._prepare_sessions()

    assert dropped.cancelled is True
    assert dropped.callbacks == []
