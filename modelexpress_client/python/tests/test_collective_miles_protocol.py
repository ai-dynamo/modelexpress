# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the concrete optional MILES collective protocol."""

import asyncio
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from modelexpress_rl.collective.integrations import miles_protocol
from modelexpress_rl.collective.integrations.miles_protocol import (
    MilesCollectiveProtocolCore,
)

try:
    import httpx
except ImportError:
    # The MX-only test environment carries the renamed httpx2 line instead of
    # httpx. MILES pins classic httpx, which is what its API client raises.
    # Aliasing keeps the fakes and the lazy import in miles_protocol on the
    # same class objects.
    import httpx2 as httpx

    sys.modules.setdefault("httpx", httpx)


def _engine_http_error(endpoint: str, status: int, message: str):
    """Raise the way MILES's ``_make_request`` does on a non-2xx SG-1 answer."""
    request = httpx.Request("POST", f"http://engine/{endpoint}")
    response = httpx.Response(
        status,
        json={"success": False, "message": message},
        request=request,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        # MILES notes the body on the error; add_note is 3.11+ and the
        # protocol's tolerance reads error.response, so it is optional.
        if hasattr(error, "add_note"):
            error.add_note(f"{response.text=}")
        raise


@pytest.fixture(autouse=True)
def _server_address(monkeypatch):
    # The protocol resolves the mx-server endpoint at construction time;
    # isolate from the deprecated alias a developer shell might export.
    monkeypatch.setenv("MX_SERVER_ADDRESS", "mx:50051")
    monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
    monkeypatch.delenv("MX_MILES_DST_LAYOUT", raising=False)
    monkeypatch.delenv("MX_NCCL_REFIT_STACK_BYTES", raising=False)
    monkeypatch.delenv("NCCL_RESHARD_PACK_BUFFSIZES", raising=False)


class _Group:
    def __init__(self, size=1, rank=0):
        self.size = size
        self.rank = rank


def _parallel_state(pp_size=1):
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
    return SimpleNamespace()


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
        """Starts the coroutine at submission, like the real background loop."""

        def __init__(self, coroutine):
            self._error = None
            self._value = None
            try:
                coroutine.send(None)
            except StopIteration as stopped:
                self._value = stopped.value
            except BaseException as error:
                self._error = error
            else:
                coroutine.close()
                raise RuntimeError(
                    "the fake async loop only supports non-suspending coroutines"
                )

        def result(self):
            if self._error is not None:
                raise self._error
            return self._value

    async_utils.submit = (
        submit if submit is not None else (lambda coroutine: Future(coroutine))
    )

    def default_wait_futures(futures):
        # Mirror MILES's wait_futures: settle every future, log each failure
        # with exc_info, then raise the first error.
        events.append("wait-generators")
        results, errors = [], []
        for index, future in enumerate(futures):
            try:
                results.append(future.result())
            except Exception as error:
                logging.getLogger("miles.utils.async_utils").warning(
                    "wait_futures index=%d failed", index, exc_info=error
                )
                results.append(None)
                errors.append(error)
        if errors:
            raise errors[0]
        return results

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


class _DroppedFuture:
    """A future stub: cancel() reports rather than raises, like the real API."""

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


def _armed_protocol(monkeypatch):
    """A connected protocol with one begin_sync round armed and a fake session."""
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.connect(
        [object()],
        [1],
        [0],
        _parallel_state(pp_size=1),
        _placement(),
        "target",
    )
    events = []

    class Session:
        membership = SimpleNamespace(group_id="group-9")

        def begin_round(self, *, version):
            events.append(("begin", version))

        def publish_group(self, *, version, layer_group_id):
            events.append(("publish", version, layer_group_id))

        def create_transfer(self, *, version):
            return "845aaea0-f64b-4f2e-b212-5af940b4c169"

        def finish_round(self, *, version, operation_id):
            assert operation_id == "845aaea0-f64b-4f2e-b212-5af940b4c169"
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
        return [_DroppedFuture()]

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


def test_endpoint_rejects_unsupported_secure_schemes(monkeypatch):
    for scheme in ("https", "grpcs"):
        monkeypatch.setenv("MX_SERVER_ADDRESS", f"{scheme}://mx.example:50051")

        with pytest.raises(ValueError, match="secure ModelExpress endpoints"):
            miles_protocol._server_endpoint()


def test_endpoint_strips_insecure_schemes(monkeypatch):
    for scheme in ("grpc", "http"):
        monkeypatch.setenv("MX_SERVER_ADDRESS", f"{scheme}://mx.example:50051")

        assert miles_protocol._server_endpoint() == "mx.example:50051"


def test_endpoint_resolves_model_express_url_with_precedence(monkeypatch):
    # The shared client resolver reads MODEL_EXPRESS_URL first (deprecated
    # alias), then MX_SERVER_ADDRESS; this path keeps that precedence but
    # requires one of them instead of defaulting to localhost.
    monkeypatch.setenv("MODEL_EXPRESS_URL", "legacy:50051")

    assert miles_protocol._server_endpoint() == "legacy:50051"

    monkeypatch.delenv("MODEL_EXPRESS_URL")

    assert miles_protocol._server_endpoint() == "mx:50051"


def test_endpoint_requires_mx_server_address(monkeypatch):
    monkeypatch.delenv("MX_SERVER_ADDRESS", raising=False)
    monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)

    with pytest.raises(ValueError, match="MX_SERVER_ADDRESS"):
        miles_protocol._server_endpoint()


def test_constructing_the_protocol_requires_mx_server_address(monkeypatch):
    monkeypatch.delenv("MX_SERVER_ADDRESS", raising=False)
    monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)

    with pytest.raises(ValueError, match="MX_SERVER_ADDRESS"):
        MilesCollectiveProtocolCore(_args())


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

    with pytest.raises(ValueError, match="parallel state 'tp'"):
        protocol.connect(
            [object()],
            [1],
            [0],
            state,
            _placement(),
            "target",
        )


def test_connect_rejects_more_than_one_pipeline_stage():
    protocol = MilesCollectiveProtocolCore(_args())

    with pytest.raises(ValueError, match="DP x TP trainer only"):
        protocol.connect(
            [object()],
            [1],
            [0],
            _parallel_state(pp_size=2),
            _placement(),
            "target",
        )

    assert protocol.rollout_engines is None
    assert protocol.is_sender is None


def test_speculative_decoding_is_rejected_at_validate_selector_and_connect():
    # SG-1 rejects receiver rounds while a draft model exists regardless of
    # the selector, so the adapter refuses the combination at connect.
    args = _args()
    args.sglang_speculative_algorithm = "EAGLE"
    protocol = MilesCollectiveProtocolCore(args)

    for selector in ("all", "target"):
        with pytest.raises(ValueError, match="speculative decoding"):
            protocol._validate_selector(selector)
        with pytest.raises(ValueError, match="speculative decoding"):
            protocol.connect(
                [object()],
                [1],
                [0],
                _parallel_state(pp_size=1),
                _placement(),
                selector,
            )

    assert protocol.rollout_engines is None


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


def test_lazy_factory_uses_the_bucket_stream_seam(monkeypatch):
    protocol_module = ModuleType("miles.backends.training_utils.weight_update.protocol")

    class WeightUpdatePlacement:
        def __init__(self, *, gather_pp):
            self.gather_pp = gather_pp

    class WeightTransferProtocol:
        # Mirrors the miles ABC default (weight_update/protocol.py): the
        # adapter inherits it instead of restating it.
        required_placement = WeightUpdatePlacement(gather_pp=False)
        supports_lora = False
        use_weight_update_session = True

        def __init__(self, args):
            self.args = args

    protocol_module.WeightTransferProtocol = WeightTransferProtocol
    module_names = [
        "miles",
        "miles.backends",
        "miles.backends.training_utils",
        "miles.backends.training_utils.weight_update",
    ]
    for module_name in module_names:
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    monkeypatch.setitem(sys.modules, protocol_module.__name__, protocol_module)

    protocol = miles_protocol.build_protocol(_args())

    assert protocol.required_placement.gather_pp is False
    assert protocol.use_weight_update_session is True
    assert protocol.supports_lora is False
    assert protocol.is_sender is None


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
    source = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)

    protocol.begin_sync(1, lambda *, materialize: iter([[("model.weight", source)]]))

    wire = protocol._tensors["model.weight"]
    assert tuple(wire.shape) == tuple(source.shape)
    assert wire.untyped_storage().nbytes() == source.untyped_storage().nbytes()
    assert [entry.name for entry in protocol._plan.bulk] == ["model.weight"]
    entry = protocol._plan.bulk[0]
    assert entry.src_mesh.shape == (1,)
    assert entry.global_shape == tuple(source.shape)
    assert entry.src_placements == (miles_protocol.Placement.replicate(),)
    assert entry.dst_mesh.shape == (generator_count,)
    assert entry.dst_placements == (miles_protocol.Placement.replicate(),)
    assert entry.group_key == "publish-group-0"
    assert protocol._publish_groups == (("model.weight",),)
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


def test_frozen_plan_is_one_publish_group_in_canonical_order():
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.connect(
        [object(), object()],
        [2, 1],
        [0, 2],
        _parallel_state(),
        _placement(),
        "target",
    )
    names = ("model.z", "model.a", "model.m")
    tensors = [
        (name, torch.full((2, 2), index, dtype=torch.bfloat16))
        for index, name in enumerate(names)
    ]

    protocol.begin_sync(1, lambda *, materialize: iter([tensors]))

    canonical = tuple(
        entry.name
        for entry in sorted(protocol._plan.bulk, key=lambda entry: entry.canonical())
    )
    # Identical shapes and dtypes, so the name decides the canonical order.
    assert canonical == ("model.a", "model.m", "model.z")
    assert tuple(entry.name for entry in protocol._plan.bulk) == canonical
    assert protocol._publish_groups == (canonical,)
    assert {entry.group_key for entry in protocol._plan.bulk} == {"publish-group-0"}
    assert {entry.partition_id for entry in protocol._plan.bulk} == {0}
    assert protocol._group_of == dict.fromkeys(canonical, 0)
    assert protocol._plan.source_partition_count == 1
    topology = protocol._topology
    assert topology.source_partition_count == 1
    assert topology.m2n_abi_version == miles_protocol.REPLICATED_DESTINATION_ABI
    run_id = protocol._run_id
    assert len(run_id) == 32
    assert topology.trainer_slots == (f"{run_id}:trainer-0",)
    assert topology.generator_slots == tuple(
        f"{run_id}:generator-{slot}" for slot in range(3)
    )
    # The replicate plan a single-rank trainer builds is unchanged.
    for entry in protocol._plan.bulk:
        assert entry.src_mesh == miles_protocol.MeshSpec((1,), rank_offset=0)
        assert entry.src_placements == (miles_protocol.Placement.replicate(),)
        assert entry.dst_mesh == miles_protocol.MeshSpec((3,), rank_offset=1)
        assert entry.dst_placements == (miles_protocol.Placement.replicate(),)


def test_begin_sync_preserves_pp_one_validation_error():
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.connect(
        [object()],
        [1],
        [0],
        _parallel_state(pp_size=1),
        _placement(),
        "target",
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
    with pytest.raises(ValueError, match="DP x TP trainer only"):
        state = _parallel_state()
        state.ep = _Group(2)
        protocol.connect(
            [object()],
            [6],
            [0],
            state,
            _placement(),
            "target",
        )
    with pytest.raises(ValueError, match="TP and EP gathering"):
        protocol.connect(
            [object()],
            [6],
            [0],
            _parallel_state(),
            SimpleNamespace(gather_pp=False, gather_tp=False),
            "target",
        )


def test_connect_names_the_field_for_a_non_integer_engine_gpu_topology():
    protocol = MilesCollectiveProtocolCore(_args())

    with pytest.raises(
        ValueError, match="engine_gpu_counts must be a list of integers"
    ):
        protocol.connect(
            [object()], ["a"], [0], _parallel_state(), _placement(), "target"
        )
    with pytest.raises(
        ValueError, match="engine_gpu_offsets must be a list of integers"
    ):
        protocol.connect(
            [object()], [1], ["a"], _parallel_state(), _placement(), "target"
        )


def test_connect_marks_the_single_trainer_rank_a_sender():
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

        def create_transfer(self, *, version):
            return "845aaea0-f64b-4f2e-b212-5af940b4c169"

        def finish_round(self, *, version, operation_id):
            assert operation_id == "845aaea0-f64b-4f2e-b212-5af940b4c169"
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


def test_bucket_stream_publishes_the_group_once_complete_and_finishes(monkeypatch):
    protocol, _session, events, tensors = _armed_protocol(monkeypatch)

    assert protocol._publish_groups == (("model.a", "model.b", "model.c"),)
    assert [entry.group_key for entry in protocol._plan.bulk] == [
        "publish-group-0",
        "publish-group-0",
        "publish-group-0",
    ]

    protocol.send_bucket([("model.a", tensors["model.a"])])
    # The single group is still incomplete: the round began but nothing
    # was published yet.
    assert ("begin", "1") in events
    assert not any(event[0] == "publish" for event in events)
    protocol.send_bucket(
        [("model.b", tensors["model.b"]), ("model.c", tensors["model.c"])]
    )
    protocol.finalize(1)

    assert events[:-1] == [
        (
            "submit",
            "run_round",
            {
                "version": "1",
                "operation_id": "845aaea0-f64b-4f2e-b212-5af940b4c169",
            },
        ),
        ("begin", "1"),
        ("publish", "1", 0),
        ("finish", "1"),
    ]
    # The finalize waits on exactly the future the fan-out submitted.
    assert events[-1][0] == "wait"
    assert len(events[-1][1]) == 1
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


def test_send_bucket_rejects_a_duplicate_within_one_bucket(monkeypatch):
    protocol, _session, _events, tensors = _armed_protocol(monkeypatch)

    with pytest.raises(ValueError, match="within one bucket"):
        protocol.send_bucket(
            [("model.a", tensors["model.a"]), ("model.a", tensors["model.a"])]
        )

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
    protocol, _session, events, tensors = _armed_protocol(monkeypatch)
    protocol.send_bucket([("model.a", tensors["model.a"])])

    with pytest.raises(RuntimeError, match="MILES NCCL M2N round failed") as raised:
        protocol.finalize(1)

    assert "bucket stream ended before publish group 0" in str(raised.value)
    assert "model.b" in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert protocol._closed is True
    assert not any(event[0] == "publish" for event in events)
    assert ("finish", "1") not in events
    assert protocol._round_version is None


def test_a_begin_round_failure_closes_and_retires_the_submitted_futures(
    monkeypatch,
):
    # The run_round submission succeeded, then the session's begin_round
    # raises: the terminal close must retire the submitted futures rather
    # than leak them, and no group may publish.
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (object(),)
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(miles_protocol.dist, "is_available", lambda: True)
    monkeypatch.setattr(miles_protocol.dist, "is_initialized", lambda: True)

    published = []

    class Session:
        membership = SimpleNamespace(group_id="group-9")

        def create_transfer(self, *, version):
            return "845aaea0-f64b-4f2e-b212-5af940b4c169"

        def begin_round(self, *, version):
            raise RuntimeError("synthetic begin_round failure")

        def publish_group(self, *, version, layer_group_id):
            published.append(layer_group_id)

        def close(self):
            pass

    class Future:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True
            return True

    submitted = []

    def generator_futures(action, **_kwargs):
        if action == "run_round":
            future = Future()
            submitted.append(future)
            return [future]
        return []

    protocol._session = Session()
    protocol._generator_futures = generator_futures
    protocol._publish_groups = (("model.weight",),)
    protocol._group_of = {"model.weight": 0}
    protocol._round_version = "1"
    protocol._round_seen = set()
    protocol._pending = [{"model.weight"}]
    protocol._next_group = 0
    protocol._round_begun = False
    protocol._round_futures = []

    with pytest.raises(RuntimeError, match="synthetic begin_round failure"):
        protocol.send_bucket([("model.weight", torch.ones((1,), dtype=torch.bfloat16))])

    assert published == []
    assert submitted[0].cancelled
    assert protocol._round_futures == []
    # The protocol closed terminally; it never reopens.
    with pytest.raises(RuntimeError, match="protocol is closed"):
        protocol.send_bucket([("model.weight", torch.ones((1,), dtype=torch.bfloat16))])


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
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 1)

    with pytest.raises(RuntimeError, match="synthetic generator close failure"):
        protocol.close()

    # The first close attempt is terminal for rounds even though teardown
    # itself failed; only the remaining cleanup may be retried.
    assert protocol._closed is True
    assert protocol._close_pending is True
    assert protocol._session is None
    assert protocol._channel is None
    assert events == ["session-close", "channel-close"]

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
    monkeypatch.setattr(miles_protocol.dist, "get_world_size", lambda: 1)

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


def test_close_retry_tolerates_engines_that_already_destroyed_the_group(
    monkeypatch,
):
    # SG-1 forgets a receiver even when its destroy() raises and answers a
    # repeat destroy over HTTP 400 with "does not exist" in the body: after a
    # partial fan-out, a retried close must treat that answer as done instead
    # of wedging on it.
    _install_fake_miles_async(monkeypatch, [])
    destroyed = set()

    class Engine:
        def __init__(self, name, *, fail_first_destroy=False):
            self.name = name
            self.fail_first_destroy = fail_first_destroy
            self.destroy_payloads = []

        async def _make_request(self, endpoint, payload=None):
            if endpoint == "init_weights_update_group":
                return {"success": True, "message": "ok"}
            assert endpoint == "destroy_weights_update_group"
            self.destroy_payloads.append(payload)
            if self.name in destroyed:
                # SG-1's non-idempotent destroy answer, as the wire sends it.
                _engine_http_error(
                    endpoint, 400, "The group to be destroyed does not exist."
                )
            # SG-1 pops the receiver before destroy, so even a raising
            # destroy leaves the group forgotten.
            destroyed.add(self.name)
            if self.fail_first_destroy:
                _engine_http_error(endpoint, 500, "receiver destroy blew up")
            return {"success": True, "message": "ok"}

    engine_a = Engine("a")
    engine_b = Engine("b", fail_first_destroy=True)
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (engine_a, engine_b)
    protocol._engine_gpu_offsets = (0, 2)
    protocol._engine_gpu_counts = (2, 2)
    _seed_real_fan_out_contract(protocol)
    protocol._wait_generator_futures(protocol._generator_futures("prepare"))
    group_name = protocol._engine_group_name
    monkeypatch.setattr(miles_protocol.dist, "is_available", lambda: True)
    monkeypatch.setattr(miles_protocol.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(miles_protocol.dist, "get_rank", lambda: 0)

    with pytest.raises(httpx.HTTPStatusError) as raised:
        protocol.close()

    assert raised.value.response.status_code == 500
    assert "receiver destroy blew up" in raised.value.response.text

    assert protocol._closed is True
    assert protocol._close_pending is True
    assert protocol._engine_group_name == group_name

    protocol.close()

    assert protocol._close_pending is False
    assert protocol._engine_group_name is None
    # Both engines were asked on each attempt; the retry converged by
    # accepting "does not exist" for the engines SG-1 already forgot.
    for engine in (engine_a, engine_b):
        assert [p["group_name"] for p in engine.destroy_payloads] == [
            group_name,
            group_name,
        ]


def test_close_on_a_never_connected_protocol_is_a_silent_no_op(caplog):
    protocol = MilesCollectiveProtocolCore(_args())

    with caplog.at_level("INFO"):
        protocol.close()

    assert protocol._closed is True
    assert protocol._close_pending is False
    assert "teardown complete" not in caplog.text


def test_destroy_tolerance_requires_the_400_does_not_exist_answer():
    protocol = MilesCollectiveProtocolCore(_args())

    class Engine:
        def __init__(self, status, message):
            self.status = status
            self.message = message

        async def _make_request(self, endpoint, payload=None):
            _engine_http_error(endpoint, self.status, self.message)

    # Already forgotten: the only answer close() may accept.
    asyncio.run(
        protocol._destroy_on_engine(
            Engine(400, "The group to be destroyed does not exist."), {}
        )
    )

    # The same body at another status, or another body at 400, must not pass.
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(
            protocol._destroy_on_engine(
                Engine(500, "The group to be destroyed does not exist."), {}
            )
        )
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(protocol._destroy_on_engine(Engine(400, "backend exploded"), {}))


def test_destroy_tolerance_covers_a_client_that_returns_the_failure_body():
    protocol = MilesCollectiveProtocolCore(_args())

    class DictEngine:
        async def call_endpoint(self, endpoint, payload=None):
            return {
                "success": False,
                "message": "The group to be destroyed does not exist.",
            }

    asyncio.run(protocol._destroy_on_engine(DictEngine(), {}))

    class RefusingEngine:
        async def call_endpoint(self, endpoint, payload=None):
            return {"success": False, "message": "backend exploded"}

    with pytest.raises(RuntimeError, match="backend exploded"):
        asyncio.run(protocol._destroy_on_engine(RefusingEngine(), {}))


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


class _RecordingEngine:
    """An engine client with only MILES's private ``_make_request`` call."""

    def __init__(self, executions, *, fail_at=None):
        self.executions = executions
        self.payloads = []
        self.fail_at = fail_at

    async def _make_request(self, endpoint, payload=None):
        offset = payload["rank_offset"] if "rank_offset" in payload else None
        self.executions.append((endpoint, offset))
        self.payloads.append((endpoint, payload))
        if self.fail_at is not None and offset in self.fail_at:
            # A real engine failure arrives as a non-2xx HTTP answer, raised
            # via raise_for_status by MILES's client.
            _engine_http_error(endpoint, 500, f"engine at offset {offset} failed")
        return {"success": True, "message": "ok"}


def test_generator_fan_out_submits_one_future_per_engine(monkeypatch):
    executions = []
    _install_fake_miles_async(monkeypatch, [])

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (
        _RecordingEngine(executions),
        _RecordingEngine(executions),
    )
    protocol._engine_gpu_offsets = (0, 4)
    protocol._engine_gpu_counts = (2, 2)
    _seed_real_fan_out_contract(protocol)

    futures = protocol._generator_futures("prepare")
    protocol._wait_generator_futures(futures)

    assert len(futures) == 2
    assert executions == [
        ("init_weights_update_group", 0),
        ("init_weights_update_group", 2),
    ]


def test_generator_fan_out_uses_dense_rank_offsets_for_gapped_gpu_layouts(
    monkeypatch,
):
    # Physical GPU offsets (0, 8) leave a gap, but the generator slot tuple
    # is flattened in engine order and the receiver indexes it positionally,
    # so rank_offset is the dense cursor (0, 2) — the same arithmetic
    # incumbent MILES protocols use — while the slot names (generator-0/1/8/9)
    # carry the physical placement.
    _install_fake_miles_async(monkeypatch, [])

    executions = []
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (
        _RecordingEngine(executions),
        _RecordingEngine(executions),
    )
    protocol._engine_gpu_offsets = (0, 8)
    protocol._engine_gpu_counts = (2, 2)
    _seed_real_fan_out_contract(protocol)

    futures = protocol._generator_futures("prepare")
    protocol._wait_generator_futures(futures)

    assert executions == [
        ("init_weights_update_group", 0),
        ("init_weights_update_group", 2),
    ]


def test_generator_fan_out_settles_every_engine_before_raising(monkeypatch):
    _install_fake_miles_async(monkeypatch, [])
    executions = []

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (
        _RecordingEngine(executions, fail_at={0}),
        _RecordingEngine(executions),
    )
    protocol._engine_gpu_offsets = (0, 4)
    protocol._engine_gpu_counts = (2, 2)
    _seed_real_fan_out_contract(protocol)

    futures = protocol._generator_futures("prepare")
    with pytest.raises(httpx.HTTPStatusError) as raised:
        protocol._wait_generator_futures(futures)

    assert raised.value.response.status_code == 500
    assert "engine at offset 0 failed" in raised.value.response.text
    assert executions == [
        ("init_weights_update_group", 0),
        ("init_weights_update_group", 2),
    ]


def test_generator_fan_out_logs_every_engine_failure(monkeypatch, caplog):
    _install_fake_miles_async(monkeypatch, [])

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (
        _RecordingEngine([], fail_at={0, 2}),
        _RecordingEngine([], fail_at={0, 2}),
    )
    protocol._engine_gpu_offsets = (0, 4)
    protocol._engine_gpu_counts = (2, 2)
    _seed_real_fan_out_contract(protocol)

    futures = protocol._generator_futures("prepare")
    with (
        caplog.at_level("WARNING"),
        pytest.raises(httpx.HTTPStatusError),
    ):
        protocol._wait_generator_futures(futures)

    # The first failure propagates; the rest must still reach the log with
    # their response bodies.
    assert "wait_futures index=1 failed" in caplog.text
    assert "engine at offset 2 failed" in caplog.text


def test_prepare_payload_carries_the_receiver_and_manifest(monkeypatch):
    _install_fake_miles_async(monkeypatch, [])
    engine = _RecordingEngine([])

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (engine,)
    protocol._engine_gpu_offsets = (0,)
    protocol._engine_gpu_counts = (2,)
    _seed_real_fan_out_contract(protocol)

    futures = protocol._generator_futures("prepare")
    protocol._wait_generator_futures(futures)

    ((endpoint, payload),) = engine.payloads
    assert endpoint == "init_weights_update_group"
    # MX_SERVER_ADDRESS is "mx:50051" from the autouse fixture.
    assert payload["master_address"] == "mx"
    assert payload["master_port"] == 50051
    assert payload["rank_offset"] == 0
    assert payload["world_size"] == 2
    assert payload["group_name"] == protocol._engine_group_name
    assert payload["group_name"].startswith("mx-m2n-")
    assert payload["receiver"] == miles_protocol.RECEIVER_PATH
    manifest = payload["receiver_init_payload"]
    from modelexpress_rl.collective.integrations.manifest import (
        MANIFEST_SCHEMA,
        manifest_from_wire,
    )

    assert manifest["schema"] == MANIFEST_SCHEMA
    plan, topology = manifest_from_wire(manifest)
    assert plan == protocol._plan
    assert topology == protocol._topology


def test_run_round_payload_is_empty_lists_plus_receiver_payload(monkeypatch):
    _install_fake_miles_async(monkeypatch, [])
    engine = _RecordingEngine([])

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (engine,)
    protocol._engine_gpu_offsets = (0,)
    protocol._engine_gpu_counts = (2,)
    _seed_real_fan_out_contract(protocol)
    protocol._wait_generator_futures(protocol._generator_futures("prepare"))
    group_name = protocol._engine_group_name

    futures = protocol._generator_futures("run_round", version="7", operation_id="op-7")
    protocol._wait_generator_futures(futures)

    endpoint, payload = engine.payloads[-1]
    assert endpoint == "update_weights_from_distributed"
    assert payload["names"] == []
    assert payload["dtypes"] == []
    assert payload["shapes"] == []
    assert payload["group_name"] == group_name
    assert payload["receiver_payload"] == {"operation_id": "op-7", "version": "7"}
    assert payload["flush_cache"] is False
    assert payload["selector"] == "target"
    for unused_by_sglang_receiver_groups in (
        "backend",
        "load_format",
        "weight_version",
    ):
        assert unused_by_sglang_receiver_groups not in payload


def test_engine_payload_rejects_an_unknown_action():
    protocol = MilesCollectiveProtocolCore(_args())
    protocol._engine_group_name = "mx-m2n-x"

    with pytest.raises(ValueError, match="unknown engine action 'bogus'"):
        protocol._engine_payload("bogus", generator_slot_offset=0, manifest=None)


def test_run_round_echoes_the_connect_time_selector(monkeypatch):
    _install_fake_miles_async(monkeypatch, [])
    engine = _RecordingEngine([])

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.connect(
        [engine],
        [2],
        [0],
        _parallel_state(),
        _placement(),
        "all",
    )
    _seed_real_fan_out_contract(protocol)
    protocol._wait_generator_futures(protocol._generator_futures("prepare"))

    futures = protocol._generator_futures("run_round", version="7", operation_id="op-7")
    protocol._wait_generator_futures(futures)

    endpoint, payload = engine.payloads[-1]
    assert endpoint == "update_weights_from_distributed"
    assert payload["selector"] == "all"


def test_close_fan_out_destroys_the_group_and_clears_its_name(monkeypatch):
    _install_fake_miles_async(monkeypatch, [])
    engine = _RecordingEngine([])

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (engine,)
    protocol._engine_gpu_offsets = (0,)
    protocol._engine_gpu_counts = (2,)
    _seed_real_fan_out_contract(protocol)
    protocol._wait_generator_futures(protocol._generator_futures("prepare"))
    group_name = protocol._engine_group_name

    protocol._close_generator_fanout()

    endpoint, payload = engine.payloads[-1]
    assert endpoint == "destroy_weights_update_group"
    assert payload == {"group_name": group_name}
    # SGLang's destroy is not idempotent, so nothing holds the name anymore.
    assert protocol._engine_group_name is None
    assert protocol._generator_futures("close") == []


def test_an_engine_without_a_generic_call_fails_closed(monkeypatch):
    _install_fake_miles_async(monkeypatch, [])

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (object(),)
    protocol._engine_gpu_offsets = (0,)
    protocol._engine_gpu_counts = (1,)
    _seed_real_fan_out_contract(protocol)

    futures = protocol._generator_futures("prepare")
    with pytest.raises(RuntimeError, match="neither call_endpoint nor _make_request"):
        protocol._wait_generator_futures(futures)


def test_call_endpoint_is_preferred_over_make_request(monkeypatch):
    _install_fake_miles_async(monkeypatch, [])
    calls = []

    class Engine:
        async def call_endpoint(self, endpoint, payload=None):
            calls.append((endpoint, payload))
            return {"success": True}

        async def _make_request(self, endpoint, payload=None):
            raise AssertionError("call_endpoint must win when both exist")

    protocol = MilesCollectiveProtocolCore(_args())
    protocol.rollout_engines = (Engine(),)
    protocol._engine_gpu_offsets = (0,)
    protocol._engine_gpu_counts = (1,)
    _seed_real_fan_out_contract(protocol)

    protocol._wait_generator_futures(protocol._generator_futures("prepare"))

    assert [endpoint for endpoint, _ in calls] == ["init_weights_update_group"]


def test_a_failed_first_begin_sync_leaves_no_committed_state(monkeypatch):
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.connect(
        [object()],
        [1],
        [0],
        _parallel_state(),
        _placement(),
        "target",
    )
    calls = 0

    def flaky(facts, layouts, tensors):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic contract failure")
        MilesCollectiveProtocolCore._build_frozen_contract(
            protocol, facts, layouts, tensors
        )

    monkeypatch.setattr(protocol, "_build_frozen_contract", flaky)
    first = torch.full((2, 2), 1, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="synthetic contract failure"):
        protocol.begin_sync(1, lambda *, materialize: iter([[("model.old", first)]]))

    assert protocol._local_shapes is None
    assert protocol._tensors is None
    assert protocol._layouts is None
    assert protocol._plan is None
    assert protocol._run_id is None

    # The retry is a fresh first round, not a shape-check against the failed
    # attempt's tensors.
    replacement = torch.full((3,), 2, dtype=torch.bfloat16)
    assert (
        protocol.begin_sync(
            2, lambda *, materialize: iter([[("model.new", replacement)]])
        )
        is True
    )
    assert [entry.name for entry in protocol._plan.bulk] == ["model.new"]
    assert protocol._round_version == "2"


def test_a_failed_later_begin_sync_keeps_the_committed_contract(monkeypatch):
    # A DP x TP world stand-in: the contract build's collectives run on both
    # ranks while a contract failure on round two leaves round one's state.
    def all_gather(value):
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], list):
            return [(rank, value[1]) for rank in range(2)]
        return [value, value]

    monkeypatch.setattr(miles_protocol, "_rank_and_world", lambda: (0, 2))
    monkeypatch.setattr(miles_protocol, "_all_gather", all_gather)
    monkeypatch.setattr(miles_protocol, "_broadcast_from_rank_zero", lambda v: v)
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.connect(
        [object()],
        [2],
        [0],
        SimpleNamespace(
            pp=SimpleNamespace(size=1, rank=0),
            tp=SimpleNamespace(size=2, rank=0),
            ep=SimpleNamespace(size=1, rank=0),
            etp=SimpleNamespace(size=1, rank=0),
            cp=SimpleNamespace(size=1, rank=0),
            intra_dp=SimpleNamespace(size=1, rank=0),
            indep_dp=SimpleNamespace(size=1, rank=0),
        ),
        _placement(),
        "target",
    )
    tensors = [
        (name, torch.full((2, 2), index + 1, dtype=torch.bfloat16))
        for index, name in enumerate(("model.a", "model.b"))
    ]
    protocol.begin_sync(1, lambda *, materialize: iter([tensors]))
    protocol._disarm_round()
    plan = protocol._plan
    shapes = dict(protocol._local_shapes)
    buffers = dict(protocol._tensors)
    layouts = dict(protocol._layouts)
    run_id = protocol._run_id

    protocol.connect(
        [object()],
        [1],
        [0],
        protocol._parallel_state,
        _placement(),
        "target",
    )
    with pytest.raises(RuntimeError, match="topology changed"):
        protocol.begin_sync(2, lambda *, materialize: iter([tensors]))

    assert protocol._plan is plan
    assert protocol._local_shapes == shapes
    assert protocol._tensors == buffers
    assert protocol._layouts == layouts
    assert protocol._run_id == run_id
    assert protocol._round_version is None


def test_retire_dropped_futures_observes_futures_that_refuse_to_cancel(caplog):
    future = _DroppedFuture(cancellable=False)

    with caplog.at_level("WARNING"):
        miles_protocol._retire_dropped_futures([future])

    assert future.cancelled is False
    assert len(future.callbacks) == 1
    future.callbacks[0](future)
    assert "failed late" in caplog.text


def test_submission_failure_reports_server_operation_before_close(monkeypatch):
    protocol, session, _events, tensors = _armed_protocol(monkeypatch)
    reports = []
    session.report_failure = lambda operation_id, error: reports.append(
        (operation_id, repr(error))
    )

    def submit(action, **kwargs):
        if action == "run_round":
            raise RuntimeError("synthetic submit failure")
        return []

    protocol._generator_futures = submit
    with pytest.raises(RuntimeError, match="synthetic submit failure"):
        protocol.send_bucket(list(tensors.items()))
    assert len(reports) == 1
    assert reports[0][0] == "845aaea0-f64b-4f2e-b212-5af940b4c169"
    assert "synthetic submit failure" in reports[0][1]
    assert protocol._closed


def test_an_empty_transfer_operation_id_fails_closed(monkeypatch):
    protocol, session, events, tensors = _armed_protocol(monkeypatch)
    session.create_transfer = lambda *, version: ""

    with pytest.raises(RuntimeError, match="did not create a transfer operation"):
        protocol.send_bucket([("model.a", tensors["model.a"])])

    assert protocol._closed is True
    assert not any(
        isinstance(event, tuple) and event[0] == "submit" for event in events
    )
    assert ("begin", "1") not in events


def test_finalize_retires_dropped_rank_zero_futures_when_the_round_fails(monkeypatch):
    protocol, session, _events, tensors = _armed_protocol(monkeypatch)
    for name in ("model.a", "model.b", "model.c"):
        protocol.send_bucket([(name, tensors[name])])
    dropped = _DroppedFuture()
    protocol._round_futures = [dropped]

    def finish_round(*, version, operation_id):
        raise RuntimeError("synthetic finish failure")

    session.finish_round = finish_round

    with pytest.raises(RuntimeError, match="MILES NCCL M2N round failed") as raised:
        protocol.finalize(1)

    assert "synthetic finish failure" in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "synthetic finish failure"
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
    monkeypatch.setattr(
        protocol, "_generator_futures", lambda action, **kwargs: [dropped]
    )

    # The failure fans out to the shared error wrapper; send_bucket closes.
    with pytest.raises(RuntimeError, match="synthetic session prepare failure"):
        protocol._prepare_sessions()

    assert dropped.cancelled is True
    assert dropped.callbacks == []


# --- equal-geometry stacking (MX_NCCL_REFIT_STACK_BYTES) -----------------------

# plan_digest of the default plan _frozen() builds, measured on the stack's
# base before stacking existed: the zero default must never move it.
_BASE_DEFAULT_DIGEST = (
    "7058181e271136afab2879dbbe0ac6957305237e82f526443615aa0a493e6688"
)


def _frozen(monkeypatch, budget=None, count=4):
    if budget is not None:
        monkeypatch.setenv("MX_NCCL_REFIT_STACK_BYTES", str(budget))
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.connect([object()], [2], [0], _parallel_state(), _placement(), "target")
    # 4 bf16 elements = 8 global bytes per tensor.
    tensors = [
        (f"model.{index}", torch.zeros((4,), dtype=torch.bfloat16))
        for index in range(count)
    ]
    protocol.begin_sync(1, lambda *, materialize: iter([tensors]))
    return protocol


def _digest_abi(protocol):
    from modelexpress_rl.collective.plan import plan_digest

    return plan_digest(
        protocol._plan, m2n_abi_version=protocol._topology.m2n_abi_version
    )


def test_the_stack_bytes_env_and_the_zero_default(monkeypatch):
    assert miles_protocol._stack_bytes() == 0
    assert MilesCollectiveProtocolCore(_args())._stack_bytes == 0
    monkeypatch.setenv("MX_NCCL_REFIT_STACK_BYTES", "64")
    assert miles_protocol._stack_bytes() == 64
    assert MilesCollectiveProtocolCore(_args())._stack_bytes == 64


@pytest.mark.parametrize("bad", ["abc", "-1"])
def test_a_bad_stack_bytes_env_names_the_variable_at_startup(monkeypatch, bad):
    monkeypatch.setenv("MX_NCCL_REFIT_STACK_BYTES", bad)
    with pytest.raises(ValueError, match="MX_NCCL_REFIT_STACK_BYTES"):
        MilesCollectiveProtocolCore(_args())


def test_stacking_off_leaves_the_plan_digest_and_abi_byte_identical(monkeypatch):
    for budget in (None, 0):
        protocol = _frozen(monkeypatch, budget)
        assert _digest_abi(protocol) == _BASE_DEFAULT_DIGEST
        assert protocol._topology.m2n_abi_version == "miles-sglang-bf16-replicated-v1"
        assert protocol._publish_groups == (
            ("model.0", "model.1", "model.2", "model.3"),
        )
        assert {entry.group_key for entry in protocol._plan.bulk} == {"publish-group-0"}
        assert protocol._stacks == ()
        assert protocol._wire_plan is None
        assert protocol._wire_groups is None
        assert protocol._stack_storage == {}
        # Every wire buffer is its own tensor.
        pointers = {
            tensor.untyped_storage().data_ptr() for tensor in protocol._tensors.values()
        }
        assert len(pointers) == 4


def test_stacking_marks_the_members_and_changes_the_wire_digest_and_abi(
    monkeypatch,
):
    from modelexpress_rl.collective.plan import plan_digest

    default = _frozen(monkeypatch)
    stacked = _frozen(monkeypatch, 16)  # 8 bytes per member: two per stack

    keys = [entry.group_key for entry in stacked._plan.bulk]
    assert keys == ["m2n-stack1-0", "m2n-stack1-0", "m2n-stack1-1", "m2n-stack1-1"]
    assert [entry.name for entry in stacked._plan.bulk] == [
        entry.name for entry in default._plan.bulk
    ]
    assert stacked._topology.m2n_abi_version == "miles-sglang-bf16-replicated-v1+stack1"
    assert [stack.members for stack in stacked._stacks] == [
        ("model.0", "model.1"),
        ("model.2", "model.3"),
    ]
    wire_names = [entry.name for entry in stacked._wire_plan.bulk]
    assert all(name.startswith("m2n-stack1/") for name in wire_names)
    wire_digest = plan_digest(
        stacked._wire_plan, m2n_abi_version=stacked._topology.m2n_abi_version
    )
    assert wire_digest != _BASE_DEFAULT_DIGEST
    assert _digest_abi(stacked) != _digest_abi(default)
    # Pending tracking names the per-tensor members; the session names wire entries.
    assert stacked._publish_groups == (
        ("model.0", "model.1"),
        ("model.2", "model.3"),
    )
    assert stacked._wire_groups == tuple((name,) for name in wire_names)
    assert sorted(stacked._group_of) == [f"model.{i}" for i in range(4)]


def test_wire_buffers_are_views_into_stack_storage_with_stable_addresses(
    monkeypatch,
):
    protocol = _frozen(monkeypatch, 1 << 20)
    (stack,) = protocol._stacks
    storage = protocol._stack_storage[stack.name]
    assert tuple(storage.shape) == (4, 4)
    for index, name in enumerate(stack.members):
        view = protocol._tensors[name]
        assert view.data_ptr() == storage.data_ptr() + index * 4 * 2
        assert view.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    pointers = {name: protocol._tensors[name].data_ptr() for name in stack.members}
    storage_ptr = storage.data_ptr()

    # Round two refills through the same views; the stack follows, no re-copy.
    refill = [
        (f"model.{index}", torch.full((4,), float(index + 7), dtype=torch.bfloat16))
        for index in range(4)
    ]
    protocol._disarm_round()
    protocol.begin_sync(2, lambda *, materialize: iter([refill]))
    assert protocol._stack_storage[stack.name].data_ptr() == storage_ptr
    assert {n: protocol._tensors[n].data_ptr() for n in stack.members} == pointers
    for index, name in enumerate(stack.members):
        assert torch.equal(storage[index], refill[int(name.split(".")[1])][1])


def test_prepare_sessions_hands_the_publisher_the_wire_plan_and_stacks(
    monkeypatch,
):
    _install_fake_miles_async(monkeypatch, [])
    seen = {}

    class Session:
        def prepare(self):
            pass

        def close(self):
            pass

    def create_session(**kwargs):
        seen["session"] = kwargs
        return Session()

    class Channel:
        def close(self):
            pass

    def publisher(**kwargs):
        seen["publisher"] = kwargs
        return object()

    monkeypatch.setattr(miles_protocol.grpc, "insecure_channel", lambda e: Channel())
    monkeypatch.setattr(miles_protocol.auth, "with_auth", lambda channel: channel)
    monkeypatch.setattr(miles_protocol, "_await_endpoint_ready", lambda *a, **k: None)
    monkeypatch.setattr(miles_protocol, "CollectiveRendezvous", lambda c: object())
    monkeypatch.setattr(miles_protocol, "MilesPublisher", publisher)
    monkeypatch.setattr(miles_protocol.MilesTrainerSession, "create", create_session)

    protocol = _frozen(monkeypatch, 16)
    monkeypatch.setattr(protocol, "_generator_futures", lambda action, **kw: [])
    monkeypatch.setattr(protocol, "_wait_generator_futures", lambda futures: None)
    protocol._prepare_sessions()

    wire_names = [entry.name for entry in protocol._wire_plan.bulk]
    assert seen["publisher"]["plan"] is protocol._wire_plan
    assert sorted(seen["publisher"]["tensors"]) == sorted(wire_names)
    for stack in protocol._stacks:
        assert (
            seen["publisher"]["tensors"][stack.name]
            is protocol._stack_storage[stack.name]
        )
    assert seen["session"]["layer_groups"] == tuple((name,) for name in wire_names)


def test_the_stacked_bucket_stream_publishes_each_stack_once_complete(monkeypatch):
    monkeypatch.setenv("MX_NCCL_REFIT_STACK_BYTES", "16")
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.connect([object()], [2], [0], _parallel_state(), _placement(), "target")
    events = []

    class Session:
        def begin_round(self, *, version):
            events.append(("begin", version))

        def publish_group(self, *, version, layer_group_id):
            events.append(("publish", layer_group_id))

        def create_transfer(self, *, version):
            return "845aaea0-f64b-4f2e-b212-5af940b4c169"

        def finish_round(self, *, version, operation_id):
            events.append(("finish", version))

        def close(self):
            pass

    monkeypatch.setattr(
        protocol, "_prepare_sessions", lambda: setattr(protocol, "_session", Session())
    )
    monkeypatch.setattr(protocol, "_generator_futures", lambda a, **k: [])
    monkeypatch.setattr(protocol, "_wait_generator_futures", lambda f: None)
    tensors = [
        (f"model.{index}", torch.zeros((4,), dtype=torch.bfloat16))
        for index in range(4)
    ]
    protocol.begin_sync(1, lambda *, materialize: iter([tensors]))
    members = protocol._publish_groups
    # Feed group 1's members first: nothing publishes until group 0 completes.
    for name in members[1]:
        protocol.send_bucket([(name, protocol._tensors[name])])
    assert [e for e in events if e[0] == "publish"] == []
    for name in members[0]:
        protocol.send_bucket([(name, protocol._tensors[name])])
    assert [e for e in events if e[0] == "publish"] == [("publish", 0), ("publish", 1)]
    protocol.finalize(1)


def test_a_stack_over_the_pack_bucket_fails_the_prepare_before_any_session(
    monkeypatch,
):
    created = []
    monkeypatch.setattr(
        miles_protocol.MilesTrainerSession,
        "create",
        lambda **kwargs: created.append(kwargs),
    )
    monkeypatch.setattr(miles_protocol, "_await_endpoint_ready", lambda *a, **k: None)
    monkeypatch.setenv("MX_NCCL_REFIT_STACK_BYTES", str(1 << 20))
    protocol = MilesCollectiveProtocolCore(_args())
    protocol.connect([object()], [2], [0], _parallel_state(), _placement(), "target")
    # Four 4-KiB tensors make one 16-KiB stack; the largest PACK bucket this
    # process runs with is 2 KiB, so the stack could never be carried.
    tensors = [
        (f"model.{index}", torch.zeros((2048,), dtype=torch.bfloat16))
        for index in range(4)
    ]
    protocol.begin_sync(1, lambda *, materialize: iter([tensors]))
    assert [stack.area_bytes for stack in protocol._stacks] == [4 * 4096]
    monkeypatch.setenv("NCCL_RESHARD_PACK_BUFFSIZES", "2048")
    monkeypatch.setattr(protocol, "_generator_futures", lambda action, **kw: [])
    with pytest.raises(RuntimeError, match="largest PACK staging bucket of 2048"):
        protocol._prepare_sessions()
    assert created == []
