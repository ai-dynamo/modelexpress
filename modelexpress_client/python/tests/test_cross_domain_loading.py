# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.adapter import EngineAdapter, StrategyRecoveryError
from modelexpress.load_strategy.context import LoadContext, LoadResult
from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
from modelexpress.load_strategy.tcp_strategy import TcpStrategy
from modelexpress.load_strategy import base, execute_load_strategies
from modelexpress.load_strategy.base import LoadStrategy
from modelexpress import load_strategy as chain
from modelexpress.load_strategy import nixl_p2p_strategy as p2p
from modelexpress.topology import TopologyPolicy, apply_policy, resolve_policy


@pytest.fixture
def adaptive(monkeypatch):
    monkeypatch.setenv("MX_P2P_TOPOLOGY_ENFORCEMENT", "adaptive")
    monkeypatch.setenv("MX_P2P_TOPOLOGY_DOMAIN", "fabric")
    monkeypatch.setenv("MX_P2P_TOPOLOGY", '{"fabric":"a"}')
    monkeypatch.setenv("MX_NIXL_BACKEND", "UCX")
    monkeypatch.setattr(
        p2p,
        "get_configured_selector",
        lambda: SimpleNamespace(name="ordered", order=lambda sources, ctx: sources),
    )


def ref(name, domain):
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=name,
        worker_id=name,
        worker_rank=0,
        accelerator="cuda",
        topology={"fabric": domain} if domain else {},
    )


def context(sources):
    return SimpleNamespace(
        identity=p2p_pb2.SourceIdentity(model_name="test", dtype="bfloat16"),
        p2p_source_identity=None,
        worker_rank=0,
        global_rank=0,
        device_id=0,
        worker_id="target",
        accelerator_backend=SimpleNamespace(name="cuda"),
        mx_client=MagicMock(
            list_sources=MagicMock(
                return_value=p2p_pb2.ListSourcesResponse(instances=sources)
            )
        ),
        p2p_state=None,
        p2p_role="main",
        is_reload=False,
        nixl_manager=None,
        tensors={},
        adapter=MagicMock(),
        p2p_enabled=True,
        skip_rdma=False,
    )


def test_full_list_retained_for_tcp(adaptive):
    ctx = context([ref("cross", "b"), ref("same", "a"), ref("unknown", "")])
    assert [s.worker_id for s in RdmaStrategy()._find_source_instances(ctx)] == ["same"]
    assert [s.worker_id for s in TcpStrategy()._find_source_instances(ctx)] == [
        "same",
        "cross",
        "unknown",
    ]
    ctx.mx_client.list_sources.assert_called_once()


@pytest.mark.parametrize("sources", [[], [ref("cross", "b")]])
def test_no_primary_source_still_uses_one_discovery(adaptive, sources):
    ctx = context(sources)
    assert RdmaStrategy()._find_source_instances(ctx) == []
    assert TcpStrategy()._find_source_instances(ctx) == sources
    ctx.mx_client.list_sources.assert_called_once()


def test_unknown_target_domain_uses_tcp_list(adaptive):
    policy = TopologyPolicy("fabric", "adaptive", None)
    sources = [ref("cross", "b")]
    assert apply_policy(sources, policy) == []
    assert apply_policy(sources, policy, channel="tcp") == sources


def test_tcp_releases_receiver_then_registers_for_publication(adaptive, monkeypatch):
    ctx = context([])
    events = []
    ctx.nixl_manager = MagicMock(shutdown=lambda: events.append("close"))
    result = LoadResult(value=torch.nn.Linear(2, 2), model=torch.nn.Linear(2, 2))
    monkeypatch.setattr(
        "modelexpress.load_strategy.tcp_strategy.register_tensors",
        lambda result, ctx: events.append("register"),
    )
    TcpStrategy().finalize(result, ctx)
    assert events == ["close", "register"]
    assert ctx.nixl_manager is None


class Adapter(EngineAdapter):
    def discover_tensors(self, result):
        return dict(result.model.named_parameters())

    def prepare_rdma_target(self, result):
        return result

    def before_rdma_receive(self, result):
        return result

    def after_rdma_receive(self, result):
        return result


class Storage(LoadStrategy):
    name = "storage"

    def load(self, result, ctx):
        with torch.no_grad():
            result.model.weight.fill_(9)
        return result

    def finalize(self, result, ctx):
        pass


@pytest.fixture
def loading(adaptive, monkeypatch):
    events = []
    adapter = Adapter()
    adapter.all_gather_state = lambda state: (state,)
    backend = MagicMock()
    backend.name = "cuda"
    ctx = LoadContext(
        model_config=SimpleNamespace(),
        load_config=None,
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        local_rank=0,
        device_id=0,
        identity=p2p_pb2.SourceIdentity(model_name="test", dtype="bfloat16"),
        mx_client=MagicMock(),
        worker_id="target",
        mx_server_url="server:8001",
        adapter=adapter,
        accelerator_backend=backend,
    )
    sources = [ref("same", "a"), ref("cross", "b")]
    ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=sources
    )

    def metadata(*, mx_source_id, worker_id):
        events.append(("metadata", worker_id))
        source = next(s for s in sources if s.worker_id == worker_id)
        return p2p_pb2.GetMetadataResponse(
            found=True,
            worker=p2p_pb2.WorkerMetadata(
                accelerator="cuda",
                topology=source.topology,
                nixl_metadata=worker_id.encode(),
                tensor_source=p2p_pb2.TensorSourceMetadata(
                    tensors=[
                        p2p_pb2.TensorDescriptor(
                            name="weight", addr=1, size=16, dtype="float32"
                        )
                    ]
                ),
            ),
        )

    ctx.mx_client.get_metadata.side_effect = metadata
    failures = set()
    managers = []

    def create(*args, ucx_mode="auto", **kwargs):
        events.append(("create", ucx_mode))
        if (ucx_mode, "create") in failures:
            raise RuntimeError("cannot initialize channel")
        manager = MagicMock()
        manager.ucx_mode = ucx_mode
        manager.listen_port = kwargs.get("listen_port")
        manager.tensor_descriptors = []

        def close(*, strict=False):
            events.append(("close", ucx_mode))
            if (ucx_mode, "close") in failures:
                raise RuntimeError("receiver close failed")

        def register(*args):
            events.append(("register", ucx_mode))
            if (ucx_mode, "register") in failures:
                raise RuntimeError("memory registration failed")
            manager.tensor_descriptors = [object()]

        manager.shutdown.side_effect = close
        manager.register_tensors.side_effect = register
        manager.register_arena.side_effect = register
        manager.register_additional_tensors.side_effect = register
        manager.add_remote_agent.side_effect = lambda value: value.decode()

        def receive(**kwargs):
            source = kwargs["remote_agent_name"]
            events.append(("read", ucx_mode, source))
            if (ucx_mode, source) in failures:
                raise TimeoutError("READ timed out")
            with torch.no_grad():
                for tensor in (kwargs["destination_tensors"] or ctx.tensors).values():
                    tensor.fill_(7)
            return 16, 1, 0

        manager.receive_from_source.side_effect = receive
        managers.append(manager)
        return manager

    def rebuild(result):
        events.append(("rebuild",))
        model = torch.nn.Linear(2, 2, bias=False)
        return LoadResult(value=model, model=model)

    adapter.reinit_for_retry = rebuild
    monkeypatch.setattr(p2p, "is_nixl_available", lambda: True)
    monkeypatch.setattr(base, "is_nixl_available", lambda: True)
    monkeypatch.setattr(p2p, "check_transfer_allowed", lambda config: (True, ""))
    monkeypatch.setattr(p2p, "_init_nixl_manager", create)
    monkeypatch.setattr(base, "_init_nixl_manager", create)
    monkeypatch.setattr(
        chain, "publish_source_if_supported", lambda *args: events.append(("publish",))
    )
    return SimpleNamespace(
        ctx=ctx, events=events, failures=failures, sources=sources, managers=managers
    )


@pytest.fixture
def draft_loading(loading, monkeypatch):
    case = loading
    ctx = case.ctx
    ctx.p2p_role = "draft"
    ctx.p2p_source_identity = ctx.identity
    ctx.identity = p2p_pb2.SourceIdentity(model_name="draft", dtype="bfloat16")
    shared = MagicMock()
    main_weight = torch.ones(2, 2)
    shared.tensors = {"main.weight": main_weight}
    ctx.shared_nixl_manager = shared
    case.shared = shared
    case.without_draft = set()
    case.incomplete_draft = set()
    metadata = ctx.mx_client.get_metadata.side_effect

    def draft_metadata(**kwargs):
        response = metadata(**kwargs)
        if kwargs["worker_id"] not in case.without_draft:
            tensor = response.worker.tensor_source.tensors.add()
            tensor.CopyFrom(response.worker.tensor_source.tensors[0])
            name = "other" if kwargs["worker_id"] in case.incomplete_draft else "weight"
            tensor.name = ctx.draft_tensor_namespace + name
        return response

    ctx.mx_client.get_metadata.side_effect = draft_metadata
    shared.register_additional_tensors.side_effect = shared.tensors.update

    def postprocess(result):
        result.model.weight = torch.nn.Parameter(result.model.weight.detach().clone())
        return result

    ctx.adapter.after_rdma_receive = postprocess

    def publish(tensors, **kwargs):
        assert ctx.nixl_manager is shared
        assert shared.tensors == {"main.weight": main_weight, **tensors}
        assert torch.all(main_weight == 1)
        assert all(manager.shutdown.called for manager in case.managers)
        shared.shutdown.assert_not_called()
        case.events.append(("publish",))
        return True

    monkeypatch.setattr(chain, "publish_source_if_supported", base.publish_source_if_supported)
    monkeypatch.setattr(base, "extend_published_tensors", publish)
    monkeypatch.setattr(Storage, "finalize", LoadStrategy.finalize)
    return case


def run_loading(case):
    return execute_load_strategies(
        torch.nn.Linear(2, 2, bias=False),
        case.ctx,
        [RdmaStrategy(), TcpStrategy(), Storage()],
    )


@pytest.mark.parametrize("domain", [None, " \t"])
@pytest.mark.parametrize("auto_fails", [False, True])
@pytest.mark.parametrize("case_fixture", ["loading", "draft_loading"])
def test_adaptive_without_domain_uses_all_sources(
    request, monkeypatch, domain, auto_fails, case_fixture
):
    loading = request.getfixturevalue(case_fixture)
    if domain is None:
        monkeypatch.delenv("MX_P2P_TOPOLOGY_DOMAIN")
    else:
        monkeypatch.setenv("MX_P2P_TOPOLOGY_DOMAIN", domain)
    assert resolve_policy(timeout=0) is None
    loading.sources.reverse()
    loading.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=loading.sources
    )
    if auto_fails:
        loading.failures.update(("auto", source.worker_id) for source in loading.sources)

    assert torch.all(run_loading(loading).weight == 7)
    expected = [("read", "auto", "cross")]
    if auto_fails:
        expected += [("read", "auto", "same"), ("read", "tcp", "cross")]
    assert [e for e in loading.events if e[0] == "read"] == expected
    loading.ctx.mx_client.list_sources.assert_called_once()
    assert loading.events[-1] == ("publish",)


@pytest.mark.parametrize("namespace", ["mx_draft::", "mx_draft::checkpoint::"])
@pytest.mark.parametrize("scenario", ["same", "cross", "retry", "peer_failure", "storage"])
def test_draft_channels_share_discovery_and_publish_final_tensors(
    draft_loading, namespace, scenario
):
    case = draft_loading
    case.ctx.draft_tensor_namespace = namespace
    if scenario == "cross":
        case.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
            instances=[case.sources[1]]
        )
    if scenario in {"retry", "storage"}:
        case.failures.add(("auto", "same"))
    if scenario == "storage":
        case.failures.update([("tcp", "same"), ("tcp", "cross")])
    if scenario == "peer_failure":
        case.ctx.adapter.all_gather_state = lambda state: [
            state, (state[0], "failed") if state[0] == "load:rdma" else state
        ]

    model = run_loading(case)

    assert torch.all(model.weight == (9 if scenario == "storage" else 7))
    reads = [event[1:] for event in case.events if event[0] == "read"]
    assert reads == {
        "same": [("auto", "same")],
        "cross": [("tcp", "cross")],
        "retry": [("auto", "same"), ("tcp", "same")],
        "peer_failure": [("auto", "same"), ("tcp", "same")],
        "storage": [("auto", "same"), ("tcp", "same"), ("tcp", "cross")],
    }[scenario]
    case.ctx.mx_client.list_sources.assert_called_once_with(
        identity=case.ctx.p2p_source_identity, status_filter=p2p_pb2.SOURCE_STATUS_READY
    )
    assert case.shared.tensors[namespace + "weight"] is model.weight
    case.shared.register_additional_tensors.assert_called_once()
    assert case.events.count(("publish",)) == 1
    assert case.ctx.draft_published
    for manager in case.managers:
        assert manager.listen_port == 0
        manager.shutdown.assert_called_once_with(strict=True)
        assert list(manager.receive_from_source.call_args.kwargs["destination_tensors"]) == [
            namespace + "weight"
        ]


@pytest.mark.parametrize("lookups", [1, 4])
def test_draft_manifest_and_receive_budgets_are_per_channel(draft_loading, monkeypatch, lookups):
    case = draft_loading
    monkeypatch.setattr(p2p, "MAX_DRAFT_SOURCE_METADATA_LOOKUPS", lookups)
    monkeypatch.setattr(p2p, "MAX_SOURCE_RETRIES", 1)
    case.sources[:] = [ref("empty", "a"), ref("same", "a"), ref("cross", "b")]
    case.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=case.sources)
    case.without_draft.add("empty")
    case.failures.update([("auto", "same"), ("tcp", "same")])

    assert torch.all(run_loading(case).weight == 9)
    assert [event[1:] for event in case.events if event[0] == "read"] == (
        [] if lookups == 1 else [("auto", "same"), ("tcp", "same")]
    )
    assert [event[1] for event in case.events if event[0] == "metadata"] == (
        ["empty", "empty"] if lookups == 1 else ["empty", "same", "empty", "same"]
    )
    case.ctx.mx_client.list_sources.assert_called_once()
    case.shared.shutdown.assert_not_called()


@pytest.mark.parametrize("next_domain", ["a", "b"])
def test_incomplete_draft_manifest_allows_next_source_or_tcp(draft_loading, next_domain):
    case = draft_loading
    case.incomplete_draft.add("same")
    case.sources[1].topology["fabric"] = next_domain
    case.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=case.sources
    )

    assert torch.all(run_loading(case).weight == 7)
    channel = "auto" if next_domain == "a" else "tcp"
    assert [event for event in case.events if event[0] == "read"] == [
        ("read", channel, "cross")
    ]
    case.ctx.mx_client.list_sources.assert_called_once()
    case.shared.shutdown.assert_not_called()


@pytest.mark.parametrize("channel", ["auto", "tcp"])
def test_draft_cleanup_failure_cannot_fall_back_or_close_main(draft_loading, channel):
    case = draft_loading
    if channel == "tcp":
        case.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
            instances=[case.sources[1]]
        )
    case.failures.add((channel, "close"))

    with pytest.raises(StrategyRecoveryError, match="receiver"):
        run_loading(case)
    case.shared.shutdown.assert_not_called()
    case.shared.register_additional_tensors.assert_not_called()
    assert ("publish",) not in case.events
    assert ("rebuild",) not in case.events
    assert case.ctx.nixl_manager is case.managers[-1]



@pytest.mark.parametrize("failure_mode", ["none", "read", "create", "all"])
def test_channel_switch_and_storage_fallback(loading, failure_mode):
    case = loading
    if failure_mode == "read":
        case.failures.add(("auto", "same"))
    elif failure_mode == "create":
        case.failures.add(("auto", "create"))
    elif failure_mode == "all":
        case.failures.update([("auto", "same"), ("tcp", "same"), ("tcp", "cross")])
    model = run_loading(case)
    assert torch.all(model.weight == (9 if failure_mode == "all" else 7))
    reads = [event[1:] for event in case.events if event[0] == "read"]
    expected = {
        "none": [("auto", "same")],
        "read": [("auto", "same"), ("tcp", "same")],
        "create": [("tcp", "same")],
        "all": [("auto", "same"), ("tcp", "same"), ("tcp", "cross")],
    }
    assert reads == expected[failure_mode]
    case.ctx.mx_client.list_sources.assert_called_once()
    assert case.events[-1] == ("publish",)
    if failure_mode == "read":
        assert case.events.index(("close", "auto")) < case.events.index(
            ("create", "tcp")
        )
        assert case.events[case.events.index(("close", "tcp")) + 1] == ("create", "auto")
        assert case.ctx.nixl_manager.ucx_mode == "auto"
    if failure_mode == "none":
        assert ("create", "tcp") not in case.events


def test_no_same_domain_candidate_starts_tcp_without_auto_agent(loading):
    case = loading
    case.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=[case.sources[1]]
    )
    assert torch.all(run_loading(case).weight == 7)
    assert [e for e in case.events if e[0] == "read"] == [("read", "tcp", "cross")]
    assert case.events.index(("create", "tcp")) < case.events.index(("create", "auto"))
    case.ctx.mx_client.list_sources.assert_called_once()


def test_each_channel_has_its_own_retry_budget(loading, monkeypatch):
    case = loading
    monkeypatch.setattr(p2p, "MAX_SOURCE_RETRIES", 2)
    case.sources[:] = [ref(str(i), "a" if i < 3 else "b") for i in range(4)]
    case.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=case.sources
    )
    case.failures.update((mode, str(i)) for mode in ("auto", "tcp") for i in range(4))
    assert torch.all(run_loading(case).weight == 9)
    assert [e for e in case.events if e[0] == "read"] == [
        ("read", "auto", "0"),
        ("read", "auto", "1"),
        ("read", "tcp", "0"),
        ("read", "tcp", "1"),
    ]
    case.ctx.mx_client.list_sources.assert_called_once()


@pytest.mark.parametrize("validation", ["prepare", "finalize"])
@pytest.mark.parametrize(
    "policy,second_domain",
    [("adaptive", "a"), ("adaptive", "b"), ("required", "a"), ("preferred", "a")],
)
def test_manifest_mismatch_retries_compatible_source(
    loading, monkeypatch, validation, policy, second_domain
):
    from modelexpress.engines.sglang.runtime_state import (
        finalize_runtime_state,
        prepare_runtime_tensors,
    )

    monkeypatch.setenv("MX_P2P_TOPOLOGY_ENFORCEMENT", policy)
    loading.sources[1].topology["fabric"] = second_domain
    loading.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=loading.sources
    )
    metadata = loading.ctx.mx_client.get_metadata.side_effect

    def source_metadata(**kwargs):
        response = metadata(**kwargs)
        if validation == "prepare" and kwargs["worker_id"] == "same":
            tensors = response.worker.tensor_source.tensors
            tensors.add().CopyFrom(tensors[0])
        return response

    def prepare(result, tensors):
        prepare_runtime_tensors(result.model, tensors)

    def finalize(result):
        if validation == "finalize":
            source = next(e[2] for e in reversed(loading.events) if e[0] == "read")
            result.model.k_scale_float = 1.0
            result.model.k_scale = torch.tensor(-1.0 if source == "same" else 1.0)
        finalize_runtime_state(result.model)
        return result

    loading.ctx.mx_client.get_metadata.side_effect = source_metadata
    loading.ctx.adapter.prepare_rdma_target_from_manifest = prepare
    loading.ctx.adapter.after_rdma_receive = finalize

    assert torch.all(run_loading(loading).weight == 7)
    channels = ["auto"] if second_domain == "a" else ["auto", "tcp"]
    expected_reads = (
        [("read", channel, "same") for channel in channels]
        if validation == "finalize" else []
    )
    expected_reads.append(("read", channels[-1], "cross"))
    assert [e for e in loading.events if e[0] == "read"] == expected_reads
    assert loading.events.count(("rebuild",)) == len(channels)
    loading.ctx.mx_client.list_sources.assert_called_once()
    assert loading.events[-1] == ("publish",)


@pytest.mark.parametrize(
    "hook,error",
    [
        ("prepare_rdma_target", RuntimeError),
        ("prepare_rdma_target_from_manifest", torch.OutOfMemoryError),
        ("after_rdma_receive", RuntimeError),
        ("discover_tensors", RuntimeError),
    ],
)
@pytest.mark.parametrize("policy", ["adaptive", "required", "preferred"])
def test_model_failure_retry_depends_on_policy(loading, monkeypatch, hook, error, policy):
    monkeypatch.setenv("MX_P2P_TOPOLOGY_ENFORCEMENT", policy)
    loading.sources[1].topology["fabric"] = "a"
    loading.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=loading.sources
    )

    def fail(result, *args):
        raise error("model processing failed")

    setattr(loading.ctx.adapter, hook, fail)
    assert torch.all(run_loading(loading).weight == 9)
    assert ("create", "tcp") not in loading.events
    assert loading.events.count(("rebuild",)) == (1 if policy == "adaptive" else 2)


@pytest.mark.parametrize("arena", [False, True])
def test_backend_registration_failure_allows_tcp(loading, arena):
    if arena:
        loading.ctx.vmm_arena = object()
    loading.failures.add(("auto", "register"))

    assert torch.all(run_loading(loading).weight == 7)
    assert [e for e in loading.events if e[0] == "read"] == [("read", "tcp", "same")]
    assert loading.events.index(("close", "auto")) < loading.events.index(("create", "tcp"))
    assert ("rebuild",) in loading.events
    loading.ctx.mx_client.list_sources.assert_called_once()


@pytest.mark.parametrize("channel", ["auto", "tcp"])
def test_registration_cleanup_failure_stops_loading(loading, channel):
    if channel == "tcp":
        loading.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
            instances=[loading.sources[1]]
        )
    loading.failures.update([(channel, "register"), (channel, "close")])

    with pytest.raises(StrategyRecoveryError, match="receiver close failed"):
        run_loading(loading)
    assert loading.ctx.nixl_manager is loading.managers[0]
    assert not any(e[0] == "read" for e in loading.events)
    assert ("rebuild",) not in loading.events
    assert ("publish",) not in loading.events


@pytest.mark.parametrize("peer_model_failure", [False, True])
def test_successful_rank_follows_peer_failure(loading, peer_model_failure):
    def gather(state):
        phase, value = state
        if phase == "load:rdma":
            return [state, (phase, "failed")]
        if phase == "prepare:tcp" and peer_model_failure:
            return [state, (phase, "failed")]
        return [state, state]

    loading.ctx.adapter.all_gather_state = gather
    assert torch.all(run_loading(loading).weight == (9 if peer_model_failure else 7))
    reads = [e for e in loading.events if e[0] == "read"]
    assert reads == (
        [("read", "auto", "same")]
        if peer_model_failure
        else [("read", "auto", "same"), ("read", "tcp", "same")]
    )
    assert ("rebuild",) in loading.events
    assert loading.events.count(("publish",)) == 1


def test_domain_change_excludes_auto_but_not_tcp(loading):
    loading.sources[0].topology["fabric"] = "b"
    assert torch.all(run_loading(loading).weight == 7)
    assert [e for e in loading.events if e[0] == "read"] == [("read", "tcp", "same")]


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_reload", True),
        ("skip_rdma", True),
        ("p2p_enabled", False),
    ],
)
def test_out_of_scope_loads_use_existing_storage(loading, field, value):
    setattr(loading.ctx, field, value)
    assert torch.all(run_loading(loading).weight == 9)
    loading.ctx.mx_client.list_sources.assert_not_called()
    assert not loading.managers


def test_list_failure_is_not_retried_by_tcp(loading):
    loading.ctx.mx_client.list_sources.side_effect = RuntimeError("server unavailable")
    assert torch.all(run_loading(loading).weight == 9)
    loading.ctx.mx_client.list_sources.assert_called_once()


@pytest.mark.parametrize("channel", ["auto", "tcp"])
def test_source_retry_cleanup_failure_remains_fatal(loading, monkeypatch, channel):
    for source in loading.sources:
        source.topology["fabric"] = "a" if channel == "auto" else "b"
    loading.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=loading.sources
    )
    loading.failures.add((channel, "same"))
    create = p2p._init_nixl_manager

    def init(*args, **kwargs):
        manager = create(*args, **kwargs)
        if len(loading.managers) == 1:
            manager.shutdown.side_effect = [RuntimeError("close failed"), None]
        return manager

    monkeypatch.setattr(p2p, "_init_nixl_manager", init)
    with pytest.raises(StrategyRecoveryError, match="close failed"):
        run_loading(loading)
    assert [e for e in loading.events if e[0] == "read"] == [("read", channel, "same")]
    assert ("rebuild",) not in loading.events
    assert ("publish",) not in loading.events


def test_tcp_handoff_failure_cannot_publish(loading):
    loading.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=[loading.sources[1]]
    )
    loading.failures.add(("tcp", "close"))
    with pytest.raises(StrategyRecoveryError, match="receiver close failed"):
        run_loading(loading)
    assert ("create", "auto") not in loading.events
    assert ("publish",) not in loading.events


@pytest.mark.parametrize("failure", [None, "read", "no_same_domain"])
def test_libfabric_keeps_same_domain_p2p_without_tcp(loading, monkeypatch, failure):
    monkeypatch.setenv("MX_NIXL_BACKEND", "LIBFABRIC")
    if failure == "read":
        loading.failures.add(("auto", "same"))
    elif failure == "no_same_domain":
        loading.ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
            instances=[loading.sources[1]]
        )

    assert torch.all(run_loading(loading).weight == (7 if failure is None else 9))
    assert [event for event in loading.events if event[0] == "read"] == (
        [] if failure == "no_same_domain" else [("read", "auto", "same")]
    )
    assert ("create", "tcp") not in loading.events
    loading.ctx.mx_client.list_sources.assert_called_once()


def test_service_discovery_skips_adaptive_p2p(loading):
    loading.ctx.mx_client.REQUIRES_P2P_METADATA = True
    assert torch.all(run_loading(loading).weight == 9)
    loading.ctx.mx_client.list_sources.assert_not_called()


@pytest.mark.parametrize("domain", [None, "fabric"])
def test_adaptive_artifacts_skip_before_allocating_receive_resources(
    adaptive, monkeypatch, domain
):
    from modelexpress.metadata.artifact_lifecycle import install_artifacts
    from modelexpress.metadata.nixl_artifact_transport import NixlArtifactTransport
    from modelexpress.metadata.artifact_transport import ArtifactCacheMiss

    if domain is None:
        monkeypatch.delenv("MX_P2P_TOPOLOGY_DOMAIN")
    factory = MagicMock()
    install_artifacts(context([]), factory, engine_label="test")
    factory.assert_not_called()
    with pytest.raises(ArtifactCacheMiss, match="adaptive"):
        NixlArtifactTransport().fetch(None, None, None)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_partial_initialization_is_closed(cleanup_fails, monkeypatch):
    from modelexpress import nixl_transfer
    from modelexpress.load_strategy.base import _init_nixl_manager

    manager = MagicMock()
    manager.initialize.side_effect = RuntimeError("initialize failed")
    if cleanup_fails:
        manager.shutdown.side_effect = RuntimeError("close failed")
    monkeypatch.setattr(
        nixl_transfer, "NixlTransferManager", MagicMock(return_value=manager)
    )
    expected = StrategyRecoveryError if cleanup_fails else RuntimeError
    with pytest.raises(expected):
        _init_nixl_manager(0, 0, "tcp", ucx_mode="tcp")
    manager.shutdown.assert_called_once()


@pytest.mark.parametrize("policy", ["required", "preferred"])
@pytest.mark.parametrize("backend", ["UCX", "LIBFABRIC"])
def test_existing_policies_keep_primary_without_tcp(loading, monkeypatch, policy, backend):
    monkeypatch.setenv("MX_P2P_TOPOLOGY_ENFORCEMENT", policy)
    monkeypatch.setenv("MX_NIXL_BACKEND", backend)
    assert RdmaStrategy().is_available(loading.ctx)
    assert not TcpStrategy().is_available(loading.ctx)


def test_unknown_target_domain_skips_auto_but_can_load_over_tcp(loading, monkeypatch):
    from modelexpress import topology

    monkeypatch.setenv("MX_P2P_TOPOLOGY", "{}")
    monkeypatch.setattr(topology, "DOMAIN_WAIT_TIMEOUT_S", 0)
    assert torch.all(run_loading(loading).weight == 7)
    assert [e for e in loading.events if e[0] == "read"] == [("read", "tcp", "same")]
    loading.ctx.mx_client.list_sources.assert_called_once()


def test_sglang_transfer_engine_skips_adaptive_peer_loading(adaptive):
    from modelexpress.engines.sglang.loader import MxModelLoader

    ctx = context([])
    assert MxModelLoader._find_transfer_engine_source(None, ctx) is None
    ctx.mx_client.list_sources.assert_not_called()
