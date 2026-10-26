# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent draft loading with known selectors and complete unknown layouts."""

import ctypes
import gc
import sys
import weakref
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from modelexpress import p2p_pb2
from modelexpress.engines.sglang import adapter as adapter_mod
from modelexpress.engines.sglang import loader as loader_mod
from modelexpress.load_strategy.base import (
    publish_metadata,
    register_tensors,
)
from modelexpress.load_strategy.context import LoadResult
from modelexpress.load_strategy.model_streamer_strategy import (
    ModelStreamerStrategy,
    _resolve_model_uri,
)
from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
from modelexpress.metadata import publish as publication
from modelexpress.metadata.payload import worker_tensor_descriptors
from modelexpress.metadata.source_id import compute_mx_source_id
from modelexpress.nixl_transfer import NixlTransferManager


@pytest.fixture(autouse=True)
def isolated_sessions(monkeypatch, mock_accelerator_backend_cls):
    for name in (
        "MX_MODEL_URI",
        "MX_MODEL_REVISION",
        "MX_DRAFT_MODEL_URI",
        "MX_DRAFT_MODEL_REVISION",
        "MX_DISABLE_DRAFT_P2P",
        "MX_DRAFT_PORT_OFFSET",
        "MX_METADATA_PORT",
        "MX_WORKER_GRPC_PORT",
        "MX_METADATA_BACKEND",
        "MX_ARTIFACT_READY_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MX_SERVER_ADDRESS", "localhost:8001")
    backend = mock_accelerator_backend_cls()
    backend.is_accel_tensor = lambda tensor: tensor.device.type == "cpu"
    monkeypatch.setattr(adapter_mod, "accelerator_backend_for", lambda device: backend)
    monkeypatch.setattr(adapter_mod.torch.cuda, "device_count", lambda: 0)
    for module, name in (
        (loader_mod, "_tensor_registry"),
        (loader_mod, "_nixl_managers"),
        (publication, "_heartbeat_threads"),
        (publication, "_worker_servers"),
        (publication, "_central_manifests"),
    ):
        monkeypatch.setattr(module, name, {})
    monkeypatch.setattr(
        loader_mod, "_heartbeat_threads", publication._heartbeat_threads
    )


def make_context(
    *, draft=True, path="org/draft", weights=None, index=0, revision=None,
    **load_options,
):
    return adapter_mod.build_sglang_load_context(
        SimpleNamespace(tp_rank=3, draft_model_idx=index, **load_options),
        SimpleNamespace(
            model_path=path,
            model_weights=weights,
            revision=revision,
            is_draft_model=draft,
            dtype=torch.float32,
            hf_config=SimpleNamespace(hidden_size=2),
        ),
        SimpleNamespace(device="cpu", gpu_id=0),
    )


@pytest.fixture
def sglang_startup_config(monkeypatch):
    """Expose resolved startup configuration independently of any main loader."""
    config = SimpleNamespace(
        main=SimpleNamespace(model_path="org/main", revision="config-sha"),
        spec=SimpleNamespace(
            speculative_draft_model_path="org/draft",
            speculative_draft_model_revision=None,
        ),
    )
    for name in ("sglang", "sglang.srt"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_model = lambda: config.main
    runtime.get_spec = lambda: config.spec
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime)
    return config


def run_nixl(ctx, model, chain):
    loader = loader_mod.MxModelLoader(ctx.load_config)
    with (
        patch.object(loader_mod, "build_sglang_load_context", return_value=ctx),
        patch.object(loader_mod, "install_sglang_cache_artifacts") as install,
        patch.object(loader_mod, "schedule_sglang_cache_artifact_publish") as artifacts,
        patch.object(loader, "_run_load_strategy_chain", side_effect=chain),
    ):
        loader._load_model_via_nixl(
            model=model,
            model_config=ctx.model_config,
            device_config=ctx.adapter.device_config,
        )
    return loader, install, artifacts


@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize("same_checkpoint", [True, False])
def test_draft_has_independent_identity_and_registration(
    pp_size, same_checkpoint, monkeypatch
):
    monkeypatch.setattr(
        adapter_mod,
        "_get_parallel_size",
        lambda name: pp_size if name == "get_pipeline_model_parallel_world_size" else 8,
    )
    monkeypatch.setattr(
        adapter_mod, "_get_sglang_worker_rank", lambda cfg: (pp_size - 1) * 8 + 3
    )
    main = make_context(draft=False, path="org/main")
    draft = make_context(path="org/main" if same_checkpoint else "org/draft")
    main_id = compute_mx_source_id(main.identity)
    draft_id = compute_mx_source_id(draft.identity)
    assert main_id != draft_id
    assert not main.identity.extra_parameters
    assert dict(draft.identity.extra_parameters) == {
        "model_role": "draft", "draft_model_idx": "0",
    }
    managers = []

    def init(*args):
        manager = MagicMock(tensor_descriptors=[])
        managers.append(manager)
        return manager

    def chain(model, ctx):
        register_tensors(model, ctx)
        return model

    with (
        patch("modelexpress.load_strategy.base.is_nixl_available", return_value=True),
        patch(
            "modelexpress.load_strategy.base._init_nixl_manager", side_effect=init
        ) as initialize,
    ):
        main_loader, _, _ = run_nixl(main, nn.Linear(2, 2), chain)
        draft_loader, install, artifacts = run_nixl(draft, nn.Linear(2, 2), chain)

    assert draft.p2p_enabled
    assert main.nixl_manager is managers[0]
    assert draft.nixl_manager is managers[1]
    assert [call.args[3] for call in initialize.call_args_list] == [5555, 15555]
    assert set(draft.tensors) == {"weight", "bias"}
    assert draft.identity.pipeline_parallel_size == pp_size
    assert draft.worker_rank == main.worker_rank == (pp_size - 1) * 8 + 3
    assert compute_mx_source_id(main.identity) == main_id
    assert compute_mx_source_id(draft.identity) == draft_id
    assert loader_mod._tensor_registry[0] is main.tensors
    assert loader_mod._tensor_registry[(0, "sglang-draft-0")] is draft.tensors
    assert loader_mod._nixl_managers[0] is main.nixl_manager
    assert loader_mod._nixl_managers[(0, "sglang-draft-0")] is draft.nixl_manager
    assert main_loader._ctx is main
    assert draft_loader._ctx is draft
    assert main.source_ready_fn is None
    assert draft.source_ready_fn is None
    install.assert_not_called()
    artifacts.assert_not_called()


def test_draft_can_load_before_main_and_without_model_whitelist():
    ctx = make_context()
    run_nixl(ctx, nn.Linear(2, 2), lambda model, ctx: model)
    assert ctx.p2p_enabled
    assert ctx.identity.revision == ""
    assert ctx.shared_nixl_manager is None


@pytest.mark.parametrize("index", [0, 1, 2])
def test_draft_index_isolates_identity_ports_and_tensors(index):
    contexts = [make_context(index=i) for i in range(index + 1)]
    identities = [compute_mx_source_id(ctx.identity) for ctx in contexts]
    assert len(set(identities)) == index + 1
    for ctx in contexts:
        run_nixl(ctx, nn.Linear(2, 2), lambda model, ctx: model)
    assert set(loader_mod._tensor_registry) == {
        (0, f"sglang-draft-{i}") for i in range(index + 1)
    }
    for i, ctx in enumerate(contexts):
        assert loader_mod._tensor_registry[(0, f"sglang-draft-{i}")] is ctx.tensors
    assert len({compute_mx_source_id(ctx.identity) for ctx in contexts}) == index + 1
    assert len({ctx.port_offset for ctx in contexts}) == index + 1
    peer = make_context(index=index)
    assert compute_mx_source_id(peer.identity) == identities[-1]


@pytest.mark.parametrize(
    "index,stride", [(-1, 10000), (False, 10000), (0, 1000), (6, 10000)]
)
def test_invalid_draft_port_assignment_is_rejected(index, stride, monkeypatch):
    monkeypatch.setenv("MX_DRAFT_PORT_OFFSET", str(stride))
    with pytest.raises(ValueError, match="draft_model_idx|MX_DRAFT_PORT_OFFSET"):
        make_context(index=index)


@pytest.mark.parametrize("same_checkpoint", [True, False])
@pytest.mark.parametrize(
    "draft_uri", [None, "s3://draft/weights", "s3://target/weights"]
)
@pytest.mark.parametrize("draft_revision", [None, "draft-sha"])
def test_draft_storage_and_revision_are_independent_of_main_load_order(
    same_checkpoint, draft_uri, draft_revision, sglang_startup_config, monkeypatch,
):
    monkeypatch.setenv("MX_MODEL_URI", "s3://target/weights")
    monkeypatch.setenv("MX_MODEL_REVISION", "target-sha")
    if draft_uri:
        monkeypatch.setenv("MX_DRAFT_MODEL_URI", draft_uri)
    if draft_revision:
        monkeypatch.setenv("MX_DRAFT_MODEL_REVISION", draft_revision)
    path = "org/main" if same_checkpoint else "org/draft"
    sglang_startup_config.spec.speculative_draft_model_path = path
    before = make_context(path=path, revision="config-sha")
    run_nixl(before, nn.Linear(2, 2), lambda model, ctx: model)
    main = make_context(draft=False, path="org/main", revision="config-sha")
    run_nixl(main, nn.Linear(2, 2), lambda model, ctx: model)
    after = make_context(path=path, revision="config-sha")
    run_nixl(after, nn.Linear(2, 2), lambda model, ctx: model)
    assert compute_mx_source_id(before.identity) == compute_mx_source_id(after.identity)
    inherited_revision = (
        "target-sha"
        if same_checkpoint and draft_uri in (None, "s3://target/weights")
        else "config-sha"
    )
    for draft in (before, after):
        assert _resolve_model_uri(draft) == (
            draft_uri or ("s3://target/weights" if same_checkpoint else path)
        )
        assert draft.identity.revision == (draft_revision or inherited_revision)
        assert compute_mx_source_id(draft.identity) != compute_mx_source_id(main.identity)
        assert draft.publication_namespace == "sglang-draft-0"
    assert _resolve_model_uri(main) == "s3://target/weights"
    assert main.identity.revision == "target-sha"


@pytest.mark.parametrize("same_checkpoint", [True, False])
def test_main_revision_change_updates_only_shared_draft_identity(
    same_checkpoint, sglang_startup_config, monkeypatch,
):
    path = "org/main" if same_checkpoint else "org/draft"
    sglang_startup_config.spec.speculative_draft_model_path = path
    monkeypatch.setenv("MX_MODEL_REVISION", "version-1")
    before = make_context(path=path, revision="config-sha")
    monkeypatch.setenv("MX_MODEL_REVISION", "version-2")
    after = make_context(path=path, revision="config-sha")

    assert before.identity.revision == ("version-1" if same_checkpoint else "config-sha")
    assert after.identity.revision == ("version-2" if same_checkpoint else "config-sha")
    assert (
        compute_mx_source_id(before.identity) != compute_mx_source_id(after.identity)
    ) is same_checkpoint


@pytest.mark.parametrize(
    "draft_uri,expected_revision",
    [("org/main", "main-sha"), ("s3://draft/weights", "config-sha")],
)
def test_revision_inheritance_compares_draft_uri_with_main_config_when_unset(
    draft_uri, expected_revision, sglang_startup_config, monkeypatch,
):
    sglang_startup_config.spec.speculative_draft_model_path = "org/main"
    monkeypatch.setenv("MX_MODEL_REVISION", "main-sha")
    monkeypatch.setenv("MX_DRAFT_MODEL_URI", draft_uri)
    draft = make_context(path="org/main", revision="config-sha")
    assert _resolve_model_uri(draft) == draft_uri
    assert draft.identity.revision == expected_revision


@pytest.mark.parametrize(
    "draft_path,draft_revision,config_path,weights",
    [
        ("org/main", "other-sha", "org/main", None),
        ("org/draft", None, "org/draft", None),
        ("s3://draft/weights", None, "/tmp/draft-config", "s3://draft/weights"),
    ],
)
def test_different_checkpoint_does_not_inherit_main_overrides(
    draft_path, draft_revision, config_path, weights, sglang_startup_config,
    monkeypatch,
):
    sglang_startup_config.spec.speculative_draft_model_path = draft_path
    sglang_startup_config.spec.speculative_draft_model_revision = draft_revision
    revision = draft_revision or sglang_startup_config.main.revision
    monkeypatch.setenv("MX_MODEL_URI", "s3://main/weights")
    monkeypatch.setenv("MX_MODEL_REVISION", "main-sha")
    draft = make_context(path=config_path, weights=weights, revision=revision)
    assert _resolve_model_uri(draft) == (weights or config_path)
    assert draft.identity.revision == revision


@pytest.mark.parametrize("revision", [None, "config-sha"])
@pytest.mark.parametrize("disable_p2p", [False, True])
@pytest.mark.parametrize("draft_path", [None, "org/main"])
def test_same_checkpoint_streamer_loads_from_inherited_uri(
    revision, disable_p2p, draft_path, sglang_startup_config, monkeypatch,
):
    sglang_startup_config.main.revision = revision
    sglang_startup_config.spec.speculative_draft_model_path = draft_path
    monkeypatch.setenv("MX_MODEL_URI", "s3://main/weights")
    monkeypatch.setenv("MX_MODEL_REVISION", "main-sha")
    if disable_p2p:
        monkeypatch.setenv("MX_DISABLE_DRAFT_P2P", "1")
    draft = make_context(path="org/main", revision=revision)
    model = MagicMock()
    model.load_weights.side_effect = list
    with (
        patch("importlib.util.find_spec", return_value=MagicMock()),
        patch.object(
            draft.adapter, "build_model_streamer_weight_iter", return_value=iter(())
        ) as stream,
    ):
        strategy = ModelStreamerStrategy()
        assert strategy.is_available(draft)
        assert strategy.load(LoadResult(model, model), draft).model is model
    stream.assert_called_once_with("s3://main/weights", model=model)
    assert draft.identity.revision == "main-sha"
    assert draft.p2p_enabled is not disable_p2p


def test_same_checkpoint_without_main_overrides_uses_own_config(sglang_startup_config):
    sglang_startup_config.spec.speculative_draft_model_path = "org/main"
    draft = make_context(path="org/main", revision="config-sha")
    assert _resolve_model_uri(draft) == "org/main"
    assert draft.identity.revision == "config-sha"


def test_remote_checkpoint_uses_startup_paths_despite_local_config_cache(
    sglang_startup_config, monkeypatch,
):
    sglang_startup_config.main.model_path = "s3://shared/checkpoint"
    sglang_startup_config.spec.speculative_draft_model_path = "s3://shared/checkpoint"
    monkeypatch.setenv("MX_MODEL_URI", "s3://mirror/checkpoint")
    monkeypatch.setenv("MX_MODEL_REVISION", "main-sha")
    draft = make_context(
        path="/tmp/draft-config", weights="s3://shared/checkpoint",
        revision="config-sha",
    )
    assert _resolve_model_uri(draft) == "s3://mirror/checkpoint"
    assert draft.identity.revision == "main-sha"


@pytest.mark.parametrize(
    "unavailable", ["missing_module", "missing_spec_api", "unpublished_main", "unpublished_spec"]
)
def test_unavailable_startup_config_keeps_draft_overrides_and_fallback(
    unavailable, sglang_startup_config, monkeypatch,
):
    def unpublished():
        raise ValueError("config not published")

    sglang_startup_config.spec.speculative_draft_model_path = "org/main"
    runtime = sys.modules["sglang.srt.runtime_context"]
    if unavailable == "missing_module":
        monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", None)
    elif unavailable == "missing_spec_api":
        monkeypatch.delattr(runtime, "get_spec")
    else:
        getter = "get_model" if unavailable == "unpublished_main" else "get_spec"
        monkeypatch.setattr(runtime, getter, unpublished)
    monkeypatch.setenv("MX_MODEL_URI", "s3://main/weights")
    monkeypatch.setenv("MX_MODEL_REVISION", "main-sha")
    draft = make_context(path="org/main", revision="config-sha")
    assert _resolve_model_uri(draft) == "org/main"
    assert draft.identity.revision == "config-sha"
    monkeypatch.setenv("MX_DRAFT_MODEL_URI", "s3://draft/weights")
    monkeypatch.setenv("MX_DRAFT_MODEL_REVISION", "draft-sha")
    explicit = make_context(path="org/main", revision="config-sha")
    assert _resolve_model_uri(explicit) == "s3://draft/weights"
    assert explicit.identity.revision == "draft-sha"


def test_draft_failure_does_not_hide_or_shutdown_main():
    main = make_context(draft=False)
    main.nixl_manager = MagicMock()
    run_nixl(main, nn.Linear(2, 2), lambda model, ctx: model)
    draft = make_context()
    draft.nixl_manager = MagicMock()
    manager = draft.nixl_manager
    RdmaStrategy().rollback(draft)
    manager.shutdown.assert_called_once()
    main.nixl_manager.shutdown.assert_not_called()
    assert main.source_ready_fn is None


@pytest.mark.parametrize("transport", ["nixl"])
def test_explicit_disable_keeps_draft_out_of_p2p(transport, monkeypatch):
    monkeypatch.setenv("MX_DISABLE_DRAFT_P2P", "1")
    ctx = make_context(modelexpress_transport=transport)
    model = nn.Linear(2, 2)
    assert not ctx.p2p_enabled
    with patch("modelexpress.load_strategy.base._init_nixl_manager") as init:
        register_tensors(model, ctx)
    init.assert_not_called()
    if transport == "transfer_engine":
        loader = loader_mod.MxModelLoader(ctx.load_config)
        with (
            patch.object(loader_mod, "build_sglang_load_context", return_value=ctx),
            patch.object(
                ctx.adapter, "load_via_native", side_effect=lambda result: result
            ) as native,
            patch.object(loader, "_find_transfer_engine_source") as find,
        ):
            assert (
                loader._load_model_via_transfer_engine(
                    model=model,
                    model_config=ctx.model_config,
                    device_config=ctx.adapter.device_config,
                )
                is model
            )
        native.assert_called_once()
        find.assert_not_called()


class DraftWithHead(nn.Module):
    def __init__(self, own_head=False):
        super().__init__()
        self.config = SimpleNamespace(draft_vocab_size=4 if own_head else None)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(4, 2)
        self.model.decoder = nn.Linear(2, 2)
        self.lm_head = nn.Linear(2, 4, bias=False)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        self.set_embed(embed)
        self.set_head(head)

    def set_embed(self, embed):
        self.model.embed_tokens.weight = embed

    def set_head(self, head):
        self.lm_head.weight = head


class NoopSetterDraft(DraftWithHead):
    def set_embed_and_head(self, embed, head):
        pass


class PackedDraft(DraftWithHead):
    def __init__(self):
        super().__init__()
        storage = torch.arange(12, dtype=torch.float32)
        self.model.embed_tokens.weight = nn.Parameter(storage[:8].view(4, 2))
        self.model.decoder.weight = nn.Parameter(storage[8:].view(2, 2))


@pytest.mark.parametrize("own_head", [False, True])
def test_unknown_draft_preserves_complete_layout(own_head):
    model = DraftWithHead(own_head)
    ctx = make_context()
    tensors = ctx.adapter.discover_tensors(LoadResult(model, model))
    assert set(tensors) == set(dict(model.named_parameters()))
    target = make_context(draft=False)
    assert set(target.adapter.discover_tensors(LoadResult(model, model))) == set(
        dict(model.named_parameters())
    )


def test_unknown_draft_keeps_embedding_while_deduplicating_storage_aliases():
    model = DraftWithHead()
    model.register_buffer(
        "embed_alias", model.model.embed_tokens.weight.detach().view(torch.uint8)
    )
    ctx = make_context()
    tensors = ctx.adapter.discover_tensors(LoadResult(model, model))
    assert set(tensors) == set(dict(model.named_parameters()))
    assert tensors["model.embed_tokens.weight"].data_ptr() == (
        model.model.embed_tokens.weight.data_ptr()
    )


@pytest.mark.parametrize(
    "model_class,attributes,excluded",
    [
        (
            "Qwen3_5ForCausalLMMTP", {},
            {"model.embed_tokens.weight", "lm_head.weight"},
        ),
        ("Step3p5MTP", {"draft_model_idx": 0}, set()),
        (
            "NemotronHForCausalLMMTP", {"_owns_lm_head": True},
            {"model.embed_tokens.weight"},
        ),
        (
            "NemotronHForCausalLMMTP", {"_owns_lm_head": False},
            {"model.embed_tokens.weight", "lm_head.weight"},
        ),
        (
            "Dots3NoteForCausalLMNextN", {"_mtp_loaded_embed": True},
            {"lm_head.weight"},
        ),
        (
            "Dots3NoteForCausalLMNextN", {"_mtp_loaded_embed": False},
            {"model.embed_tokens.weight", "lm_head.weight"},
        ),
        ("DeepseekV4ForCausalLMDSpark", {}, set()),
    ],
)
def test_known_draft_uses_existing_publishable_rules(
    model_class, attributes, excluded,
):
    model = type(model_class, (DraftWithHead,), {})()
    model.config.num_hidden_layers = 43
    model.config.num_nextn_predict_layers = 1
    for name, value in attributes.items():
        setattr(model, name, value)
    all_names = set(dict(model.named_parameters()))
    result = LoadResult(model, model)

    draft = make_context()
    assert set(draft.adapter.discover_tensors(result)) == all_names - excluded
    main = make_context(draft=False)
    assert set(main.adapter.discover_tensors(result)) == all_names


@pytest.mark.parametrize("transport", ["nixl"])
@pytest.mark.parametrize("binding", ["parameter", "data"])
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("ready_url", ["", "http://localhost:30000/health"])
def test_publication_keeps_tensor_views_after_loader_collection(
    transport, binding, draft, ready_url, monkeypatch
):
    monkeypatch.setenv("MX_P2P_METADATA", "0")
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", ready_url)
    engine = MagicMock()
    engine.register_memory.return_value = 0
    client = MagicMock()
    client.publish_metadata.return_value = "model-source"
    ready = False
    monkeypatch.setattr(loader_mod, "_sglang_health_ready", lambda ctx: ready)
    monkeypatch.setattr(publication.PublisherThread, "start", lambda self: None)
    model = DraftWithHead()
    expected = {
        name: (tensor.data_ptr(), tensor.detach().numpy().tobytes())
        for name, tensor in model.named_parameters()
    }

    def load():
        ctx = make_context(
            draft=draft,
            modelexpress_transport=transport,
            remote_instance_weight_loader_transfer_engine=engine,
            remote_instance_weight_loader_transfer_engine_session_id="model-session",
        )
        ctx.mx_client = client
        if transport == "nixl":
            manager = NixlTransferManager(
                "draft", 0,
                listen_port=15555,
                accelerator_backend=ctx.accelerator_backend,
            )

            def chain(model, ctx):
                register_tensors(model, ctx)
                publish_metadata(ctx)
                return model

            with (
                patch(
                    "modelexpress.load_strategy.base.is_nixl_available",
                    return_value=True,
                ),
                patch(
                    "modelexpress.load_strategy.base._init_nixl_manager",
                    return_value=manager,
                ),
                patch.object(
                    manager, "register_tensors",
                    side_effect=manager._build_tensor_descriptors,
                ),
            ):
                loader, _, _ = run_nixl(ctx, model, chain)
        else:
            loader = loader_mod.MxModelLoader(ctx.load_config)
            with (
                patch.object(
                    loader_mod, "build_sglang_load_context", return_value=ctx,
                ),
                patch.object(
                    loader, "_find_transfer_engine_source", return_value=None,
                ),
                patch.object(
                    ctx.adapter, "load_via_native", side_effect=lambda result: result,
                ),
            ):
                loader._load_model_via_transfer_engine(
                    model=model,
                    model_config=ctx.model_config,
                    device_config=ctx.adapter.device_config,
                )
        views = {name: weakref.ref(tensor) for name, tensor in ctx.tensors.items()}
        key = publication.publication_key(ctx.worker_rank, ctx.publication_namespace)
        return views, weakref.ref(loader), weakref.ref(ctx), key

    views, loader_ref, ctx_ref, key = load()
    embed = nn.Parameter(torch.full((4, 2), 42.0))
    head = nn.Parameter(torch.full((4, 2), 43.0))
    if binding == "parameter":
        model.set_embed_and_head(embed, head)
    else:
        model.model.embed_tokens.weight.data = embed.data
        model.lm_head.weight.data = head.data
    gc.collect()
    assert loader_ref() is None
    assert model.model.embed_tokens.weight.data_ptr() != (
        expected["model.embed_tokens.weight"][0]
    )
    assert model.lm_head.weight.data_ptr() != expected["lm_head.weight"][0]
    assert set(views) == set(expected)
    assert all(view() is not None for view in views.values())

    publisher = publication._heartbeat_threads[key]
    client.publish_metadata.assert_not_called()
    if ready_url:
        assert not publisher._try_publish()
        client.publish_metadata.assert_not_called()
    ready = True
    assert publisher._try_publish()
    published = client.publish_metadata.call_args.args[1]
    descriptors = worker_tensor_descriptors(published)
    assert {tensor.name for tensor in descriptors} == set(expected)
    for tensor in descriptors:
        pointer, data = expected[tensor.name]
        assert tensor.addr == pointer
        assert tensor.size == len(data)
        assert ctypes.string_at(tensor.addr, tensor.size) == data

    # Ensure test fixtures do not keep the context or published views alive.
    del publisher
    loader_mod._tensor_registry.clear()
    loader_mod._nixl_managers.clear()
    publication._heartbeat_threads.clear()
    gc.collect()
    assert ctx_ref() is None
    assert all(view() is None for view in views.values())


def test_nixl_draft_receives_with_exact_catalog_policy_and_own_manager():
    ctx = make_context()
    model = DraftWithHead()
    manager = MagicMock(tensor_descriptors=[])
    tensors = dict(model.named_parameters())
    worker = p2p_pb2.WorkerMetadata(
        tensors=publication.build_tensor_protos(tensors, 0, ctx.worker_rank),
        nixl_metadata=b"draft-metadata",
    )
    manager.receive_from_source.return_value = (64, len(tensors), None)
    with (
        patch("modelexpress.load_strategy.base.is_nixl_available", return_value=True),
        patch(
            "modelexpress.load_strategy.base._init_nixl_manager", return_value=manager
        ),
        patch.object(
            ctx.adapter, "before_rdma_receive", side_effect=lambda result: result
        ),
    ):
        RdmaStrategy()._load_as_target(
            LoadResult(model, model), ctx, worker, "draft-id", "peer"
        )

    received = manager.receive_from_source.call_args.kwargs
    assert received["require_exact_match"] is True
    assert received["destination_tensors"] is None  # The manager owns only this draft.
    assert {t.name for t in received["source_tensors"]} == set(tensors)
    assert set(ctx.tensors) == set(tensors)
    manager.register_tensors.assert_called_once()
    manager.register_additional_tensors.assert_not_called()


@pytest.mark.parametrize("change", [{"path": "org/other"}, {"revision": "other-sha"}])
def test_draft_identity_distinguishes_checkpoints(change):
    source, target = make_context(), make_context(**change)
    assert compute_mx_source_id(source.identity) != compute_mx_source_id(
        target.identity
    )


def test_explicit_health_check_gates_each_model_session(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "http://localhost:30000/health")
    main, draft = make_context(draft=False), make_context()
    for ctx in (main, draft):
        run_nixl(ctx, nn.Linear(2, 2), lambda model, ctx: model)
        with patch.object(
            loader_mod, "_sglang_health_ready", return_value=False
        ) as health:
            assert not ctx.source_ready_fn()
            health.assert_called_once_with(ctx)
