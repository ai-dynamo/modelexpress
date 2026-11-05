# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved serving configuration reaches artifact publishers across workers."""

import copy
import inspect
import pickle
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from types import ModuleType, SimpleNamespace

import pytest

from modelexpress.engines.sglang import artifacts as sglang_artifacts
from modelexpress.engines.sglang import readiness as sglang_readiness
from modelexpress.engines.vllm import artifacts as vllm_artifacts
from modelexpress.engines.vllm.patches.patch_artifact_ready_url import (
    patch_artifact_ready_url,
)
from modelexpress.metadata.artifact_lifecycle import artifact_health_ready
from modelexpress.metadata.readiness import health_url_from_server_args


@pytest.fixture(autouse=True)
def clear_readiness_env(monkeypatch):
    monkeypatch.delenv("MX_ARTIFACT_READY_URL", raising=False)
    monkeypatch.delenv("LWS_LEADER_ADDRESS", raising=False)


@pytest.mark.parametrize(
    "host,expected",
    [
        (None, "127.0.0.1"),
        ("0.0.0.0", "127.0.0.1"),
        ("::", "[::1]"),
        ("::1", "[::1]"),
        ("10.0.0.7", "10.0.0.7"),
        ("fd00::7", "[fd00::7]"),
    ],
)
def test_health_url_uses_http_port_and_a_reachable_bind_address(host, expected):
    args = SimpleNamespace(host=host, port=9007, master_port=29500, nccl_port=29501)
    assert health_url_from_server_args(args) == f"http://{expected}:9007/health"


@pytest.mark.parametrize("port", [None, True, 0, -1, 65536, "9007"])
def test_missing_or_invalid_http_port_does_not_use_rpc_ports(port):
    args = SimpleNamespace(port=port, master_port=29500, nccl_port=29501)
    assert health_url_from_server_args(args) is None


def test_framework_tls_settings_select_https():
    args = SimpleNamespace(port=9443, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
    assert health_url_from_server_args(args) == "https://127.0.0.1:9443/health"


@pytest.mark.parametrize(
    "settings",
    [
        {"uds": "/tmp/model.sock"},
        {"grpc": True},
        {"grpc_mode": True},
        {"smg_grpc_mode": True},
    ],
)
def test_non_http_transports_do_not_advertise_an_unused_tcp_port(settings):
    args = SimpleNamespace(host="0.0.0.0", port=9007, **settings)
    assert health_url_from_server_args(args) is None


def test_http_server_with_an_additional_grpc_port_keeps_http_readiness():
    args = SimpleNamespace(host="0.0.0.0", port=9007, grpc_port=50051)
    assert health_url_from_server_args(args) == "http://127.0.0.1:9007/health"


@pytest.fixture
def patched_engine_args(monkeypatch):
    class EngineArgs:
        @classmethod
        def from_cli_args(cls, args):
            result = cls()
            result.model = args.model
            return result

        def create_engine_config(self, usage_context=None):
            return SimpleNamespace(
                model=self.model,
                usage_context=usage_context,
                load_config=SimpleNamespace(load_format="modelexpress"),
                additional_config={"existing": "preserved"},
            )

    module = ModuleType("vllm.engine.arg_utils")
    module.EngineArgs = EngineArgs
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert patch_artifact_ready_url()
    return EngineArgs


@pytest.mark.parametrize("port", [8000, 9007])
def test_vllm_carries_resolved_frontend_configuration_to_worker(
    patched_engine_args, port
):
    class AsyncEngineArgs(patched_engine_args):
        pass

    args = SimpleNamespace(model="example-model", port=port, host="0.0.0.0")
    engine_args = AsyncEngineArgs.from_cli_args(args)
    assert isinstance(engine_args, AsyncEngineArgs)
    config = engine_args.create_engine_config(usage_context="serve")
    assert config.model == "example-model"
    assert config.usage_context == "serve"
    assert config.additional_config == {"existing": "preserved"}
    # Both local worker copies and spawn/Ray serialization keep the endpoint.
    for worker_config in (copy.deepcopy(config), pickle.loads(pickle.dumps(config))):
        ctx = SimpleNamespace(
            artifact_ready_url=worker_config.load_config._mx_artifact_ready_url,
            head_addr="127.0.0.1",
        )
        assert vllm_artifacts._vllm_health_url(ctx) == f"http://127.0.0.1:{port}/health"
    assert patch_artifact_ready_url() is False


def test_programmatic_vllm_engine_does_not_invent_a_serving_endpoint(
    patched_engine_args,
):
    args = patched_engine_args()
    args.model = "offline-model"
    config = args.create_engine_config()
    assert not hasattr(config.load_config, "_mx_artifact_ready_url")


@pytest.mark.parametrize("settings", [{"headless": True}, {"api_server_count": 0}])
def test_headless_vllm_redirects_automatic_bind_ip_to_head(
    patched_engine_args, monkeypatch, settings
):
    args = SimpleNamespace(
        model="example-model", port=9007, host="10.0.0.2", **settings
    )
    config = patched_engine_args.from_cli_args(args).create_engine_config()
    ctx = SimpleNamespace(
        artifact_ready_url=config.load_config._mx_artifact_ready_url,
        head_addr="10.0.0.1",
        node_rank=1,
    )
    assert vllm_artifacts._vllm_health_url(ctx) == "http://10.0.0.1:9007/health"
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "http://custom-health:9010/ready")
    assert vllm_artifacts._vllm_health_url(ctx) == "http://custom-health:9010/ready"


@pytest.mark.parametrize("getter_name", ["get_serving", "get_global_server_args"])
@pytest.mark.parametrize("port", [30000, 9008])
def test_sglang_reads_resolved_configuration_from_supported_versions(
    monkeypatch, getter_name, port
):
    module = SimpleNamespace(**{
        getter_name: lambda: SimpleNamespace(host="0.0.0.0", port=port),
    })
    module.get_parallel = lambda: SimpleNamespace(node_rank=0)
    monkeypatch.setattr(sglang_readiness, "import_module", lambda name: module)
    assert sglang_artifacts._sglang_health_url() == f"http://127.0.0.1:{port}/health"


def test_sglang_falls_back_when_runtime_context_is_not_published(monkeypatch):
    def not_published():
        raise ValueError("Serving configuration is not published")

    def module(name):
        if name.endswith("runtime_context"):
            return SimpleNamespace(get_serving=not_published)
        return SimpleNamespace(
            get_global_server_args=lambda: SimpleNamespace(port=9010, node_rank=0),
        )

    monkeypatch.setattr(sglang_readiness, "import_module", module)
    assert sglang_artifacts._sglang_health_url() == "http://127.0.0.1:9010/health"


@pytest.mark.parametrize(
    "getter_name,mode_name",
    [("get_global_server_args", "grpc_mode"), ("get_serving", "smg_grpc_mode")],
)
def test_sglang_grpc_only_configuration_has_no_inferred_http_endpoint(
    monkeypatch, getter_name, mode_name
):
    args = SimpleNamespace(host="0.0.0.0", port=9007, **{mode_name: True})
    module = SimpleNamespace(**{getter_name: lambda: args})
    monkeypatch.setattr(sglang_readiness, "import_module", lambda name: module)
    assert sglang_readiness.sglang_server_readiness().url is None


def test_sglang_follower_keeps_its_local_dummy_health_listener():
    ctx = SimpleNamespace(
        artifact_ready_url="http://10.0.0.2:9007/health",
        head_addr="10.0.0.1",
        node_rank=1,
    )
    assert sglang_artifacts._sglang_health_url(ctx) == "http://10.0.0.2:9007/health"


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.2"])
def test_sglang_tls_follower_probes_its_local_plain_http_listener(
    monkeypatch, modern, host
):
    monkeypatch.delenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN", raising=False)
    args = SimpleNamespace(
        host=host, port=9007, ssl_keyfile="key.pem", ssl_certfile="cert.pem"
    )
    if modern:
        modules = {
            "sglang.srt.runtime_context": SimpleNamespace(
                get_serving=lambda: args,
                get_parallel=lambda: SimpleNamespace(node_rank=1),
            ),
        }
    else:
        args.node_rank = 1
        modules = {
            "sglang.srt.server_args": SimpleNamespace(get_global_server_args=lambda: args),
        }
    modules["sglang.srt.environ"] = SimpleNamespace(
        envs=SimpleNamespace(SGLANG_RUST_SERVER=SimpleNamespace(get=lambda: False))
    )

    def module(name):
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(sglang_readiness, "import_module", module)
    expected = f"http://{host}:9007/health"
    assert sglang_artifacts._sglang_health_url() == expected
    # A locally available follower listener must not become a cleartext probe
    # of the TLS head merely because an orchestrator supplies its address.
    monkeypatch.setenv("LWS_LEADER_ADDRESS", "tls-head.internal")
    assert sglang_artifacts._sglang_health_url() == expected
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "https://custom-health:9443/ready")
    assert sglang_artifacts._sglang_health_url() == "https://custom-health:9443/ready"


@pytest.mark.parametrize(
    "node_rank,rust,owns_port,scheme,local_dummy",
    [
        (0, False, False, "https", False),
        (1, False, False, "http", True),
        (1, True, True, "https", False),
        (1, True, False, "http", True),
    ],
)
def test_sglang_listener_selection_uses_native_rust_placement(
    monkeypatch, node_rank, rust, owns_port, scheme, local_dummy
):
    monkeypatch.delenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN", raising=False)
    args = SimpleNamespace(
        host="127.0.0.1", port=9443, ssl_keyfile="key.pem", ssl_certfile="cert.pem"
    )
    placement_calls = []

    def node_hosts_rust_server():
        placement_calls.append(True)
        return owns_port

    modules = {
        "sglang.srt.runtime_context": SimpleNamespace(
            get_serving=lambda: args,
            get_parallel=lambda: SimpleNamespace(node_rank=node_rank),
        ),
        "sglang.srt.environ": SimpleNamespace(
            envs=SimpleNamespace(SGLANG_RUST_SERVER=SimpleNamespace(get=lambda: rust))
        ),
        "sglang.srt.entrypoints.engine": SimpleNamespace(
            node_hosts_rust_server=node_hosts_rust_server
        ),
    }

    def module(name):
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(sglang_readiness, "import_module", module)
    readiness = sglang_readiness.sglang_server_readiness()
    assert readiness.url == f"{scheme}://127.0.0.1:9443/health"
    assert readiness.local_dummy is local_dummy
    assert placement_calls == ([True] if rust and node_rank > 0 else [])


@pytest.mark.parametrize("unknown_rust", [False, True])
def test_sglang_does_not_claim_a_listener_when_native_placement_is_unavailable(
    monkeypatch, unknown_rust
):
    # An API follower can return without starting any dummy server. A new
    # Rust version without its placement helper is also not safe to guess.
    monkeypatch.setenv(
        "SGLANG_BLOCK_NONZERO_RANK_CHILDREN", "1" if unknown_rust else "0"
    )
    args = SimpleNamespace(host="127.0.0.1", port=9007, node_rank=1)
    modules = {
        "sglang.srt.server_args": SimpleNamespace(get_global_server_args=lambda: args),
        "sglang.srt.environ": SimpleNamespace(
            envs=SimpleNamespace(
                SGLANG_RUST_SERVER=SimpleNamespace(get=lambda: unknown_rust)
            )
        ),
    }

    def module(name):
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(sglang_readiness, "import_module", module)
    assert sglang_readiness.sglang_server_readiness().url is None


def test_sglang_publisher_uses_captured_local_listener_without_runtime_getters(
    monkeypatch,
):
    monkeypatch.setenv("LWS_LEADER_ADDRESS", "tls-head.internal")

    def unavailable():
        raise AssertionError("The worker already captured its listener configuration")

    monkeypatch.setattr(sglang_artifacts, "sglang_server_readiness", unavailable)
    ctx = SimpleNamespace(
        artifact_ready_url="http://127.0.0.1:9007/health", artifact_ready_local=True
    )
    assert sglang_artifacts._sglang_health_url(ctx) == "http://127.0.0.1:9007/health"


@pytest.mark.parametrize(
    "resolve", [vllm_artifacts._vllm_health_url, sglang_artifacts._sglang_health_url]
)
def test_explicit_url_wins_over_framework_configuration(monkeypatch, resolve):
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "https://health.internal:9443/ready")
    ctx = SimpleNamespace(artifact_ready_url="http://127.0.0.1:9007/health")
    assert resolve(ctx) == "https://health.internal:9443/ready"


def test_explicit_sglang_url_does_not_require_runtime_configuration(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "http://health.internal:9009/ready")

    def should_not_read_config():
        raise AssertionError("The explicit URL must take precedence")

    monkeypatch.setattr(
        sglang_artifacts, "sglang_server_readiness", should_not_read_config
    )
    assert sglang_artifacts._sglang_health_url() == "http://health.internal:9009/ready"


@pytest.mark.parametrize(
    "resolve", [vllm_artifacts._vllm_health_url, sglang_artifacts._sglang_health_url]
)
def test_multi_node_health_keeps_resolved_http_port(monkeypatch, resolve):
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "   ")
    ctx = SimpleNamespace(
        artifact_ready_url="http://127.0.0.1:9007/health", head_addr="head.internal"
    )
    assert resolve(ctx) == "http://head.internal:9007/health"


def test_probe_reaches_the_resolved_server_port():
    paths = []

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), HealthHandler) as server:
        thread = Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            url = health_url_from_server_args(SimpleNamespace(port=server.server_port))
            ctx = SimpleNamespace(artifact_ready_url=url)
            assert artifact_health_ready(vllm_artifacts._vllm_health_url(ctx))
        finally:
            thread.join(timeout=3)
    assert paths == ["/health"]


@pytest.mark.parametrize(
    "use_yaml,cli_port,expected_port,extra_args",
    [
        (False, None, 8000, []),
        (True, None, 9100, []),
        (True, 9101, 9101, []),
        (True, 9101, 9101, ["--headless", "--host", "10.0.0.2"]),
        (False, 9007, None, ["--uds", "/tmp/mx-test-no-created.sock"]),
    ],
)
def test_native_vllm_parser_preserves_config_precedence_and_cache_hash(
    monkeypatch, tmp_path, use_yaml, cli_port, expected_port, extra_args
):
    arg_utils = pytest.importorskip("vllm.engine.arg_utils")
    if not isinstance(arg_utils, ModuleType):
        pytest.skip("Requires an installed vLLM, rather than the conftest mocks")
    serve = pytest.importorskip("vllm.entrypoints.cli.serve")
    parser_utils = pytest.importorskip("vllm.utils.argparse_utils")
    load_module = pytest.importorskip("vllm.config.load")

    # Parser construction loads general plugins. Keep their logging setup from
    # attaching pytest's capture handlers to other tests in this process.
    monkeypatch.setattr(
        "modelexpress.engines.vllm.configure_vllm_logging", lambda: None
    )
    parser = parser_utils.FlexibleArgumentParser()
    serve.ServeSubcommand().subparser_init(parser.add_subparsers(dest="subparser"))
    engine_cls = arg_utils.EngineArgs
    monkeypatch.setattr(
        engine_cls,
        "from_cli_args",
        classmethod(inspect.unwrap(engine_cls.from_cli_args.__func__)),
    )
    before_hash = []

    def create_without_loading_a_model(self, *args, **kwargs):
        load_config = load_module.LoadConfig(load_format="modelexpress")
        before_hash.append(load_config.compute_hash())
        return SimpleNamespace(load_config=load_config, additional_config={})

    monkeypatch.setattr(engine_cls, "create_engine_config", create_without_loading_a_model)
    patch_artifact_ready_url()
    argv = ["serve", "/tmp/mx-test-no-model-loading", "--load-format", "modelexpress"]
    if use_yaml:
        config = tmp_path / "serve.yaml"
        config.write_text("port: 9100\n")
        argv.extend(["--config", str(config)])
    if cli_port is not None:
        argv.extend(["--port", str(cli_port)])
    argv.extend(extra_args)
    args = parser.parse_args(argv)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    # Avoid constructor-side model lookup: this case tests the real parser,
    # conversion and LoadConfig serialization, not model discovery or loading.
    monkeypatch.setattr(engine_cls, "__post_init__", lambda self: None)
    config = engine_cls.from_cli_args(args).create_engine_config()
    worker_load_config = pickle.loads(pickle.dumps(config.load_config))
    if expected_port is None:
        assert not hasattr(worker_load_config, "_mx_artifact_ready_url")
        assert worker_load_config.compute_hash() == before_hash[0]
        return
    assert worker_load_config._mx_artifact_ready_url == (
        f"http://127.0.0.1:{expected_port}/health"
    )
    ctx = SimpleNamespace(
        artifact_ready_url=worker_load_config._mx_artifact_ready_url,
        head_addr="10.0.0.1" if args.headless else "127.0.0.1",
    )
    expected_host = "10.0.0.1" if args.headless else "127.0.0.1"
    assert vllm_artifacts._vllm_health_url(ctx) == (
        f"http://{expected_host}:{expected_port}/health"
    )
    assert worker_load_config.compute_hash() == before_hash[0]
    assert config.additional_config == {}
