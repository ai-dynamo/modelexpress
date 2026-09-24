"""Render a model profile for an explicit Kubernetes/storage environment; no deployment."""

import argparse
import json
import os
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REDIS_IMAGE = "redis:7.4.5-alpine@sha256:bb186d083732f669da90be8b0f975a37812b15e913465bb14d845db72a4e3e08"


def environment_config(environment, overrides):
    if isinstance(environment, dict):
        env = json.loads(json.dumps(environment))
    else:
        path = Path(environment)
        if not path.is_file():
            path = ROOT / "environments" / (str(environment) + ".json")
        env = json.loads(path.read_text())
    aliases = {
        "context": ("KUBE_CONTEXT", "WORKER_CONTEXT"),
        "namespace": ("NAMESPACE",),
        "kubeconfig": ("KUBECONFIG",),
        "region": ("MX_CI_S3_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
        "bucket": ("MX_CI_S3_BUCKET",),
        "endpoint_url": ("AWS_ENDPOINT_URL",),
    }
    for key, names in aliases.items():
        for name in names:
            if os.environ.get(name):
                env[key] = os.environ[name]
                break
    env.update({k: v for k, v in overrides.items() if v is not None})
    for key in ["context", "namespace", "region", "bucket"]:
        if not env.get(key) or not isinstance(env[key], str):
            raise ValueError(f"Environment requires an explicit {key}")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", env["namespace"]):
        raise ValueError("Invalid Kubernetes namespace")
    env.setdefault("endpoint_url", None)
    env.setdefault("addressing_style", "auto")
    if env["endpoint_url"] == "":
        env["endpoint_url"] = None
    if env["addressing_style"] not in ["auto", "virtual", "path"]:
        raise ValueError("Invalid S3 addressing_style")
    return env


def prepare(model, output, run, paths=None, *, environment, **overrides):
    models = json.loads((ROOT / "profiles.json").read_text())["models"]
    if model not in models:
        raise ValueError(f"Unknown model profile: {model}")
    config = json.loads((ROOT / "profiles" / model / "profile.json").read_text())
    env = environment_config(environment, overrides)
    allowed_models = env.get("allowed_models")
    if allowed_models is not None and model not in allowed_models:
        raise ValueError(
            f"Environment allows only: {', '.join(allowed_models)}; requested {model}"
        )
    if not re.fullmatch(model + r"-[a-z0-9-]{1,32}", run):
        raise ValueError(
            f"RUN_ID must be {model}- followed by 1–32 lowercase letters/digits/hyphens"
        )
    paths = paths or env.get("default_paths", "s3")
    if paths not in ["s3", "both"]:
        raise ValueError("paths must be s3 or both")
    if paths == "both" and not (
        env.get("extra_worker_resources")
        and env.get("worker_env", {}).get("MX_NIXL_BACKEND")
        and env.get("peer_transfer_marker")
    ):
        raise ValueError(
            "Peer coverage requires explicit extra_worker_resources, "
            "worker_env.MX_NIXL_BACKEND, and peer_transfer_marker"
        )
    for key in ["tp", "cpu", "memory", "seed_prefix", "delta_prefix"]:
        if key in env:
            config[key] = env[key]
    if not isinstance(config["tp"], int) or config["tp"] < 1:
        raise ValueError("tp must be a positive integer")
    config.update(
        run=run,
        resource_prefix="mx-" + run,
        roles=["s3", "peer"] if paths == "both" else ["s3"],
        bucket=env["bucket"],
        storage={k: env[k] for k in ["endpoint_url", "region", "addressing_style"]},
        gpu_memory_utilization=env.get("gpu_memory_utilization", 0.70),
        peer_transfer_marker=env.get("peer_transfer_marker", "RDMA transfer complete:"),
    )
    if not 0 < config["gpu_memory_utilization"] <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    config["second_update_allocation_tracing"] = env.get(
        "second_update_allocation_tracing", False
    )
    if type(config["second_update_allocation_tracing"]) is not bool:
        raise ValueError("second_update_allocation_tracing must be a boolean")
    prefix = config["resource_prefix"]
    images = {}
    for kind, aliases in {
        "server": ["SERVER_IMAGE", "MX_MAIN_SERVER_IMAGE"],
        "runtime": ["WORKER_IMAGE", "MX_MAIN_RUNTIME_IMAGE"],
    }.items():
        value = next(
            (os.environ[a] for a in aliases if os.environ.get(a)),
            env.get("images", {}).get(kind, ""),
        )
        # Upstream CI builds commit-SHA tags. Record actual pod imageIDs in evidence too.
        if not re.fullmatch(
            r"[a-zA-Z0-9./:_-]+(?:@sha256:[0-9a-f]{64}|:[0-9a-f]{40,64})", value
        ):
            raise ValueError(
                f"{kind} image must be digest-qualified or tagged with a full commit SHA"
            )
        images[
            "MX_MAIN_" + ("SERVER" if kind == "server" else "RUNTIME") + "_IMAGE"
        ] = value
    storage_env = {
        "AWS_DEFAULT_REGION": env["region"],
        "AWS_REGION": env["region"],
        "AWS_CONFIG_FILE": "/etc/aws/config",
        "PYTHONPATH": "/opt/benchmark",
    }
    if env["endpoint_url"] is not None:
        storage_env["AWS_ENDPOINT_URL"] = env["endpoint_url"]
    pod_env = env.get("pod_env", [])
    control_env = [
        {"name": k, "value": v} for k, v in {**storage_env, "DELTA_RUN": run}.items()
    ] + pod_env
    resources = {
        "cpu": str(config["cpu"]),
        "memory": config["memory"],
        "nvidia.com/gpu": str(config["tp"]),
        **env.get("extra_worker_resources", {}),
    }
    # Extended resource overrides cannot change the model's requested GPU count.
    if resources["nvidia.com/gpu"] != str(config["tp"]):
        raise ValueError("Use tp to change GPU count")
    values = {
        **images,
        "NAMESPACE": env["namespace"],
        "CONTROL_NAME": prefix + "-control",
        "CONTROL_LABELS": {"app": prefix + "-control", "mx-regression": run},
        "RUN_LABELS": {"mx-regression": run},
        "HARNESS_NAME": prefix + "-harness",
        "AWS_CONFIG_NAME": prefix + "-aws",
        "AWS_CONFIG": f"[default]\nregion = {env['region']}\ns3 =\n  addressing_style = {env['addressing_style']}\n",
        "SERVICE_ACCOUNT": env.get("service_account", "default"),
        "WORKER_NODE_SELECTOR": env.get("worker_node_selector", {}),
        "CONTROL_NODE_SELECTOR": env.get("control_node_selector", {}),
        "WORKER_TOLERATIONS": env.get("worker_tolerations", []),
        "CONTROL_TOLERATIONS": env.get("control_tolerations", []),
        "PRIORITY_CLASS": env.get("priority_class", ""),
        "IMAGE_PULL_SECRETS": env.get("image_pull_secrets", []),
        "WORKER_SECURITY_CONTEXT": env.get(
            "worker_security_context",
            {"runAsUser": 0, "capabilities": {"add": ["IPC_LOCK"]}},
        ),
        "WORKER_RESOURCES": resources,
        "CONTROL_RESOURCES": env.get(
            "control_resources",
            {
                "requests": {"cpu": "4", "memory": "32Gi"},
                "limits": {"cpu": "16", "memory": "64Gi"},
            },
        ),
        "ENV_FROM": env.get("pod_env_from", []),
        "CONTROL_ENV": control_env,
        "WORKER_MATCH_LABELS": {"mx-regression": run, "mx-worker": "true"},
        "REDIS_IMAGE": env.get("redis_image", REDIS_IMAGE),
    }
    output.mkdir(parents=True, exist_ok=False)
    for role in ["control", *config["roles"]]:
        worker_env = {
            "VLLM_PLUGINS": "modelexpress",
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "MX_INSTANT_TENSOR": "0",
            "MX_MS_DISTRIBUTED": "0",
            "MX_P2P_METADATA": "1",
            "MX_POOL_REG": "1",
            "MX_METRICS_ENABLED": "1",
            "PROMETHEUS_MULTIPROC_DIR": "/tmp/mx-metrics",
            "MX_METRICS_PORT": "9402",
            "MX_TRANSFER_TIMEOUT": "1800",
            "MODEL_EXPRESS_LOG_LEVEL": "INFO",
            "NIXL_LOG_LEVEL": "INFO",
            "MX_GENERATOR_SOURCE_ORDER": "OBJECT_STORAGE",
            "DYN_RL_INIT_WEIGHTS_TIMEOUT_S": "3600",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            **env.get("worker_env", {}),
            **storage_env,
            "MODEL_EXPRESS_URL": prefix + "-control:8000",
            "MX_SERVER_ADDRESS": prefix + "-control:8000",
            "MX_MODEL_URI": f"s3://{config['bucket']}/{config['seed_prefix'].rstrip('/')}",
            "BENCH_MODEL": config["model"],
            "BENCH_REVISION": config["revision"],
            "BENCH_PREFIX": config["seed_prefix"],
            "BENCH_KEY": model,
            "BENCH_ROLE": role,
            "BENCH_RUN": run,
        }
        fields = {
            **values,
            "WORKER_NAME": prefix + "-" + role,
            "WORKER_LABELS": {
                "app": prefix + "-" + role,
                "mx-regression": run,
                "mx-worker": "true",
            },
            "WORKER_ENV": [{"name": k, "value": str(v)} for k, v in worker_env.items()]
            + pod_env,
        }
        template = "control" if role == "control" else "worker"
        text = (ROOT / "yaml" / (template + ".yaml.in")).read_text()
        # JSON values are valid YAML scalars/collections and preserve types/escaping.
        text = re.sub(
            r"\$\{([A-Z_]+)\}", lambda m, fields=fields: json.dumps(fields[m[1]]), text
        )
        (
            output / ("control.yaml" if role == "control" else f"worker-{role}.yaml")
        ).write_text(text)
    data = {p.name: p.read_text() for p in (ROOT / "common").glob("*.py")}
    for module in ["model_checks.py", "model_adapter.py"]:
        source = ROOT / "profiles" / model / module
        if source.exists():
            data[module] = source.read_text()
    data["config.json"] = json.dumps(config, indent=2)
    for name, text in data.items():
        if name.endswith(".py"):
            compile(text, name, "exec")
        (output / name).write_text(text)
    harness = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": prefix + "-harness",
            "namespace": env["namespace"],
            "labels": {"mx-regression": run},
        },
        "data": data,
    }
    (output / "harness.json").write_text(json.dumps(harness, indent=2) + "\n")
    (output / "images.json").write_text(json.dumps(images, indent=2) + "\n")
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    shell_env = {
        "WORKER_CONTEXT": env["context"],
        "NAMESPACE": env["namespace"],
        "KUBECONFIG": env.get("kubeconfig", ""),
        "RUN_ID": run,
        "RESOURCE_PREFIX": prefix,
        "RESULTS_DIR": str(output.resolve()),
        "ROLES": " ".join(config["roles"]),
    }
    (output / "run.env").write_text(
        "".join(f"export {k}={shlex.quote(v)}\n" for k, v in shell_env.items())
    )
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", choices=json.loads((ROOT / "profiles.json").read_text())["models"]
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--environment", required=True, help="Preset name or JSON file")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--paths", choices=["s3", "both"])
    for name in [
        "context",
        "namespace",
        "kubeconfig",
        "region",
        "bucket",
        "endpoint-url",
        "service-account",
        "seed-prefix",
        "delta-prefix",
        "cpu",
        "memory",
    ]:
        parser.add_argument("--" + name)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    args = vars(parser.parse_args())
    keys = ["model", "output", "run_id", "paths", "environment"]
    fixed = {k: args.pop(k) for k in keys}
    prepare(
        fixed["model"],
        fixed["output"],
        fixed["run_id"],
        fixed["paths"],
        environment=fixed["environment"],
        **args,
    )
    print(fixed["output"])
