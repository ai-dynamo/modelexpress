"""Wire the runbook's dry-run configs to the allocated Slurm hosts."""

import json
import os
from pathlib import Path
import shutil


def configure(env, model_is_moe=True):
    roots = list(Path(env["CFG_BASE"]).glob("*/configs/latest/resolved"))
    if len(roots) != 1:
        raise ValueError(f"Expected one resolved config directory, found {roots}")
    target = Path(env["CONFIG_ROOT"])
    shutil.copytree(roots[0], target)
    configs = {role: json.loads((target / f"{role}.json").read_text()) for role in ("trainer", "inference", "orchestrator")}
    out = env["OUT"]
    hosts = env["HOSTS_CSV"].split(",")
    infer_nodes, train_nodes, gpus, tp = (int(env[k]) for k in ("INFER_NODES", "TRAIN_NODES", "GPUS_PER_NODE", "TP"))
    if len(hosts) != train_nodes + infer_nodes:
        raise ValueError("Allocated host count differs from configuration")
    ray = env.get("EXECUTOR", "mp") == "ray"
    nodes_per_replica = int(env.get("NODES_PER_REPLICA", infer_nodes))
    admin = (
        [f"http://{hosts[n]}:8100/v1" for n in range(0, infer_nodes, nodes_per_replica)]
        if ray else [f"http://{host}:{8100+d}/v1" for host in hosts[:infer_nodes] for d in range(gpus // tp)]
    )
    for role in ("trainer", "orchestrator"):
        cfg = configs[role]
        if env.get("SAVE_CHECKPOINTS") == "false":
            cfg["ckpt"] = None
        cfg["output_dir"] = out
        cfg["monitors"]["file"]["path"] = f"{out}/{role}-metrics.jsonl"
        cfg["rollout_transport"]["host"] = env["ORCHESTRATOR_HOST"] if role == "trainer" else "0.0.0.0"
        cfg["rollout_transport"]["port"] = 5655
        if cfg["weight_broadcast"]["type"] != "mx_refit":
            raise ValueError("Wrong weight transport in generated config")
    configs["trainer"]["dist_timeout_seconds"] = 3600
    configs["trainer"]["weight_broadcast"]["staging_mode"] = env["TRAINER_STAGING_MODE"]
    if "REFIT_HANDSHAKE_MODE" in env:
        mode = env["REFIT_HANDSHAKE_MODE"]
        if mode not in ("object", "tensor"):
            raise ValueError("Invalid refit handshake mode")
        configs["trainer"]["weight_broadcast"]["handshake_mode"] = mode
    if "REFIT_HANDSHAKE_BARRIER" in env:
        barrier = env["REFIT_HANDSHAKE_BARRIER"]
        if barrier not in ("true", "false"):
            raise ValueError("Invalid refit handshake barrier setting")
        configs["trainer"]["weight_broadcast"]["handshake_barrier"] = barrier == "true"
    configs["trainer"]["model"]["conversion_dir"] = env.get("CONVERSION_DIR") or f"{out}/model-conversion"
    if "ENABLE_THINKING" in env:
        configs["orchestrator"]["renderer"]["enable_thinking"] = env["ENABLE_THINKING"] == "true"
    configs["orchestrator"]["model"]["client"].update(
        base_url=f"http://{env['INFER_HEAD']}:8000/v1", admin_base_url=admin,
    )
    for role, cfg in configs.items():
        (target / f"{role}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for path in (target / "envs").glob("*/*.json"):
        cfg = json.loads(path.read_text())
        split, name = path.parent.name, path.stem
        cfg["address_file"] = str(Path(out) / "configs/latest/resolved/envs" / split / f"{name}.address")
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    for local_rank in range(1 if ray else gpus // tp):
        cfg = json.loads(json.dumps(configs["inference"]))
        cfg["router"] = None
        cfg["server"].update(host="0.0.0.0", port=8100 + local_rank)
        v = cfg["vllm"]
        v.update(tensor_parallel_size=tp, data_parallel_size=1 if ray else infer_nodes*gpus//tp,
                 data_parallel_size_local=1, api_server_count=1, enable_expert_parallel=model_is_moe,
                 dtype="bfloat16", quantization=None, enable_eplb=False)
        if "ENABLE_PREFIX_CACHING" in env:
            v["enable_prefix_caching"] = env["ENABLE_PREFIX_CACHING"] == "true"
        if ray:
            v["distributed_executor_backend"] = "ray"
        if v["data_parallel_size"] > 1:
            v.update(data_parallel_rank=int(env["ROLE_RANK"])*(gpus//tp)+local_rank,
                     data_parallel_address=env["INFER_HEAD"], data_parallel_rpc_port=13345)
        for key in ("deployment", "slurm", "dry_run", "output_dir"):
            cfg.pop(key, None)
        (target / f"inference-{local_rank}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("PR3487_CONFIGS_OK", target, "admin_endpoints", len(admin))


if __name__ == "__main__":
    model_config = json.loads((Path(os.environ["MODEL_PATH"]) / "config.json").read_text())
    configure(os.environ, bool(model_config.get("num_experts", model_config.get("n_routed_experts", 0))))
