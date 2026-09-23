"""Nemotron #782 regression checks, outside shared distribution accounting."""


def _validate_rank(config, result):
    report = {}
    if "peer" not in config["roles"]:
        return report
    before = result("peer", "baseline-host-scales")[0]["scales"]
    after = result("peer", "immediate-post-refit-host-scales")[0]["scales"]
    assert [(r["module"], r["scale"]) for r in before] == [
        (r["module"], r["scale"]) for r in after
    ]
    changed = sum(a["host"] != b["host"] for a, b in zip(before, after))
    assert changed == 11, "Did not exercise the expected Nemotron scale updates"
    report["changed_peer_host_entries"] = changed
    assert all(
        v is None
        for r in result("peer", "immediate-post-refit-host-scales")[0][
            "flashinfer_caches"
        ]
        for v in r["values"].values()
    )
    caches = {
        role: {
            r["module"]: r["values"]
            for r in result(role, "post-inference-host-scales")[0]["flashinfer_caches"]
        }
        for role in config["roles"]
    }
    report["post_inference_launch_cache_differences"] = [
        {
            "module": name,
            "cache": key,
            "s3": value,
            "peer": caches["peer"][name][key],
        }
        for name, values in caches["s3"].items()
        for key, value in values.items()
        if value != caches["peer"][name][key]
    ]
    return report


def validate(config, result):
    import validation

    if "peer" not in config["roles"]:
        return {}
    steps = [
        "baseline-host-scales",
        "immediate-post-refit-host-scales",
        "post-inference-host-scales",
    ]
    rows = {
        (role, step): validation.ranks(result(role, step), config)
        for role in config["roles"]
        for step in steps
    }
    reports = {}
    for rank in range(config["tp"]):
        reports[rank] = _validate_rank(
            config, lambda role, step, rank=rank: [rows[role, step][rank]]
        )
    return {**reports[0], "host_scale_checks_by_rank": reports}
