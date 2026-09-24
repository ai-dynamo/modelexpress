"""Validation contracts shared by the online driver and offline report."""

if not __debug__:
    raise RuntimeError(
        "Benchmark validation requires Python assertions; remove -O/PYTHONOPTIMIZE"
    )


def ranks(rows, config):
    assert isinstance(rows, list) and len(rows) == config["tp"], (
        "Incomplete rank results"
    )
    by_rank = {r["rank"]: r for r in rows}
    assert set(by_rank) == set(range(config["tp"])), "Missing or duplicate ranks"
    for row in by_rank.values():
        assert "error" not in row and not row["phase"].endswith("-failed"), row
    return by_rank


def hashes(rows, config):
    result = {}
    for rank, row in ranks(rows, config).items():
        tensors = row["tensors"]
        assert tensors, "Empty tensor inventory"
        count = config["expected_tensors_per_rank"]
        if count is not None:
            assert len(tensors) == count, (rank, len(tensors), count)
        result[rank] = tensors
    return result


def scales(rows, config):
    for row in ranks(rows, config).values():
        assert row["enforce_eager"] is True
        expected = config["expected_host_scales_per_rank"]
        if expected is not None:
            assert len(row["scales"]) == expected
        assert all(
            x["gpu"] == x["host"] and (x["cpu"] is None or x["gpu"] == x["cpu"])
            for x in row["scales"]
        ), row


def refit(rows, config, role, version=None):
    for row in ranks(rows, config).values():
        assert (
            row["phase"]
            == row["version"]
            == row["serving_version"]
            == (version or config["run"] + "-d1")
        ), row
        assert row["source"] == ("OBJECT_STORAGE" if role == "s3" else "GENERATOR")
        assert row["weight_addresses_preserved"]


def inference(rows):
    assert rows
    for row in rows:
        assert row["token_ids"] and row["logprob_count"] == len(row["token_ids"]), row


def trials(publication, config):
    assert publication["run"] == config["run"]
    assert publication["model_revision"] == config["revision"]
    rows = publication["trials"]
    assert len(rows) == 2, "Expected two published updates"
    for step, row in enumerate(rows, 1):
        assert row["version"] == config["run"] + f"-d{step}"
        assert row["base_version"] == config["run"] + ("-base" if step == 1 else "-d1")
        assert row["payload_bytes"] > 0
        assert row["expected_hashes"] and all(row["expected_hashes"].values())
        assert row["expected_sha256"] == row["expected_hashes"][config["embedding"]]
    return rows


def checkpoint(before, after, config, trial, *, reuse):
    before, after = ranks(before, config), ranks(after, config)
    for rank, old in before.items():
        new = after[rank]
        assert old["version"] == trial["base_version"]
        assert new["version"] == trial["version"]
        assert old["files"] and old["files"].keys() == new["files"].keys()
        for name, identity in old["files"].items():
            current = new["files"][name]
            assert identity["bytes"] == current["bytes"] > 0
            same = (identity["device"], identity["inode"]) == (
                current["device"],
                current["inode"],
            )
            assert same == reuse, ("Unexpected checkpoint copy/reuse", rank, name)


def refitted(config, trial, result, previous, sessions, *, reuse):
    version = trial["version"]
    updated = {}
    for role in config["roles"]:
        rows = result(role, "refit")
        refit(rows, config, role, version)
        for rank, row in ranks(rows, config).items():
            assert row["refit_session"] == sessions[role][rank], "Refit client changed"
            expected_trace = (
                config["second_update_allocation_tracing"] if reuse else True
            )
            assert row["allocation_tracing"] is expected_trace
        updated[role] = hashes(result(role, "updated-hashes"), config)
        for rank, tensors in updated[role].items():
            assert tensors.keys() == previous[role][rank].keys(), (
                "Tensor inventory changed"
            )
        assert updated[role] != previous[role], ("No updated tensors", role)
        if config["expected_host_scales_per_rank"] is not None:
            for step in [
                "baseline-host-scales",
                "immediate-post-refit-host-scales",
            ]:
                scales(result(role, step), config)
    check = ranks(result("s3", "verify-checkpoint"), config)
    for row in check.values():
        assert row["version"] == version and row["verified"] is True
        assert row["sha256"] == row["expected_sha256"]
    assert check[0]["full_checkpoint_sha256"] == trial["expected_sha256"]
    assert check[0]["publisher_hashes"] == trial["expected_hashes"], (
        "Checkpoint differs from publisher"
    )
    checkpoint(
        result("s3", "layout-before"),
        result("s3", "layout-after"),
        config,
        trial,
        reuse=reuse,
    )
    if "peer" in updated:
        assert updated["s3"] == updated["peer"], "Refit peer tensors differ per TP rank"
    return updated


def update(config, trial, result, previous, sessions, *, reuse):
    updated = refitted(config, trial, result, previous, sessions, reuse=reuse)
    for role in config["roles"]:
        inference(result(role, "post-refit-inference"))
        if config["expected_host_scales_per_rank"] is not None:
            scales(result(role, "post-inference-host-scales"), config)
    if "peer" in updated:
        assert result("s3", "post-refit-inference") == result(
            "peer", "post-refit-inference"
        ), "Resumed inference differs"
    return updated


def sessions(config, result):
    values = {}
    for role in config["roles"]:
        values[role] = {}
        for rank, row in ranks(result(role, "init"), config).items():
            assert row["phase"] == "init" and row["version"] == config["run"] + "-base"
            assert row["refit_session"]
            values[role][rank] = row["refit_session"]
    return values
