"""Validation contracts shared by the online driver and offline report."""


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


def refit(rows, config, role):
    for row in ranks(rows, config).values():
        assert (
            row["phase"]
            == row["version"]
            == row["serving_version"]
            == config["run"] + "-d1"
        ), row
        assert row["source"] == ("OBJECT_STORAGE" if role == "s3" else "GENERATOR")
        assert row["weight_addresses_preserved"]


def inference(rows):
    assert rows
    for row in rows:
        assert row["token_ids"] and row["logprob_count"] == len(row["token_ids"]), row
