"""Require per-rank Kimi MLA refresh evidence for each sequential update."""

import validation


def validate(config, result):
    for role in config["roles"]:
        before = validation.ranks(result(role, "derived-baseline"), config)
        after = validation.ranks(result(role, "derived-updated"), config)
        for rank, row in after.items():
            assert before[rank]["verified"] is True and row["verified"] is True
            baseline = {t["name"]: t for t in before[rank]["tensors"]}
            updated = {t["name"]: t for t in row["tensors"]}
            assert baseline and baseline.keys() == updated.keys()
            assert len(updated) == len(row["tensors"])
            assert all(
                t["pointer"] == baseline[n]["pointer"] for n, t in updated.items()
            )
            changed = {
                n for n, t in updated.items() if t["sha256"] != baseline[n]["sha256"]
            }
            targeted = {n for n in updated if ".layers.0.self_attn." in n}
            assert len(targeted) == 2 and changed == targeted == set(row["changed"])
    return {"mla_derived_weights_verified": True}
