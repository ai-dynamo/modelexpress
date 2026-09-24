"""Default model hook; shared all-rank tensor/refit checks still apply."""


def validate(config, result):
    assert config["expected_host_scales_per_rank"] is None, (
        "Missing model-specific checks"
    )
    assert not config.get("derived_weight_check"), "Missing derived-weight checks"
    return {}
