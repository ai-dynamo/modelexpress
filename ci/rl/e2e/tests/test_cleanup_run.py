"""Incomplete publications must clean only their run's objects."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "cleanup_run", Path(__file__).resolve().parents[1] / "common/cleanup_run.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Storage:
    def __init__(self):
        self.objects = {
            "deltas/nemotron-test/partial": 1,
            "deltas/nemotron-test2/keep": 1,
            "models/keep": 1,
        }

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):
        assert Bucket == "ci"
        for key in list(self.objects):
            if key.startswith(Prefix):
                yield {"Contents": [{"Key": key}]}

    def delete_objects(self, *, Bucket, Delete):
        assert Bucket == "ci"
        for row in Delete["Objects"]:
            del self.objects[row["Key"]]
        return {}

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
        return {
            "Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)][
                :MaxKeys
            ]
        }


def config():
    return {
        "key": "nemotron",
        "run": "nemotron-test",
        "delta_prefix": "deltas/",
        "seed_prefix": "models/",
        "bucket": "ci",
    }


def test_partial_publication_cleanup_preserves_snapshot_and_neighbor_run():
    storage = Storage()
    report = module.cleanup(storage, config())
    assert report["verified_absent"] and report["deleted"] == 1
    assert set(storage.objects) == {"deltas/nemotron-test2/keep", "models/keep"}
    assert module.cleanup(storage, config())["deleted"] == 0


@pytest.mark.parametrize(
    "change", [{"delta_prefix": ""}, {"run": "../"}, {"seed_prefix": "deltas/"}]
)
def test_unsafe_cleanup_scope_is_rejected(change):
    storage = Storage()
    with pytest.raises(ValueError):
        module.cleanup(storage, {**config(), **change})
    assert len(storage.objects) == 3
