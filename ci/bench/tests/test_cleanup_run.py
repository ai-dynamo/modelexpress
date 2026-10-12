# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incomplete publications must clean only their run's objects."""

import pytest
from harness import cleanup as module


class Storage:
    class exceptions:
        class NoSuchBucket(Exception):
            pass

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
        "artifact_prefix": "deltas/nemotron-test/",
        "seed_prefix": "models/",
        "bucket": "ci",
    }


def test_partial_publication_cleanup_preserves_snapshot_and_neighbor_run():
    storage = Storage()
    report = module.cleanup(storage, config())
    assert report["verified_absent"] and report["deleted"] == 1
    assert set(storage.objects) == {"deltas/nemotron-test2/keep", "models/keep"}
    assert module.cleanup(storage, config())["deleted"] == 0


@pytest.mark.parametrize("missing_bucket", [True, False])
def test_cleanup_before_bucket_creation_requires_explicit_absence(missing_bucket):
    class UnavailableStorage(Storage):
        def paginate(self, **kwargs):
            if missing_bucket:
                raise self.exceptions.NoSuchBucket("NoSuchBucket")
            raise RuntimeError("AccessDenied")

    storage = UnavailableStorage()
    if missing_bucket:
        report = module.cleanup(storage, config())
        assert report["verified_absent"] and report["deleted"] == 0
    else:
        with pytest.raises(RuntimeError, match="AccessDenied"):
            module.cleanup(storage, config())


@pytest.mark.parametrize(
    "change", [{"artifact_prefix": ""}, {"run": "../"}, {"seed_prefix": "deltas/"}]
)
def test_unsafe_cleanup_scope_is_rejected(change):
    storage = Storage()
    with pytest.raises(ValueError):
        module.cleanup(storage, {**config(), **change})
    assert len(storage.objects) == 3


def test_partial_delete_failure_can_retry_remaining_objects():
    class PartiallyFailingStorage(Storage):
        def __init__(self):
            super().__init__()
            self.objects["deltas/nemotron-test/remaining"] = 1
            self.fail = True

        def paginate(self, *, Bucket, Prefix):
            yield {
                "Contents": [
                    {"Key": key} for key in self.objects if key.startswith(Prefix)
                ]
            }

        def delete_objects(self, *, Bucket, Delete):
            if self.fail:
                self.fail = False
                del self.objects[Delete["Objects"][0]["Key"]]
                return {
                    "Errors": [
                        {"Key": Delete["Objects"][1]["Key"], "Code": "AccessDenied"}
                    ]
                }
            return super().delete_objects(Bucket=Bucket, Delete=Delete)

    storage = PartiallyFailingStorage()
    with pytest.raises(RuntimeError, match="AccessDenied"):
        module.cleanup(storage, config())
    assert "deltas/nemotron-test/remaining" in storage.objects
    report = module.cleanup(storage, config())
    assert report["deleted"] == 1 and report["verified_absent"]
    assert set(storage.objects) == {"deltas/nemotron-test2/keep", "models/keep"}
