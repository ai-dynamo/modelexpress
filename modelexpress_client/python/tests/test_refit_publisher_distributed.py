# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from modelexpress.refit import (
    PublicationMode,
    Publisher,
    PublisherConfig,
    RevisionRecord,
    RevisionState,
    S3Config,
)
from modelexpress.refit.publisher import PublisherError
from modelexpress.refit.source.canonical import CanonicalTensorSpec
from modelexpress.refit.source.megatron_bridge import (
    MegatronBridgeHfBucketConfig,
    for_each_megatron_hf_bucket,
)


Rewrite = Callable[[object], tuple[object, bool]]


class _DistributedRanks:
    def __init__(self) -> None:
        self.remote_rewrite: Rewrite = lambda value: (value, False)
        self.mutation_count = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(
            torch.distributed,
            "get_rank",
            lambda _group=None: 0,
        )
        monkeypatch.setattr(
            torch.distributed,
            "get_world_size",
            lambda _group=None: 2,
        )
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            self.all_gather_object,
        )
        monkeypatch.setattr(
            torch.distributed,
            "broadcast_object_list",
            lambda _values, src=0, group=None: None,
        )
        monkeypatch.setattr(
            torch.distributed,
            "barrier",
            lambda group=None: None,
        )

    def all_gather_object(
        self,
        output: list[object],
        local: object,
        group: object = None,
    ) -> None:
        del group
        remote, changed = self.remote_rewrite(local)
        if changed:
            self.mutation_count += 1
        output[:] = [local, remote]


class _Catalog:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], RevisionRecord] = {}
        self.published: list[object] = []
        self.get_calls: list[tuple[str, str]] = []

    def publish_revision(self, manifest: Any) -> RevisionRecord:
        self.published.append(manifest)
        record = RevisionRecord(manifest, RevisionState.READY)
        self.records[(manifest.model_id, manifest.target_version)] = record
        return record

    def get_revision(self, model_id: str, target_version: str) -> RevisionRecord:
        self.get_calls.append((model_id, target_version))
        return self.records[(model_id, target_version)]


class _S3:
    def __init__(self) -> None:
        self.put_calls: list[dict[str, object]] = []

    def put_object(self, **kwargs: object) -> dict[str, str]:
        self.put_calls.append(kwargs)
        raise AssertionError("distributed disagreement must precede S3 upload")


class _Bridge:
    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        self.weights = weights
        self.capture_calls = 0

    def get_conversion_tasks(self, _model: object) -> list[object]:
        return [object()]

    def export_hf_weights(self, _model: object, **_kwargs: object):
        self.capture_calls += 1
        for name in reversed(sorted(self.weights)):
            yield name, self.weights[name]


@contextmanager
def _rank_teardown_context(rank: int):
    yield
    if rank == 0:
        raise RuntimeError("rank-zero teardown failed")


def _real_teardown_worker(
    rank: int,
    init_file: str,
    spool_root: str,
    results: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        bridge = _Bridge({"weight": torch.tensor([1.0], dtype=torch.float32)})
        try:
            for_each_megatron_hf_bucket(
                object(),
                MegatronBridgeHfBucketConfig(
                    bridge=bridge,
                    canonical_schema=(
                        CanonicalTensorSpec("weight", (1,), torch.float32),
                    ),
                    bucket_bytes=16,
                    spool_directory=Path(spool_root) / f"rank-{rank}",
                    model_context=partial(_rank_teardown_context, rank),
                ),
                lambda _bucket: None,
            )
        except Exception as exc:
            results.put((rank, type(exc).__name__, str(exc)))
        else:
            results.put((rank, "success", ""))
    finally:
        dist.destroy_process_group()


def _rewrite_tree(
    value: object,
    rewrite_scalar: Callable[[object, str | None], tuple[object, bool]],
    field_name: str | None = None,
) -> tuple[object, bool]:
    rewritten, changed = rewrite_scalar(value, field_name)
    if changed:
        return rewritten, True
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        updates: dict[str, object] = {}
        any_changed = False
        for field in dataclasses.fields(value):
            item, item_changed = _rewrite_tree(
                getattr(value, field.name),
                rewrite_scalar,
                field.name,
            )
            if item_changed:
                updates[field.name] = item
                any_changed = True
        if any_changed:
            return dataclasses.replace(value, **updates), True
        return value, False
    if isinstance(value, dict):
        result: dict[object, object] = {}
        any_changed = False
        for key, item in value.items():
            rewritten_item, item_changed = _rewrite_tree(
                item,
                rewrite_scalar,
                key if isinstance(key, str) else None,
            )
            result[key] = rewritten_item
            any_changed = any_changed or item_changed
        return (result, True) if any_changed else (value, False)
    if isinstance(value, tuple):
        items = []
        any_changed = False
        for item in value:
            rewritten_item, item_changed = _rewrite_tree(item, rewrite_scalar)
            items.append(rewritten_item)
            any_changed = any_changed or item_changed
        return (tuple(items), True) if any_changed else (value, False)
    if isinstance(value, list):
        items = []
        any_changed = False
        for item in value:
            rewritten_item, item_changed = _rewrite_tree(item, rewrite_scalar)
            items.append(rewritten_item)
            any_changed = any_changed or item_changed
        return (items, True) if any_changed else (value, False)
    return value, False


def _different_launch_attestation(value: object) -> tuple[object, bool]:
    def rewrite_digest(
        item: object,
        _field_name: str | None,
    ) -> tuple[object, bool]:
        if isinstance(item, str) and item.startswith("sha256:"):
            return f"sha256:{'f' * 64}", True
        return item, False

    return _rewrite_tree(value, rewrite_digest)


def _different_publish_request(
    remote_version: str,
    remote_base_version: str,
) -> Rewrite:
    def rewrite(value: object) -> tuple[object, bool]:
        def rewrite_version(
            item: object,
            field_name: str | None,
        ) -> tuple[object, bool]:
            if not isinstance(item, str):
                return item, False
            if field_name is not None:
                if "base_version" in field_name and item == "0":
                    return remote_base_version, remote_base_version != item
                if "version" in field_name and item == "1":
                    return remote_version, remote_version != item
            if field_name is None and item == "1":
                return remote_version, remote_version != item
            if field_name is None and item == "0":
                return remote_base_version, remote_base_version != item
            return item, False

        return _rewrite_tree(value, rewrite_version)

    return rewrite


def _checkpoint(tmp_path: Any) -> tuple[Any, dict[str, torch.Tensor]]:
    path = tmp_path / "hf"
    path.mkdir()
    weights = {
        "model.a.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "model.b.weight": torch.ones((2, 2), dtype=torch.float32),
    }
    save_file(weights, path / "model.safetensors")
    return path, weights


def _publisher(
    tmp_path: Any,
    catalog: _Catalog,
    s3: _S3,
    bridge: _Bridge,
) -> Publisher:
    checkpoint, _weights = _checkpoint(tmp_path)
    return Publisher(
        model=object(),
        launch_checkpoint=checkpoint,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )


def _config() -> PublisherConfig:
    return PublisherConfig(
        model_id="model",
        catalog_endpoint="mx:8001",
        s3=S3Config(bucket="bucket", prefix="run"),
        publication_mode=PublicationMode.ASYNC,
    )


def test_launch_attestation_mismatch_prevents_catalog_publication(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranks = _DistributedRanks()
    ranks.remote_rewrite = _different_launch_attestation
    ranks.install(monkeypatch)
    catalog = _Catalog()
    s3 = _S3()
    bridge = _Bridge({})
    publisher = _publisher(tmp_path, catalog, s3, bridge)

    with pytest.raises(PublisherError):
        publisher.initialize(_config())
        publisher.publish_version("0")

    assert ranks.mutation_count > 0
    assert catalog.published == []
    assert s3.put_calls == []
    assert bridge.capture_calls == 0


@pytest.mark.parametrize(
    ("remote_version", "remote_base_version"),
    [("rank-1-target", "0"), ("1", "rank-1-base")],
)
def test_divergent_publish_request_fails_before_capture_or_catalog_side_effects(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    remote_version: str,
    remote_base_version: str,
) -> None:
    ranks = _DistributedRanks()
    ranks.install(monkeypatch)
    catalog = _Catalog()
    s3 = _S3()
    checkpoint, weights = _checkpoint(tmp_path)
    bridge = _Bridge({name: tensor.clone() for name, tensor in weights.items()})
    publisher = Publisher(
        model=object(),
        launch_checkpoint=checkpoint,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(_config())
    publisher.publish_version("0")
    ranks.remote_rewrite = _different_publish_request(
        remote_version,
        remote_base_version,
    )

    with pytest.raises(PublisherError):
        publisher.publish_version("1", base_version="0")

    assert ranks.mutation_count > 0
    assert bridge.capture_calls == 0
    assert s3.put_calls == []
    assert catalog.get_calls == []
    assert [manifest.target_version for manifest in catalog.published] == ["0"]


def test_real_gloo_teardown_failure_reaches_every_rank_without_deadlock(tmp_path):
    context = mp.get_context("spawn")
    results = context.Queue()
    init_file = tmp_path / "gloo-init"
    processes = [
        context.Process(
            target=_real_teardown_worker,
            args=(rank, str(init_file), str(tmp_path / "spool"), results),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5)

    assert alive == []
    assert [process.exitcode for process in processes] == [0, 0]
    observed = sorted(results.get(timeout=5) for _process in processes)
    assert [item[:2] for item in observed] == [
        (0, "CanonicalError"),
        (1, "CanonicalError"),
    ]
    assert all("rank 0: rank-zero teardown failed" in item[2] for item in observed)
