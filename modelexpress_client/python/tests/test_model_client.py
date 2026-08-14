# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ModelExpress model-cache client and its stream validation."""

import pytest

from modelexpress import model_pb2
from modelexpress.model_client import ModelCacheClient, ModelCacheError
from modelexpress.model_snapshot import ModelSnapshotCache, ModelSnapshotError

COMMIT = "c" * 40
MODEL = "org/model"


def chunk(
    relative_path,
    data,
    *,
    offset=0,
    total_size=None,
    is_last_chunk=True,
    is_last_file=False,
    commit_hash=None,
):
    payload = model_pb2.FileChunk(
        relative_path=relative_path,
        data=data,
        offset=offset,
        total_size=len(data) if total_size is None else total_size,
        is_last_chunk=is_last_chunk,
        is_last_file=is_last_file,
    )
    if commit_hash is not None:
        payload.commit_hash = commit_hash
    return payload


def whole_file(relative_path, data, *, is_last_file=False, commit_hash=None):
    return chunk(
        relative_path,
        data,
        is_last_chunk=True,
        is_last_file=is_last_file,
        commit_hash=commit_hash,
    )


class FakeStub:
    """Records requests and replays canned ModelService responses."""

    def __init__(self, *, files=None, chunks=None, updates=None):
        self.files = files or {}
        self.chunks = chunks or []
        self.updates = updates
        self.stream_requests = []
        self.list_requests = []
        self.download_requests = []

    def EnsureModelDownloaded(self, request):
        self.download_requests.append(request)
        updates = self.updates
        if updates is None:
            updates = [
                model_pb2.ModelStatusUpdate(
                    model_name=request.model_name, status=model_pb2.DOWNLOADED
                )
            ]
        return iter(updates)

    def ListModelFiles(self, request):
        self.list_requests.append(request)
        return model_pb2.ModelFileList(
            model_name=request.model_name,
            files=[
                model_pb2.ModelFileInfo(relative_path=path, size=size)
                for path, size in self.files.items()
            ],
            total_size=sum(self.files.values()),
        )

    def StreamModelFiles(self, request):
        self.stream_requests.append(request)
        return iter(self.chunks)


def make_client(tmp_path, stub, **kwargs):
    client = ModelCacheClient(server_url="localhost:1", cache_directory=tmp_path, **kwargs)
    client._stub = stub
    return client


@pytest.fixture(autouse=True)
def no_cache_env(monkeypatch):
    monkeypatch.delenv("MODEL_EXPRESS_CACHE_DIRECTORY", raising=False)


class TestConstruction:
    def test_rejects_zero_chunk_size(self, tmp_path):
        with pytest.raises(ValueError):
            ModelCacheClient(cache_directory=tmp_path, chunk_size=0)

    def test_rejects_zero_max_message_size(self, tmp_path):
        with pytest.raises(ValueError):
            ModelCacheClient(cache_directory=tmp_path, max_message_size=0)


class TestEnsureDownloaded:
    def test_returns_on_downloaded(self, tmp_path):
        stub = FakeStub(
            updates=[
                model_pb2.ModelStatusUpdate(model_name=MODEL, status=model_pb2.DOWNLOADING),
                model_pb2.ModelStatusUpdate(model_name=MODEL, status=model_pb2.DOWNLOADED),
            ]
        )
        make_client(tmp_path, stub).ensure_downloaded(MODEL)

        assert stub.download_requests[0].ignore_weights is False

    def test_raises_on_error_status(self, tmp_path):
        stub = FakeStub(
            updates=[
                model_pb2.ModelStatusUpdate(
                    model_name=MODEL, status=model_pb2.ERROR, message="no disk"
                )
            ]
        )
        with pytest.raises(ModelCacheError, match="no disk"):
            make_client(tmp_path, stub).ensure_downloaded(MODEL)

    def test_raises_when_stream_ends_early(self, tmp_path):
        stub = FakeStub(
            updates=[
                model_pb2.ModelStatusUpdate(model_name=MODEL, status=model_pb2.DOWNLOADING)
            ]
        )
        with pytest.raises(ModelCacheError, match="ended before"):
            make_client(tmp_path, stub).ensure_downloaded(MODEL)


class TestListFiles:
    def test_returns_manifest(self, tmp_path):
        stub = FakeStub(files={"config.json": 2, "model.safetensors": 7})
        assert make_client(tmp_path, stub).list_files(MODEL) == {
            "config.json": 2,
            "model.safetensors": 7,
        }

    def test_rejects_empty_manifest(self, tmp_path):
        stub = FakeStub(files={})
        with pytest.raises(ModelCacheError, match="empty model file manifest"):
            make_client(tmp_path, stub).list_files(MODEL)

    def test_rejects_total_size_mismatch(self, tmp_path, monkeypatch):
        stub = FakeStub(files={"config.json": 2})
        original = stub.ListModelFiles

        def lying_list(request):
            response = original(request)
            response.total_size = 999
            return response

        stub.ListModelFiles = lying_list
        with pytest.raises(ModelCacheError, match="total mismatch"):
            make_client(tmp_path, stub).list_files(MODEL)


class TestInstallMetadataSnapshot:
    def test_requests_only_non_weight_files(self, tmp_path):
        stub = FakeStub(
            files={"config.json": 2, "model.safetensors": 7, "tokenizer.json": 2},
            chunks=[
                whole_file("config.json", b"{}", commit_hash=COMMIT),
                whole_file("tokenizer.json", b"[]", is_last_file=True),
            ],
        )
        snapshot = make_client(tmp_path, stub).install_metadata_snapshot(MODEL)

        assert list(stub.stream_requests[0].file_selector.paths) == [
            "config.json",
            "tokenizer.json",
        ]
        assert (snapshot / "config.json").read_bytes() == b"{}"
        assert not (snapshot / "model.safetensors").exists()
        assert snapshot.name == COMMIT

    def test_writes_main_ref(self, tmp_path):
        stub = FakeStub(
            files={"config.json": 2},
            chunks=[whole_file("config.json", b"{}", is_last_file=True, commit_hash=COMMIT)],
        )
        make_client(tmp_path, stub).install_metadata_snapshot(MODEL)

        cache = ModelSnapshotCache(MODEL, tmp_path)
        assert cache.read_main_ref() == COMMIT

    def test_reuses_existing_snapshot(self, tmp_path):
        stub = FakeStub(
            files={"config.json": 2},
            chunks=[whole_file("config.json", b"{}", is_last_file=True, commit_hash=COMMIT)],
        )
        client = make_client(tmp_path, stub)
        first = client.install_metadata_snapshot(MODEL)
        second = client.install_metadata_snapshot(MODEL)

        assert first == second
        assert len(stub.stream_requests) == 1

    def test_metadata_phase_asks_for_a_metadata_only_download(self, tmp_path):
        """A cold server must not fetch the weights before RdmaStrategy runs.

        The server keys its registry entry on the weight mode, so this claim
        does not satisfy the weight phase's later full-weight request.
        """
        stub = FakeStub(
            files={"config.json": 2},
            chunks=[whole_file("config.json", b"{}", is_last_file=True, commit_hash=COMMIT)],
        )
        make_client(tmp_path, stub).install_metadata_snapshot(MODEL)

        assert [r.ignore_weights for r in stub.download_requests] == [True]
        assert [r.ignore_weights for r in stub.list_requests] == [True]

    def test_rejects_manifest_without_metadata(self, tmp_path):
        stub = FakeStub(files={"model.safetensors": 7})
        with pytest.raises(ModelCacheError, match="no non-weight files"):
            make_client(tmp_path, stub).install_metadata_snapshot(MODEL)

    def test_leaves_no_snapshot_when_stream_fails(self, tmp_path):
        stub = FakeStub(
            files={"config.json": 2, "tokenizer.json": 2},
            chunks=[whole_file("config.json", b"{}", is_last_file=True, commit_hash=COMMIT)],
        )
        with pytest.raises(ModelCacheError, match="missing files"):
            make_client(tmp_path, stub).install_metadata_snapshot(MODEL)

        cache = ModelSnapshotCache(MODEL, tmp_path)
        assert cache.read_main_ref() is None
        leftovers = [
            p.name for p in cache.repo_root.iterdir() if p.name.startswith(".modelexpress-")
        ]
        assert leftovers == []


    def test_installed_snapshot_resolves_offline(self, tmp_path):
        """What the engine actually does with the snapshot, end to end.

        vLLM resolves the model through snapshot_download(local_files_only=True)
        while parsing engine args, well before the weight loader runs.
        """
        from huggingface_hub import snapshot_download

        stub = FakeStub(
            files={"config.json": 2, "model.safetensors": 7},
            chunks=[whole_file("config.json", b"{}", is_last_file=True, commit_hash=COMMIT)],
        )
        snapshot = make_client(tmp_path, stub).install_metadata_snapshot(MODEL)

        resolved = snapshot_download(MODEL, cache_dir=str(tmp_path), local_files_only=True)

        assert resolved == str(snapshot)


class TestInstallWeightFiles:
    def _snapshot(self, tmp_path):
        stub = FakeStub(
            files={"config.json": 2},
            chunks=[whole_file("config.json", b"{}", is_last_file=True, commit_hash=COMMIT)],
        )
        return make_client(tmp_path, stub).install_metadata_snapshot(MODEL)

    def test_requests_only_weight_files(self, tmp_path):
        snapshot = self._snapshot(tmp_path)
        stub = FakeStub(
            files={"config.json": 2, "model.safetensors": 7},
            chunks=[
                whole_file("model.safetensors", b"weights", is_last_file=True, commit_hash=COMMIT)
            ],
        )
        make_client(tmp_path, stub).install_weight_files(MODEL, snapshot)

        assert list(stub.stream_requests[0].file_selector.paths) == ["model.safetensors"]
        assert (snapshot / "model.safetensors").read_bytes() == b"weights"
        assert (snapshot / "config.json").read_bytes() == b"{}"

    def test_skips_when_weights_present(self, tmp_path):
        snapshot = self._snapshot(tmp_path)
        (snapshot / "model.safetensors").write_bytes(b"weights")
        stub = FakeStub(files={"config.json": 2, "model.safetensors": 7})

        make_client(tmp_path, stub).install_weight_files(MODEL, snapshot)

        assert stub.stream_requests == []

    def test_weight_phase_asks_for_the_weights(self, tmp_path):
        snapshot = self._snapshot(tmp_path)
        stub = FakeStub(
            files={"config.json": 2, "model.safetensors": 7},
            chunks=[
                whole_file("model.safetensors", b"weights", is_last_file=True, commit_hash=COMMIT)
            ],
        )
        make_client(tmp_path, stub).install_weight_files(MODEL, snapshot)

        assert [r.ignore_weights for r in stub.download_requests] == [False]

    def test_wrong_commit_aborts_before_transferring(self, tmp_path):
        """Reject on the first chunk, not after the whole checkpoint arrives.

        A sharded model is tens of gigabytes; noticing the mismatch only at the
        end means throwing all of it away.
        """
        snapshot = self._snapshot(tmp_path)
        produced = []

        def counting_stream(request):
            for payload in (
                whole_file("a.safetensors", b"A", commit_hash="d" * 40),
                whole_file("b.safetensors", b"B"),
                whole_file("c.safetensors", b"C", is_last_file=True),
            ):
                produced.append(payload.relative_path)
                yield payload

        stub = FakeStub(
            files={"config.json": 2, "a.safetensors": 1, "b.safetensors": 1, "c.safetensors": 1}
        )
        stub.StreamModelFiles = counting_stream

        with pytest.raises(ModelCacheError, match="refusing to mix revisions"):
            make_client(tmp_path, stub).install_weight_files(MODEL, snapshot)

        assert produced == ["a.safetensors"]
        assert list(snapshot.iterdir()) == [snapshot / "config.json"]

    def test_refuses_weights_from_a_different_commit(self, tmp_path):
        """Pinned revisions are addressed by directory name, so commits must match."""
        snapshot = self._snapshot(tmp_path)
        stub = FakeStub(
            files={"config.json": 2, "model.safetensors": 7},
            chunks=[
                whole_file(
                    "model.safetensors", b"weights", is_last_file=True, commit_hash="d" * 40
                )
            ],
        )
        with pytest.raises(ModelCacheError, match="refusing to mix revisions"):
            make_client(tmp_path, stub).install_weight_files(MODEL, snapshot)

        assert list(snapshot.iterdir()) == [snapshot / "config.json"]

    def test_rejects_manifest_without_weights(self, tmp_path):
        snapshot = self._snapshot(tmp_path)
        stub = FakeStub(files={"config.json": 2})
        with pytest.raises(ModelCacheError, match="no weight files"):
            make_client(tmp_path, stub).install_weight_files(MODEL, snapshot)

    def test_leaves_no_partial_file_when_stream_fails(self, tmp_path):
        snapshot = self._snapshot(tmp_path)
        stub = FakeStub(
            files={"config.json": 2, "model.safetensors": 7},
            chunks=[
                chunk(
                    "model.safetensors",
                    b"weig",
                    total_size=7,
                    is_last_chunk=False,
                    commit_hash=COMMIT,
                )
            ],
        )
        with pytest.raises(ModelCacheError, match="final file marker"):
            make_client(tmp_path, stub).install_weight_files(MODEL, snapshot)

        assert list(snapshot.iterdir()) == [snapshot / "config.json"]

    def test_rolls_back_completed_files_when_a_later_file_fails(self, tmp_path):
        """A half-applied weight set would load as if it were complete."""
        snapshot = self._snapshot(tmp_path)
        stub = FakeStub(
            files={"config.json": 2, "a.safetensors": 1, "b.safetensors": 1},
            chunks=[whole_file("a.safetensors", b"A", commit_hash=COMMIT)],
        )
        with pytest.raises(ModelCacheError, match="final file marker"):
            make_client(tmp_path, stub).install_weight_files(MODEL, snapshot)

        assert list(snapshot.iterdir()) == [snapshot / "config.json"]

    def test_ensures_the_server_has_the_model(self, tmp_path):
        snapshot = self._snapshot(tmp_path)
        stub = FakeStub(
            files={"config.json": 2, "model.safetensors": 7},
            chunks=[
                whole_file("model.safetensors", b"weights", is_last_file=True, commit_hash=COMMIT)
            ],
        )
        make_client(tmp_path, stub).install_weight_files(MODEL, snapshot)

        assert len(stub.download_requests) == 1


class TestStreamValidation:
    """One canned bad stream per protocol rule the client has to enforce."""

    def _install(self, tmp_path, chunks, files=None):
        stub = FakeStub(files=files or {"config.json": 2}, chunks=chunks)
        return make_client(tmp_path, stub).install_metadata_snapshot(MODEL)

    def test_first_chunk_must_carry_commit_hash(self, tmp_path):
        with pytest.raises(ModelCacheError, match="commit hash"):
            self._install(tmp_path, [whole_file("config.json", b"{}", is_last_file=True)])

    def test_commit_hash_may_not_change(self, tmp_path):
        with pytest.raises(ModelCacheError, match="changed the commit hash"):
            self._install(
                tmp_path,
                [
                    whole_file("config.json", b"{}", commit_hash=COMMIT),
                    whole_file(
                        "tokenizer.json", b"[]", is_last_file=True, commit_hash="d" * 40
                    ),
                ],
                files={"config.json": 2, "tokenizer.json": 2},
            )

    def test_rejects_unrequested_file(self, tmp_path):
        with pytest.raises(ModelCacheError, match="unrequested file"):
            self._install(
                tmp_path,
                [whole_file("secret.json", b"{}", is_last_file=True, commit_hash=COMMIT)],
            )

    def test_rejects_weight_file_in_metadata_stream(self, tmp_path):
        with pytest.raises(ModelCacheError, match="unrequested file"):
            self._install(
                tmp_path,
                [
                    whole_file(
                        "model.safetensors", b"weights", is_last_file=True, commit_hash=COMMIT
                    )
                ],
                files={"config.json": 2, "model.safetensors": 7},
            )

    def test_rejects_size_mismatch_against_manifest(self, tmp_path):
        with pytest.raises(ModelCacheError, match="Size mismatch"):
            self._install(
                tmp_path,
                [
                    chunk(
                        "config.json",
                        b"{}",
                        total_size=99,
                        is_last_chunk=False,
                        commit_hash=COMMIT,
                    )
                ],
            )

    def test_rejects_non_zero_first_offset(self, tmp_path):
        with pytest.raises(ModelCacheError, match="offset"):
            self._install(
                tmp_path,
                [
                    chunk(
                        "config.json",
                        b"{}",
                        offset=1,
                        total_size=2,
                        is_last_file=True,
                        commit_hash=COMMIT,
                    )
                ],
            )

    def test_rejects_offset_gap(self, tmp_path):
        with pytest.raises(ModelCacheError, match="Unexpected offset"):
            self._install(
                tmp_path,
                [
                    chunk(
                        "config.json", b"{", total_size=4, is_last_chunk=False, commit_hash=COMMIT
                    ),
                    chunk("config.json", b"}", offset=3, total_size=4, is_last_file=True),
                ],
                files={"config.json": 4},
            )

    def test_rejects_data_beyond_total_size(self, tmp_path):
        with pytest.raises(ModelCacheError, match="exceeds its advertised size"):
            self._install(
                tmp_path,
                [
                    chunk(
                        "config.json",
                        b"{oversized}",
                        total_size=2,
                        is_last_file=True,
                        commit_hash=COMMIT,
                    )
                ],
            )

    def test_rejects_interleaved_files(self, tmp_path):
        with pytest.raises(ModelCacheError, match="before"):
            self._install(
                tmp_path,
                [
                    chunk(
                        "config.json", b"{", total_size=2, is_last_chunk=False, commit_hash=COMMIT
                    ),
                    whole_file("tokenizer.json", b"[]"),
                ],
                files={"config.json": 2, "tokenizer.json": 2},
            )

    def test_rejects_duplicate_file(self, tmp_path):
        with pytest.raises(ModelCacheError, match="twice"):
            self._install(
                tmp_path,
                [
                    whole_file("config.json", b"{}", commit_hash=COMMIT),
                    whole_file("config.json", b"{}", is_last_file=True),
                ],
            )

    def test_rejects_final_file_marker_before_final_chunk(self, tmp_path):
        with pytest.raises(ModelCacheError, match="Final-file marker"):
            self._install(
                tmp_path,
                [
                    chunk(
                        "config.json",
                        b"{",
                        total_size=2,
                        is_last_chunk=False,
                        is_last_file=True,
                        commit_hash=COMMIT,
                    )
                ],
            )

    def test_rejects_data_after_final_marker(self, tmp_path):
        with pytest.raises(ModelCacheError, match="after the final stream marker"):
            self._install(
                tmp_path,
                [
                    whole_file("config.json", b"{}", is_last_file=True, commit_hash=COMMIT),
                    whole_file("tokenizer.json", b"[]"),
                ],
                files={"config.json": 2, "tokenizer.json": 2},
            )

    def test_rejects_missing_final_marker(self, tmp_path):
        with pytest.raises(ModelCacheError, match="final file marker"):
            self._install(
                tmp_path, [whole_file("config.json", b"{}", commit_hash=COMMIT)]
            )

    def test_rejects_empty_stream(self, tmp_path):
        with pytest.raises(ModelCacheError, match="no model files"):
            self._install(tmp_path, [])

    def test_rejects_incomplete_file(self, tmp_path):
        with pytest.raises(ModelCacheError, match="Incomplete file"):
            self._install(
                tmp_path,
                [
                    chunk(
                        "config.json",
                        b"{",
                        total_size=2,
                        is_last_chunk=True,
                        is_last_file=True,
                        commit_hash=COMMIT,
                    )
                ],
            )

    def test_unsafe_path_is_caught_by_the_manifest_check(self, tmp_path):
        with pytest.raises(ModelCacheError, match="unrequested file"):
            self._install(
                tmp_path,
                [whole_file("../escape.json", b"{}", is_last_file=True, commit_hash=COMMIT)],
            )

    def test_unsafe_path_in_the_manifest_never_reaches_the_filesystem(self, tmp_path):
        """Defense in depth: a compromised manifest must not steer the writer."""
        with pytest.raises(ModelSnapshotError, match="Unsafe model file path"):
            self._install(
                tmp_path,
                [whole_file("../escape.json", b"{}", is_last_file=True, commit_hash=COMMIT)],
                files={"../escape.json": 2},
            )
        assert not (tmp_path / "escape.json").exists()

    def test_accepts_multi_chunk_file(self, tmp_path):
        snapshot = self._install(
            tmp_path,
            [
                chunk("config.json", b"{", total_size=4, is_last_chunk=False, commit_hash=COMMIT),
                chunk("config.json", b"a", offset=1, total_size=4, is_last_chunk=False),
                chunk("config.json", b"b}", offset=2, total_size=4, is_last_file=True),
            ],
            files={"config.json": 4},
        )
        assert (snapshot / "config.json").read_bytes() == b"{ab}"

    def test_accepts_empty_file(self, tmp_path):
        snapshot = self._install(
            tmp_path,
            [
                whole_file("config.json", b"{}", commit_hash=COMMIT),
                whole_file(".gitattributes", b"", is_last_file=True),
            ],
            files={"config.json": 2, ".gitattributes": 0},
        )
        assert (snapshot / ".gitattributes").read_bytes() == b""
