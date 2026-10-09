# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from modelexpress_rl import control, refit_pb2


@pytest.fixture
def redis_command(tmp_path: Path) -> Iterator[Callable[..., str]]:
    server = shutil.which("redis-server")
    client = shutil.which("redis-cli")
    if server is None or client is None:
        pytest.skip("Redis executables are required for atomic publication tests")
    socket = str(tmp_path / "redis.sock")
    process = subprocess.Popen(
        [
            server,
            "--port",
            "0",
            "--unixsocket",
            socket,
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def command(*args: str) -> str:
        return subprocess.check_output(
            [client, "-s", socket, "--raw", *args], text=True
        ).strip()

    try:
        for _ in range(100):
            if Path(socket).exists() and command("PING") == "PONG":
                break
            if process.poll() is not None:
                raise RuntimeError("Redis failed to start")
            time.sleep(0.01)
        else:
            raise RuntimeError("Redis socket did not become ready")
        yield command
    finally:
        process.terminate()
        process.wait(timeout=5)


def _script(
    command: Callable[..., str], name: str, keys: list[str], args: list[str]
) -> str:
    scripts = (
        Path(__file__).resolve().parents[3]
        / "modelexpress_server/src/refit/backend/redis/scripts"
    )
    return command("EVAL", (scripts / name).read_text(), str(len(keys)), *keys, *args)


def _create(command: Callable[..., str], uid: str, *, mesh_id: str = "mesh") -> str:
    return _script(
        command,
        "create_weight_version.lua",
        [f"version:{uid}", f"request:{uid}", "mesh", "mesh:versions"],
        [
            uid,
            "model",
            uid,
            "1",
            "",
            "" if mesh_id else "s3://weights/model",
            "1",
            "1",
            "123",
            mesh_id,
            "",
        ],
    )


def test_creation_captures_generation_without_a_caller_generation(
    redis_command: Callable[..., str],
) -> None:
    redis_command("HSET", "mesh", "model_name", "model", "generation", "1")
    assert _create(redis_command, "first") == "CREATED"
    assert redis_command("HGET", "version:first", "trainer_mesh_generation") == "1"
    redis_command("HSET", "mesh", "generation", "2")
    assert _create(redis_command, "first") == "EXISTING:first"
    assert redis_command("HGET", "version:first", "trainer_mesh_generation") == "1"
    assert _create(redis_command, "second") == "CREATED"
    assert redis_command("HGET", "version:second", "trainer_mesh_generation") == "2"
    assert _create(redis_command, "s3", mesh_id="") == "CREATED"
    assert redis_command("HGET", "version:s3", "trainer_mesh_generation") == "0"


@pytest.mark.parametrize("generation", [None, "0"])
def test_creation_rejects_mesh_without_generation(
    redis_command: Callable[..., str], generation: str | None
) -> None:
    redis_command("HSET", "mesh", "model_name", "model")
    if generation is not None:
        redis_command("HSET", "mesh", "generation", generation)
    assert _create(redis_command, "missing") == "MESH_GENERATION_MISSING"
    assert redis_command("EXISTS", "version:missing", "request:missing") == "0"


@pytest.mark.parametrize("initial_state", ["1", "2"])
def test_publication_rejects_stale_generation_before_any_write(
    redis_command: Callable[..., str], initial_state: str
) -> None:
    worker = {"worker": {"logical_shard_id": "shard", "metadata_endpoint": "endpoint"}}
    redis_command(
        "HSET",
        "mesh",
        "model_name",
        "model",
        "generation",
        "1",
        "workers",
        json.dumps(worker),
        "logical_shards",
        '["shard"]',
    )
    redis_command(
        "HSET",
        "mx:refit:worker:worker",
        "model_name",
        "model",
        "role",
        "1",
        "refit_endpoint",
        "endpoint",
    )
    assert _create(redis_command, "version") == "CREATED"
    redis_command("HSET", "version:version", "state", initial_state)
    keys = ["version:version", "mx:refit:worker:worker", "shards", "mesh", "endpoints"]
    args = [
        "6:workershard",
        "manifest",
        "model",
        "shard",
        "1",
        "2",
        "worker",
        "1",
        "endpoint",
    ]
    redis_command("HSET", "mesh", "generation", "2")
    assert (
        _script(redis_command, "create_weight_version_shard.lua", keys, args)
        == "MESH_GENERATION_MISMATCH"
    )
    assert redis_command("EXISTS", "shards", "endpoints") == "0"
    assert redis_command("HGET", "version:version", "state") == initial_state
    redis_command("HSET", "mesh", "generation", "1")
    assert (
        _script(redis_command, "create_weight_version_shard.lua", keys, args) == "OK:2"
    )
    redis_command("HSET", "mesh", "generation", "2")
    assert (
        _script(redis_command, "create_weight_version_shard.lua", keys, args)
        == "MESH_GENERATION_MISMATCH"
    )
    assert redis_command("HGET", "shards", "6:workershard") == "manifest"


@pytest.mark.parametrize("generation", [0, 1, 2**64 - 1])
def test_mesh_generation_is_decoded_from_response(generation: int) -> None:
    version = refit_pb2.WeightVersion(
        uid="version",
        trainer_mesh_id="mesh",
        trainer_mesh_generation=generation,
        payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_TENSOR,
        state=refit_pb2.WEIGHT_VERSION_STATE_STAGING,
    )
    if generation == 0:
        with pytest.raises(ValueError, match="recreate"):
            control._response_version(
                refit_pb2.CreateWeightVersionResponse(version=version),
                "CreateWeightVersion",
            )
    else:
        assert (
            control._response_version(
                refit_pb2.CreateWeightVersionResponse(version=version),
                "CreateWeightVersion",
            ).trainer_mesh_generation
            == generation
        )


@pytest.mark.parametrize("generation", [True, -1, 2**64])
def test_weight_version_rejects_invalid_generation(generation: int) -> None:
    with pytest.raises(ValueError, match="generation"):
        control.WeightVersion(
            version_id="version",
            model_name="model",
            payload_format=control.WeightPayloadFormat.FULL_TENSOR,
            layout_signature="",
            state=control.WeightVersionState.STAGING,
            created_at_unix_ms=0,
            trainer_mesh_id="mesh",
            trainer_mesh_generation=generation,
        )
