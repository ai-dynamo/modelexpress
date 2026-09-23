"""Publish synthetic embedding delta for regression test."""

import concurrent.futures
import gc
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import boto3
import numpy as np
import torch
import torch.distributed as dist
from botocore.config import Config
from config import CONFIG
from modelexpress_rl import (
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    ObjectStorageConfig,
    ObjectStorageSource,
    ObjectStorageType,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionState,
)
from modelexpress_rl.utils import compress_delta
from safetensors.torch import save_file

root = Path("/tmp/mx-delta")
root.mkdir(exist_ok=True)
run = os.environ["DELTA_RUN"]
bucket = CONFIG["bucket"]
model = CONFIG["model"]
prefix = CONFIG["delta_prefix"] + run + "/"
uri = f"s3://{bucket}/{prefix}"
name = CONFIG["embedding"]
seed_prefix = CONFIG["seed_prefix"]
s3 = boto3.client(
    "s3",
    endpoint_url=CONFIG["storage"]["endpoint_url"],
    region_name=CONFIG["storage"]["region"],
    config=Config(
        max_pool_connections=32,
        s3={"addressing_style": CONFIG["storage"]["addressing_style"]},
    ),
)
idx = json.loads(
    s3.get_object(Bucket=bucket, Key=seed_prefix + "model.safetensors.index.json")[
        "Body"
    ].read()
)
while True:
    try:
        s3.head_object(Bucket=bucket, Key=seed_prefix + idx["weight_map"][name])
        break
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise
        print("Waiting for layer-zero checkpoint shard", flush=True)
        time.sleep(30)
key = seed_prefix + idx["weight_map"][name]
header_length = struct.unpack(
    "<Q", s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-7")["Body"].read()
)[0]
header = json.loads(
    s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=8-{7 + header_length}")[
        "Body"
    ].read()
)
entry = header[name]
assert entry["dtype"] == "BF16"
begin, end = entry["data_offsets"]
size = end - begin
data_start = 8 + header_length + begin
raw_path = root / "seed.raw"
fd = os.open(raw_path, os.O_CREAT | os.O_RDWR, 0o600)
os.ftruncate(fd, size)


def fetch(offset):
    stop = min(size, offset + 32 * 1024**2)
    body = s3.get_object(
        Bucket=bucket,
        Key=key,
        Range=f"bytes={data_start + offset}-{data_start + stop - 1}",
    )["Body"]
    try:
        data = body.read()
    finally:
        body.close()
    assert len(data) == stop - offset
    assert os.pwrite(fd, data, offset) == len(data)


t = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(fetch, range(0, size, 32 * 1024**2)))
os.close(fd)
print("SEED_DOWNLOAD", time.perf_counter() - t, size, flush=True)
raw = np.memmap(raw_path, dtype=np.uint8, mode="r+")
tensor = torch.from_numpy(raw).view(torch.bfloat16).reshape(entry["shape"])
seed = root / "seed"
seed.mkdir(exist_ok=True)
save_file({name: tensor}, str(seed / "model.safetensors"))
seed_index = {
    "metadata": {"total_size": size},
    "weight_map": {name: "model.safetensors"},
}
(seed / "model.safetensors.index.json").write_text(json.dumps(seed_index))
dist.init_process_group(
    "gloo", init_method="tcp://127.0.0.1:29642", rank=0, world_size=1
)
control = ModelExpressControlClient.connect(server_url="127.0.0.1:8000")
base = run + "-base"
version = run + "-d1"
target = CONFIG["delta_bytes"]
control.create_weight_version(
    uid=base,
    model_name=model,
    idempotency_key=base,
    payload_format=WeightPayloadFormat.FULL_HF_CHECKPOINT,
    state=WeightVersionState.READY,
    object_storage=ObjectStorageSource(
        storage_type=ObjectStorageType.S3,
        uri=f"s3://{bucket}/{seed_prefix}model.safetensors.index.json",
    ),
)
trainer = ModelExpressTrainerClient.initialize(
    ModelExpressTrainerConfig(
        model_name=model,
        server_url="127.0.0.1:8000",
        staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
        payload_format=WeightPayloadFormat.XOR_DELTA,
        process_group=dist.group.WORLD,
        object_storage=ObjectStorageConfig(
            storage_type=ObjectStorageType.S3,
            uri_prefix=uri.rstrip("/"),
            initial_base_version_id=base,
            seed_checkpoint_path=str(seed),
            endpoint_url=CONFIG["storage"]["endpoint_url"],
            region_name=CONFIG["storage"]["region"],
        ),
    )
)
trainer.prepare_delta_base(hf_tensor_iter=[[(name, tensor)]])
sample = np.zeros(32 * 1024**2, dtype=np.uint8)
sample[::2] = np.random.default_rng(15).integers(
    0, 128, len(sample) // 2, dtype=np.uint8
)
ratio = len(compress_delta(sample)) / len(sample)
del sample
active_bytes = min(size, int(target / ratio) // 2 * 2)
xor = np.zeros(size, dtype=np.uint8)
xor[:active_bytes:2] = np.random.default_rng(20260915).integers(
    0, 128, active_bytes // 2, dtype=np.uint8
)
encoded = compress_delta(xor)
predicted = len(encoded)
del encoded
assert abs(predicted - target) / target < 0.03, (predicted, target)
print(
    "CALIBRATED",
    json.dumps(
        {
            "target_bytes": target,
            "predicted_bytes": predicted,
            "active_tensor_bytes": active_bytes,
        }
    ),
    flush=True,
)
np.bitwise_xor(raw, xor, out=raw)
changed_bytes = int(np.count_nonzero(xor))
del xor
gc.collect()
expected = hashlib.sha256(raw).hexdigest()
v = control.create_weight_version(
    uid=version,
    model_name=model,
    idempotency_key=version,
    payload_format=WeightPayloadFormat.XOR_DELTA,
    base_version_id=base,
    object_storage=ObjectStorageSource(
        storage_type=ObjectStorageType.S3, uri=uri + "d1/model.safetensors.index.json"
    ),
)
t = time.perf_counter()
staged = trainer.stage_shard(version=v.ref, hf_tensor_iter=[[(name, tensor)]])
encode_seconds = time.perf_counter() - t
t = time.perf_counter()
staged.publish()
publication_seconds = time.perf_counter() - t
control.update_weight_version_state(version, WeightVersionState.READY)
publisher_metrics = trainer.pop_metrics()
trainer.close()
control.close()
dist.destroy_process_group()
del trainer, staged, tensor, raw
gc.collect()
objects = [
    {"key": x["Key"], "bytes": x["Size"]}
    for x in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
]
payload_bytes = sum(
    x["bytes"]
    for x in objects
    if "/d1/" in x["key"] and x["key"].endswith(".safetensors")
)
assert abs(payload_bytes - target) / target < 0.03, payload_bytes
report = {
    "run": run,
    "model_source": model,
    "model_revision": CONFIG["revision"],
    "tensor": name,
    "tensor_shape": entry["shape"],
    "tensor_bytes": size,
    "target_payload_bytes": target,
    "payload_bytes": payload_bytes,
    "active_tensor_bytes": active_bytes,
    "changed_bytes": changed_bytes,
    "expected_sha256": expected,
    "encode_seconds": encode_seconds,
    "publication_seconds": publication_seconds,
    "publisher_metrics": publisher_metrics,
    "objects": objects,
    "trials": [],
}
(root / "report.json").write_text(json.dumps(report, indent=2))
print(
    "PUBLISHED",
    json.dumps(
        {
            k: report[k]
            for k in ["payload_bytes", "encode_seconds", "publication_seconds"]
        }
    ),
    flush=True,
)
print("DELTA_PUBLICATION_COMPLETE", flush=True)
