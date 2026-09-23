import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from config import CONFIG

s = boto3.client(
    "s3",
    endpoint_url=CONFIG["storage"]["endpoint_url"],
    region_name=CONFIG["storage"]["region"],
    config=Config(
        s3={"addressing_style": CONFIG["storage"]["addressing_style"]},
        max_pool_connections=512,
        retries={"max_attempts": 8},
    ),
)
b = CONFIG["bucket"]
prefix = os.environ["BENCH_PREFIX"]
m = json.loads(
    s.get_object(Bucket=b, Key=prefix + "snapshot-manifest.json")["Body"].read()
)
root = Path("/models")
root.mkdir(exist_ok=True)
# Remove only incomplete files from this run's previous boto downloader.
known = {f["name"] for f in m["files"]}
for partial in root.iterdir():
    if (
        partial.is_file()
        and partial.name not in known
        and any(partial.name.startswith(n + ".") for n in known)
    ):
        partial.unlink()
print("CLASSIC_BOOTSTRAP_STARTED", flush=True)


def copy(f):
    p = root / f["name"]
    p.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 3600
    while True:
        try:
            head = s.head_object(Bucket=b, Key=prefix + f["name"])
            f["bytes"] = head["ContentLength"]
            break
        except s.exceptions.ClientError as e:
            if (
                e.response["Error"]["Code"] not in ("404", "NoSuchKey")
                or time.monotonic() > deadline
            ):
                raise
            time.sleep(15)
    if p.exists() and p.stat().st_size == f["bytes"]:
        return
    s.download_file(
        b,
        prefix + f["name"],
        str(p),
        Config=TransferConfig(
            multipart_chunksize=16 * 1024**2,
            max_concurrency=8,
            preferred_transfer_client="classic",
        ),
    )
    assert p.stat().st_size == f["bytes"]
    print("Downloaded", f["name"], flush=True)


t = time.monotonic()
with ThreadPoolExecutor(max_workers=16) as e:
    list(e.map(copy, m["files"]))
print("SEED_READY", time.monotonic() - t, flush=True)
