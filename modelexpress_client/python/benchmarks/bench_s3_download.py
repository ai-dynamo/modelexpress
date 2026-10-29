# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare growing/preallocated TransferManager output with parallel buffer writes.

Run against one immutable object using the standard AWS credential chain:
    PYTHONPATH=. python benchmarks/bench_s3_download.py s3://bucket/key
Results describe object download time, not full-model load or refit latency.
"""

import argparse
import hashlib
import json
from contextlib import ExitStack, closing
from io import BytesIO
from time import perf_counter

import boto3
from botocore.config import Config

from modelexpress_rl import envs
from modelexpress_rl.s3 import (
    _content_length,
    _parse_uri,
    _PreallocatedDownloader,
    _required_header,
    _TransferManagerDownloader,
)


class _PreallocatedTransferManager(_TransferManagerDownloader):
    def __init__(self, client):
        super().__init__(client)
        self._client = client

    def get(self, bucket, key):
        metadata = self._client.head_object(Bucket=bucket, Key=key)
        total = _content_length(metadata, bucket, key)
        with BytesIO() as target:
            if total:
                target.seek(total - 1)
                target.write(b"\0")
                target.seek(0)
            self._manager.download(bucket, key, target).result()
            return target.getvalue()


def run(client, uri, *, iterations=3, warmup=1):
    """Rotate three arms over the same object and verify every returned payload."""
    bucket, key = _parse_uri(uri)
    metadata = client.head_object(Bucket=bucket, Key=key)
    etag = _required_header(metadata, "ETag", bucket, key)
    size = _content_length(metadata, bucket, key)
    reference_digest = None
    with ExitStack() as cleanup:
        arms = [
            ("transfer_manager", _TransferManagerDownloader(client)),
            ("transfer_manager_preallocated", _PreallocatedTransferManager(client)),
            ("preallocated", _PreallocatedDownloader(client)),
        ]
        for _, downloader in arms:
            cleanup.callback(downloader.close)
        for iteration in range(-warmup, iterations):
            offset = (iteration + warmup) % len(arms)
            for name, downloader in arms[offset:] + arms[:offset]:
                started = perf_counter()
                data = downloader.get(bucket, key)
                elapsed = perf_counter() - started
                digest = hashlib.sha256(data).hexdigest()
                if reference_digest is None:
                    reference_digest = digest
                if len(data) != size or digest != reference_digest:
                    raise ValueError("benchmark arms returned different payloads")
                del data
                current = client.head_object(Bucket=bucket, Key=key)
                if _required_header(current, "ETag", bucket, key) != etag:
                    raise ValueError("benchmark object changed between downloads")
                if iteration >= 0:
                    print(
                        json.dumps(
                            {
                                "backend": name,
                                "iteration": iteration,
                                "bytes": size,
                                "seconds": elapsed,
                                "decimal_MB_per_s": size / elapsed / 1e6,
                                "sha256": digest,
                            }
                        ),
                        flush=True,
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uri")
    parser.add_argument("--endpoint-url")
    parser.add_argument("--region-name")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be nonnegative")
    with closing(
        boto3.client(
            "s3",
            endpoint_url=args.endpoint_url,
            region_name=args.region_name,
            config=Config(
                max_pool_connections=envs.MX_S3_MAX_POOL_CONNECTIONS,
                retries={
                    "total_max_attempts": envs.MX_S3_MAX_ATTEMPTS,
                    "mode": "standard",
                },
                tcp_keepalive=envs.MX_S3_TCP_KEEPALIVE,
            ),
        )
    ) as client:
        run(client, args.uri, iterations=args.iterations, warmup=args.warmup)


if __name__ == "__main__":
    main()
