# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delete the exact benchmark run prefix, including incomplete publications."""

import re


def cleanup(s3, config):
    run = config["run"]
    key = config["key"]
    if not re.fullmatch(r"[a-z][a-z0-9-]*", key) or not re.fullmatch(
        re.escape(key) + r"-[a-z0-9-]{1,32}", run
    ):
        raise ValueError("Invalid cleanup run ID")
    prefix = config["artifact_prefix"]
    if not prefix.endswith("/" + run + "/") or prefix == run + "/":
        raise ValueError("Cleanup requires an exact run artifact prefix")
    seed = config["seed_prefix"]
    if prefix.startswith(seed) or seed.startswith(prefix):
        raise ValueError("Cleanup prefix overlaps the model snapshot")
    bucket = config["bucket"]
    deleted = 0
    try:
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            objects = [{"Key": row["Key"]} for row in page.get("Contents", [])]
            if any(not row["Key"].startswith(prefix) for row in objects):
                raise ValueError("Object outside exact run prefix")
            if objects:
                response = s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
                if response.get("Errors"):
                    raise RuntimeError(response["Errors"])
                deleted += len(objects)
        if s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1).get("Contents"):
            raise RuntimeError("Run objects remain after cleanup")
    except s3.exceptions.NoSuchBucket:
        pass
    return {"prefix": prefix, "deleted": deleted, "verified_absent": True}


def main():
    import json

    import boto3
    from botocore.config import Config

    from harness.config import load_config

    config = load_config()

    client = boto3.client(
        "s3",
        region_name=config["storage"]["region"],
        endpoint_url=config["storage"]["endpoint_url"],
        config=Config(s3={"addressing_style": config["storage"]["addressing_style"]}),
    )
    print(json.dumps(cleanup(client, config)))


if __name__ == "__main__":
    main()
