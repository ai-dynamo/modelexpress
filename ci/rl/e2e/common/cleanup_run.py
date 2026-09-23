"""CI cleanup of the exact run prefix, including incomplete publications."""

import re


def cleanup(s3, config):
    run = config["run"]
    key = config["key"]
    if not re.fullmatch(r"[a-z][a-z0-9-]*", key) or not re.fullmatch(
        re.escape(key) + r"-[a-z0-9-]{1,32}", run
    ):
        raise ValueError("Invalid cleanup run ID")
    base = config["delta_prefix"]
    if not base or not base.endswith("/"):
        raise ValueError("Cleanup requires a nonempty delta prefix ending in /")
    prefix = base + run + "/"
    seed = config["seed_prefix"]
    if prefix.startswith(seed) or seed.startswith(prefix):
        raise ValueError("Cleanup prefix overlaps the model snapshot")
    bucket = config["bucket"]
    deleted = 0
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
    return {"prefix": prefix, "deleted": deleted, "verified_absent": True}


if __name__ == "__main__":
    import json

    import boto3
    from botocore.config import Config
    from config import CONFIG

    client = boto3.client(
        "s3",
        region_name=CONFIG["storage"]["region"],
        endpoint_url=CONFIG["storage"]["endpoint_url"],
        config=Config(s3={"addressing_style": CONFIG["storage"]["addressing_style"]}),
    )
    print(json.dumps(cleanup(client, CONFIG)))
