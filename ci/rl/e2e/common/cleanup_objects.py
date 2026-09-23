"""Run in the control pod after saving its publication report; delete only this delta."""

import json
import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from config import CONFIG

report = json.loads(Path("/tmp/mx-delta/report.json").read_text())
assert report["run"] == os.environ["DELTA_RUN"]
prefix = CONFIG["delta_prefix"] + report["run"] + "/"
objects = report["objects"]
assert objects and all(x["key"].startswith(prefix) for x in objects)
s3 = boto3.client(
    "s3",
    endpoint_url=CONFIG["storage"]["endpoint_url"],
    region_name=CONFIG["storage"]["region"],
    config=Config(s3={"addressing_style": CONFIG["storage"]["addressing_style"]}),
)
bucket = CONFIG["bucket"]
for item in objects:
    assert (
        s3.head_object(Bucket=bucket, Key=item["key"])["ContentLength"] == item["bytes"]
    )
response = s3.delete_objects(
    Bucket=bucket, Delete={"Objects": [{"Key": x["key"]} for x in objects]}
)
assert not response.get("Errors"), response
for item in objects:
    try:
        s3.head_object(Bucket=bucket, Key=item["key"])
    except ClientError as error:
        assert error.response["Error"]["Code"] in ("404", "NoSuchKey")
    else:
        raise AssertionError("Object remains: " + item["key"])
print(json.dumps({"deleted_objects": objects, "verified_absent": True}))
