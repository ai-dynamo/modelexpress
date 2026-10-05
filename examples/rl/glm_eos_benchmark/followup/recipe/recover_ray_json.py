"""Recover one bounded Ray forwarding split without editing JSON payload bytes."""

import hashlib
import json
import re


class RecoveryError(ValueError):
    """A record cannot be recovered with an unambiguous producer identity."""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


DECODER = json.JSONDecoder(object_pairs_hook=unique_object)
ANSI = r"(?:\x1b\[[0-9;]*m)*"
PRODUCER = re.compile(
    r"^(?P<prefix>\(EngineCore pid=(?P<engine>\d+)\) "
    + ANSI
    + r"\(RayWorkerProc pid=(?P<pid>\d+)(?:, ip=(?P<ip>[0-9a-fA-F:.]+))?\)"
    + ANSI
    + r" )"
)
WORKER = re.compile(r"\(Worker_TP(?P<rank>\d+)_EP\d+ pid=(?P<pid>\d+)\) ")
MARKER = '{"record"'
MAX_RECORD_BYTES = 1_000_000


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def recover_log(raw, captured_records=None):
    """Return derived bytes and audit; only consecutive same-producer pairs qualify.

    The only removed bytes are a single newline and the exact repeated forwarding
    prefix. Values are never parsed and reserialized into the output transcript.
    Missing/truncated/ambiguous records fail closed. No joins across other output.
    """
    text = raw.decode("utf-8", errors="strict")
    lines = text.splitlines(keepends=True)
    output, repairs = [], []
    input_index = 0
    valid_records = 0
    while input_index < len(lines):
        line = lines[input_index]
        start = line.find(MARKER)
        repaired = False
        while start >= 0:
            try:
                _, end = DECODER.raw_decode(line[start:])
            except RecoveryError:
                raise
            except ValueError as error:
                producer = PRODUCER.match(line)
                if producer is None or line.count(MARKER) != 1:
                    raise RecoveryError("Malformed record without one Ray producer") from error
                worker = WORKER.fullmatch(line[producer.end():start])
                if worker is None and line[producer.end():start]:
                    raise RecoveryError("Missing or ambiguous worker identity") from error
                if worker is not None and worker["pid"] != producer["pid"]:
                    raise RecoveryError("Missing or ambiguous worker identity") from error
                if not line.endswith("\n") or line.endswith("\r\n"):
                    raise RecoveryError("Unsupported or truncated record boundary") from error
                if input_index + 1 >= len(lines):
                    raise RecoveryError("Missing continuation line") from error
                following = lines[input_index + 1]
                next_producer = PRODUCER.match(following)
                if next_producer is None or next_producer["prefix"] != producer["prefix"]:
                    raise RecoveryError("Continuation belongs to a different producer") from error
                suffix = following[next_producer.end():]
                if MARKER in suffix or WORKER.match(suffix):
                    raise RecoveryError("Continuation contains a new record or worker message") from error
                payload = line[start:-1] + suffix
                if len(payload.encode()) > MAX_RECORD_BYTES:
                    raise RecoveryError("Continuation exceeds bounded record size") from error
                try:
                    record, end = DECODER.raw_decode(payload)
                except RecoveryError:
                    raise
                except ValueError as joined_error:
                    raise RecoveryError("Two fragments do not form one complete JSON record") from joined_error
                if payload[end:].strip():
                    raise RecoveryError("Ambiguous content after reconstructed record")
                captured_identity = None
                if worker is None:
                    matches = (captured_records or {}).get(sha(payload[:end].encode()), [])
                    matches = [entry for entry in matches if entry["pid"] == int(producer["pid"])]
                    if len(matches) != 1:
                        raise RecoveryError("Missing or ambiguous captured worker identity")
                    captured_identity = matches[0]
                    worker = {"rank": captured_identity["rank"], "pid": captured_identity["pid"]}
                if not (
                    record.get("record") == "mx-refit-phases-v1"
                    and record.get("role") == "generator"
                    and type(record.get("marks", {}).get("rank")) is int
                    and record["marks"]["rank"] == int(worker["rank"])
                ):
                    raise RecoveryError("Recovered phase identity differs from producer worker")
                derived = line[:-1] + suffix
                repairs.append({
                    "input_start_line": input_index + 1,
                    "input_end_line": input_index + 2,
                    "output_line": len(output) + 1,
                    "engine_pid": int(producer["engine"]),
                    "ray_worker_pid": int(producer["pid"]),
                    "ray_worker_ip": producer["ip"],
                    "rank": record["marks"]["rank"],
                    "step": record.get("step"),
                    "version_uid": record.get("version_uid"),
                    "source_fragment_sha256": sha((line + following).encode()),
                    "reconstructed_json_sha256": sha(payload[:end].encode()),
                    "removed_forwarding_prefix": producer["prefix"],
                    "removed_bytes": len(("\n" + producer["prefix"]).encode()),
                    "operation": "Concatenate exact payload fragments; remove only newline and repeated same-producer prefix",
                    "captured_worker_identity": captured_identity,
                })
                output.append(derived)
                valid_records += 1
                input_index += 2
                repaired = True
                break
            valid_records += 1
            start = line.find(MARKER, start + end)
        if not repaired:
            output.append(line)
            input_index += 1
    derived = "".join(output).encode()
    assert len(raw) - len(derived) == sum(r["removed_bytes"] for r in repairs)
    return derived, {
        "schema": "ray-json-continuation-recovery-v1",
        "original_bytes": len(raw), "original_sha256": sha(raw),
        "derived_bytes": len(derived), "derived_sha256": sha(derived),
        "records_parsed": valid_records, "repair_count": len(repairs),
        "repairs": repairs,
        "policy": "Only two consecutive fragments, exact EngineCore/Ray PID/IP prefix, matching TP worker rank; no value edits or relaxed record-completeness rules",
    }


def load_captured_records(capture_root):
    """Index intact worker payloads after the independent capture verifier passes.

    A missing forwarded Worker_TP prefix is recoverable only when the exact JSON
    bytes occur once in a checksummed capture carrying its PID and rank prefix.
    Duplicate matches remain ambiguous, even if their parsed values agree.
    """
    records = {}
    for manifest in sorted(capture_root.glob("*/capture.json")):
        capture = json.loads(manifest.read_text())
        for item in capture["files"]:
            name = item["name"]
            if not (name.startswith("worker-") and name.endswith(".out")):
                continue
            path = manifest.parent / item["relative_path"]
            if item["relative_path"] != "files/" + name or path.name != name or "/" in name:
                raise RecoveryError("Invalid captured worker path")
            if path.is_symlink():
                raise RecoveryError("Captured worker source must be a regular file")
            raw = path.read_bytes()
            if sha(raw) != item["sha256"] or len(raw) != item["bytes_copied"]:
                raise RecoveryError("Captured worker checksum or size differs")
            for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                worker = WORKER.match(line)
                if worker is None:
                    continue
                payload = line[worker.end():]
                if not payload.startswith(MARKER):
                    continue
                record, end = DECODER.raw_decode(payload)
                if record.get("record") != "mx-refit-phases-v1" or record.get("role") != "generator":
                    continue
                if payload[end:].strip():
                    raise RecoveryError("Ambiguous trailing captured worker content")
                rank = record.get("marks", {}).get("rank")
                if type(rank) is not int or rank != int(worker["rank"]):
                    raise RecoveryError("Captured phase rank differs from worker")
                if not name.endswith("-" + worker["pid"] + ".out"):
                    raise RecoveryError("Captured worker PID differs from filename")
                if record.get("hostname", "").split(".")[0] != capture["hostname"].split(".")[0]:
                    raise RecoveryError("Captured worker host differs from manifest")
                records.setdefault(sha(payload[:end].encode()), []).append({
                    "pid": int(worker["pid"]), "rank": rank,
                    "source_path": str(path), "source_sha256": sha(raw),
                    "source_line": number,
                    "manifest_sha256": sha(manifest.read_bytes()),
                })
    return records
