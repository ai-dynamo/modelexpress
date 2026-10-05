"""Verify imported source bytes match the frozen candidate."""

import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path

receipt = json.loads(Path("/validation/snapshot.json").read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for key, root in (("candidate_files", "/candidate"), ("mx_files", "/mx-source")):
    for name, expected in receipt[key].items():
        assert digest(Path(root) / name) == expected, name

server_root = Path("/server-build")
server_receipt = json.loads((server_root / "source-receipt.json").read_text())
assert receipt["server_inputs_identical"] is True
assert receipt["server_binary_source_head"] == server_receipt["mx_source_head"]
assert (server_root / "exit-code.txt").read_text().strip() == "0"
assert digest(server_root / "modelexpress-server") == (server_root / "server-binary.sha256").read_text().split()[0]

from prime_rl.utils.mx_compat import require_mx_refit

require_mx_refit(staging_mode="COPY_TO_HOST", streaming=True)
loaded = {}
for name in (
    "prime_rl.utils.mx_compat",
    "prime_rl.configs.shared",
    "prime_rl.transports.weights.mx_refit",
    "prime_rl.inference.vllm.worker.mx_refit",
):
    path = Path(inspect.getfile(importlib.import_module(name))).resolve()
    assert path.is_relative_to("/candidate"), (name, str(path))
    relative = str(path.relative_to("/candidate"))
    assert digest(path) == receipt["candidate_files"][relative], name
    loaded[name] = {"path": str(path), "sha256": digest(path)}

distribution = importlib.metadata.distribution("modelexpress")
installed = {}
for name, expected in receipt["mx_files"].items():
    relative = Path(name).relative_to("modelexpress_client/python")
    if (
        len(relative.parts) < 2
        or not relative.parts[0].startswith("modelexpress")
        or relative.suffix != ".py"
    ):
        continue
    path = Path(distribution.locate_file(str(relative)))
    assert path.is_file() and digest(path) == expected, (name, str(path))
    installed[str(relative)] = {"path": str(path), "sha256": expected}
assert installed
for legacy in (
    "direct_copy.py",
    "direct_glm.py",
    "direct_glm_profile.py",
    "direct_mla.py",
):
    assert not Path(
        distribution.locate_file("modelexpress_rl/inference/engines/vllm") / legacy
    ).exists(), legacy

installer = importlib.import_module("modelexpress_rl.inference.engines.vllm.installer")
installer_text = Path(inspect.getfile(installer)).read_text()
assert "_alias_guard" not in installer_text
assert "_dict_key_kind" not in installer_text

import torch
import vllm

assert vllm.__version__ == "0.30.0", vllm.__version__
assert torch.cuda.device_count() == 1
report = {
    "passed": True,
    "prime_source_head": receipt["prime_base_head"],
    "mx_source_head": receipt["mx_source_head"],
    "pr749_source_head": receipt["pr749_source_head"],
    "server_binary_source_head": receipt["server_binary_source_head"],
    "server_inputs_identical": receipt["server_inputs_identical"],
    "candidate_module_sources": loaded,
    "installed_mx_sources": installed,
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "scope": receipt["scope"],
}
Path("/evidence/source-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "passed": True,
            "candidate_modules": len(loaded),
            "installed_mx_files": len(installed),
        }
    )
)
