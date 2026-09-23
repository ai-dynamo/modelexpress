"""Resolved per-run profile, mounted with the benchmark code."""

import json
from pathlib import Path

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())
