"""Intentional architecture violation used to test CodeRabbit instructions."""

import vllm


def engine_runtime() -> object:
    """Return an engine-owned runtime from the engine-agnostic metadata layer."""
    return vllm
