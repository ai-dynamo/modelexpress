# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patch the Dynamo TRT-LLM backend for ModelExpress P2P support.

Why this script exists (TODO: delete after upstream merges):

  This is a *temporary* shim so users can layer MX support on top of
  the released ``nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime`` image
  (or any other Dynamo-bundled image) without rebuilding Dynamo from
  source. The actual code changes live in
  https://github.com/ai-dynamo/dynamo/pull/8037 — once that PR merges
  and a new tensorrtllm-runtime image is published, this script
  becomes a no-op and this folder can be deleted.

  Why patch instead of just COPY the rebased Dynamo files?
  We tried that in earlier iterations of this image. It breaks because
  the rebased ``llm_worker.py`` from the dynamo PR branch transitively
  imports symbols (e.g., ``register_embedding_cache_metrics``) that
  don't exist in the base image's older Dynamo. Patching only the
  three lines we need (model_express_url plumbing) avoids dragging in
  unrelated upstream churn.

What this script patches:

1. ``backend_args.py``: adds ``--model-express-url`` CLI arg
2. ``engine.py``: when model_express_url is set, sets
   ``engine_args["checkpoint_format"] = "MX"`` so the new
   ``MXCheckpointLoader`` activates
3. ``workers/llm_worker.py``:
   - plumbs ``model_express_url`` through to ``get_llm_engine``
   - adds compat guards for ``exclude_tools_when_tool_choice_none``
     and ``RequestHandlerConfig`` kwargs so the script also works on
     older base images where those fields don't exist

Each ``str.replace()`` is validated; the script exits non-zero on any
unmatched pattern so partial patching can't slip through to the image.
"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.find_spec("dynamo.trtllm")
TRTLLM_PKG = Path(spec.submodule_search_locations[0]) if spec else Path(
    "/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm"
)


def _replace_or_die(src: str, old: str, new: str, *, file: Path, label: str) -> str:
    """Apply str.replace and fail loudly if the pattern was not found.

    Catches Dynamo source drift instead of silently writing a
    partially-patched file.
    """
    if old not in src:
        print(
            f"\n[ERROR] {file}: pattern not found for '{label}'.\n"
            f"  This usually means the upstream Dynamo source has changed.\n"
            f"  Expected pattern (first 100 chars): {old[:100]!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    return src.replace(old, new, 1)


def patch_backend_args():
    p = TRTLLM_PKG / "backend_args.py"
    src = p.read_text()
    if "model_express_url" in src:
        print("backend_args.py: already patched")
        return

    src = _replace_or_die(
        src,
        '        add_argument(\n            g,\n            flag_name="--disaggregation-mode"',
        '        add_argument(\n'
        '            g,\n'
        '            flag_name="--model-express-url",\n'
        '            env_var="MODEL_EXPRESS_URL",\n'
        '            default=None,\n'
        '            help="ModelExpress P2P server URL (e.g., modelexpress-server:8001). "\n'
        '            "When set, auto-detects source/target role based on existing sources.",\n'
        '        )\n'
        '        add_argument(\n            g,\n            flag_name="--disaggregation-mode"',
        file=p, label="--model-express-url CLI arg",
    )

    src = _replace_or_die(
        src,
        "    disaggregation_mode: DisaggregationMode",
        "    model_express_url: Optional[str] = None\n\n"
        "    disaggregation_mode: DisaggregationMode",
        file=p, label="model_express_url config field",
    )

    p.write_text(src)
    print("backend_args.py: patched (added --model-express-url)")


def patch_engine():
    p = TRTLLM_PKG / "engine.py"
    src = p.read_text()
    if "model_express_url" in src:
        print("engine.py: already patched")
        return

    if "import os" not in src:
        src = _replace_or_die(
            src, "import enum", "import enum\nimport os",
            file=p, label="import os",
        )

    src = _replace_or_die(
        src,
        "        disaggregation_mode: Optional[DisaggregationMode] = None,\n"
        "    ) -> None:",
        "        disaggregation_mode: Optional[DisaggregationMode] = None,\n"
        "        model_express_url: Optional[str] = None,\n"
        "    ) -> None:",
        file=p, label="TensorRTLLMEngine.__init__ signature",
    )

    src = _replace_or_die(
        src,
        "        self.engine_args = engine_args\n",
        "        self.engine_args = engine_args\n"
        "        self._model_express_url = model_express_url\n",
        file=p, label="store _model_express_url on engine",
    )

    src = _replace_or_die(
        src,
        "    async def initialize(self) -> None:\n"
        "        if not self._llm:\n",
        "    async def initialize(self) -> None:\n"
        "        if not self._llm:\n"
        "            if self._model_express_url:\n"
        '                self.engine_args["checkpoint_format"] = "MX"\n'
        '                os.environ.setdefault("MODEL_EXPRESS_URL",\n'
        "                                      self._model_express_url)\n"
        "                logger.info(\n"
        '                    "ModelExpress P2P enabled: checkpoint_format=MX, server=%s",\n'
        "                    self._model_express_url,\n"
        "                )\n\n",
        file=p, label="checkpoint_format=MX in initialize()",
    )

    src = _replace_or_die(
        src,
        "    disaggregation_mode: Optional[DisaggregationMode] = None,\n"
        "    component_gauges: Any = None,\n"
        ") -> AsyncGenerator[TensorRTLLMEngine, None]:",
        "    disaggregation_mode: Optional[DisaggregationMode] = None,\n"
        "    component_gauges: Any = None,\n"
        "    model_express_url: Optional[str] = None,\n"
        ") -> AsyncGenerator[TensorRTLLMEngine, None]:",
        file=p, label="get_llm_engine signature",
    )

    src = _replace_or_die(
        src,
        "    engine = TensorRTLLMEngine(engine_args, disaggregation_mode)",
        "    engine = TensorRTLLMEngine(engine_args, disaggregation_mode, model_express_url)",
        file=p, label="get_llm_engine constructor call",
    )

    p.write_text(src)
    print("engine.py: patched (added model_express_url)")


def patch_llm_worker():
    p = TRTLLM_PKG / "workers" / "llm_worker.py"
    src = p.read_text()
    if "model_express_url" in src:
        print("llm_worker.py: already patched")
        return

    src = _replace_or_die(
        src,
        "        component_gauges=component_gauges,\n"
        "    ) as engine:",
        "        component_gauges=component_gauges,\n"
        "        model_express_url=config.model_express_url,\n"
        "    ) as engine:",
        file=p, label="model_express_url in get_llm_engine call",
    )

    if "hasattr(runtime_config" not in src and "exclude_tools_when_tool_choice_none" in src:
        src = _replace_or_die(
            src,
            "        runtime_config.exclude_tools_when_tool_choice_none = (\n"
            "            config.exclude_tools_when_tool_choice_none\n"
            "        )",
            '        if hasattr(runtime_config, "exclude_tools_when_tool_choice_none"):\n'
            "            runtime_config.exclude_tools_when_tool_choice_none = getattr(\n"
            '                config, "exclude_tools_when_tool_choice_none", True\n'
            "            )",
            file=p, label="exclude_tools_when_tool_choice_none compat guard",
        )

    if "inspect.signature(RequestHandlerConfig" not in src:
        lines = src.split('\n')
        new_lines = []
        in_handler_call = False
        paren_depth = 0
        for line in lines:
            if not in_handler_call:
                if "handler_config = RequestHandlerConfig(" in line:
                    line = line.replace(
                        "handler_config = RequestHandlerConfig(",
                        "handler_kwargs = dict(")
                    in_handler_call = True
                    paren_depth = line.count('(') - line.count(')')
                new_lines.append(line)
            else:
                new_lines.append(line)
                paren_depth += line.count('(') - line.count(')')
                if paren_depth <= 0:
                    in_handler_call = False
                    indent = "        "
                    new_lines.append(f"{indent}import inspect")
                    new_lines.append(f"{indent}valid_params = set(inspect.signature(RequestHandlerConfig.__init__).parameters.keys())")
                    new_lines.append(f"{indent}handler_kwargs = {{k: v for k, v in handler_kwargs.items() if k in valid_params}}")
                    new_lines.append(f"{indent}handler_config = RequestHandlerConfig(**handler_kwargs)")
        src = '\n'.join(new_lines)

    p.write_text(src)
    print("llm_worker.py: patched (model_express_url + compat guards)")


if __name__ == "__main__":
    patch_backend_args()
    patch_engine()
    patch_llm_worker()
    print("All Dynamo MX patches applied.")
