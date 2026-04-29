# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patch Dynamo TRT-LLM backend for ModelExpress P2P support.

Applies minimal patches to add --model-express-url CLI arg and
checkpoint_format="MX" engine integration. Compatible with any
Dynamo version that has the TRT-LLM backend.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("dynamo.trtllm")
TRTLLM_PKG = Path(spec.submodule_search_locations[0]) if spec else Path(
    "/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm"
)


def patch_backend_args():
    p = TRTLLM_PKG / "backend_args.py"
    src = p.read_text()
    if "model_express_url" in src:
        print("backend_args.py: already patched")
        return

    src = src.replace(
        '        add_argument(\n            g,\n            flag_name="--disaggregation-mode"',
        '        add_argument(\n'
        '            g,\n'
        '            flag_name="--model-express-url",\n'
        '            env_var="MODEL_EXPRESS_URL",\n'
        '            default=None,\n'
        '            help="ModelExpress P2P server URL (e.g., modelexpress-server:8001). "\n'
        '            "When set, auto-detects source/target role based on existing sources.",\n'
        '        )\n'
        '        add_argument(\n            g,\n            flag_name="--disaggregation-mode"'
    )

    src = src.replace(
        "    disaggregation_mode: DisaggregationMode",
        "    model_express_url: Optional[str] = None\n\n"
        "    disaggregation_mode: DisaggregationMode"
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
        src = src.replace("import enum", "import enum\nimport os")

    src = src.replace(
        "        disaggregation_mode: Optional[DisaggregationMode] = None,\n"
        "    ) -> None:",
        "        disaggregation_mode: Optional[DisaggregationMode] = None,\n"
        "        model_express_url: Optional[str] = None,\n"
        "    ) -> None:"
    )

    src = src.replace(
        "        self.engine_args = engine_args\n",
        "        self.engine_args = engine_args\n"
        "        self._model_express_url = model_express_url\n"
    )

    src = src.replace(
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
        "                )\n\n"
    )

    src = src.replace(
        "    disaggregation_mode: Optional[DisaggregationMode] = None,\n"
        "    component_gauges: Any = None,\n"
        ") -> AsyncGenerator[TensorRTLLMEngine, None]:",
        "    disaggregation_mode: Optional[DisaggregationMode] = None,\n"
        "    component_gauges: Any = None,\n"
        "    model_express_url: Optional[str] = None,\n"
        ") -> AsyncGenerator[TensorRTLLMEngine, None]:"
    )

    src = src.replace(
        "    engine = TensorRTLLMEngine(engine_args, disaggregation_mode)",
        "    engine = TensorRTLLMEngine(engine_args, disaggregation_mode, model_express_url)"
    )

    p.write_text(src)
    print("engine.py: patched (added model_express_url)")


def patch_llm_worker():
    p = TRTLLM_PKG / "workers" / "llm_worker.py"
    src = p.read_text()
    if "model_express_url" in src:
        print("llm_worker.py: already patched")
        return

    if "model_express_url=config.model_express_url" not in src:
        src = src.replace(
            "        component_gauges=component_gauges,\n"
            "    ) as engine:",
            "        component_gauges=component_gauges,\n"
            "        model_express_url=config.model_express_url,\n"
            "    ) as engine:"
        )

    if "hasattr(runtime_config" not in src and "exclude_tools_when_tool_choice_none" in src:
        src = src.replace(
            "        runtime_config.exclude_tools_when_tool_choice_none = (\n"
            "            config.exclude_tools_when_tool_choice_none\n"
            "        )",
            '        if hasattr(runtime_config, "exclude_tools_when_tool_choice_none"):\n'
            "            runtime_config.exclude_tools_when_tool_choice_none = getattr(\n"
            '                config, "exclude_tools_when_tool_choice_none", True\n'
            "            )"
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
