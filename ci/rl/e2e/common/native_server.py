import json
import os
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main():
    from config import CONFIG
    from vllm import LLM, SamplingParams

    root = Path("/refit/benchmark")
    root.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    llm = LLM(
        model=os.environ["BENCH_MODEL"],
        revision=os.environ["BENCH_REVISION"],
        load_format="modelexpress",
        model_loader_extra_config={"memory_limit": 8589934592, "concurrency": 8},
        tensor_parallel_size=CONFIG["tp"],
        limit_mm_per_prompt=CONFIG["limit_mm_per_prompt"],
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
        max_model_len=4096,
        max_num_seqs=8,
        worker_extension_cls="regression_worker.RegressionWorker",
    )
    (root / "startup.json").write_text(
        json.dumps(
            {
                "seconds": time.perf_counter() - t,
                "role": os.environ["BENCH_ROLE"],
                "model": os.environ["BENCH_MODEL"],
            }
        )
    )
    print("BENCH_READY", flush=True)
    paused = False

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ready":true}')

        def do_POST(self):
            nonlocal paused
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0"))) or "{}"
            )
            t = time.perf_counter()
            try:
                if self.path == "/pause":
                    paused = True
                    out = {"paused": True}
                elif self.path == "/resume":
                    paused = False
                    out = {"paused": False}
                elif self.path == "/rpc":
                    assert paused or body["method"] in [
                        "hotload_init",
                        "hotload_verify",
                    ], "pause required"
                    out = llm.collective_rpc(
                        body["method"], kwargs=body.get("kwargs", {})
                    )
                elif self.path == "/generate":
                    assert not paused, "paused"
                    results = llm.generate(
                        [body.get("prompt", "The capital of France is")],
                        SamplingParams(temperature=0, max_tokens=32, logprobs=1),
                    )
                    out = [
                        {
                            "text": x.outputs[0].text,
                            "token_ids": list(x.outputs[0].token_ids),
                            "logprob_count": len(x.outputs[0].logprobs or []),
                        }
                        for x in results
                    ]
                else:
                    raise ValueError(self.path)
                data = {"ok": True, "seconds": time.perf_counter() - t, "result": out}
                status = 200
            except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
                data = {
                    "ok": False,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()


if __name__ == "__main__":
    main()
