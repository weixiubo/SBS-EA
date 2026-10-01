from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)


def _script_sha256() -> str:
    with open(__file__, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _configure_logging() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    faulthandler.enable(all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _load_model(model_path: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=_dtype(),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    return "vision", processor, None, model


def _load_text_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=_dtype(),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    return "text", None, tokenizer, model


class ModelRuntime:
    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device
        logging.info("runtime_init_start model=%s device=%s", model_path, device)
        try:
            logging.info("runtime_load_processor_start")
            self.mode, self.processor, self.tokenizer, self.model = _load_model(model_path)
            logging.info("runtime_load_processor_done mode=vision")
        except Exception as exc:
            logging.info("runtime_load_vision_failed error=%s", exc)
            self.mode, self.processor, self.tokenizer, self.model = _load_text_model(model_path)
            logging.info("runtime_load_text_done mode=text")
        if torch.cuda.is_available():
            logging.info("runtime_move_to_device_start device=%s", device)
            self.model.to(device)
            logging.info("runtime_move_to_device_done device=%s", device)
        self.model.eval()
        logging.info("runtime_eval_done")

    def generate(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        top_p: float | None,
        seed: int | None,
        max_new_tokens: int,
    ) -> dict:
        messages = []
        if system_prompt:
            if self.mode == "vision":
                messages.append(
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
                )
            else:
                messages.append({"role": "system", "content": system_prompt})
        if self.mode == "vision":
            messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        else:
            messages.append({"role": "user", "content": prompt})
        adapter = self.processor if self.mode == "vision" else self.tokenizer
        assert adapter is not None
        chat_text = adapter.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = adapter(text=[chat_text], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        do_sample = temperature > 0
        generated = self.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=1.0 if top_p is None else float(top_p),
            max_new_tokens=max_new_tokens,
        )
        generated = generated[:, inputs["input_ids"].shape[-1] :]
        text = adapter.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return {"text": text}


@dataclass
class RuntimeState:
    model_path: str
    device: str
    script_sha256: str
    status: str = "loading"
    stage: str = "booting"
    error: str | None = None
    ready_at: float | None = None
    started_at: float = field(default_factory=time.time)
    runtime: ModelRuntime | None = None
    requests_served: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "status": self.status,
                "stage": self.stage,
                "error": self.error,
                "model_path": self.model_path,
                "device": self.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "script_sha256": self.script_sha256,
                "uptime_seconds": round(time.time() - self.started_at, 3),
                "ready_seconds": None
                if self.ready_at is None
                else round(time.time() - self.ready_at, 3),
                "requests_served": self.requests_served,
            }

    def mark_stage(self, stage: str) -> None:
        with self.lock:
            self.stage = stage

    def mark_ready(self, runtime: ModelRuntime) -> None:
        with self.lock:
            self.runtime = runtime
            self.status = "ok"
            self.stage = "ready"
            self.error = None
            self.ready_at = time.time()

    def mark_error(self, message: str) -> None:
        with self.lock:
            self.status = "error"
            self.stage = "error"
            self.error = message

    def acquire_runtime(self) -> tuple[str, ModelRuntime | None, str | None]:
        with self.lock:
            return self.status, self.runtime, self.error

    def note_request(self) -> None:
        with self.lock:
            self.requests_served += 1


def _boot_runtime(state: RuntimeState) -> None:
    try:
        state.mark_stage("loading_model")
        runtime = ModelRuntime(model_path=state.model_path, device=state.device)
        state.mark_ready(runtime)
        logging.info("runtime_ready model=%s device=%s", state.model_path, state.device)
    except Exception as exc:
        message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        state.mark_error(message)
        logging.exception("runtime_boot_failed")


def build_handler(state: RuntimeState):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send_json(404, {"error": "not_found"})
                return
            snapshot = state.snapshot()
            code = 200 if snapshot["status"] == "ok" else 503
            self._send_json(code, snapshot)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/generate":
                self._send_json(404, {"error": "not_found"})
                return
            status, runtime, error = state.acquire_runtime()
            if status != "ok" or runtime is None:
                self._send_json(
                    503,
                    {
                        "error": "model_not_ready",
                        "status": status,
                        "detail": error,
                    },
                )
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            try:
                result = runtime.generate(
                    prompt=payload["prompt"],
                    system_prompt=payload.get("system_prompt"),
                    temperature=float(payload.get("temperature", 0.0)),
                    top_p=None if payload.get("top_p") is None else float(payload.get("top_p")),
                    seed=None if payload.get("seed") is None else int(payload.get("seed")),
                    max_new_tokens=int(payload.get("max_new_tokens", 256)),
                )
                state.note_request()
            except Exception as exc:
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._send_json(200, result)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return Handler


def main() -> None:
    _configure_logging()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    state = RuntimeState(
        model_path=args.model_path,
        device=args.device,
        script_sha256=_script_sha256(),
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(state))
    logging.info("server_bound host=%s port=%s", args.host, args.port)
    boot_thread = threading.Thread(target=_boot_runtime, args=(state,), daemon=True)
    boot_thread.start()
    server.serve_forever()


if __name__ == "__main__":
    main()
