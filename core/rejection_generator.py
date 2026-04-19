"""
core/rejection_generator.py
Subprocess-isolated rejected response generator for auto SFT→ORPO/DPO conversion.
Runs mlx_lm (Apple Silicon) or llama-cli (other) in a long-lived subprocess
to avoid Metal command buffer conflicts with the training process.
"""

import json
import os
import subprocess
import sys
import threading
from typing import Optional, Callable

LogFn = Callable[[str], None]

# MLX worker script — runs in subprocess, loads model once, processes prompts via stdin/stdout
# Model stays resident in memory. Only reloads on explicit "reload" command (checkpoint refresh).
# Memory is explicitly freed before reload via del + mx.clear_cache().
_MLX_WORKER_SCRIPT = r'''
import sys, json, os, gc

model_path = sys.argv[1]
adapter_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "__none__" else None
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 512
temp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

def load_model(path, ap=None):
    """Load model + optional adapters, return (model, tokenizer)."""
    m, tok = load(path)
    if ap and os.path.isdir(ap):
        try:
            from mlx_lm.utils import load_adapters
            m = load_adapters(m, ap)
            mx.eval(m.parameters())
        except Exception:
            pass
    return m, tok

def free_model(m, tok):
    """Explicitly free model memory and clear Metal cache."""
    del m
    del tok
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

model, tokenizer = load_model(model_path, adapter_path)
sampler = make_sampler(temp=temp)
print(json.dumps({"status": "ready"}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        print(json.dumps({"error": "invalid json"}), flush=True)
        continue

    cmd = data.get("cmd", "generate")

    if cmd == "reload":
        ap = data.get("adapter_path")
        try:
            # Free old model before loading new one
            free_model(model, tokenizer)
            model, tokenizer = load_model(model_path, ap if ap else None)
            sampler = make_sampler(temp=temp)
            print(json.dumps({"status": "reloaded"}), flush=True)
        except Exception as e:
            print(json.dumps({"status": "reload_error", "error": str(e)}), flush=True)
        continue

    if cmd == "generate":
        prompt = data.get("prompt", "")
        mt = data.get("max_tokens", max_tokens)
        try:
            resp = generate(model, tokenizer, prompt=prompt,
                           max_tokens=mt, verbose=False, sampler=sampler)
            # Clear inference cache after each generation to prevent memory creep
            try:
                mx.clear_cache()
            except Exception:
                pass
            print(json.dumps({"response": resp}), flush=True)
        except Exception as e:
            print(json.dumps({"response": "", "error": str(e)}), flush=True)
        continue

    if cmd == "quit":
        free_model(model, tokenizer)
        break

    print(json.dumps({"error": f"unknown cmd: {cmd}"}), flush=True)
'''


class RejectionGenerator:
    """Generate rejected responses via subprocess-isolated inference."""

    def __init__(
        self,
        model_path: str,
        backend: str = "mlx",
        max_tokens: int = 512,
        temperature: float = 0.7,
        adapter_path: Optional[str] = None,
        log_fn: Optional[LogFn] = None,
    ):
        self.model_path = model_path
        self.backend = backend  # "mlx" | "gguf"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.adapter_path = adapter_path
        self._log_fn = log_fn
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _log(self, msg: str):
        if self._log_fn:
            self._log_fn(msg)

    def start(self):
        """Start the inference subprocess."""
        if self.backend == "mlx":
            self._start_mlx()
        else:
            self._log("GGUF backend for rejection generation not yet implemented, using MLX")
            self._start_mlx()

    def _start_mlx(self):
        ap = self.adapter_path or "__none__"
        self._process = subprocess.Popen(
            [sys.executable, "-c", _MLX_WORKER_SCRIPT,
             self.model_path, ap,
             str(self.max_tokens), str(self.temperature)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Wait for ready signal
        line = self._process.stdout.readline()
        try:
            status = json.loads(line)
            if status.get("status") == "ready":
                self._log(f"Rejection generator ready (model: {os.path.basename(self.model_path)})")
            else:
                self._log(f"Rejection generator warning: {line.strip()}")
        except Exception:
            self._log(f"Rejection generator startup output: {line.strip()}")

    def generate(self, prompt: str) -> str:
        """Send a prompt, get a rejected response. Thread-safe, blocking."""
        with self._lock:
            if not self._process or self._process.poll() is not None:
                raise RuntimeError("Rejection generator subprocess not running")
            try:
                self._process.stdin.write(
                    json.dumps({"cmd": "generate", "prompt": prompt,
                                "max_tokens": self.max_tokens}) + "\n"
                )
                self._process.stdin.flush()
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError("Subprocess closed stdout")
                data = json.loads(line)
                if data.get("error"):
                    self._log(f"Generation error: {data['error']}")
                return data.get("response", "")
            except Exception as e:
                raise RuntimeError(f"Rejection generation failed: {e}")

    def reload_model(self, adapter_path: Optional[str] = None):
        """Reload model with updated adapter weights."""
        with self._lock:
            if not self._process or self._process.poll() is not None:
                return
            self._log(f"Reloading rejection model" +
                      (f" with adapters: {adapter_path}" if adapter_path else ""))
            self._process.stdin.write(
                json.dumps({"cmd": "reload", "adapter_path": adapter_path or ""}) + "\n"
            )
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            try:
                data = json.loads(line)
                if data.get("status") == "reloaded":
                    self._log("Rejection model reloaded successfully")
                else:
                    self._log(f"Reload warning: {data}")
            except Exception:
                self._log(f"Reload response: {line.strip()}")

    def stop(self):
        """Terminate the subprocess."""
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._process.stdin.flush()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
            self._log("Rejection generator stopped")

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None
