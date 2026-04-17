"""
core/exporter.py
模型导出模块：支持 LoRA 适配器、合并完整模型、GGUF 量化、HuggingFace Hub 推送。
"""

import json
import os
import shutil
import subprocess
import sys
from typing import Optional, Callable


LogFn = Callable[[str], None]


def _log(fn: Optional[LogFn], msg: str) -> None:
    if fn:
        fn(msg)
    else:
        print(msg)


def _ensure_adapter_config(adapters_dir: str, training_config: dict, log_fn: Optional[LogFn] = None) -> None:
    """Always regenerate adapter_config.json in adapters_dir (mlx_lm.fuse format)."""
    cfg_path = os.path.join(adapters_dir, "adapter_config.json")
    lora_r = training_config.get("lora_r", 16)
    lora_alpha = training_config.get("lora_alpha", 32)
    # Get actual num_hidden_layers from model config.json
    num_layers = 32  # safe default
    model_id = training_config.get("model_id", "")
    if model_id:
        model_cfg_path = os.path.join(model_id, "config.json")
        if os.path.isfile(model_cfg_path):
            try:
                with open(model_cfg_path, "r", encoding="utf-8") as f:
                    mcfg = json.load(f)
                num_layers = mcfg.get("num_hidden_layers", num_layers)
            except Exception:
                pass
    cfg = {
        "fine_tune_type": "lora",
        "base_model_name_or_path": model_id,
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": lora_r,
            "alpha": lora_alpha,
            "dropout": training_config.get("lora_dropout", 0.0),
            "scale": lora_alpha / lora_r if lora_r else 1.0,
        },
        "target_modules": training_config.get("target_modules", []),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    _log(log_fn, f"Generated adapter_config.json (num_layers={num_layers}) at {cfg_path}")


def _ensure_adapters_safetensors(ckpt_path: str, adapters_dir: str, log_fn: Optional[LogFn] = None) -> None:
    """mlx_lm.fuse expects 'adapters.safetensors' — copy the selected checkpoint file if needed."""
    dst = os.path.join(adapters_dir, "adapters.safetensors")
    if not os.path.isfile(dst) or os.path.abspath(ckpt_path) != os.path.abspath(dst):
        shutil.copy2(ckpt_path, dst)
        _log(log_fn, f"Copied {os.path.basename(ckpt_path)} → adapters.safetensors")


def _run_streaming(cmd: list, log_fn: Optional[LogFn]) -> None:
    """Run a subprocess and stream stdout+stderr line by line to log_fn."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            _log(log_fn, line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")




def save_lora_adapter_mlx(
    ckpt_path: str,
    adapters_dir: str,
    output_dir: str,
    training_config: dict,
    log_fn: Optional[LogFn] = None,
) -> str:
    """Copy MLX adapter weights + generate adapter_config.json to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    _ensure_adapter_config(adapters_dir, training_config, log_fn)
    _ensure_adapters_safetensors(ckpt_path, adapters_dir, log_fn)
    dst = os.path.join(output_dir, os.path.basename(ckpt_path))
    shutil.copy2(ckpt_path, dst)
    shutil.copy2(os.path.join(adapters_dir, "adapter_config.json"),
                 os.path.join(output_dir, "adapter_config.json"))
    _log(log_fn, f"✓ Adapters saved to {output_dir}")
    return output_dir


def save_merged_model_mlx(
    model_id: str,
    ckpt_path: str,
    adapters_dir: str,
    output_dir: str,
    training_config: dict,
    log_fn: Optional[LogFn] = None,
) -> str:
    """Fuse MLX LoRA adapters into base model using mlx_lm.fuse."""
    os.makedirs(output_dir, exist_ok=True)
    _ensure_adapter_config(adapters_dir, training_config, log_fn)
    _ensure_adapters_safetensors(ckpt_path, adapters_dir, log_fn)
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_id,
        "--adapter-path", adapters_dir,
        "--save-path", output_dir,
    ]
    _log(log_fn, f"Running: {' '.join(cmd)}")
    _run_streaming(cmd, log_fn)
    _log(log_fn, f"✓ Merged model saved to {output_dir}")
    return output_dir


def save_gguf_mlx(
    model_id: str,
    ckpt_path: str,
    adapters_dir: str,
    output_path: str,
    training_config: dict,
    quantization: str = "q4_k_m",
    log_fn: Optional[LogFn] = None,
) -> str:
    """Export fused MLX model to GGUF.

    Strategy:
    1. Try mlx_lm fuse --export-gguf (fast, but not all model types supported)
    2. Fallback: fuse to merged HF model, then convert with llama.cpp
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _ensure_adapter_config(adapters_dir, training_config, log_fn)
    _ensure_adapters_safetensors(ckpt_path, adapters_dir, log_fn)

    # Strategy 1: mlx_lm fuse --export-gguf
    cmd1 = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_id,
        "--adapter-path", adapters_dir,
        "--export-gguf",
        "--gguf-path", output_path,
    ]
    _log(log_fn, f"Running: {' '.join(cmd1)}")
    proc = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    out_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            _log(log_fn, line)
            out_lines.append(line)
    proc.wait()
    if proc.returncode == 0:
        _log(log_fn, f"✓ GGUF saved to {output_path}")
        return output_path

    # Check if failure is "not supported for GGUF conversion"
    combined = "\n".join(out_lines)
    if "not supported for GGUF" not in combined and "not supported" not in combined:
        raise RuntimeError(f"mlx_lm fuse --export-gguf failed:\n{combined}")

    _log(log_fn, "⚠️  Direct GGUF export not supported for this model type. Falling back: fuse → llama.cpp convert")

    # Strategy 2: fuse to merged model, then llama.cpp
    merged_dir = os.path.join(os.path.dirname(output_path), "merged_for_gguf")
    save_merged_model_mlx(model_id, ckpt_path, adapters_dir, merged_dir, training_config, log_fn)

    convert_script = _find_llama_cpp_convert()
    if convert_script is None:
        raise RuntimeError(
            "llama.cpp convert_hf_to_gguf.py not found.\n"
            f"Merged model saved at: {merged_dir}\n"
            "To finish GGUF conversion manually:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            f"  python llama.cpp/convert_hf_to_gguf.py {merged_dir} --outfile {output_path}\n"
            f"  llama.cpp/llama-quantize {output_path} {output_path}.q4_k_m.gguf Q4_K_M"
        )

    # convert_hf_to_gguf.py natively supports: f32, f16, bf16, q8_0
    # k-quants (Q4_K_M, Q5_K_M, Q2_K, etc.) require llama-quantize
    _NATIVE_TYPES = {"F16", "F32", "BF16", "Q8_0"}
    quant_upper = quantization.upper()

    if quant_upper in _NATIVE_TYPES:
        # Direct conversion — no llama-quantize needed
        outtype = quant_upper.lower().replace("_", "")  # q8_0 → q80... actually keep as-is
        # convert_hf_to_gguf uses "q8_0" as the outtype string
        outtype_arg = quant_upper.lower()
        cmd2 = [sys.executable, convert_script, merged_dir,
                "--outfile", output_path, "--outtype", outtype_arg]
        _log(log_fn, f"Running: {' '.join(cmd2)}")
        _run_streaming(cmd2, log_fn)
        _log(log_fn, f"✓ GGUF saved to {output_path}")
        return output_path

    # k-quant: convert to F16 first, then quantize
    f16_path = output_path.replace(".gguf", "_f16.gguf")
    cmd2 = [sys.executable, convert_script, merged_dir, "--outfile", f16_path, "--outtype", "f16"]
    _log(log_fn, f"Running: {' '.join(cmd2)}")
    _run_streaming(cmd2, log_fn)

    _log(log_fn, f"Quantizing to {quant_upper} (building llama-quantize from cmake if needed)...")
    quantize_bin = _find_llama_cpp_quantize(convert_script)
    if quantize_bin:
        _log(log_fn, f"Using: {quantize_bin}")
        # Try requested k-quant; if it fails skip to Q8_0 (natively supported, no llama-quantize)
        succeeded = False
        try:
            cmd3 = [quantize_bin, f16_path, output_path, quant_upper]
            _log(log_fn, f"Running: {' '.join(cmd3)}")
            _run_streaming(cmd3, log_fn)
            succeeded = True
        except Exception as e:
            _log(log_fn, f"⚠️  {quant_upper} failed ({e}), falling back to Q8_0 via native convert...")
            # Q8_0 is natively supported by convert_hf_to_gguf.py — no llama-quantize needed
            try:
                cmd_q8 = [sys.executable, convert_script, merged_dir,
                          "--outfile", output_path, "--outtype", "q8_0"]
                _log(log_fn, f"Running: {' '.join(cmd_q8)}")
                _run_streaming(cmd_q8, log_fn)
                _log(log_fn, "ℹ️  Saved as Q8_0 (k-quant unavailable)")
                succeeded = True
            except Exception as e2:
                _log(log_fn, f"⚠️  Q8_0 also failed ({e2}), falling back to F16")
        if succeeded:
            if os.path.isfile(f16_path):
                try:
                    os.remove(f16_path)
                except Exception:
                    pass
        else:
            import shutil as _shutil
            _shutil.move(f16_path, output_path)
            _log(log_fn, "⚠️  All quantization levels failed — saved as F16")
    else:
        import shutil as _shutil
        _shutil.move(f16_path, output_path)
        _log(log_fn, "⚠️  llama-quantize not found and cmake build failed — saved as F16")

    _log(log_fn, f"✓ GGUF saved to {output_path}")
    return output_path


# ────────────────────────────────────────────────────────────────
# LoRA Adapter
# ────────────────────────────────────────────────────────────────

def save_lora_adapter(
    model,
    tokenizer,
    output_dir: str,
    log_fn: Optional[LogFn] = None,
) -> str:
    """保存 LoRA 适配器（仅增量权重，10-100MB）。"""
    os.makedirs(output_dir, exist_ok=True)
    _log(log_fn, f"保存 LoRA 适配器到: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _log(log_fn, "LoRA 适配器保存完成。")
    return output_dir


# ────────────────────────────────────────────────────────────────
# Merged Model
# ────────────────────────────────────────────────────────────────

def save_merged_model(
    model,
    tokenizer,
    output_dir: str,
    log_fn: Optional[LogFn] = None,
) -> str:
    """合并 LoRA 权重到基础模型并保存完整模型。"""
    os.makedirs(output_dir, exist_ok=True)
    _log(log_fn, f"合并模型到: {output_dir}（可能需要几分钟...）")

    # 尝试 Unsloth 方式
    try:
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
        _log(log_fn, "合并完成（Unsloth merged_16bit）。")
        return output_dir
    except AttributeError:
        pass

    # 标准 PEFT 方式
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        _log(log_fn, "合并完成（PEFT merge_and_unload）。")
        return output_dir
    except Exception as e:
        raise RuntimeError(f"合并模型失败: {e}") from e


# ────────────────────────────────────────────────────────────────
# GGUF
# ────────────────────────────────────────────────────────────────

GGUF_QUANTIZATIONS = {
    "q4_k_m": "Q4_K_M（推荐，4bit，质量/大小均衡）",
    "q5_k_m": "Q5_K_M（5bit，质量更高）",
    "q8_0":   "Q8_0（8bit，接近全精度）",
    "f16":    "F16（半精度，最大质量）",
    "q2_k":   "Q2_K（2bit，极小体积，质量较低）",
}


def save_gguf(
    model,
    tokenizer,
    output_dir: str,
    quantization: str = "q4_k_m",
    log_fn: Optional[LogFn] = None,
) -> str:
    """
    将模型导出为 GGUF 格式。
    优先使用 Unsloth 内置方法；若不可用，尝试通过 llama.cpp 转换。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 方式 1：Unsloth 内置 GGUF 导出
    try:
        _log(log_fn, f"导出 GGUF ({quantization}) 到: {output_dir}")
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization)
        gguf_files = [f for f in os.listdir(output_dir) if f.endswith(".gguf")]
        if gguf_files:
            path = os.path.join(output_dir, gguf_files[0])
            _log(log_fn, f"GGUF 导出完成: {path}")
            return path
    except AttributeError:
        _log(log_fn, "Unsloth GGUF 方法不可用，尝试先合并模型再转换...")
    except Exception as e:
        _log(log_fn, f"Unsloth GGUF 失败: {e}，尝试备用方法...")

    # 方式 2：先合并为完整模型，再调用 llama.cpp 转换
    merged_dir = os.path.join(output_dir, "merged_for_gguf")
    save_merged_model(model, tokenizer, merged_dir, log_fn)

    # 查找 llama.cpp convert 脚本
    convert_script = _find_llama_cpp_convert()
    if convert_script is None:
        raise RuntimeError(
            "未找到 llama.cpp convert 脚本。\n"
            "请先安装 llama.cpp 或使用 Unsloth CUDA 后端（支持内置 GGUF 导出）。"
        )

    gguf_out = os.path.join(output_dir, f"model-{quantization}.gguf")
    cmd = [
        sys.executable, convert_script,
        merged_dir,
        "--outfile", gguf_out,
        "--outtype", quantization,
    ]
    _log(log_fn, f"运行 llama.cpp 转换: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"llama.cpp 转换失败:\n{result.stderr}")

    _log(log_fn, f"GGUF 导出完成: {gguf_out}")
    return gguf_out


def _find_llama_cpp_convert() -> Optional[str]:
    """查找 llama.cpp convert_hf_to_gguf.py 或 convert.py 脚本。"""
    candidates = [
        os.path.join(os.getcwd(), "llama.cpp", "convert_hf_to_gguf.py"),
        os.path.join(os.getcwd(), "llama.cpp", "convert.py"),
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        os.path.expanduser("~/llama.cpp/convert.py"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _find_llama_cpp_quantize(convert_script: str) -> Optional[str]:
    """Find or build llama-quantize binary."""
    import shutil as _shutil

    # 1. PATH + common Homebrew locations
    for candidate in ["llama-quantize", "/opt/homebrew/bin/llama-quantize",
                      "/usr/local/bin/llama-quantize"]:
        p = _shutil.which(candidate) or (candidate if os.path.isfile(candidate) else None)
        if p and os.access(p, os.X_OK):
            return p

    # 2. Next to convert script in build subdirs
    llama_dir = os.path.dirname(convert_script)
    for name in ("llama-quantize", "quantize"):
        for subdir in ("", "build", os.path.join("build", "bin"),
                       os.path.join("build", "Release")):
            p = os.path.join(llama_dir, subdir, name)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p

    # 3. cmake build from source (use full path to avoid conda PATH issues)
    cmake_bin = _shutil.which("cmake") or "/opt/homebrew/bin/cmake" or "/usr/local/bin/cmake"
    if not os.path.isfile(cmake_bin):
        return None
    build_dir = os.path.join(llama_dir, "build")
    try:
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run([
            cmake_bin, "..",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_METAL=ON" if sys.platform == "darwin" else "-DLLAMA_METAL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
        ], cwd=build_dir, check=True, timeout=120,
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([
            cmake_bin, "--build", ".", "--config", "Release",
            "--target", "llama-quantize", "-j4",
        ], cwd=build_dir, check=True, timeout=600,
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None

    for name in ("llama-quantize", "quantize"):
        for subdir in ("bin", "", os.path.join("bin", "Release")):
            p = os.path.join(build_dir, subdir, name)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    return None


# Quantization fallback order: preferred → less compressed
_QUANT_FALLBACK = ["Q2_K", "Q4_K_M", "Q5_K_M", "Q8_0", "F16"]


# ────────────────────────────────────────────────────────────────
# HuggingFace Hub
# ────────────────────────────────────────────────────────────────

def push_to_hub(
    model,
    tokenizer,
    repo_id: str,
    token: str,
    private: bool = True,
    log_fn: Optional[LogFn] = None,
) -> str:
    """将模型和 tokenizer 推送到 HuggingFace Hub。"""
    _log(log_fn, f"推送到 HuggingFace Hub: {repo_id}")
    try:
        model.push_to_hub(repo_id, token=token, private=private)
        tokenizer.push_to_hub(repo_id, token=token, private=private)
        url = f"https://huggingface.co/{repo_id}"
        _log(log_fn, f"推送完成: {url}")
        return url
    except Exception as e:
        raise RuntimeError(f"推送失败: {e}") from e


# ────────────────────────────────────────────────────────────────
# Quick Inference
# ────────────────────────────────────────────────────────────────

def _find_llama_cli(convert_script: str) -> Optional[str]:
    """Find or build llama-cli binary alongside llama-quantize."""
    import shutil as _shutil
    for candidate in ["llama-cli", "/opt/homebrew/bin/llama-cli"]:
        p = _shutil.which(candidate) or (candidate if os.path.isfile(candidate) else None)
        if p and os.access(p, os.X_OK):
            return p
    llama_dir = os.path.dirname(convert_script)
    for name in ("llama-cli", "main"):
        for subdir in ("", "build", os.path.join("build", "bin")):
            p = os.path.join(llama_dir, subdir, name)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    # Build from source (targets both llama-quantize and llama-cli)
    cmake_bin = _shutil.which("cmake") or "/opt/homebrew/bin/cmake"
    if not os.path.isfile(cmake_bin):
        return None
    build_dir = os.path.join(llama_dir, "build")
    try:
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run([cmake_bin, "..", "-DCMAKE_BUILD_TYPE=Release",
                        "-DLLAMA_METAL=ON", "-DLLAMA_BUILD_TESTS=OFF"],
                       cwd=build_dir, check=True, timeout=120,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([cmake_bin, "--build", ".", "--config", "Release",
                        "--target", "llama-cli", "-j4"],
                       cwd=build_dir, check=True, timeout=600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    for name in ("llama-cli", "main"):
        for subdir in ("bin", ""):
            p = os.path.join(build_dir, subdir, name)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    return None


def run_inference_gguf(
    gguf_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    convert_script: Optional[str] = None,
) -> str:
    """Run inference on a GGUF file using llama-cli subprocess."""
    convert_script = convert_script or _find_llama_cpp_convert()
    if not convert_script:
        raise RuntimeError("llama.cpp not found — cannot run GGUF inference")
    cli = _find_llama_cli(convert_script)
    if not cli:
        raise RuntimeError("llama-cli not found and cmake build failed")
    cmd = [
        cli, "-m", gguf_path,
        "-p", prompt,
        "-n", str(max_new_tokens),
        "--temp", str(temperature),
        "--no-display-prompt",
        "-e",  # escape newlines
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"llama-cli failed:\n{result.stderr[:500]}")
    return result.stdout.strip()


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    gguf_path: Optional[str] = None,
) -> str:
    """使用已加载模型进行快速推理（供导出 Tab 测试用）。

    If gguf_path is provided, uses llama-cli for inference (avoids segfault).
    Otherwise runs MLX generate in-process.
    """
    # Always use GGUF via llama-cli — in-process MLX inference causes Metal crashes
    if not gguf_path or not os.path.isfile(gguf_path):
        raise RuntimeError(
            "Please export the model to GGUF first, then run inference.\n"
            "(Export tab → check GGUF → click Export, then try inference again)"
        )
    convert_script = _find_llama_cpp_convert()
    if not convert_script:
        raise RuntimeError("llama.cpp not found — cannot run GGUF inference")
    return run_inference_gguf(gguf_path, prompt, max_new_tokens, temperature, convert_script)
