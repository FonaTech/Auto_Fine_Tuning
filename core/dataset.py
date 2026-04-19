"""
core/dataset.py
数据集加载、验证、格式化模块。
支持 JSONL / JSON / CSV，处理 Alpaca 格式与 <think> 推理块。
"""

import json
import re
import os
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


# ────────────────────────────────────────────────────────────────
# Prompt 模板
# ────────────────────────────────────────────────────────────────

ALPACA_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)

ALPACA_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{output}"
)

CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}{input_suffix}<|im_end|>\n"
    "<|im_start|>assistant\n{output}<|im_end|>"
)

LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}{input_suffix}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{output}<|eot_id|>"
)

TEMPLATES = {
    "alpaca": "Alpaca 格式",
    "chatml": "ChatML 格式",
    "llama3": "Llama3 格式",
}

# SFT 必须字段（至少需要 instruction 和 output）
SFT_REQUIRED_FIELDS = {"instruction", "output"}
# DPO / ORPO 必须字段
PREF_REQUIRED_FIELDS = {"prompt", "chosen", "rejected"}


# ────────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────────

@dataclass
class DatasetStats:
    total: int
    train_size: int
    eval_size: int
    fields: List[str]
    avg_instruction_len: float
    avg_output_len: float
    has_think_blocks: bool
    sample_count_with_input: int


@dataclass
class ValidationResult:
    valid: bool
    message: str
    missing_fields: List[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────────────────────

def load_raw(path: str) -> List[Dict[str, Any]]:
    """读取 JSONL / JSON / CSV 文件，返回 dict 列表。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 可能是 {"data": [...]} 形式
            for key in ("data", "examples", "train", "samples"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        raise ValueError("JSON 格式不支持：需要数组或含 data 键的对象")
    elif ext == ".csv":
        records = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records
    else:
        raise ValueError(f"不支持的文件格式：{ext}（支持 .jsonl / .json / .csv）")


def detect_fields(records: List[Dict]) -> List[str]:
    """从前 10 条记录中收集所有字段名。"""
    fields = set()
    for r in records[:10]:
        fields.update(r.keys())
    return sorted(fields)


# ────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────

def validate_fields(records: List[Dict], training_type: str = "sft") -> ValidationResult:
    if not records:
        return ValidationResult(valid=False, message="Dataset is empty")

    fields = set(detect_fields(records))
    required = SFT_REQUIRED_FIELDS if training_type == "sft" else PREF_REQUIRED_FIELDS
    missing = list(required - fields)

    if missing:
        return ValidationResult(
            valid=False,
            message=f"Missing required fields: {missing}. Detected fields: {sorted(fields)}",
            missing_fields=missing,
        )
    return ValidationResult(valid=True, message=f"✅ Validation passed: {len(records)} records")


# ────────────────────────────────────────────────────────────────
# Think block handling
# ────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """移除 <think>...</think> 块，保留后续答案文本。"""
    result = _THINK_RE.sub("", text).strip()
    return result


def has_think_blocks(text: str) -> bool:
    return bool(_THINK_RE.search(text))


# ────────────────────────────────────────────────────────────────
# Formatting
# ────────────────────────────────────────────────────────────────

def format_sft_record(
    record: Dict,
    template: str = "alpaca",
    think_mode: str = "keep",
    eos_token: str = "</s>",
) -> str:
    """
    将单条 Alpaca 格式记录格式化为训练文本。
    think_mode: "keep" | "strip"
    """
    instruction = record.get("instruction", "").strip()
    input_text = record.get("input", "").strip()
    output = record.get("output", "").strip()
    system = record.get("system", "").strip() or "You are a helpful assistant."

    if think_mode == "strip":
        output = strip_think_blocks(output)

    if template == "alpaca":
        if input_text:
            text = ALPACA_TEMPLATE_WITH_INPUT.format(
                instruction=instruction, input=input_text, output=output
            )
        else:
            text = ALPACA_TEMPLATE_NO_INPUT.format(
                instruction=instruction, output=output
            )
    elif template == "chatml":
        input_suffix = f"\n{input_text}" if input_text else ""
        text = CHATML_TEMPLATE.format(
            system=system,
            instruction=instruction,
            input_suffix=input_suffix,
            output=output,
        )
    elif template == "llama3":
        input_suffix = f"\n{input_text}" if input_text else ""
        text = LLAMA3_TEMPLATE.format(
            system=system,
            instruction=instruction,
            input_suffix=input_suffix,
            output=output,
        )
    else:
        raise ValueError(f"未知模板：{template}")

    return text + eos_token


def format_dataset_sft(
    records: List[Dict],
    template: str = "alpaca",
    think_mode: str = "keep",
    eos_token: str = "</s>",
    max_samples: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    将记录列表格式化为 SFT 训练格式（每条包含 "text" 字段）。
    max_chars: 字符数上限，超长序列在字符级截断（约等于 max_seq_length * 4）。
    """
    if max_samples:
        records = records[:max_samples]
    result = []
    for r in records:
        try:
            text = format_sft_record(r, template, think_mode, eos_token)
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]
            result.append({"text": text})
        except Exception:
            continue
    return result


# ────────────────────────────────────────────────────────────────
# Splitting
# ────────────────────────────────────────────────────────────────

def split_records(
    records: List[Dict],
    train_ratio: float = 0.95,
    max_samples: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """按比例分割训练集/验证集。"""
    if max_samples:
        records = records[:max_samples]
    n = len(records)
    split_idx = max(1, int(n * train_ratio))
    return records[:split_idx], records[split_idx:]


# ────────────────────────────────────────────────────────────────
# SFT → Preference conversion
# ────────────────────────────────────────────────────────────────

def _build_prompt_only(
    record: Dict,
    template: str = "alpaca",
    think_mode: str = "keep",
) -> str:
    """Build the prompt portion (without the output) for rejection generation."""
    instruction = record.get("instruction", "").strip()
    input_text = record.get("input", "").strip()
    system = record.get("system", "").strip() or "You are a helpful assistant."

    if template == "alpaca":
        if input_text:
            return (
                "Below is an instruction that describes a task, paired with an input that provides "
                "further context. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            )
        else:
            return (
                "Below is an instruction that describes a task. Write a response that "
                "appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:\n"
            )
    elif template == "chatml":
        input_suffix = f"\n{input_text}" if input_text else ""
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}{input_suffix}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    elif template == "llama3":
        input_suffix = f"\n{input_text}" if input_text else ""
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}{input_suffix}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        return f"{instruction}\n{input_text}\n" if input_text else f"{instruction}\n"


def detect_dataset_type(records: List[Dict], sample_size: int = 10) -> str:
    """Classify a dataset as 'preference' (has chosen+rejected on at least one
    sampled record) or 'sft' (instruction/output style). Used by both the
    trainer and auto-tuner to decide whether SFT→preference conversion is
    needed for DPO/ORPO runs.
    """
    if not records:
        return "sft"
    sample = records[:max(1, sample_size)]
    has_chosen = any(str(r.get("chosen", "")).strip() for r in sample)
    has_rejected = any(str(r.get("rejected", "")).strip() for r in sample)
    if has_chosen and has_rejected:
        return "preference"
    return "sft"


def sft_to_preference_prompts(
    records: List[Dict],
    template: str = "alpaca",
    think_mode: str = "keep",
    max_chars: Optional[int] = None,
) -> List[Dict]:
    """
    Convert SFT records to preference prompt pairs.
    Returns list of {"prompt": str, "chosen": str} — rejected to be generated later.
    """
    result = []
    for r in records:
        instruction = r.get("instruction", "").strip()
        output = r.get("output", "").strip()
        if not instruction or not output:
            continue
        if think_mode == "strip":
            output = strip_think_blocks(output)
        prompt = _build_prompt_only(r, template, think_mode)
        if max_chars and len(prompt) > max_chars:
            prompt = prompt[:max_chars]
        result.append({"prompt": prompt, "chosen": output})
    return result


# ────────────────────────────────────────────────────────────────
# Statistics & Preview
# ────────────────────────────────────────────────────────────────

def compute_statistics(
    records: List[Dict],
    train_size: int = 0,
    eval_size: int = 0,
) -> DatasetStats:
    if not records:
        return DatasetStats(
            total=0, train_size=0, eval_size=0,
            fields=[], avg_instruction_len=0, avg_output_len=0,
            has_think_blocks=False, sample_count_with_input=0,
        )

    fields = detect_fields(records)
    sample = records[:500]  # 采样前500条统计

    avg_instr = sum(len(r.get("instruction", "")) for r in sample) / len(sample)
    avg_out = sum(len(r.get("output", "")) for r in sample) / len(sample)
    has_think = any(has_think_blocks(r.get("output", "")) for r in sample[:50])
    with_input = sum(1 for r in sample if r.get("input", "").strip())

    return DatasetStats(
        total=len(records),
        train_size=train_size,
        eval_size=eval_size,
        fields=fields,
        avg_instruction_len=round(avg_instr, 1),
        avg_output_len=round(avg_out, 1),
        has_think_blocks=has_think,
        sample_count_with_input=with_input,
    )


def preview_samples(records: List[Dict], n: int = 5) -> List[Dict]:
    """返回前 n 条记录的预览（截断长字段）。"""
    result = []
    for r in records[:n]:
        preview = {}
        for k, v in r.items():
            s = str(v)
            preview[k] = s[:200] + "..." if len(s) > 200 else s
        result.append(preview)
    return result


def format_preview_prompt(record: Dict, template: str = "alpaca") -> str:
    """格式化单条样本为完整 prompt（不含 EOS），用于 UI 展示。"""
    try:
        return format_sft_record(record, template=template, think_mode="keep", eos_token="")
    except Exception as e:
        return f"格式化失败：{e}"
