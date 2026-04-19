"""
core/trainer.py
训练编排核心模块。
支持三种后端：Unsloth+CUDA / HuggingFace+MPS / HuggingFace+CPU
支持三种训练方式：SFT / DPO / ORPO
训练在后台线程运行，通过 TrainingMonitor 与 UI 通信。
"""

import os
import json
import time
import threading
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from core.monitor import TrainingMonitor


# ────────────────────────────────────────────────────────────────
# TrainingConfig
# ────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # 模型
    model_id: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    load_in_4bit: bool = True
    max_seq_length: int = 2048

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_rslora: bool = False
    use_gradient_checkpointing: bool = True

    # 训练方式
    training_type: str = "sft"          # "sft" | "dpo" | "orpo"

    # 超参数
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "auto"                  # "auto" = 根据后端自动选择
    seed: int = 3407

    # 数据集
    dataset_path: str = ""
    train_ratio: float = 0.95
    max_samples: int = 0                 # 0 = 全量
    prompt_template: str = "alpaca"      # "alpaca" | "chatml" | "llama3"
    think_mode: str = "keep"            # "keep" | "strip"
    packing: bool = False
    neftune_noise_alpha: float = 0.0

    # Output
    output_dir: str = "outputs"
    save_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 20
    report_to: str = "none"
    wandb_project: str = ""

    # Resume
    resume_from_checkpoint: str = ""     # path to checkpoint dir (empty = fresh start)

    # DPO / ORPO
    beta: float = 0.1

    # Auto-generate rejected responses (SFT→ORPO/DPO)
    auto_generate_rejected: bool = True       # auto-detect and generate if SFT dataset
    rejection_refresh_steps: int = 1000       # refresh rejection model every N steps (0=never)
    rejection_refresh_epochs: bool = True     # also refresh at epoch boundaries
    rejection_max_tokens: int = 512
    rejection_temperature: float = 0.7

    # How rejected samples are produced when auto_generate_rejected is True.
    #   "dynamic"      — current behaviour: generate in background subprocess,
    #                    train on sliding window as samples arrive
    #   "pre_generate" — generate ALL rejected samples first, save preference
    #                    dataset to disk, then run standard ORPO/DPO training
    rejection_mode: str = "dynamic"
    # Output path for the pre-generated preference dataset (when mode=pre_generate).
    # Empty = auto-derive as `<output_dir>/preference_dataset.jsonl`.
    pre_generated_dataset_path: str = ""

    # Opt-in extreme memory-saving for preference training (DPO/ORPO).
    # When True the trainer auto-halves per_device_train_batch_size, doubles
    # gradient_accumulation_steps (to preserve effective batch), and caps
    # max_seq_length at 1024. OFF by default so training dynamics exactly
    # match the user's configured hyperparameters — use only when VRAM/RAM
    # is genuinely insufficient, as these caps change gradient noise
    # statistics and can truncate long sequences.
    aggressive_memory_save: bool = False

    def to_dict(self) -> Dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingConfig":
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ────────────────────────────────────────────────────────────────
# Metrics Callback
# ────────────────────────────────────────────────────────────────

class MetricsCallback:
    """HuggingFace TrainerCallback，将指标推送到 TrainingMonitor。"""

    def __init__(self, monitor: TrainingMonitor, stop_event: threading.Event,
                 config: "TrainingConfig" = None):
        self.monitor = monitor
        self.stop_event = stop_event
        self._config = config
        self._start_time = time.time()
        self._last_step_time = time.time()

    def _get_trainer_callback(self):
        """返回 transformers.TrainerCallback 子类实例。"""
        from transformers import TrainerCallback

        monitor = self.monitor
        stop_event = self.stop_event
        start_time = self._start_time
        config = self._config

        class _CB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                # GPU 内存
                gpu_mem = None
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_mem = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
                except Exception:
                    pass

                # 训练速度（samples/s）
                speed = None
                try:
                    speed = logs.get("train_samples_per_second") or logs.get("samples_per_second")
                except Exception:
                    pass

                monitor.put({
                    "type": "metrics",
                    "step": state.global_step,
                    "loss": logs.get("loss") or logs.get("train_loss"),
                    "eval_loss": logs.get("eval_loss"),
                    "learning_rate": logs.get("learning_rate"),
                    "epoch": state.epoch,
                    "speed": speed,
                    "gpu_mem_gb": gpu_mem,
                })
                monitor.put({
                    "type": "log",
                    "line": f"[step {state.global_step}] " + " | ".join(
                        f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in logs.items()
                    ),
                })

            def on_save(self, args, state, control, **kwargs):
                ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                monitor.put({"type": "checkpoint", "path": ckpt})
                # Auto-save training_config.json alongside the checkpoint
                if config is not None:
                    try:
                        cfg_path = os.path.join(ckpt, "training_config.json")
                        with open(cfg_path, "w", encoding="utf-8") as f:
                            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass

            def on_train_end(self, args, state, control, **kwargs):
                monitor.put({"type": "status", "status": "finished",
                              "total_steps": state.global_step})

            def on_step_end(self, args, state, control, **kwargs):
                if stop_event.is_set():
                    control.should_training_stop = True
                monitor.put({
                    "type": "status",
                    "status": "running",
                    "total_steps": state.max_steps,
                })
                return control

        return _CB()


# ────────────────────────────────────────────────────────────────
# Training Orchestrator
# ────────────────────────────────────────────────────────────────

class TrainingOrchestrator:
    """
    训练编排器：管理训练生命周期，在后台线程中运行训练。
    """

    def __init__(self, monitor: TrainingMonitor):
        self.monitor = monitor
        self._stop_event = threading.Event()
        self._queue_event = threading.Event()  # set by session_manager when slot available
        self._thread: Optional[threading.Thread] = None
        self._model = None
        self._tokenizer = None
        self._config: Optional[TrainingConfig] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, config: TrainingConfig) -> None:
        if self.is_running:
            raise RuntimeError("训练已在运行中，请先停止当前训练。")
        self._config = config
        self.monitor.reset()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(config,),
            daemon=True,
            name="training-thread",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.monitor.put({"type": "status", "status": "stopped"})

    # ── Internal ──────────────────────────────────────────────────

    def _run(self, config: TrainingConfig) -> None:
        try:
            self.monitor.put({"type": "status", "status": "loading"})
            self.monitor.put({"type": "log", "line": "=== Preparing training ==="})

            backend = self._detect_backend()
            self.monitor.put({"type": "log", "line": f"Backend: {backend}"})

            self.monitor.put({"type": "log", "line": f"Loading dataset: {config.dataset_path}"})
            self._auto_gen_prompts = None  # reset sentinel
            train_dataset, eval_dataset = self._prepare_datasets(config, backend)

            # Check if auto-generation mode was triggered
            auto_gen = self._auto_gen_prompts is not None

            if not auto_gen:
                n_train = len(train_dataset)
                n_eval = len(eval_dataset) if eval_dataset else 0
            else:
                train_prompts, eval_prompts = self._auto_gen_prompts
                n_train = len(train_prompts)
                n_eval = len(eval_prompts) if eval_prompts else 0

            eff_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
            steps_per_epoch = max(1, n_train // eff_batch)
            total_steps = steps_per_epoch * config.num_epochs
            self.monitor.put({
                "type": "status", "status": "loading", "total_steps": total_steps
            })
            self.monitor.put({
                "type": "log",
                "line": (f"Train: {n_train} | Eval: {n_eval} | "
                         f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}"),
            })

            self.monitor.put({"type": "log", "line": f"Loading model: {config.model_id}"})
            if backend == "unsloth":
                model, tokenizer = self._load_unsloth_model(config)
            elif backend == "mlx":
                model, tokenizer = self._load_mlx_model(config)
            else:
                model, tokenizer = self._load_hf_model(config, backend)

            self._model = model
            self._tokenizer = tokenizer
            self.monitor.put({"type": "log", "line": "Model loaded. Starting training..."})
            self.monitor.put({"type": "status", "status": "running"})

            # Save training_config.json to output root so checkpoint resume can find it
            try:
                os.makedirs(config.output_dir, exist_ok=True)
                cfg_root_path = os.path.join(config.output_dir, "training_config.json")
                with open(cfg_root_path, "w", encoding="utf-8") as f:
                    json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            except Exception:
                pass

            if auto_gen:
                if getattr(config, "rejection_mode", "dynamic") == "pre_generate":
                    # Synchronously generate ALL rejected samples, persist the
                    # preference dataset to disk, then fall through to a
                    # standard (non-rolling) ORPO/DPO training pass.
                    train_dataset, eval_dataset = self._run_pre_generate_preference(
                        config, backend, train_prompts, eval_prompts
                    )
                    del train_prompts, eval_prompts
                    self._auto_gen_prompts = None
                    import gc as _gc_local
                    _gc_local.collect()
                    # Recompute steps from the materialised dataset size
                    n_train = len(train_dataset)
                    steps_per_epoch = max(1, n_train // eff_batch)
                    total_steps = steps_per_epoch * config.num_epochs
                    self.monitor.put({
                        "type": "status", "status": "running", "total_steps": total_steps
                    })
                    self.monitor.put({
                        "type": "log",
                        "line": (f"Pre-generated preference dataset: Train={n_train}, "
                                 f"Steps/epoch={steps_per_epoch}, Total steps={total_steps}"),
                    })
                    callback = MetricsCallback(self.monitor, self._stop_event, config)
                    trainer = self._build_trainer(
                        config, backend, model, tokenizer,
                        train_dataset, eval_dataset, callback, total_steps
                    )
                    if backend == "mlx":
                        self._run_mlx_training(config, model, trainer)
                    else:
                        resume_ckpt = config.resume_from_checkpoint or None
                        trainer.train(resume_from_checkpoint=resume_ckpt)
                else:
                    # Dynamic ORPO/DPO: generate rejected responses and train
                    self._run_dynamic_preference(config, backend, model, tokenizer,
                                                 train_prompts, eval_prompts, total_steps)
                    # Drop large prompt lists once dynamic training has consumed them
                    del train_prompts, eval_prompts
                    self._auto_gen_prompts = None
                    import gc as _gc_local
                    _gc_local.collect()
            else:
                callback = MetricsCallback(self.monitor, self._stop_event, config)
                trainer = self._build_trainer(
                    config, backend, model, tokenizer,
                    train_dataset, eval_dataset, callback, total_steps
                )

                if backend == "mlx":
                    self._run_mlx_training(config, model, trainer)
                else:
                    resume_ckpt = config.resume_from_checkpoint or None
                    if resume_ckpt:
                        self.monitor.put({"type": "log", "line": f"Resuming from checkpoint: {resume_ckpt}"})
                    trainer.train(resume_from_checkpoint=resume_ckpt)

            if not self._stop_event.is_set():
                final_dir = os.path.join(config.output_dir, "final")
                os.makedirs(final_dir, exist_ok=True)
                if backend == "mlx":
                    model.save_pretrained(final_dir)
                    tokenizer.save_pretrained(final_dir)
                else:
                    model.save_pretrained(final_dir)
                    tokenizer.save_pretrained(final_dir)
                self.monitor.put({"type": "checkpoint", "path": final_dir})
                self.monitor.put({"type": "log", "line": f"Model saved to: {final_dir}"})
                self.monitor.put({"type": "status", "status": "finished"})

        except Exception as e:
            tb = traceback.format_exc()
            self.monitor.put({"type": "status", "status": "error", "error": str(e)})
            self.monitor.put({"type": "log", "line": f"[ERROR] {e}\n{tb}"})

    def _detect_backend(self) -> str:
        """检测可用后端：unsloth | hf_cuda | mlx | hf_mps | hf_cpu"""
        import importlib.util

        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False

        if cuda:
            if importlib.util.find_spec("unsloth") is not None:
                return "unsloth"
            return "hf_cuda"

        # macOS / Apple Silicon — prefer MLX-Tune over vanilla MPS
        if importlib.util.find_spec("mlx_tune") is not None:
            return "mlx"

        try:
            import torch
            if torch.backends.mps.is_available():
                return "hf_mps"
        except Exception:
            pass
        return "hf_cpu"

    # ── Dataset preparation ───────────────────────────────────────

    def _prepare_datasets(self, config: TrainingConfig, backend: str):
        from datasets import Dataset as HFDataset
        from core.dataset import (
            load_raw, split_records, format_dataset_sft,
        )

        records = load_raw(config.dataset_path)
        max_s = config.max_samples if config.max_samples > 0 else None
        train_recs, eval_recs = split_records(records, config.train_ratio, max_s)

        # 尝试获取 EOS token（此时 tokenizer 未加载，用默认值）
        # 实际 EOS 在加载 tokenizer 后传入，此处先格式化 text 字段
        # 真正的 eos_token 会在 _format_with_tokenizer 中处理
        eos = "</s>"

        if config.training_type == "sft":
            max_chars = config.max_seq_length * 4
            train_formatted = format_dataset_sft(
                train_recs, config.prompt_template, config.think_mode, eos,
                max_chars=max_chars,
            )
            eval_formatted = format_dataset_sft(
                eval_recs, config.prompt_template, config.think_mode, eos,
                max_chars=max_chars,
            ) if eval_recs else None
        else:
            # DPO / ORPO: detect whether the dataset already carries preference
            # pairs, or whether we need to auto-generate rejected responses.
            from core.dataset import detect_dataset_type
            sample = train_recs[0] if train_recs else {}
            ds_type = detect_dataset_type(train_recs)

            if ds_type == "preference":
                # Native preference dataset — use as-is
                train_formatted = [
                    {"prompt": r.get("prompt", ""), "chosen": r.get("chosen", ""), "rejected": r.get("rejected", "")}
                    for r in train_recs
                    if r.get("chosen", "").strip() and r.get("rejected", "").strip()
                ]
                eval_formatted = [
                    {"prompt": r.get("prompt", ""), "chosen": r.get("chosen", ""), "rejected": r.get("rejected", "")}
                    for r in eval_recs
                    if r.get("chosen", "").strip() and r.get("rejected", "").strip()
                ] if eval_recs else None
            elif config.auto_generate_rejected:
                # SFT dataset → auto-generate preference pairs
                # Return a sentinel so _run() knows to use dynamic generation
                from core.dataset import sft_to_preference_prompts
                max_chars = config.max_seq_length * 4
                train_prompts = sft_to_preference_prompts(
                    train_recs, config.prompt_template, config.think_mode, max_chars)
                eval_prompts = sft_to_preference_prompts(
                    eval_recs, config.prompt_template, config.think_mode, max_chars) if eval_recs else None
                self.monitor.put({"type": "log",
                    "line": f"Auto-generating preference dataset: {len(train_prompts)} prompts "
                            f"(chosen from SFT output, rejected will be generated by base model)"})
                # Store for _run() to pick up
                self._auto_gen_prompts = (train_prompts, eval_prompts)
                return None, None  # sentinel: _run() handles dynamic dataset
            else:
                raise ValueError(
                    f"{config.training_type.upper()} training requires a dataset with "
                    f"'chosen' and 'rejected' fields, but the loaded dataset appears to be "
                    f"SFT format (fields: {list(sample.keys())}). "
                    f"Please use a preference dataset or switch training type to SFT."
                )

        train_ds = HFDataset.from_list(train_formatted)
        eval_ds = HFDataset.from_list(eval_formatted) if eval_formatted else None
        return train_ds, eval_ds

    # ── Model loading ─────────────────────────────────────────────

    def _load_unsloth_model(self, config: TrainingConfig):
        """Load a model with Unsloth on CUDA."""
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_id,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=config.load_in_4bit,
        )
        gc = "unsloth" if config.use_gradient_checkpointing else False
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=gc,
            random_state=config.seed,
            use_rslora=config.use_rslora,
            loftq_config=None,
        )
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def _load_mlx_model(self, config: TrainingConfig):
        """使用 mlx_tune.FastLanguageModel 加载模型（Apple Silicon 原生）。"""
        # Apply in-process memory patches to mlx_tune (stream log-probs via
        # logsumexp + clear Metal cache per step). Safe no-op if mlx_tune not
        # installed or already patched.
        try:
            from core.mlx_patches import apply_mlx_tune_patches
            apply_mlx_tune_patches(
                log_fn=lambda m: self.monitor.put({"type": "log", "line": m})
            )
        except Exception:
            pass

        from mlx_tune import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_id,
            max_seq_length=config.max_seq_length,
            dtype=None,         # MLX 自动管理精度
            load_in_4bit=False, # MLX 不使用 bitsandbytes
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.target_modules or None,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            random_state=config.seed,
            use_rslora=config.use_rslora,
        )
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def _load_hf_model(self, config: TrainingConfig, backend: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        if backend == "hf_mps":
            dtype = torch.bfloat16
            device_map = {"": "mps"}
        elif backend == "hf_cuda":
            dtype = torch.float16
            device_map = "auto"
        else:
            dtype = torch.float32
            device_map = "cpu"

        model_kwargs = dict(
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        # 4bit 仅在 CUDA + bitsandbytes 可用时启用
        if config.load_in_4bit and backend in ("hf_cuda",):
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs.pop("torch_dtype", None)
            except ImportError:
                pass

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=True
        )
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)

        if config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=config.use_rslora,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model, tokenizer

    # ── Dynamic preference training (SFT→ORPO/DPO auto-convert) ──

    def _run_pre_generate_preference(self, config, backend, train_prompts, eval_prompts):
        """Synchronously generate rejected samples for every prompt, persist
        the resulting {prompt,chosen,rejected} preference dataset to disk,
        and return (train_ds, eval_ds) ready for standard ORPO/DPO training.

        This is the "pre_generate" rejection_mode — memory-friendly vs the
        rolling "dynamic" mode because the rejection subprocess is fully done
        (and terminated) before training starts.
        """
        import json as _json
        import os as _os
        import time as _time
        from datasets import Dataset as HFDataset
        from core.rejection_generator import RejectionGenerator

        monitor = self.monitor
        stop_event = self._stop_event

        out_path = (config.pre_generated_dataset_path or "").strip()
        if not out_path:
            _os.makedirs(config.output_dir, exist_ok=True)
            out_path = _os.path.join(config.output_dir, "preference_dataset.jsonl")

        def _generate_split(split_prompts, split_name):
            if not split_prompts:
                return []
            gen_backend = "mlx" if backend == "mlx" else "gguf"
            generator = RejectionGenerator(
                model_path=config.model_id,
                backend=gen_backend,
                max_tokens=config.rejection_max_tokens,
                temperature=config.rejection_temperature,
                log_fn=lambda msg: monitor.put({"type": "log", "line": msg}),
            )
            results = []
            try:
                generator.start()
                total = len(split_prompts)
                monitor.put({"type": "log",
                             "line": f"[pre_generate] {split_name}: generating "
                                     f"{total} rejected responses..."})
                t0 = _time.time()
                for i, item in enumerate(split_prompts):
                    if stop_event.is_set():
                        break
                    try:
                        rejected = generator.generate(item["prompt"])
                        results.append({
                            "prompt": item["prompt"],
                            "chosen": item["chosen"],
                            "rejected": rejected,
                        })
                    except Exception as e:
                        monitor.put({"type": "log",
                                     "line": f"[pre_generate] sample {i}: {e}"})
                    if (i + 1) % 10 == 0 or (i + 1) == total:
                        elapsed = _time.time() - t0
                        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                        eta = (total - (i + 1)) / rate if rate > 0 else 0.0
                        monitor.put({"type": "log",
                                     "line": f"[pre_generate] {split_name}: "
                                             f"{i+1}/{total}  {rate:.2f} it/s  "
                                             f"ETA {int(eta)}s"})
            finally:
                generator.stop()
            return results

        train_records = _generate_split(train_prompts, "train")
        if stop_event.is_set():
            raise RuntimeError("Training stopped during preference pre-generation")
        eval_records = _generate_split(eval_prompts, "eval") if eval_prompts else []

        # Persist the preference dataset so it can be reused / inspected.
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in train_records:
                    f.write(_json.dumps({**rec, "_split": "train"}, ensure_ascii=False) + "\n")
                for rec in eval_records:
                    f.write(_json.dumps({**rec, "_split": "eval"}, ensure_ascii=False) + "\n")
            monitor.put({"type": "log",
                         "line": f"[pre_generate] saved preference dataset -> {out_path}"})
        except Exception as e:
            monitor.put({"type": "log",
                         "line": f"[pre_generate] WARN: failed to save dataset: {e}"})

        train_ds = HFDataset.from_list(train_records) if train_records else None
        eval_ds = HFDataset.from_list(eval_records) if eval_records else None
        return train_ds, eval_ds

    def _run_dynamic_preference(self, config, backend, model, tokenizer,
                                train_prompts, eval_prompts, total_steps):
        """
        Generate rejected responses in background, train ORPO/DPO in batches.
        The rejection model stays loaded in the subprocess — only reloads at
        checkpoint refresh points (every N steps or epoch boundary).
        """
        import time
        import gc as _gc
        from datasets import Dataset as HFDataset
        from core.rejection_generator import RejectionGenerator
        from core.dynamic_dataset import DynamicPreferenceDataset

        monitor = self.monitor
        stop_event = self._stop_event
        eff_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps

        # Create dynamic dataset buffer.
        # After this call, the buffer owns the prompt list — null the local
        # parameter references so the caller's `del` can actually free them.
        n_train_prompts = len(train_prompts)
        dataset = DynamicPreferenceDataset(train_prompts, batch_size=eff_batch)
        train_prompts = None
        eval_prompts = None
        _gc.collect()

        # Start rejection generator in subprocess (model loaded once, stays resident)
        gen_backend = "mlx" if backend == "mlx" else "gguf"
        generator = RejectionGenerator(
            model_path=config.model_id,
            backend=gen_backend,
            max_tokens=config.rejection_max_tokens,
            temperature=config.rejection_temperature,
            log_fn=lambda msg: monitor.put({"type": "log", "line": msg}),
        )

        try:
            generator.start()

            # Background thread: generate rejected responses (subprocess stays alive)
            def _generate_thread():
                while not stop_event.is_set():
                    pending = dataset.pop_pending(1)
                    if not pending:
                        break
                    for item in pending:
                        if stop_event.is_set():
                            break
                        try:
                            rejected = generator.generate(item["prompt"])
                            item["rejected"] = rejected
                            dataset.add(item)
                        except Exception as e:
                            dataset.add_error()
                            monitor.put({"type": "log",
                                         "line": f"[WARN] Rejection generation error: {e}"})
                        gen_count, gen_total, gen_errors = dataset.get_progress()
                        if gen_count % 10 == 0 or gen_count == gen_total:
                            monitor.put({"type": "log",
                                         "line": f"Generating rejected: {gen_count}/{gen_total}"
                                                 + (f" ({gen_errors} errors)" if gen_errors else "")})
                dataset.mark_done()

            gen_thread = threading.Thread(target=_generate_thread, daemon=True)
            gen_thread.start()

            # Wait for first batch
            monitor.put({"type": "log",
                         "line": f"Waiting for first batch of rejected responses ({eff_batch} samples)..."})
            if not dataset.wait_for_batch(timeout=600):
                raise RuntimeError("Timeout waiting for rejected response generation")

            if stop_event.is_set():
                return

            # Train in rounds with a sliding window: each round consumes up to
            # WINDOW_MULTIPLIER * eff_batch freshly generated samples, trains on
            # them, then discards them from the buffer so memory stays bounded.
            monitor.put({"type": "log", "line": "Starting dynamic ORPO/DPO training..."})
            global_step = 0
            last_refresh_step = 0
            epoch = 0
            steps_per_epoch = max(1, n_train_prompts // eff_batch)
            WINDOW_MULTIPLIER = 8
            window_size = eff_batch * WINDOW_MULTIPLIER

            while global_step < total_steps and not stop_event.is_set():
                ready_n = dataset.peek_ready_count()
                if ready_n < eff_batch:
                    if dataset.is_done:
                        break
                    time.sleep(1)
                    continue

                # Consume a bounded window — samples are removed from the buffer
                take = min(window_size, ready_n)
                batch_samples = dataset.consume_ready(take)
                if not batch_samples:
                    time.sleep(1)
                    continue

                consumed_n = len(batch_samples)
                train_ds = HFDataset.from_list(batch_samples)
                # Release the Python list; Arrow now owns the data
                del batch_samples
                eval_ds = None

                # Expected optimiser steps this round = num_epochs passes
                # over the consumed window, with the user-configured batch.
                # This is our best a-priori estimate; we let the trainer run
                # its natural iter count (do NOT pass a max_steps cap) so
                # mlx_tune's in-loop "Step N/M | Loss:" print is reached
                # and the loss curve callback keeps firing.
                expected_iters = max(
                    1,
                    consumed_n * max(1, config.num_epochs) // eff_batch,
                )
                round_steps = min(
                    config.rejection_refresh_steps if config.rejection_refresh_steps > 0 else steps_per_epoch,
                    total_steps - global_step,
                    expected_iters,
                )

                # Auto-scale logging_steps so mlx_tune actually emits at
                # least one "Step N/M | Loss:" line per round (which the
                # `_capturing_print` hook in _run_mlx_training parses into
                # metrics for the live loss / LR curves).
                orig_logging_steps = config.logging_steps
                round_logging = max(1, min(orig_logging_steps, expected_iters // 2 or 1))
                config.logging_steps = round_logging

                # Build and run trainer for this round (no max_steps override —
                # let the trainer's natural iter count drive mlx_tune's
                # logging_steps cadence).
                callback = MetricsCallback(self.monitor, self._stop_event, config)
                trainer = self._build_trainer(
                    config, backend, model, tokenizer,
                    train_ds, eval_ds, callback, 0
                )
                # Restore user-configured logging_steps on the Python object
                # (the trainer has captured the value it needs already).
                config.logging_steps = orig_logging_steps

                # Resolve actual iter count the trainer will execute — MLX
                # exposes `self.iters`; HF Trainer we fall back to the
                # estimate. This keeps global_step aligned with real work.
                actual_iters = getattr(trainer, "iters", None)
                if not isinstance(actual_iters, int) or actual_iters <= 0:
                    actual_iters = round_steps

                if backend == "mlx":
                    self._run_mlx_training(config, model, trainer)
                else:
                    trainer.train()

                # Free trainer/dataset references between rounds to prevent memory creep
                del trainer, train_ds, eval_ds, callback
                _gc.collect()
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                    elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                        try:
                            _torch.mps.empty_cache()
                        except Exception:
                            pass
                except ImportError:
                    pass
                if backend == "mlx":
                    try:
                        import mlx.core as mx
                        mx.clear_cache()
                    except Exception:
                        pass

                global_step += actual_iters
                monitor.put({"type": "log",
                             "line": (f"Dynamic training round complete: "
                                      f"step {global_step}/{total_steps} "
                                      f"(+{actual_iters} steps this round; "
                                      f"consumed {consumed_n} samples; "
                                      f"buffer remaining: {dataset.peek_ready_count()})")})

                # Check epoch boundary
                new_epoch = global_step // steps_per_epoch
                if new_epoch > epoch:
                    epoch = new_epoch
                    monitor.put({"type": "log", "line": f"Epoch {epoch} complete"})

                # Refresh rejection model if needed (only time subprocess reloads)
                should_refresh = False
                if config.rejection_refresh_steps > 0 and (global_step - last_refresh_step) >= config.rejection_refresh_steps:
                    should_refresh = True
                if config.rejection_refresh_epochs and new_epoch > (last_refresh_step // steps_per_epoch if steps_per_epoch else 0):
                    should_refresh = True

                if should_refresh and not dataset.is_done:
                    last_refresh_step = global_step
                    adapters_dir = os.path.join(config.output_dir, "adapters")
                    os.makedirs(adapters_dir, exist_ok=True)
                    try:
                        model.save_pretrained(adapters_dir)
                        monitor.put({"type": "log",
                                     "line": f"Refreshing rejection model at step {global_step}..."})
                        generator.reload_model(adapters_dir)
                    except Exception as e:
                        monitor.put({"type": "log",
                                     "line": f"[WARN] Rejection model refresh failed: {e}"})

        finally:
            # Always clean up subprocess
            generator.stop()
            if 'gen_thread' in dir() and gen_thread.is_alive():
                gen_thread.join(timeout=10)

        gen_count, gen_total, gen_errors = dataset.get_progress()
        monitor.put({"type": "log",
                     "line": f"Dynamic preference training complete. "
                             f"Generated: {gen_count}/{gen_total}, Errors: {gen_errors}"})

    # ── MLX training ──────────────────────────────────────────────

    def _run_mlx_training(self, config: TrainingConfig, model, trainer):
        """
        执行 MLX 训练，通过 monkey-patch 将指标回调注入 mlx_lm 训练循环。
        DPO/ORPO 采用自定义子类覆写训练循环。
        """
        monitor = self.monitor
        stop_event = self._stop_event

        # ── Resume: load adapter weights ────────────────────────────
        if config.resume_from_checkpoint:
            ckpt = config.resume_from_checkpoint
            adapter_file = None
            if os.path.isfile(ckpt) and ckpt.endswith(".safetensors"):
                adapter_file = ckpt
            elif os.path.isdir(ckpt):
                # Find the highest-numbered adapter file in the dir
                import glob
                candidates = sorted(glob.glob(os.path.join(ckpt, "*_adapters.safetensors")))
                if not candidates:
                    candidates = sorted(glob.glob(os.path.join(ckpt, "adapters.safetensors")))
                if candidates:
                    adapter_file = candidates[-1]
            if adapter_file:
                try:
                    import mlx.core as mx
                    model.load_weights(adapter_file, strict=False)
                    mx.eval(model.parameters())
                    monitor.put({"type": "log",
                                 "line": f"Resumed adapter weights from: {adapter_file}"})
                except Exception as e:
                    monitor.put({"type": "log",
                                 "line": f"[WARN] Could not load adapter weights: {e}"})
            else:
                monitor.put({"type": "log",
                             "line": f"[WARN] No .safetensors found at resume path: {ckpt}"})

        if config.training_type == "sft":
            # SFT：patch mlx_tune.sft_trainer.mlx_train 注入 TrainingCallback
            try:
                import mlx_tune.sft_trainer as _sft_mod
                from mlx_lm.tuner.callbacks import TrainingCallback

                _reported_ckpts: set = set()

                class _MLXMetricCB(TrainingCallback):
                    def on_train_loss_report(self, info: dict):
                        if stop_event.is_set():
                            raise KeyboardInterrupt("Training stopped by user")
                        monitor.put({
                            "type": "metrics",
                            "step": info.get("iteration", 0),
                            "loss": info.get("train_loss"),
                            "learning_rate": info.get("learning_rate"),
                            "epoch": None,
                            "speed": info.get("iterations_per_second"),
                            "gpu_mem_gb": info.get("peak_memory"),
                        })
                        monitor.put({
                            "type": "log",
                            "line": (
                                f"[step {info.get('iteration', 0)}] "
                                f"loss={info.get('train_loss', 0):.5g} | "
                                f"lr={info.get('learning_rate', 0):.3e} | "
                                f"{info.get('tokens_per_second', 0):.0f} tok/s"
                            ),
                        })
                        # Report any newly saved adapter files to the monitor
                        adapters_dir = os.path.join(config.output_dir, "adapters")
                        if os.path.isdir(adapters_dir):
                            import glob as _glob
                            for f in sorted(_glob.glob(os.path.join(adapters_dir, "*_adapters.safetensors"))):
                                if f not in _reported_ckpts:
                                    _reported_ckpts.add(f)
                                    monitor.put({"type": "checkpoint", "path": f})

                    def on_val_loss_report(self, info: dict):
                        # 验证结束后主动释放 Metal 缓存，防止内存阶梯式增长
                        try:
                            import mlx.core as mx
                            mx.clear_cache()
                        except Exception:
                            pass

                _orig = _sft_mod.mlx_train
                _cb = _MLXMetricCB()

                def _patched_train(*args, **kwargs):
                    kwargs["training_callback"] = _cb
                    return _orig(*args, **kwargs)

                _sft_mod.mlx_train = _patched_train
                try:
                    trainer.train()
                finally:
                    _sft_mod.mlx_train = _orig

            except Exception:
                # 如果 patch 失败，直接跑（无实时指标）
                trainer.train()

        else:
            # DPO / ORPO：训练循环在 rl_trainers.py 中自定义，
            # 通过 monkey-patch 打印钩子注入指标

            # Fix: mlx_tune passes float token IDs to embedding layer — cast to int32
            try:
                import mlx.core as mx
                import mlx_tune.losses as _losses_mod
                _orig_log_probs = _losses_mod.compute_log_probs_with_lengths

                def _patched_log_probs(model, input_ids, lengths):
                    import mlx.core as _mx
                    if hasattr(input_ids, 'dtype') and input_ids.dtype != _mx.int32:
                        input_ids = input_ids.astype(_mx.int32)
                    return _orig_log_probs(model, input_ids, lengths)

                _losses_mod.compute_log_probs_with_lengths = _patched_log_probs
            except Exception:
                pass
            import builtins
            _orig_print = builtins.print
            step_buf: dict = {"step": 0, "loss": None}

            def _capturing_print(*args, **kwargs):
                _orig_print(*args, **kwargs)
                msg = " ".join(str(a) for a in args)
                # 解析 "Step N/M | Loss: X.XXXX" 格式
                if "Step " in msg and "Loss:" in msg:
                    try:
                        parts = msg.split("|")
                        step_part = parts[0].strip()   # "Step 10/100"
                        loss_part = parts[1].strip()   # "Loss: 0.6789"
                        step_num = int(step_part.split("/")[0].replace("Step", "").strip())
                        loss_val = float(loss_part.split(":")[1].strip())
                        step_buf["step"] = step_num
                        step_buf["loss"] = loss_val
                        monitor.put({
                            "type": "metrics",
                            "step": step_num,
                            "loss": loss_val,
                            "learning_rate": None,
                            "epoch": None,
                            "speed": None,
                            "gpu_mem_gb": None,
                        })
                        monitor.put({"type": "log", "line": f"[step {step_num}] loss={loss_val:.5g}"})
                    except Exception:
                        pass

            builtins.print = _capturing_print
            try:
                trainer.train()
            finally:
                builtins.print = _orig_print

    def _build_training_args(self, config: TrainingConfig, backend: str, total_steps: int):
        import torch
        from transformers import TrainingArguments

        # 选择优化器
        if config.optim == "auto":
            if backend == "unsloth" or backend == "hf_cuda":
                try:
                    import bitsandbytes
                    optim = "adamw_8bit"
                except ImportError:
                    optim = "adamw_torch"
            else:
                optim = "adamw_torch"
        else:
            optim = config.optim

        # 精度
        use_bf16 = False
        use_fp16 = False
        if backend in ("unsloth", "hf_cuda"):
            try:
                use_bf16 = torch.cuda.is_bf16_supported()
                use_fp16 = not use_bf16
            except Exception:
                use_fp16 = True
        elif backend == "hf_mps":
            use_bf16 = True
        # CPU: 两者都 False（fp32）

        os.makedirs(config.output_dir, exist_ok=True)

        return TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.lr_scheduler_type,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            fp16=use_fp16,
            bf16=use_bf16,
            optim=optim,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            output_dir=config.output_dir,
            report_to=config.report_to if config.report_to != "none" else "none",
            seed=config.seed,
            dataloader_num_workers=0,   # Windows/macOS 兼容
            remove_unused_columns=False,
            # neftune
            **({"neftune_noise_alpha": config.neftune_noise_alpha}
               if config.neftune_noise_alpha > 0 else {}),
        )

    def _build_trainer(
        self, config, backend, model, tokenizer,
        train_dataset, eval_dataset, callback_helper, total_steps
    ):
        if backend == "mlx":
            return self._build_mlx_trainer(config, model, tokenizer,
                                           train_dataset, eval_dataset,
                                           max_steps=total_steps)

        from trl import SFTTrainer, DPOTrainer, ORPOTrainer

        training_args = self._build_training_args(config, backend, total_steps)
        cb = callback_helper._get_trainer_callback()

        # 更新 EOS token 到已格式化的 text 字段
        if config.training_type == "sft":
            # 如果数据集 text 字段缺少正确 eos_token，重新格式化
            eos = tokenizer.eos_token or "</s>"
            if train_dataset[0]["text"][-len(eos):] != eos:
                from core.dataset import load_raw, split_records, format_dataset_sft
                from datasets import Dataset as HFDataset
                records = load_raw(config.dataset_path)
                max_s = config.max_samples if config.max_samples > 0 else None
                train_recs, eval_recs = split_records(records, config.train_ratio, max_s)
                train_fmt = format_dataset_sft(train_recs, config.prompt_template, config.think_mode, eos)
                eval_fmt = format_dataset_sft(eval_recs, config.prompt_template, config.think_mode, eos) if eval_recs else None
                train_dataset = HFDataset.from_list(train_fmt)
                eval_dataset = HFDataset.from_list(eval_fmt) if eval_fmt else None

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",
                max_seq_length=config.max_seq_length,
                packing=config.packing,
                args=training_args,
                callbacks=[cb],
            )
        elif config.training_type == "dpo":
            trainer = DPOTrainer(
                model=model,
                ref_model=None,    # 使用隐式参考模型（PEFT 模式）
                args=training_args,
                beta=config.beta,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                max_length=config.max_seq_length,
                callbacks=[cb],
            )
        elif config.training_type == "orpo":
            from trl import ORPOConfig
            pref_batch, pref_accum, pref_max_seq = self._preference_memory_caps(config)
            orpo_args = ORPOConfig(
                per_device_train_batch_size=pref_batch,
                gradient_accumulation_steps=pref_accum,
                num_train_epochs=config.num_epochs,
                learning_rate=config.learning_rate,
                warmup_ratio=config.warmup_ratio,
                lr_scheduler_type=config.lr_scheduler_type,
                weight_decay=config.weight_decay,
                max_grad_norm=config.max_grad_norm,
                fp16=training_args.fp16,
                bf16=training_args.bf16,
                optim=training_args.optim,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                save_total_limit=config.save_total_limit,
                output_dir=config.output_dir,
                seed=config.seed,
                beta=config.beta,
                max_length=pref_max_seq,
                max_prompt_length=pref_max_seq // 2,
                remove_unused_columns=True,
            )
            trainer = ORPOTrainer(
                model=model,
                args=orpo_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                callbacks=[cb],
            )
        else:
            raise ValueError(f"未知训练类型：{config.training_type}")

        return trainer

    def _preference_memory_caps(self, config: TrainingConfig):
        """Return (batch, grad_accum, max_seq) for DPO/ORPO.

        Default: return the user's exact configured values — preference
        training runs at the same hyperparameters as SFT so training
        dynamics are not silently altered.

        Opt-in (`config.aggressive_memory_save=True`): halve the batch,
        double gradient accumulation (to keep effective batch), and cap
        sequence length at 1024. Intended only for genuinely
        memory-constrained hosts; this changes gradient noise statistics
        and may truncate long sequences.
        """
        if not getattr(config, "aggressive_memory_save", False):
            return (
                config.per_device_train_batch_size,
                config.gradient_accumulation_steps,
                config.max_seq_length,
            )
        pref_batch = max(1, config.per_device_train_batch_size // 2)
        pref_accum = max(1, config.gradient_accumulation_steps * 2)
        pref_max_seq = min(config.max_seq_length, 1024)
        if (pref_batch != config.per_device_train_batch_size
                or pref_max_seq != config.max_seq_length):
            try:
                self.monitor.put({
                    "type": "log",
                    "line": (
                        "[Preference memory caps] "
                        f"batch={config.per_device_train_batch_size}->{pref_batch}, "
                        f"grad_accum={config.gradient_accumulation_steps}->{pref_accum}, "
                        f"max_seq_length={config.max_seq_length}->{pref_max_seq} "
                        "(aggressive_memory_save=True)"
                    ),
                })
            except Exception:
                pass
        return pref_batch, pref_accum, pref_max_seq

    def _build_mlx_trainer(self, config, model, tokenizer,
                           train_dataset, eval_dataset,
                           max_steps: int = 0):
        """构建 mlx_tune 训练器（SFT / DPO / ORPO）。

        When ``max_steps > 0`` the trainer is forced to execute exactly that
        many optimisation steps. Used by the dynamic preference rolling loop
        to keep the outer ``global_step`` counter in sync with the actual
        number of iterations the MLX trainer performs each round (otherwise
        mlx_tune falls back to ``num_train_epochs × len(dataset) / batch``
        and silently out-runs our accounting).
        """
        import os

        os.makedirs(config.output_dir, exist_ok=True)

        # 将 HFDataset 转为 list[dict]（mlx_tune 接受 list）
        # 转换后立即释放 HFDataset 引用，避免内存中存在两份副本
        def _to_list(ds):
            if ds is None:
                return None
            if hasattr(ds, "to_list"):
                return ds.to_list()
            return list(ds)

        train_list = _to_list(train_dataset)
        eval_list = _to_list(eval_dataset)
        del train_dataset, eval_dataset

        # Optionally force an exact iter count (used by the dynamic preference
        # rolling loop). Pass max_steps into whichever mlx_tune config accepts
        # it; we add it conditionally so SFTConfig/ORPOConfig/DPOConfig that
        # don't expose the field won't break.
        extra_args = {"max_steps": int(max_steps)} if max_steps and max_steps > 0 else {}

        if config.training_type == "sft":
            from mlx_tune import SFTTrainer, SFTConfig
            args = SFTConfig(
                output_dir=config.output_dir,
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                lr_scheduler_type=config.lr_scheduler_type,
                warmup_ratio=config.warmup_ratio,
                num_train_epochs=config.num_epochs,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                save_total_limit=config.save_total_limit,
                weight_decay=config.weight_decay,
                max_seq_length=config.max_seq_length,
                packing=config.packing,
                use_native_training=True,
                **extra_args,
            )
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_list,
                eval_dataset=eval_list,
                args=args,
                dataset_text_field="text",
            )

        elif config.training_type == "dpo":
            from mlx_tune import DPOTrainer, DPOConfig
            # DPO/ORPO do two forward passes per step and retain both activation
            # graphs for backward — halve the effective batch to keep MLX peak
            # memory roughly comparable to SFT.
            pref_batch, pref_accum, pref_max_seq = self._preference_memory_caps(config)
            args = DPOConfig(
                beta=config.beta,
                output_dir=config.output_dir,
                learning_rate=config.learning_rate,
                per_device_train_batch_size=pref_batch,
                gradient_accumulation_steps=pref_accum,
                num_train_epochs=config.num_epochs,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                max_seq_length=pref_max_seq,
                max_prompt_length=pref_max_seq // 2,
                **extra_args,
            )
            trainer = DPOTrainer(
                model=model,
                train_dataset=train_list,
                ref_model=None,
                tokenizer=tokenizer,
                args=args,
            )

        elif config.training_type == "orpo":
            from mlx_tune import ORPOTrainer, ORPOConfig
            pref_batch, pref_accum, pref_max_seq = self._preference_memory_caps(config)
            args = ORPOConfig(
                beta=config.beta,
                output_dir=config.output_dir,
                learning_rate=config.learning_rate,
                per_device_train_batch_size=pref_batch,
                gradient_accumulation_steps=pref_accum,
                num_train_epochs=config.num_epochs,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                max_seq_length=pref_max_seq,
                max_prompt_length=pref_max_seq // 2,
                **extra_args,
            )
            trainer = ORPOTrainer(
                model=model,
                train_dataset=train_list,
                tokenizer=tokenizer,
                args=args,
            )

        else:
            raise ValueError(f"未知训练类型：{config.training_type}")

        # If max_steps was requested but the mlx_tune config rejected the kwarg
        # (older versions), force it directly on the trainer's iter count so the
        # outer step accounting remains accurate.
        if max_steps and max_steps > 0:
            for attr in ("iters", "max_steps"):
                if hasattr(trainer, attr):
                    try:
                        setattr(trainer, attr, int(max_steps))
                    except Exception:
                        pass

        return trainer
