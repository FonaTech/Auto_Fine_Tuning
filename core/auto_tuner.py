"""
core/auto_tuner.py
AI 自动研究内核：基于 Optuna TPE（Tree-structured Parzen Estimator）贝叶斯优化，
自动搜索最优训练超参数，减少人工调参成本。

设计原则（对齐 karpathy/autoresearch 的"自动实验循环"理念）：
  1. 从候选空间抽样参数组合
  2. 在数据子集上运行短时间训练（probe trial）
  3. 用 Optuna TPE 从历史结果中学习，生成下一批更优参数
  4. 收集所有 trial 结果，输出最优配置 + 参数重要性分析
  5. 一键将最优参数填入主训练配置

线程架构：
  - 优化循环运行在后台 daemon 线程
  - 每个 trial 在同一线程内同步执行（避免嵌套线程）
  - UI 通过 event_queue 轮询进度事件
"""

import os
import gc
import time
import queue
import threading
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable

# Optuna（贝叶斯优化库）
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# ────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────

@dataclass
class SearchSpaceConfig:
    """定义哪些超参数参与贝叶斯搜索及其范围。"""
    # LoRA rank
    tune_lora_r: bool = True
    lora_r_choices: List[int] = field(default_factory=lambda: [4, 8, 16, 32])

    # Learning rate（对数均匀分布）
    tune_lr: bool = True
    lr_min: float = 5e-5
    lr_max: float = 5e-4

    # Batch size
    tune_batch: bool = True
    batch_size_choices: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # Gradient accumulation
    tune_grad_accum: bool = False
    grad_accum_choices: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # Warmup ratio
    tune_warmup: bool = True
    warmup_min: float = 0.01
    warmup_max: float = 0.15

    # LR scheduler
    tune_scheduler: bool = True
    scheduler_choices: List[str] = field(default_factory=lambda: ["cosine", "linear", "constant"])

    # LoRA alpha multiplier（alpha = r × multiplier）
    tune_lora_alpha: bool = True
    alpha_multiplier_choices: List[int] = field(default_factory=lambda: [1, 2])


@dataclass
class TrialResult:
    trial_number: int
    params: Dict[str, Any]
    train_loss: float          # 本 trial 的训练损失（越小越好）
    duration_s: float
    status: str                # "complete" | "pruned" | "failed"
    error: str = ""


@dataclass
class AutoTuneResult:
    best_params: Dict[str, Any]
    best_loss: float
    trials: List[TrialResult]
    param_importances: Dict[str, float]   # Optuna 参数重要性分析
    n_completed: int
    n_pruned: int
    n_failed: int
    elapsed_s: float


# ────────────────────────────────────────────────────────────────
# Auto Tuner
# ────────────────────────────────────────────────────────────────

class AutoTuner:
    """
    Optuna TPE 贝叶斯超参数优化器。

    工作流程：
      1. 加载基础模型（一次，所有 trial 共享）
      2. 对每个 trial：
         a. Optuna TPE 从搜索空间采样参数
         b. 应用参数到 LoRA 配置，在数据子集上训练 probe_steps 步
         c. 记录最终训练损失作为目标值
         d. Optuna 学习该结果，更新 TPE 内核，指导下一次采样
      3. 所有 trial 完成后，输出最优参数 + 参数重要性
    """

    def __init__(self):
        self._event_queue: queue.Queue = queue.Queue(maxsize=5000)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.study: Optional["optuna.Study"] = None
        self.result: Optional[AutoTuneResult] = None
        self.trials: List[TrialResult] = []
        self.status: str = "idle"   # idle|loading|running|finished|error|stopped

    # ── Public API ────────────────────────────────────────────────

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        model_id: str,
        dataset_path: str,
        train_ratio: float,
        prompt_template: str,
        think_mode: str,
        search_space: SearchSpaceConfig,
        n_trials: int = 20,
        probe_steps: int = 80,
        probe_samples: int = 300,
        load_in_4bit: bool = True,
        max_seq_length: int = 1024,
        base_lora_r: int = 16,
        base_lora_alpha: int = 32,
        base_lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        output_dir: str = "auto_tune_cache",
        seed: int = 42,
        training_type: str = "sft",
        auto_generate_rejected: bool = True,
        rejection_max_tokens: int = 512,
        rejection_temperature: float = 0.7,
        beta: float = 0.1,
        aggressive_memory_save: bool = False,
    ) -> None:
        if self.is_running():
            raise RuntimeError("自动优化已在运行中，请先停止。")
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna 未安装，请运行：pip install optuna")

        self._stop_event.clear()
        self.trials.clear()
        self.result = None
        self.status = "loading"

        kwargs = dict(
            model_id=model_id,
            dataset_path=dataset_path,
            train_ratio=train_ratio,
            prompt_template=prompt_template,
            think_mode=think_mode,
            search_space=search_space,
            n_trials=n_trials,
            probe_steps=probe_steps,
            probe_samples=probe_samples,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
            base_lora_r=base_lora_r,
            base_lora_alpha=base_lora_alpha,
            base_lora_dropout=base_lora_dropout,
            target_modules=target_modules or [
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj",
            ],
            output_dir=output_dir,
            seed=seed,
            training_type=training_type,
            auto_generate_rejected=auto_generate_rejected,
            rejection_max_tokens=rejection_max_tokens,
            rejection_temperature=rejection_temperature,
            beta=beta,
            aggressive_memory_save=aggressive_memory_save,
        )
        self._thread = threading.Thread(
            target=self._run, kwargs=kwargs, daemon=True, name="auto-tune-thread"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._put({"type": "status", "status": "stopped"})

    def poll(self) -> List[Dict]:
        """非阻塞拉取所有待处理事件（Gradio 主线程调用）。"""
        events = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    # ── Internal ──────────────────────────────────────────────────

    def _put(self, event: Dict) -> None:
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            pass

    def _run(
        self,
        model_id, dataset_path, train_ratio, prompt_template, think_mode,
        search_space, n_trials, probe_steps, probe_samples,
        load_in_4bit, max_seq_length,
        base_lora_r, base_lora_alpha, base_lora_dropout, target_modules,
        output_dir, seed,
        training_type="sft",
        auto_generate_rejected=True,
        rejection_max_tokens=512,
        rejection_temperature=0.7,
        beta=0.1,
        aggressive_memory_save=False,
    ) -> None:
        self._put({"type": "log", "line": "=== 自动优化开始 ==="})
        start_time = time.time()

        try:
            # ── 1. 加载数据集 ──────────────────────────────────────
            self._put({"type": "log", "line": f"加载数据集: {dataset_path}"})
            train_records, data_flavor = self._load_probe_dataset(
                dataset_path, train_ratio, probe_samples, prompt_template, think_mode,
                training_type=training_type,
                auto_generate_rejected=auto_generate_rejected,
                model_id=model_id,
                rejection_max_tokens=rejection_max_tokens,
                rejection_temperature=rejection_temperature,
                output_dir=output_dir,
            )
            self._put({"type": "log",
                       "line": f"探针数据集: {len(train_records)} 条 (flavor={data_flavor})"})

            # ── 2. 加载基础模型（所有 trial 共享，避免重复加载）────
            self._put({"type": "status", "status": "loading"})
            self._put({"type": "log", "line": f"加载基础模型: {model_id}"})
            base_model, tokenizer, backend = self._load_base_model(
                model_id, load_in_4bit, max_seq_length
            )
            self._put({"type": "log", "line": f"模型加载完成，后端: {backend}"})
            self._put({"type": "status", "status": "running"})

            # ── 3. 创建 Optuna Study（TPE + MedianPruner）────────────
            sampler = TPESampler(seed=seed, n_startup_trials=max(5, n_trials // 5))
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=probe_steps // 4)
            self.study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                study_name="unsloth_auto_tune",
            )
            self._put({
                "type": "log",
                "line": (f"Optuna Study 已创建: TPE采样器 "
                         f"(startup={sampler._n_startup_trials}), "
                         f"MedianPruner, {n_trials} trials, "
                         f"每 trial {probe_steps} steps"),
            })

            # ── 4. 定义 objective，运行优化 ───────────────────────────
            def objective(trial: "optuna.Trial") -> float:
                if self._stop_event.is_set():
                    raise optuna.exceptions.OptunaError("用户停止")

                trial_start = time.time()
                params = self._sample_params(trial, search_space, base_lora_r, base_lora_alpha)
                self._put({
                    "type": "trial_start",
                    "number": trial.number,
                    "params": params,
                    "total": n_trials,
                })
                self._put({
                    "type": "log",
                    "line": f"[Trial {trial.number+1}/{n_trials}] 参数: {_fmt_params(params)}",
                })

                try:
                    loss = self._run_probe_trial(
                        base_model, tokenizer, backend,
                        train_records, params,
                        max_seq_length, base_lora_dropout, target_modules,
                        probe_steps, output_dir, trial.number, trial, seed,
                        training_type=training_type, data_flavor=data_flavor, beta=beta,
                        aggressive_memory_save=aggressive_memory_save,
                    )
                    status = "complete"
                except optuna.exceptions.TrialPruned:
                    loss = float("inf")
                    status = "pruned"
                except Exception as e:
                    self._put({"type": "log", "line": f"[Trial {trial.number}] 失败: {e}"})
                    loss = float("inf")
                    status = "failed"

                duration = time.time() - trial_start
                tr = TrialResult(
                    trial_number=trial.number,
                    params=params,
                    train_loss=loss,
                    duration_s=round(duration, 1),
                    status=status,
                )
                self.trials.append(tr)

                best_so_far = (
                    min(t.train_loss for t in self.trials if t.status == "complete")
                    if any(t.status == "complete" for t in self.trials) else float("inf")
                )
                self._put({
                    "type": "trial_end",
                    "number": trial.number,
                    "loss": loss,
                    "best_loss": best_so_far,
                    "status": status,
                    "duration_s": duration,
                    "params": params,
                })
                self._put({
                    "type": "log",
                    "line": (f"[Trial {trial.number+1}/{n_trials}] "
                             f"loss={loss:.4f} | best={best_so_far:.4f} | "
                             f"{duration:.0f}s | {status}"),
                })

                if status == "pruned":
                    raise optuna.exceptions.TrialPruned()
                return loss

            self.study.optimize(
                objective,
                n_trials=n_trials,
                callbacks=[self._make_stop_callback()],
                gc_after_trial=True,
            )

            # ── 5. 计算参数重要性 & 输出结果 ──────────────────────────
            try:
                importances = optuna.importance.get_param_importances(self.study)
            except Exception:
                importances = {}

            completed = [t for t in self.trials if t.status == "complete"]
            pruned = [t for t in self.trials if t.status == "pruned"]
            failed = [t for t in self.trials if t.status == "failed"]

            best_params = self.study.best_params if completed else {}
            best_loss = self.study.best_value if completed else float("inf")

            self.result = AutoTuneResult(
                best_params=best_params,
                best_loss=best_loss,
                trials=self.trials,
                param_importances=importances,
                n_completed=len(completed),
                n_pruned=len(pruned),
                n_failed=len(failed),
                elapsed_s=time.time() - start_time,
            )
            self.status = "finished"

            self._put({
                "type": "finished",
                "best_params": best_params,
                "best_loss": best_loss,
                "importances": importances,
                "n_completed": len(completed),
                "n_pruned": len(pruned),
                "n_failed": len(failed),
                "elapsed_s": self.result.elapsed_s,
            })
            self._put({
                "type": "log",
                "line": (f"=== 优化完成 === "
                         f"最优 loss: {best_loss:.4f} | "
                         f"完成: {len(completed)} | 剪枝: {len(pruned)} | 失败: {len(failed)} | "
                         f"用时: {self.result.elapsed_s:.0f}s\n"
                         f"最优参数: {_fmt_params(best_params)}"),
            })

            # 清理模型内存
            del base_model, tokenizer
            _gc_collect()

        except Exception as e:
            import traceback
            self.status = "error"
            self._put({"type": "status", "status": "error", "error": str(e)})
            self._put({"type": "log", "line": f"[ERROR] {e}\n{traceback.format_exc()}"})

    # ── Dataset loading ───────────────────────────────────────────

    def _load_probe_dataset(
        self, path: str, train_ratio: float,
        probe_samples: int, template: str, think_mode: str,
        training_type: str = "sft",
        auto_generate_rejected: bool = True,
        model_id: str = "",
        rejection_max_tokens: int = 512,
        rejection_temperature: float = 0.7,
        output_dir: str = "auto_tune_cache",
    ):
        """Load a probe dataset shaped for the requested training_type.

        Returns `(records, flavor)` where flavor is one of:
          - "sft"                → records have a "text" field
          - "preference_native"  → records have prompt/chosen/rejected
          - "preference_cached"  → same as native, but auto-generated once
                                    and cached at <output_dir>/probe_preference.jsonl
        """
        from core.dataset import (
            load_raw, split_records, format_dataset_sft,
            sft_to_preference_prompts, detect_dataset_type,
        )
        import os as _os
        import json as _json

        records = load_raw(path)
        train_recs, _ = split_records(records, train_ratio)
        recs = train_recs[:probe_samples] if probe_samples > 0 else train_recs

        if training_type == "sft":
            return format_dataset_sft(recs, template, think_mode, "</s>"), "sft"

        # DPO / ORPO probe
        if detect_dataset_type(recs) == "preference":
            pairs = [
                {"prompt": r.get("prompt", ""),
                 "chosen": r.get("chosen", ""),
                 "rejected": r.get("rejected", "")}
                for r in recs
                if str(r.get("chosen", "")).strip() and str(r.get("rejected", "")).strip()
            ]
            return pairs, "preference_native"

        if not auto_generate_rejected:
            raise ValueError(
                f"{training_type.upper()} auto-tune requires preference data but the dataset "
                f"is SFT-style and auto_generate_rejected is disabled."
            )

        # SFT → synchronously pre-generate rejected for the probe subset, cache once.
        _os.makedirs(output_dir, exist_ok=True)
        cache_path = _os.path.join(output_dir, "probe_preference.jsonl")
        if _os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = [_json.loads(l) for l in f if l.strip()]
                if len(cached) >= len(recs) * 0.8:  # reuse if sufficiently covered
                    self._put({"type": "log",
                               "line": f"[auto-tune] reusing cached probe preference "
                                       f"dataset ({len(cached)} samples) -> {cache_path}"})
                    return cached[: len(recs) if probe_samples > 0 else len(cached)], "preference_cached"
            except Exception:
                pass

        self._put({"type": "log",
                   "line": f"[auto-tune] generating {len(recs)} rejected samples for probe..."})
        from core.rejection_generator import RejectionGenerator
        prompts = sft_to_preference_prompts(recs, template, think_mode)
        generator = RejectionGenerator(
            model_path=model_id,
            backend="mlx",  # subprocess auto-falls-back
            max_tokens=rejection_max_tokens,
            temperature=rejection_temperature,
            log_fn=lambda m: self._put({"type": "log", "line": m}),
        )
        out_pairs = []
        try:
            generator.start()
            for i, item in enumerate(prompts):
                if self._stop_event.is_set():
                    break
                try:
                    rj = generator.generate(item["prompt"])
                except Exception as e:
                    self._put({"type": "log", "line": f"[auto-tune] gen err {i}: {e}"})
                    continue
                out_pairs.append({"prompt": item["prompt"],
                                  "chosen": item["chosen"], "rejected": rj})
                if (i + 1) % 10 == 0:
                    self._put({"type": "log",
                               "line": f"[auto-tune] rejected gen {i+1}/{len(prompts)}"})
        finally:
            generator.stop()
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                for rec in out_pairs:
                    f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            self._put({"type": "log",
                       "line": f"[auto-tune] cached probe preference -> {cache_path}"})
        except Exception:
            pass
        return out_pairs, "preference_cached"

    # ── Model loading ─────────────────────────────────────────────

    def _load_base_model(self, model_id: str, load_in_4bit: bool, max_seq_length: int):
        """加载不带 LoRA 的基础模型（作为所有 trial 的起点）。"""
        import importlib.util
        import torch

        cuda = torch.cuda.is_available()
        mps = False
        try:
            mps = not cuda and torch.backends.mps.is_available()
        except Exception:
            pass

        if cuda and importlib.util.find_spec("unsloth") is not None:
            backend = "unsloth"
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
        elif cuda:
            backend = "hf_cuda"
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        elif importlib.util.find_spec("mlx_tune") is not None:
            # Apple Silicon MLX backend
            backend = "mlx"
            from mlx_tune import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=False,
            )
        elif mps:
            backend = "hf_mps"
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
                device_map={"": "mps"}, trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        else:
            backend = "hf_cpu"
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32,
                device_map="cpu", trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer, backend

    # ── Probe trial ───────────────────────────────────────────────

    def _run_probe_trial(
        self,
        base_model, tokenizer, backend: str,
        train_records: list,
        params: dict,
        max_seq_length: int,
        base_lora_dropout: float,
        target_modules: list,
        probe_steps: int,
        output_dir: str,
        trial_num: int,
        trial: "optuna.Trial",
        seed: int,
        training_type: str = "sft",
        data_flavor: str = "sft",
        beta: float = 0.1,
    ) -> float:
        """在探针数据集上运行 probe_steps 步，返回训练损失。"""
        import torch
        from datasets import Dataset as HFDataset

        lora_r = params["lora_r"]
        lora_alpha = params["lora_alpha"]
        lr = params["learning_rate"]
        batch_size = params["per_device_train_batch_size"]
        grad_accum = params.get("gradient_accumulation_steps", 4)
        warmup_ratio = params["warmup_ratio"]
        scheduler = params["lr_scheduler_type"]

        trial_output_dir = os.path.join(output_dir, f"trial_{trial_num}")
        os.makedirs(trial_output_dir, exist_ok=True)

        # ── MLX backend probe trial ───────────────────────────────
        if backend == "mlx":
            return self._run_mlx_probe_trial(
                base_model, tokenizer, train_records, params,
                max_seq_length, target_modules, probe_steps,
                trial_output_dir, trial_num, trial, seed,
                training_type=training_type, data_flavor=data_flavor, beta=beta,
                aggressive_memory_save=aggressive_memory_save,
            )

        train_ds = HFDataset.from_list(train_records)

        # ── 应用 LoRA ────────────────────────────────────────────
        if backend == "unsloth":
            from unsloth import FastLanguageModel
            model_with_lora = FastLanguageModel.get_peft_model(
                base_model,
                r=lora_r,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=base_lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=seed,
            )
        else:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=base_lora_dropout,
                target_modules=target_modules, bias="none", task_type=TaskType.CAUSAL_LM,
            )
            model_with_lora = get_peft_model(base_model, lora_cfg)
            model_with_lora.enable_input_require_grads()

        # ── TrainingArguments ─────────────────────────────────────
        from transformers import TrainingArguments
        use_bf16 = backend in ("hf_mps",) or (backend != "hf_cpu" and _bf16_supported())
        use_fp16 = backend == "hf_cuda" and not use_bf16

        if backend in ("unsloth", "hf_cuda"):
            optim = "adamw_8bit" if _bnb_available() else "adamw_torch"
        else:
            optim = "adamw_torch"

        training_args = TrainingArguments(
            output_dir=trial_output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            max_steps=probe_steps,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=scheduler,
            fp16=use_fp16, bf16=use_bf16,
            optim=optim,
            logging_steps=max(1, probe_steps // 10),
            save_steps=99999,
            seed=seed,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
        )

        # Optuna 中间值回调（支持 MedianPruner）
        step_losses: list = []

        from transformers import TrainerCallback
        class _PruneCallback(TrainerCallback):
            def on_log(self_, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    loss_val = logs["loss"]
                    step_losses.append(loss_val)
                    trial.report(loss_val, step=state.global_step)
                    if trial.should_prune():
                        control.should_training_stop = True

        # ── Trainer dispatch (SFT / DPO / ORPO) ───────────────────
        if training_type == "sft":
            from trl import SFTTrainer
            trainer = SFTTrainer(
                model=model_with_lora,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                packing=False,
                args=training_args,
                callbacks=[_PruneCallback()],
            )
        elif training_type == "orpo":
            from trl import ORPOTrainer, ORPOConfig
            orpo_batch = max(1, batch_size // 2) if aggressive_memory_save else batch_size
            orpo_accum = grad_accum * 2 if aggressive_memory_save else grad_accum
            orpo_max_len = min(max_seq_length, 1024) if aggressive_memory_save else max_seq_length
            orpo_args = ORPOConfig(
                per_device_train_batch_size=orpo_batch,
                gradient_accumulation_steps=orpo_accum,
                max_steps=probe_steps,
                learning_rate=lr,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=scheduler,
                fp16=use_fp16, bf16=use_bf16,
                optim=optim,
                logging_steps=max(1, probe_steps // 10),
                save_steps=99999,
                seed=seed,
                dataloader_num_workers=0,
                remove_unused_columns=True,
                output_dir=trial_output_dir,
                beta=beta,
                max_length=orpo_max_len,
                max_prompt_length=orpo_max_len // 2,
                report_to="none",
            )
            trainer = ORPOTrainer(
                model=model_with_lora,
                args=orpo_args,
                train_dataset=train_ds,
                tokenizer=tokenizer,
                callbacks=[_PruneCallback()],
            )
        elif training_type == "dpo":
            from trl import DPOTrainer
            dpo_max_len = min(max_seq_length, 1024) if aggressive_memory_save else max_seq_length
            trainer = DPOTrainer(
                model=model_with_lora,
                ref_model=None,
                args=training_args,
                beta=beta,
                train_dataset=train_ds,
                tokenizer=tokenizer,
                max_length=dpo_max_len,
                callbacks=[_PruneCallback()],
            )
        else:
            raise ValueError(f"unsupported training_type: {training_type}")

        train_result = trainer.train()
        final_loss = train_result.training_loss

        # 清理 LoRA 权重（恢复基础模型状态）
        try:
            model_with_lora.disable_adapter_layers()
        except Exception:
            pass

        del model_with_lora
        _gc_collect()

        # 若被 pruner 停止，用 median 作为 loss
        if trial.should_prune() or self._stop_event.is_set():
            if step_losses:
                final_loss = sum(step_losses) / len(step_losses)
            raise optuna.exceptions.TrialPruned()

        return final_loss

    def _run_mlx_probe_trial(
        self,
        base_model, tokenizer, train_records: list,
        params: dict, max_seq_length: int, target_modules: list,
        probe_steps: int, trial_output_dir: str, trial_num: int,
        trial, seed: int,
        training_type: str = "sft",
        data_flavor: str = "sft",
        beta: float = 0.1,
        aggressive_memory_save: bool = False,
    ) -> float:
        """MLX backend 探针训练：SFT / DPO / ORPO。"""
        # Apply memory-saving mlx_tune monkey-patches (logsumexp + clear_cache)
        try:
            from core.mlx_patches import apply_mlx_tune_patches
            apply_mlx_tune_patches(
                log_fn=lambda m: self._put({"type": "log", "line": m})
            )
        except Exception:
            pass
        from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig

        lora_r = params["lora_r"]
        lora_alpha = params["lora_alpha"]
        lr = params["learning_rate"]
        batch_size = params["per_device_train_batch_size"]
        warmup_ratio = params["warmup_ratio"]
        scheduler = params["lr_scheduler_type"]

        # 应用 LoRA 到基础模型（mlx_tune 的 get_peft_model 是延迟的，需要重新应用）
        model_with_lora = FastLanguageModel.get_peft_model(
            base_model,
            r=lora_r,
            target_modules=target_modules or None,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=seed,
        )

        step_losses: list = []

        # 通过 monkey-patch 捕获训练指标
        try:
            import mlx_tune.sft_trainer as _sft_mod
            from mlx_lm.tuner.callbacks import TrainingCallback

            class _MLXPruneCallback(TrainingCallback):
                def on_train_loss_report(self_, info: dict):
                    loss_val = info.get("train_loss", float("inf"))
                    step = info.get("iteration", 0)
                    step_losses.append(loss_val)
                    trial.report(loss_val, step=step)

            _orig = _sft_mod.mlx_train
            _cb = _MLXPruneCallback()

            def _patched(*a, **kw):
                kw["training_callback"] = _cb
                return _orig(*a, **kw)

            _sft_mod.mlx_train = _patched

            if training_type == "sft":
                args = SFTConfig(
                    output_dir=trial_output_dir,
                    per_device_train_batch_size=batch_size,
                    learning_rate=lr,
                    lr_scheduler_type=scheduler,
                    warmup_ratio=warmup_ratio,
                    max_steps=probe_steps,
                    logging_steps=max(1, probe_steps // 10),
                    save_steps=99999,
                    max_seq_length=max_seq_length,
                    use_native_training=True,
                )
                trainer = SFTTrainer(
                    model=model_with_lora,
                    tokenizer=tokenizer,
                    train_dataset=train_records,
                    args=args,
                    dataset_text_field="text",
                )
            elif training_type == "orpo":
                from mlx_tune import ORPOTrainer, ORPOConfig
                pref_max_seq = (min(max_seq_length, 1024)
                                if aggressive_memory_save else max_seq_length)
                pref_batch = (max(1, batch_size // 2)
                              if aggressive_memory_save else batch_size)
                args = ORPOConfig(
                    beta=beta,
                    output_dir=trial_output_dir,
                    learning_rate=lr,
                    per_device_train_batch_size=pref_batch,
                    num_train_epochs=1,
                    logging_steps=max(1, probe_steps // 10),
                    save_steps=99999,
                    max_seq_length=pref_max_seq,
                    max_prompt_length=pref_max_seq // 2,
                )
                trainer = ORPOTrainer(
                    model=model_with_lora,
                    train_dataset=train_records,
                    tokenizer=tokenizer,
                    args=args,
                )
            elif training_type == "dpo":
                from mlx_tune import DPOTrainer, DPOConfig
                pref_max_seq = (min(max_seq_length, 1024)
                                if aggressive_memory_save else max_seq_length)
                pref_batch = (max(1, batch_size // 2)
                              if aggressive_memory_save else batch_size)
                args = DPOConfig(
                    beta=beta,
                    output_dir=trial_output_dir,
                    learning_rate=lr,
                    per_device_train_batch_size=pref_batch,
                    num_train_epochs=1,
                    logging_steps=max(1, probe_steps // 10),
                    save_steps=99999,
                    max_seq_length=pref_max_seq,
                    max_prompt_length=pref_max_seq // 2,
                )
                trainer = DPOTrainer(
                    model=model_with_lora,
                    train_dataset=train_records,
                    ref_model=None,
                    tokenizer=tokenizer,
                    args=args,
                )
            else:
                raise ValueError(f"unsupported training_type: {training_type}")
            trainer.train()

        finally:
            _sft_mod.mlx_train = _orig
            # 关键：用 mlx_lm.remove_lora_layers 将 LoRALinear 还原为 Linear，
            # 否则下一个 trial 调用 _apply_lora 时会报 "Can't convert LoRALinear to LoRA"
            try:
                from mlx_lm.tuner.utils import remove_lora_layers
                actual_model = base_model.model if hasattr(base_model, "model") else base_model
                remove_lora_layers(actual_model)
                base_model._lora_applied = False
                base_model.lora_enabled = False
            except Exception:
                pass

        if trial.should_prune() or self._stop_event.is_set():
            if step_losses:
                return sum(step_losses) / len(step_losses)
            raise optuna.exceptions.TrialPruned()

        return step_losses[-1] if step_losses else float("inf")

    # ── Parameter sampling ────────────────────────────────────────

    def _sample_params(
        self,
        trial: "optuna.Trial",
        ss: SearchSpaceConfig,
        base_lora_r: int,
        base_lora_alpha: int,
    ) -> dict:
        params = {}

        if ss.tune_lora_r:
            params["lora_r"] = trial.suggest_categorical("lora_r", ss.lora_r_choices)
        else:
            params["lora_r"] = base_lora_r

        if ss.tune_lora_alpha:
            mult = trial.suggest_categorical("alpha_multiplier", ss.alpha_multiplier_choices)
            params["lora_alpha"] = params["lora_r"] * mult
        else:
            params["lora_alpha"] = base_lora_alpha

        if ss.tune_lr:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", ss.lr_min, ss.lr_max, log=True
            )
        else:
            params["learning_rate"] = 2e-4

        if ss.tune_batch:
            params["per_device_train_batch_size"] = trial.suggest_categorical(
                "per_device_train_batch_size", ss.batch_size_choices
            )
        else:
            params["per_device_train_batch_size"] = 4

        if ss.tune_grad_accum:
            params["gradient_accumulation_steps"] = trial.suggest_categorical(
                "gradient_accumulation_steps", ss.grad_accum_choices
            )
        else:
            params["gradient_accumulation_steps"] = 4

        if ss.tune_warmup:
            params["warmup_ratio"] = trial.suggest_float(
                "warmup_ratio", ss.warmup_min, ss.warmup_max
            )
        else:
            params["warmup_ratio"] = 0.05

        if ss.tune_scheduler:
            params["lr_scheduler_type"] = trial.suggest_categorical(
                "lr_scheduler_type", ss.scheduler_choices
            )
        else:
            params["lr_scheduler_type"] = "cosine"

        return params

    def _make_stop_callback(self):
        stop_event = self._stop_event
        def _cb(study, trial):
            if stop_event.is_set():
                study.stop()
        return _cb

    # ── Best params → TrainingConfig dict ────────────────────────

    def get_best_config_patch(self) -> Dict[str, Any]:
        """返回可直接 apply 到 TrainingConfig 的最优参数字典。"""
        if self.result is None or not self.result.best_params:
            return {}
        p = self.result.best_params
        patch = {}
        if "lora_r" in p:
            patch["lora_r"] = p["lora_r"]
        if "lora_alpha" in p:
            patch["lora_alpha"] = p["lora_alpha"]
        elif "lora_r" in p and "alpha_multiplier" in p:
            patch["lora_alpha"] = p["lora_r"] * p["alpha_multiplier"]
        if "learning_rate" in p:
            patch["learning_rate"] = p["learning_rate"]
        if "per_device_train_batch_size" in p:
            patch["per_device_train_batch_size"] = p["per_device_train_batch_size"]
        if "gradient_accumulation_steps" in p:
            patch["gradient_accumulation_steps"] = p["gradient_accumulation_steps"]
        if "warmup_ratio" in p:
            patch["warmup_ratio"] = p["warmup_ratio"]
        if "lr_scheduler_type" in p:
            patch["lr_scheduler_type"] = p["lr_scheduler_type"]
        return patch

    # ── Visualization data ────────────────────────────────────────

    def get_history_df(self):
        """返回用于绘制优化历史的 Pandas DataFrame。"""
        try:
            import pandas as pd
        except ImportError:
            return None
        if not self.trials:
            return None
        rows = []
        best = float("inf")
        for t in self.trials:
            if t.status == "complete" and t.train_loss < best:
                best = t.train_loss
            rows.append({
                "trial": t.trial_number + 1,
                "loss": t.train_loss if t.train_loss < float("inf") else None,
                "best_loss": best if best < float("inf") else None,
                "status": t.status,
            })
        return pd.DataFrame(rows)

    def get_importance_df(self):
        """返回参数重要性 DataFrame。"""
        try:
            import pandas as pd
        except ImportError:
            return None
        if self.result is None or not self.result.param_importances:
            return None
        items = sorted(self.result.param_importances.items(), key=lambda x: -x[1])
        return pd.DataFrame({"参数": [k for k, _ in items],
                              "重要性": [round(v, 4) for _, v in items]})


# ────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────

def _fmt_params(p: dict) -> str:
    """紧凑格式化参数字典供日志输出。"""
    parts = []
    for k, v in p.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.2e}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _gc_collect():
    import gc as _gc
    _gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _bf16_supported() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _bnb_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("bitsandbytes") is not None
