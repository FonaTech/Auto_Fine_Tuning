"""
ui/tabs/auto_tune_tab.py
Tab 7: AI hyperparameter auto-tuning (Bayesian search)
"""

import time
import gradio as gr
from core.auto_tuner import SearchSpaceConfig, OPTUNA_AVAILABLE
from core.session_manager import session_manager
from ui.i18n import tr, register_translatable, t


def build_auto_tune_tab(
    dataset_state: gr.State,
    model_state: gr.State,
    config_components: dict,
) -> None:
    with gr.Tab(tr("tab.autotune"), elem_classes="workspace-tab", render_children=True) as tab:
        register_translatable(tab, label_key="tab.autotune")
        if not OPTUNA_AVAILABLE:
            gr.Markdown(tr("autotune.optuna_missing"))
            return

        title_md = gr.Markdown(tr("autotune.title"))
        register_translatable(title_md, label_key="autotune.title")

        # ── 搜索空间配置 ──────────────────────────────────────────
        with gr.Accordion(tr("autotune.accordion.search"), open=True) as acc_search:
            register_translatable(acc_search, label_key="autotune.accordion.search")
            search_hint = gr.Markdown(tr("autotune.search.hint"))
            register_translatable(search_hint, label_key="autotune.search.hint")
            with gr.Row():
                with gr.Column():
                    tune_lora_r = gr.Checkbox(value=True, label="LoRA Rank (r)")
                    lora_r_choices = gr.CheckboxGroup(
                        choices=[4, 8, 16, 32, 64], value=[4, 8, 16, 32],
                        label=tr("tune.lora_choices"),
                    )
                    register_translatable(lora_r_choices, label_key="tune.lora_choices")
                with gr.Column():
                    tune_lr = gr.Checkbox(value=True, label=tr("tune.lr"))
                    register_translatable(tune_lr, label_key="tune.lr")
                    with gr.Row():
                        lr_min = gr.Textbox(value="5e-5", label=tr("tune.min"))
                        lr_max = gr.Textbox(value="5e-4", label=tr("tune.max"))
                with gr.Column():
                    tune_batch = gr.Checkbox(value=True, label=tr("tune.batch"))
                    register_translatable(tune_batch, label_key="tune.batch")
                    batch_choices = gr.CheckboxGroup(
                        choices=[1, 2, 4, 8], value=[1, 2, 4],
                        label=tr("tune.batch_choices"),
                    )
                    register_translatable(batch_choices, label_key="tune.batch_choices")

            with gr.Row():
                with gr.Column():
                    tune_warmup = gr.Checkbox(value=True, label=tr("tune.warmup"))
                    register_translatable(tune_warmup, label_key="tune.warmup")
                    with gr.Row():
                        warmup_min = gr.Number(value=0.01, label=tr("tune.min"), precision=None)
                        warmup_max = gr.Number(value=0.15, label=tr("tune.max"), precision=None)
                with gr.Column():
                    tune_scheduler = gr.Checkbox(value=True, label=tr("tune.scheduler"))
                    register_translatable(tune_scheduler, label_key="tune.scheduler")
                    scheduler_choices = gr.CheckboxGroup(
                        choices=["cosine", "linear", "constant"],
                        value=["cosine", "linear"],
                        label=tr("tune.scheduler_choices"),
                    )
                    register_translatable(scheduler_choices, label_key="tune.scheduler_choices")
                with gr.Column():
                    tune_grad_accum = gr.Checkbox(value=False, label=tr("tune.grad_accum"))
                    register_translatable(tune_grad_accum, label_key="tune.grad_accum")
                    grad_accum_choices = gr.CheckboxGroup(
                        choices=[1, 2, 4, 8], value=[2, 4],
                        label=tr("tune.grad_accum_choices"),
                    )
                    register_translatable(grad_accum_choices, label_key="tune.grad_accum_choices")

        # ── 优化参数 ───────────────────────────────────────────────
        with gr.Accordion(tr("autotune.accordion.params"), open=True) as acc_params:
            register_translatable(acc_params, label_key="autotune.accordion.params")
            with gr.Row():
                n_trials = gr.Slider(
                    minimum=5, maximum=100, value=20, step=5,
                    label=tr("tune.n_trials"),
                    info=tr("tune.n_trials.info"),
                )
                register_translatable(n_trials, label_key="tune.n_trials", info_key="tune.n_trials.info")
                probe_steps = gr.Slider(
                    minimum=20, maximum=300, value=80, step=10,
                    label=tr("tune.probe_steps"),
                    info=tr("tune.probe_steps.info"),
                )
                register_translatable(probe_steps, label_key="tune.probe_steps", info_key="tune.probe_steps.info")
                probe_samples = gr.Slider(
                    minimum=100, maximum=2000, value=300, step=100,
                    label=tr("tune.probe_samples"),
                    info=tr("tune.probe_samples.info"),
                )
                register_translatable(probe_samples, label_key="tune.probe_samples", info_key="tune.probe_samples.info")
            with gr.Row():
                probe_max_seq = gr.Dropdown(
                    choices=[256, 512, 1024, 2048], value=512,
                    label=tr("tune.probe_max_seq"),
                )
                register_translatable(probe_max_seq, label_key="tune.probe_max_seq")
                auto_tune_output_dir = gr.Textbox(
                    value="auto_tune_cache",
                    label=tr("tune.output_dir"),
                )
                register_translatable(auto_tune_output_dir, label_key="tune.output_dir")

        # ── Control buttons ────────────────────────────────────────
        with gr.Row():
            start_tune_btn = gr.Button(tr("autotune.start"), variant="primary", scale=2)
            register_translatable(start_tune_btn, label_key="autotune.start")
            stop_tune_btn = gr.Button(tr("autotune.stop"), variant="stop", scale=1, interactive=False)
            register_translatable(stop_tune_btn, label_key="autotune.stop")

        # ── Status ─────────────────────────────────────────────────
        tune_status_html = gr.HTML(value=_status_html("idle"))
        tune_progress = gr.Slider(
            minimum=0, maximum=100, value=0,
            label=tr("autotune.progress"), interactive=False,
        )
        register_translatable(tune_progress, label_key="autotune.progress")
        tune_progress_text = gr.Textbox(
            value="", interactive=False, label="", show_label=False,
        )

        # ── Live charts ────────────────────────────────────────────
        with gr.Row():
            history_plot = gr.LinePlot(
                value=None,
                x="trial", y="loss",
                title=tr("autotune.history_plot"),
                x_title="Trial #",
                y_title="Training Loss",
                height=280,
            )
            best_history_plot = gr.LinePlot(
                value=None,
                x="trial", y="best_loss",
                title=tr("autotune.best_plot"),
                x_title="Trial #",
                y_title="Best Loss",
                height=280,
            )

        # ── Trial history table ────────────────────────────────────
        trial_table = gr.Dataframe(
            headers=["Trial#", "loss", "lora_r", "lr", "batch", "scheduler", "status", "time(s)"],
            label=tr("autotune.trials_table"),
            interactive=False,
            wrap=True,
        )
        register_translatable(trial_table, label_key="autotune.trials_table")

        # ── Parameter importance ───────────────────────────────────
        with gr.Row():
            importance_plot = gr.BarPlot(
                value=None,
                x="param", y=tr("autotune.importance_label"),
                title=tr("autotune.importance_plot"),
                x_title="Hyperparameter",
                y_title=tr("autotune.importance_label"),
                height=260,
            )
            best_params_box = gr.Textbox(
                label=tr("autotune.best_params"),
                lines=10,
                interactive=False,
                value=tr("autotune.best_params_placeholder"),
            )
            register_translatable(best_params_box, label_key="autotune.best_params")

        # ── Apply best params ──────────────────────────────────────
        with gr.Row():
            apply_best_btn = gr.Button(
                tr("autotune.apply_btn"),
                variant="primary",
                interactive=False,
            )
            register_translatable(apply_best_btn, label_key="autotune.apply_btn")
            apply_status = gr.Textbox(value="", label="", interactive=False)

        tune_log = gr.Textbox(
            value="", label=tr("autotune.log"), lines=10, interactive=False,
            buttons=["copy"],
        )
        register_translatable(tune_log, label_key="autotune.log")

        # ── Event handlers ────────────────────────────────────────────

        def start_and_stream(
            dataset_info, model_info,
            do_lora_r, lora_r_vals,
            do_lr, lr_min_v, lr_max_v,
            do_batch, batch_vals,
            do_warmup, warmup_min_v, warmup_max_v,
            do_scheduler, sched_vals,
            do_grad_accum, grad_accum_vals,
            n_trials_v, probe_steps_v, probe_samples_v,
            probe_max_seq_v, output_dir_v,
            request: gr.Request,
        ):
            import pandas as pd
            session_id = getattr(request, "session_hash", "__singleton__")
            auto_tuner = session_manager.get_or_create(session_id).auto_tuner

            # Validation
            if not dataset_info or not dataset_info.get("path"):
                yield (
                    _status_html("error"),
                    gr.update(interactive=True), gr.update(interactive=False),
                    0, t("autotune.err.no_dataset"),
                    None, None, [], t("autotune.err.no_dataset"), None, "", gr.update(interactive=False), "",
                )
                return
            if not model_info or not model_info.get("model_id"):
                yield (
                    _status_html("error"),
                    gr.update(interactive=True), gr.update(interactive=False),
                    0, t("autotune.err.no_model"),
                    None, None, [], t("autotune.err.no_model"), None, "", gr.update(interactive=False), "",
                )
                return

            # 构建搜索空间
            ss = SearchSpaceConfig(
                tune_lora_r=bool(do_lora_r),
                lora_r_choices=sorted([int(x) for x in lora_r_vals]) if lora_r_vals else [8, 16],
                tune_lr=bool(do_lr),
                lr_min=float(lr_min_v or 5e-5),
                lr_max=float(lr_max_v or 5e-4),
                tune_batch=bool(do_batch),
                batch_size_choices=sorted([int(x) for x in batch_vals]) if batch_vals else [2, 4],
                tune_warmup=bool(do_warmup),
                warmup_min=float(warmup_min_v or 0.01),
                warmup_max=float(warmup_max_v or 0.15),
                tune_scheduler=bool(do_scheduler),
                scheduler_choices=list(sched_vals) if sched_vals else ["cosine", "linear"],
                tune_grad_accum=bool(do_grad_accum),
                grad_accum_choices=sorted([int(x) for x in grad_accum_vals]) if grad_accum_vals else [2, 4],
            )

            # 从主配置获取基础 LoRA 参数
            try:
                base_r = int(config_components["lora_r"].value)
                base_alpha = int(config_components["lora_alpha"].value)
                base_dropout = float(config_components["lora_dropout"].value)
                base_targets = list(config_components["target_modules"].value or [])
            except Exception:
                base_r, base_alpha, base_dropout = 16, 32, 0.0
                base_targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

            # Read training_type and rejection settings from the shared config so
            # auto-tune runs the same training regime the user will eventually launch.
            try:
                tt_val = str(config_components["training_type"].value or "sft").lower()
                auto_gen = bool(config_components.get("auto_generate_rejected",
                                                     type("_", (), {"value": True})()).value)
                rej_tokens = int(config_components.get("rejection_max_tokens",
                                                      type("_", (), {"value": 512})()).value)
                rej_temp = float(config_components.get("rejection_temperature",
                                                      type("_", (), {"value": 0.7})()).value)
                beta_v = float(config_components.get("beta",
                                                    type("_", (), {"value": 0.1})()).value)
                ams = bool(config_components.get("aggressive_memory_save",
                                                 type("_", (), {"value": False})()).value)
            except Exception:
                tt_val, auto_gen, rej_tokens, rej_temp, beta_v, ams = \
                    "sft", True, 512, 0.7, 0.1, False

            auto_tuner.start(
                model_id=model_info["model_id"],
                dataset_path=dataset_info["path"],
                train_ratio=float(dataset_info.get("train_ratio", 0.95)),
                prompt_template=dataset_info.get("template", "alpaca"),
                think_mode=dataset_info.get("think_mode", "keep"),
                search_space=ss,
                n_trials=int(n_trials_v),
                probe_steps=int(probe_steps_v),
                probe_samples=int(probe_samples_v),
                load_in_4bit=model_info.get("load_in_4bit", True),
                max_seq_length=int(probe_max_seq_v),
                base_lora_r=base_r,
                base_lora_alpha=base_alpha,
                base_lora_dropout=base_dropout,
                target_modules=base_targets,
                output_dir=str(output_dir_v) or "auto_tune_cache",
                training_type=tt_val,
                auto_generate_rejected=auto_gen,
                rejection_max_tokens=rej_tokens,
                rejection_temperature=rej_temp,
                beta=beta_v,
                aggressive_memory_save=ams,
            )

            yield (
                _status_html("loading"),
                gr.update(interactive=False), gr.update(interactive=True),
                0, t("autotune.status.loading"),
                None, None, [], t("autotune.status.running"), None, "", gr.update(interactive=False), "",
            )

            # Polling loop
            log_lines = []
            n_total = int(n_trials_v)
            n_done = 0

            while auto_tuner.is_running() or auto_tuner.status in ("loading", "running"):
                events = auto_tuner.poll()
                for ev in events:
                    etype = ev.get("type")
                    if etype == "log":
                        log_lines.append(ev.get("line", ""))
                        log_lines = log_lines[-200:]
                    elif etype == "trial_end" and ev.get("status") == "complete":
                        n_done += 1
                    elif etype == "status":
                        auto_tuner.status = ev.get("status", auto_tuner.status)

                pct = min(100, n_done / n_total * 100) if n_total > 0 else 0
                prog_text = f"Trial {n_done}/{n_total} ({pct:.0f}%)"

                hist_df = auto_tuner.get_history_df()
                loss_df = hist_df[["trial", "loss"]].dropna() if hist_df is not None else None
                best_df = hist_df[["trial", "best_loss"]].dropna() if hist_df is not None else None

                table_rows = _build_trial_table(auto_tuner.trials)

                yield (
                    _status_html(auto_tuner.status),
                    gr.update(interactive=(auto_tuner.status in ("finished", "error", "stopped"))),
                    gr.update(interactive=(auto_tuner.status in ("loading", "running"))),
                    pct, prog_text,
                    loss_df, best_df,
                    table_rows,
                    t("autotune.status.running"),
                    None,
                    "\n".join(log_lines[-100:]),
                    gr.update(interactive=False),
                    "",
                )

                if auto_tuner.status in ("finished", "error", "stopped"):
                    break
                time.sleep(2)

            # Final output
            events = auto_tuner.poll()
            for ev in events:
                if ev.get("type") == "log":
                    log_lines.append(ev.get("line", ""))

            hist_df = auto_tuner.get_history_df()
            loss_df = hist_df[["trial", "loss"]].dropna() if hist_df is not None else None
            best_df = hist_df[["trial", "best_loss"]].dropna() if hist_df is not None else None
            importance_df = auto_tuner.get_importance_df()
            table_rows = _build_trial_table(auto_tuner.trials)

            best_text = t("autotune.err.no_result")
            if auto_tuner.result and auto_tuner.result.best_params:
                r = auto_tuner.result
                lines = [
                    f"Best Loss: {r.best_loss:.4f}",
                    f"Completed: {r.n_completed} trials | Pruned: {r.n_pruned} | Failed: {r.n_failed}",
                    f"Time: {r.elapsed_s:.0f}s",
                    "",
                    "── Best Parameters ──",
                ]
                for k, v in r.best_params.items():
                    if isinstance(v, float):
                        lines.append(f"  {k}: {v:.2e}")
                    else:
                        lines.append(f"  {k}: {v}")
                best_text = "\n".join(lines)

            can_apply = auto_tuner.result is not None and bool(auto_tuner.result.best_params)

            yield (
                _status_html(auto_tuner.status),
                gr.update(interactive=True),
                gr.update(interactive=False),
                100 if auto_tuner.status == "finished" else (n_done / n_total * 100 if n_total else 0),
                f"Trial {n_done}/{n_total}",
                loss_df, best_df,
                table_rows,
                best_text,
                importance_df,
                "\n".join(log_lines[-150:]),
                gr.update(interactive=can_apply),
                "",
            )

        # 所有输出组件
        _stream_outputs = [
            tune_status_html, start_tune_btn, stop_tune_btn,
            tune_progress, tune_progress_text,
            history_plot, best_history_plot,
            trial_table, best_params_box,
            importance_plot, tune_log,
            apply_best_btn, apply_status,
        ]

        start_tune_btn.click(
            fn=start_and_stream,
            inputs=[
                dataset_state, model_state,
                tune_lora_r, lora_r_choices,
                tune_lr, lr_min, lr_max,
                tune_batch, batch_choices,
                tune_warmup, warmup_min, warmup_max,
                tune_scheduler, scheduler_choices,
                tune_grad_accum, grad_accum_choices,
                n_trials, probe_steps, probe_samples,
                probe_max_seq, auto_tune_output_dir,
            ],
            outputs=_stream_outputs,
            concurrency_limit=None,
        )

        def on_stop(request: gr.Request):
            session_id = getattr(request, "session_hash", "__singleton__")
            auto_tuner = session_manager.get_or_create(session_id).auto_tuner
            auto_tuner.stop()
            return (
                _status_html("stopped"),
                gr.update(interactive=True),
                gr.update(interactive=False),
            )

        stop_tune_btn.click(
            fn=on_stop,
            inputs=[],
            outputs=[tune_status_html, start_tune_btn, stop_tune_btn],
        )

        # ── Apply best params to training config ──────────────────
        _APPLY_KEYS = [
            "lora_r", "lora_alpha", "learning_rate",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "warmup_ratio", "lr_scheduler_type",
        ]
        _apply_outputs = [config_components[k] for k in _APPLY_KEYS] + [apply_status]

        def apply_best_params(request: gr.Request):
            session_id = getattr(request, "session_hash", "__singleton__")
            auto_tuner = session_manager.get_or_create(session_id).auto_tuner
            if auto_tuner.result is None:
                return [gr.update()] * len(_APPLY_KEYS) + [t("autotune.err.no_result")]
            patch = auto_tuner.get_best_config_patch()
            if not patch:
                return [gr.update()] * len(_APPLY_KEYS) + [t("autotune.err.no_best")]
            updates = []
            for k in _APPLY_KEYS:
                if k in patch:
                    updates.append(gr.update(value=str(patch[k]) if k == "learning_rate" else patch[k]))
                else:
                    updates.append(gr.update())
            applied = [f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}" for k, v in patch.items()]
            return updates + [f"✅ Applied: {', '.join(applied)}"]

        apply_best_btn.click(
            fn=apply_best_params,
            inputs=[],
            outputs=_apply_outputs,
        )


# ── Helpers ────────────────────────────────────────────────────────

def _build_trial_table(trials) -> list:
    rows = []
    for t in trials:
        p = t.params
        rows.append([
            t.trial_number + 1,
            f"{t.train_loss:.4f}" if t.train_loss < float("inf") else "—",
            p.get("lora_r", "—"),
            f"{p.get('learning_rate', 0):.2e}" if "learning_rate" in p else "—",
            p.get("per_device_train_batch_size", "—"),
            p.get("lr_scheduler_type", "—"),
            t.status,
            f"{t.duration_s:.0f}s",
        ])
    return rows


def _status_html(status: str) -> str:
    from ui.i18n import t
    colors = {
        "idle":     ("#94a3b8", "⚪"),
        "loading":  ("#f59e0b", "🟡"),
        "running":  ("#22c55e", "🟢"),
        "stopped":  ("#64748b", "⏹"),
        "finished": ("#3b82f6", "✅"),
        "error":    ("#ef4444", "❌"),
    }
    color, icon = colors.get(status, ("#94a3b8", "❓"))
    label = t(f"autotune.status.{status}")
    return (
        f'<div style="padding:10px 16px;border-radius:8px;background:{color}20;'
        f'border:2px solid {color};font-size:15px;font-weight:600;">'
        f'{icon} {label}'
        f'</div>'
    )


def _status_label(status: str) -> str:
    from ui.i18n import t
    return t(f"autotune.status.{status}")
