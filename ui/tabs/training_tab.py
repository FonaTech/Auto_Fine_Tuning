"""
ui/tabs/training_tab.py
Tab 5: Training monitor — start/stop, live loss curve, log stream, progress & ETA.
Uses Gradio Generator mode for real-time updates (no gr.Timer needed).
"""

import os
import time
import gradio as gr

from core.trainer import TrainingConfig
from core.checkpoint import scan_checkpoints, load_checkpoint_config, configs_compatible, load_training_config_raw
from core.session_manager import session_manager
from ui.i18n import tr, register_translatable, t


def build_training_tab(
    config_components: dict,
    dataset_state: gr.State,
    model_state: gr.State,
) -> gr.State:
    with gr.Tab(tr("tab.training"), elem_classes="workspace-tab", render_children=True) as tab:
        register_translatable(tab, label_key="tab.training")
        title_md = gr.Markdown(tr("training.title"))
        register_translatable(title_md, label_key="training.title")

        # ── Control buttons (top, always visible) ───────────────────
        with gr.Row(elem_classes="training-btn-row"):
            start_btn = gr.Button(tr("training.start"), variant="primary", scale=2)
            register_translatable(start_btn, label_key="training.start")
            stop_btn = gr.Button(tr("training.stop"), variant="secondary", scale=1, interactive=False)
            register_translatable(stop_btn, label_key="training.stop")

        # ── Status & progress ────────────────────────────────────────
        status_html = gr.HTML(value=_status_html("idle"))
        progress_bar = gr.Slider(
            minimum=0, maximum=100, value=0,
            label=tr("training.progress"),
            interactive=False,
        )
        register_translatable(progress_bar, label_key="training.progress")
        progress_text = gr.Textbox(
            value="", interactive=False, label="", show_label=False,
        )

        # ── Charts ───────────────────────────────────────────────────
        with gr.Row():
            loss_plot = gr.LinePlot(
                value=None, x="step", y="loss",
                title="Training Loss", x_title="Step", y_title="Loss", height=280,
            )
            lr_plot = gr.LinePlot(
                value=None, x="step", y="lr",
                title="Learning Rate", x_title="Step", y_title="Learning Rate", height=280,
            )

        # ── GPU status ───────────────────────────────────────────────
        with gr.Row():
            gpu_mem_box = gr.Textbox(value="—", label=tr("training.gpu_mem"), interactive=False, scale=1)
            register_translatable(gpu_mem_box, label_key="training.gpu_mem")
            speed_box = gr.Textbox(value="—", label=tr("training.speed"), interactive=False, scale=1)
            register_translatable(speed_box, label_key="training.speed")
            eta_box = gr.Textbox(value="—", label=tr("training.eta"), interactive=False, scale=1)
            register_translatable(eta_box, label_key="training.eta")

        # ── Log ──────────────────────────────────────────────────────
        log_box = gr.Textbox(
            value="", label=tr("training.log"),
            lines=12, max_lines=20, interactive=False, buttons=["copy"],
        )
        register_translatable(log_box, label_key="training.log")

        # ── Checkpoint list ──────────────────────────────────────────
        checkpoints_box = gr.Textbox(
            value="", label=tr("training.checkpoints"), interactive=False, lines=4,
        )
        register_translatable(checkpoints_box, label_key="training.checkpoints")

        # ── Checkpoint Resume (below main controls) ──────────────────
        with gr.Accordion(tr("training.accordion.resume"), open=False) as acc_resume:
            register_translatable(acc_resume, label_key="training.accordion.resume")
            with gr.Row():
                resume_scan_dir = gr.Textbox(
                    value="outputs", label=tr("training.resume.scan_dir"), scale=2,
                )
                register_translatable(resume_scan_dir, label_key="training.resume.scan_dir")
                resume_scan_btn = gr.Button(tr("training.resume.scan"), variant="secondary", scale=1)
                register_translatable(resume_scan_btn, label_key="training.resume.scan")
            resume_checkpoint_dd = gr.Dropdown(
                choices=[], value=None, label=tr("training.resume.checkpoint"), interactive=True,
                allow_custom_value=True,
            )
            register_translatable(resume_checkpoint_dd, label_key="training.resume.checkpoint")
            # Mirror State: keeps the selected path accessible even inside a closed accordion
            resume_ckpt_path_state = gr.State("")
            resume_compat_status = gr.Textbox(
                value="", label=tr("training.resume.compat"), interactive=False,
            )
            register_translatable(resume_compat_status, label_key="training.resume.compat")
            resume_enabled = gr.Checkbox(value=False, label=tr("training.resume.enable"))
            register_translatable(resume_enabled, label_key="training.resume.enable")
            with gr.Row():
                apply_config_btn = gr.Button(tr("training.resume.apply_config"), variant="secondary")
                register_translatable(apply_config_btn, label_key="training.resume.apply_config")
            apply_config_status = gr.Textbox(
                value="", label=tr("training.resume.apply_config.status"), interactive=False,
            )
            register_translatable(apply_config_status, label_key="training.resume.apply_config.status")

        # ── Config keys ──────────────────────────────────────────────
        _CONFIG_KEYS = [
            "lora_r", "lora_alpha", "lora_dropout", "use_rslora",
            "target_modules", "training_type", "num_epochs",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "learning_rate", "lr_scheduler_type", "warmup_ratio",
            "weight_decay", "max_grad_norm", "beta",
            "load_in_4bit", "use_gradient_checkpointing", "packing",
            "max_seq_length", "neftune_noise_alpha",
            "output_dir", "save_steps", "save_total_limit",
            "logging_steps", "report_to",
        ]
        config_inputs = [config_components[k] for k in _CONFIG_KEYS]

        # ── Event handlers ───────────────────────────────────────────

        def on_resume_scan(scan_dir, model_info, lora_r, lora_alpha, target_modules):
            found = scan_checkpoints(scan_dir.strip() or "outputs")
            if not found:
                return gr.update(choices=[], value=None), t("training.resume.scan") + ": 0"
            labels = [(c.label, c.path) for c in found]
            return gr.update(choices=labels, value=found[-1].path), f"Found {len(found)}"

        def on_resume_select(ckpt_path, model_info, lora_r, lora_alpha, target_modules):
            if not ckpt_path:
                return "", False
            cfg = load_checkpoint_config(ckpt_path)
            if cfg is None:
                return "⚠️ No adapter_config.json or training_config.json found near this checkpoint", False
            source = cfg.get("_source", "adapter_config")
            source_label = "adapter_config.json" if source == "adapter_config" else "training_config.json"
            model_id = (model_info or {}).get("model_id", "")
            modules = list(target_modules) if target_modules else []
            ok, reason = configs_compatible(cfg, model_id, int(lora_r), int(lora_alpha), modules)
            prefix = f"✅ {reason}" if ok else f"❌ {reason}"
            return f"{prefix}  (source: {source_label})", ok

        resume_scan_btn.click(
            fn=on_resume_scan,
            inputs=[resume_scan_dir, model_state,
                    config_components["lora_r"], config_components["lora_alpha"],
                    config_components["target_modules"]],
            outputs=[resume_checkpoint_dd, resume_compat_status],
        )

        def _on_resume_dd_change(ckpt_path, model_info, lora_r, lora_alpha, target_modules):
            """Wraps on_resume_select and also updates the mirror State."""
            compat_text, is_ok = on_resume_select(ckpt_path, model_info, lora_r, lora_alpha, target_modules)
            return compat_text, is_ok, ckpt_path or ""

        resume_checkpoint_dd.change(
            fn=_on_resume_dd_change,
            inputs=[resume_checkpoint_dd, model_state,
                    config_components["lora_r"], config_components["lora_alpha"],
                    config_components["target_modules"]],
            outputs=[resume_compat_status, resume_enabled, resume_ckpt_path_state],
        )

        # ── Apply config from checkpoint to UI ───────────────────────
        _CONFIG_KEYS_FOR_APPLY = [
            "lora_r", "lora_alpha", "lora_dropout", "use_rslora",
            "target_modules", "training_type", "num_epochs",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "learning_rate", "lr_scheduler_type", "warmup_ratio",
            "weight_decay", "max_grad_norm", "beta",
            "load_in_4bit", "use_gradient_checkpointing", "packing",
            "max_seq_length", "neftune_noise_alpha",
            "output_dir", "save_steps", "save_total_limit",
            "logging_steps", "report_to",
        ]
        _config_apply_outputs = [config_components[k] for k in _CONFIG_KEYS_FOR_APPLY]
        _N_CFG = len(_CONFIG_KEYS_FOR_APPLY)

        def on_apply_config(ckpt_path):
            no_update = [gr.update()] * _N_CFG
            if not ckpt_path:
                return no_update + [{}, {}, t("training.resume.apply_config.no_ckpt")]
            tc = load_training_config_raw(ckpt_path)
            if tc is None:
                return no_update + [{}, {}, t("training.resume.apply_config.no_cfg")]

            # 25 config field values (same order as _CONFIG_KEYS_FOR_APPLY)
            config_vals = [
                tc.get("lora_r", 16),
                tc.get("lora_alpha", 32),
                tc.get("lora_dropout", 0.0),
                tc.get("use_rslora", False),
                tc.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"]),
                tc.get("training_type", "sft"),
                tc.get("num_epochs", 3),
                tc.get("per_device_train_batch_size", 4),
                tc.get("gradient_accumulation_steps", 4),
                str(tc.get("learning_rate", 2e-4)),
                tc.get("lr_scheduler_type", "cosine"),
                tc.get("warmup_ratio", 0.05),
                tc.get("weight_decay", 0.01),
                tc.get("max_grad_norm", 1.0),
                tc.get("beta", 0.1),
                tc.get("load_in_4bit", False),
                tc.get("use_gradient_checkpointing", True),
                tc.get("packing", False),
                tc.get("max_seq_length", 2048),
                tc.get("neftune_noise_alpha", 0),
                tc.get("output_dir", "outputs"),
                tc.get("save_steps", 200),
                tc.get("save_total_limit", 3),
                tc.get("logging_steps", 20),
                tc.get("report_to", "none"),
            ]

            # Reconstruct model_state from training_config
            model_id = tc.get("model_id", "")
            new_model_state = {
                "model_id": model_id,
                "display_name": model_id,
                "load_in_4bit": tc.get("load_in_4bit", False),
                "default_target_modules": tc.get("target_modules", []),
                "context_length": 4096,
            } if model_id else {}

            # Reconstruct dataset_state from training_config
            dataset_path = tc.get("dataset_path", "")
            new_dataset_state = {
                "path": dataset_path,
                "train_ratio": tc.get("train_ratio", 0.95),
                "max_samples": tc.get("max_samples", 0),
                "template": tc.get("prompt_template", "alpaca"),
                "think_mode": tc.get("think_mode", "keep"),
            } if dataset_path else {}

            model_display = model_id or "?"
            dataset_display = os.path.basename(dataset_path) if dataset_path else "?"
            warn = ""
            if dataset_path and not os.path.isfile(dataset_path):
                warn = " " + t("training.resume.apply_config.warn_dataset")
            status = t("training.resume.apply_config.ok").format(
                model=model_display,
                dataset=dataset_display,
                r=tc.get("lora_r", "?"),
                lr=tc.get("learning_rate", "?"),
            ) + warn

            return config_vals + [new_model_state, new_dataset_state, status]

        apply_config_btn.click(
            fn=on_apply_config,
            inputs=[resume_ckpt_path_state],
            outputs=_config_apply_outputs + [model_state, dataset_state, apply_config_status],
        )

        # ── Button update helpers: always include translated value ───────
        # Other buttons (scan, apply_config) work because they're never in
        # _stream_outputs — their text is always set by build_language_update.
        # start_btn/stop_btn ARE in _stream_outputs, so every yield must
        # explicitly set value= or Gradio clears them to None.
        def _start_idle():
            return gr.update(value=t("training.start"), interactive=True)

        def _start_busy():
            return gr.update(value=t("training.start"), interactive=False)

        def _stop_idle():
            return gr.update(value=t("training.stop"), interactive=False)

        def _stop_busy():
            return gr.update(value=t("training.stop"), interactive=True)

        def start_and_stream(
            dataset_info, model_info, resume_on, resume_ckpt,
            lora_r, lora_alpha, lora_dropout, use_rslora,
            target_modules, training_type, num_epochs,
            batch_size, grad_accum,
            learning_rate, lr_scheduler, warmup_ratio,
            weight_decay, max_grad_norm, beta,
            load_in_4bit, use_gc, packing,
            max_seq_length, neftune,
            output_dir, save_steps, save_total_limit,
            logging_steps, report_to,
            request: gr.Request,
        ):
            session_id = getattr(request, "session_hash", "__singleton__")
            sess = session_manager.get_or_create(session_id)
            monitor = sess.monitor
            orchestrator = sess.orchestrator

            if not dataset_info or not dataset_info.get("path"):
                yield (_status_html("error"), _start_idle(), _stop_idle(),
                       0, t("autotune.err.no_dataset"), None, None, "—", "—", "—", "", "")
                return
            if not model_info or not model_info.get("model_id"):
                yield (_status_html("error"), _start_idle(), _stop_idle(),
                       0, t("autotune.err.no_model"), None, None, "—", "—", "—", "", "")
                return

            try:
                lr_val = float(learning_rate)
            except ValueError:
                lr_val = 2e-4

            cfg = TrainingConfig(
                model_id=model_info["model_id"],
                load_in_4bit=bool(load_in_4bit),
                max_seq_length=int(max_seq_length),
                lora_r=int(lora_r), lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=list(target_modules) if target_modules else [
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_rslora=bool(use_rslora),
                use_gradient_checkpointing=bool(use_gc),
                training_type=str(training_type),
                num_epochs=int(num_epochs),
                per_device_train_batch_size=int(batch_size),
                gradient_accumulation_steps=int(grad_accum),
                learning_rate=lr_val,
                warmup_ratio=float(warmup_ratio),
                lr_scheduler_type=str(lr_scheduler),
                weight_decay=float(weight_decay),
                max_grad_norm=float(max_grad_norm),
                beta=float(beta),
                dataset_path=dataset_info["path"],
                train_ratio=float(dataset_info.get("train_ratio", 0.95)),
                max_samples=int(dataset_info.get("max_samples", 0)),
                prompt_template=dataset_info.get("template", "alpaca"),
                think_mode=dataset_info.get("think_mode", "keep"),
                packing=bool(packing),
                neftune_noise_alpha=float(neftune),
                output_dir=str(output_dir),
                save_steps=int(save_steps),
                save_total_limit=int(save_total_limit),
                logging_steps=int(logging_steps),
                report_to=str(report_to),
                resume_from_checkpoint=(resume_ckpt if resume_on and resume_ckpt else ""),
            )

            # In per_session mode, isolate output to a session-specific subdirectory
            # so concurrent sessions don't overwrite each other's checkpoints.
            if session_manager.mode == "per_session" and session_id != "__singleton__":
                import re as _re
                safe_id = _re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)[:32]
                cfg.output_dir = os.path.join(cfg.output_dir, safe_id)

            # Request training slot (may queue)
            slot = session_manager.request_training(session_id)
            if slot == "queued":
                pos = session_manager.queue_position(session_id)
                yield (_status_html("queued"), _start_busy(), _stop_busy(),
                       0, t("status.queued_msg").format(pos=pos), None, None, "—", "—", "—", "", "")
                # Wait for slot
                orchestrator._queue_event.clear()
                while not orchestrator._queue_event.wait(timeout=3):
                    pos = session_manager.queue_position(session_id)
                    if pos == 0:
                        break
                    yield (_status_html("queued"), _start_busy(), _stop_busy(),
                           0, t("status.queued_msg").format(pos=pos), None, None, "—", "—", "—", "", "")

            monitor.reset()
            orchestrator.start(cfg)

            yield (_status_html("loading"), _start_busy(), _stop_busy(),
                   0, t("training.status.loading"), None, None, "—", "—", "—", "", "")

            import pandas as pd
            while True:
                monitor.poll()
                if monitor.status in ("finished", "error", "stopped"):
                    break
                c, total, pct = monitor.get_progress()
                yield (
                    _status_html(monitor.status),
                    _start_busy(), _stop_busy(),
                    *_poll_outputs(monitor),
                )
                time.sleep(2)

            monitor.poll()
            session_manager.on_training_done(session_id)
            sess.status = monitor.status
            yield (
                _status_html(monitor.status),
                _start_idle(), _stop_idle(),
                *_poll_outputs(monitor),
            )

        _stream_outputs = [
            status_html, start_btn, stop_btn,
            progress_bar, progress_text,
            loss_plot, lr_plot,
            gpu_mem_box, speed_box, eta_box,
            log_box, checkpoints_box,
        ]

        start_btn.click(
            fn=start_and_stream,
            inputs=[dataset_state, model_state, resume_enabled, resume_checkpoint_dd] + config_inputs,
            outputs=_stream_outputs,
            concurrency_limit=None,
        )

        def on_stop(request: gr.Request):
            session_id = getattr(request, "session_hash", "__singleton__")
            sess = session_manager.get_or_create(session_id)
            sess.orchestrator.stop()
            session_manager.on_training_done(session_id)
            return (_status_html("stopped"), _start_idle(), _stop_idle())

        stop_btn.click(fn=on_stop, inputs=[], outputs=[status_html, start_btn, stop_btn])

        # ── Reconnect trigger ────────────────────────────────────────
        reconnect_trigger = gr.State(False)

        def reconnect_stream(should_reconnect: bool, request: gr.Request):
            session_id = getattr(request, "session_hash", "__singleton__")
            sess = session_manager.get_or_create(session_id)
            monitor = sess.monitor
            if not should_reconnect or monitor.status not in ("loading", "running"):
                # Not running — always emit button text so Gradio can't clear it
                yield (_status_html(monitor.status), _start_idle(), _stop_idle(),
                       *_poll_outputs(monitor))
                return
            yield (_status_html(monitor.status), _start_busy(), _stop_busy(),
                   *_poll_outputs(monitor))
            while True:
                monitor.poll()
                if monitor.status in ("finished", "error", "stopped"):
                    break
                yield (_status_html(monitor.status), _start_busy(), _stop_busy(),
                       *_poll_outputs(monitor))
                time.sleep(2)
            monitor.poll()
            yield (_status_html(monitor.status), _start_idle(), _stop_idle(),
                   *_poll_outputs(monitor))

        reconnect_trigger.change(
            fn=reconnect_stream,
            inputs=[reconnect_trigger],
            outputs=_stream_outputs,
            concurrency_limit=None,
        )

        return reconnect_trigger


def _poll_outputs(monitor) -> tuple:
    import pandas as pd
    c, total, pct = monitor.get_progress()
    steps, losses = monitor.get_loss_curve()
    loss_df = pd.DataFrame({"step": steps, "loss": losses}) if steps else None
    lr_steps, lrs = monitor.get_lr_curve()
    lr_df = pd.DataFrame({"step": lr_steps, "lr": lrs}) if lr_steps else None
    checkpoints = "\n".join(monitor.get_checkpoints()) or "(none)"
    return (
        pct,
        f"Step {c} / {total} ({pct:.1f}%)" if total else t("training.status.loading"),
        loss_df, lr_df,
        monitor.get_current_gpu_mem(),
        monitor.get_current_speed(),
        monitor.get_eta(),
        monitor.get_log_text(),
        checkpoints,
    )


def _status_html(status: str, label: str = "") -> str:
    colors = {
        "idle":     ("#94a3b8", "⚪"),
        "loading":  ("#f59e0b", "🟡"),
        "running":  ("#22c55e", "🟢"),
        "queued":   ("#a855f7", "⏳"),
        "paused":   ("#6366f1", "⏸"),
        "stopped":  ("#64748b", "⏹"),
        "finished": ("#3b82f6", "✅"),
        "error":    ("#ef4444", "❌"),
    }
    color, icon = colors.get(status, ("#94a3b8", "❓"))
    if not label:
        key = f"training.status.{status}"
        label = t(key) if status not in ("queued",) else t("training.queued")
    return (
        f'<div style="padding:10px 16px;border-radius:8px;background:{color}20;'
        f'border:2px solid {color};font-size:15px;font-weight:600;">'
        f'{icon} {label}'
        f'</div>'
    )
