"""
ui/tabs/training_tab.py
Tab 5: Training monitor — start/stop, live loss curve, log stream, progress & ETA.
Uses Gradio Generator mode for real-time updates (no gr.Timer needed).
"""

import os
import time
import gradio as gr

from core.trainer import TrainingConfig
from core.checkpoint import scan_checkpoints, load_checkpoint_config, configs_compatible
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
        with gr.Row():
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
            )
            register_translatable(resume_checkpoint_dd, label_key="training.resume.checkpoint")
            resume_compat_status = gr.Textbox(
                value="", label=tr("training.resume.compat"), interactive=False,
            )
            register_translatable(resume_compat_status, label_key="training.resume.compat")
            resume_enabled = gr.Checkbox(value=False, label=tr("training.resume.enable"))
            register_translatable(resume_enabled, label_key="training.resume.enable")

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
        resume_checkpoint_dd.change(
            fn=on_resume_select,
            inputs=[resume_checkpoint_dd, model_state,
                    config_components["lora_r"], config_components["lora_alpha"],
                    config_components["target_modules"]],
            outputs=[resume_compat_status, resume_enabled],
        )

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
                yield (_status_html("error"), gr.update(interactive=True), gr.update(interactive=False),
                       0, t("autotune.err.no_dataset"), None, None, "—", "—", "—", "", "")
                return
            if not model_info or not model_info.get("model_id"):
                yield (_status_html("error"), gr.update(interactive=True), gr.update(interactive=False),
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
                yield (_status_html("queued"), gr.update(interactive=False), gr.update(interactive=True),
                       0, t("status.queued_msg").format(pos=pos), None, None, "—", "—", "—", "", "")
                # Wait for slot
                orchestrator._queue_event.clear()
                while not orchestrator._queue_event.wait(timeout=3):
                    pos = session_manager.queue_position(session_id)
                    if pos == 0:
                        break
                    yield (_status_html("queued"), gr.update(interactive=False), gr.update(interactive=True),
                           0, t("status.queued_msg").format(pos=pos), None, None, "—", "—", "—", "", "")

            monitor.reset()
            orchestrator.start(cfg)

            yield (_status_html("loading"), gr.update(interactive=False), gr.update(interactive=True),
                   0, t("training.status.loading"), None, None, "—", "—", "—", "", "")

            import pandas as pd
            while True:
                monitor.poll()
                if monitor.status in ("finished", "error", "stopped"):
                    break
                c, total, pct = monitor.get_progress()
                yield (
                    _status_html(monitor.status),
                    gr.update(interactive=False), gr.update(interactive=True),
                    *_poll_outputs(monitor),
                )
                time.sleep(2)

            monitor.poll()
            session_manager.on_training_done(session_id)
            sess.status = monitor.status
            yield (
                _status_html(monitor.status),
                gr.update(interactive=True), gr.update(interactive=False),
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
            return (_status_html("stopped"), gr.update(interactive=True), gr.update(interactive=False))

        stop_btn.click(fn=on_stop, inputs=[], outputs=[status_html, start_btn, stop_btn])

        # ── Reconnect trigger ────────────────────────────────────────
        reconnect_trigger = gr.State(False)

        def reconnect_stream(should_reconnect: bool, request: gr.Request):
            if not should_reconnect:
                return
            session_id = getattr(request, "session_hash", "__singleton__")
            sess = session_manager.get_or_create(session_id)
            monitor = sess.monitor
            if monitor.status not in ("loading", "running"):
                # Not running — just show current state and return immediately (no spin)
                yield (_status_html(monitor.status), gr.update(interactive=True), gr.update(interactive=False),
                       *_poll_outputs(monitor))
                return
            yield (_status_html(monitor.status), gr.update(interactive=False), gr.update(interactive=True),
                   *_poll_outputs(monitor))
            while True:
                monitor.poll()
                if monitor.status in ("finished", "error", "stopped"):
                    break
                yield (_status_html(monitor.status), gr.update(interactive=False), gr.update(interactive=True),
                       *_poll_outputs(monitor))
                time.sleep(2)
            monitor.poll()
            yield (_status_html(monitor.status), gr.update(interactive=True), gr.update(interactive=False),
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
