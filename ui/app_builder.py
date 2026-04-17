"""
ui/app_builder.py
Gradio application assembly.
"""

import gradio as gr

from core.environment import detect_environment
from core.session_manager import session_manager

from ui.tabs.env_tab import build_env_tab
from ui.tabs.dataset_tab import build_dataset_tab
from ui.tabs.model_tab import build_model_tab
from ui.tabs.config_tab import build_config_tab
from ui.tabs.training_tab import build_training_tab
from ui.tabs.export_tab import build_export_tab
from ui.tabs.auto_tune_tab import build_auto_tune_tab
from ui.i18n import (
    DEFAULT_LANGUAGE, LANGUAGE_CHOICES, tr, normalize_language,
    build_language_update, get_registered_components, set_current_lang,
    register_translatable, ts, t,
)
from ui.theme import (
    APP_CSS,
    APP_THEME,
    COLOR_THEME_CHOICES,
    DEFAULT_COLOR_THEME,
    THEME_SWITCH_JS,
    make_prefs_restore_js,
)

# Global environment info (detected once at startup)
_env_info = None


def get_env_info():
    global _env_info
    if _env_info is None:
        _env_info = detect_environment()
    return _env_info


def _hero_html(env_info) -> str:
    from core.environment import get_backend_display
    backend_label = get_backend_display(env_info.backend)
    gpu_name = env_info.gpu_name or "Not detected"
    if env_info.gpu_name and env_info.gpu_vram_gb:
        gpu_name = f"{env_info.gpu_name} ({env_info.gpu_vram_gb}GB)"
    return (
        "<div class='app-header'>"
        "<div class='app-header-kicker'>Local Fine-Tuning Workbench</div>"
        "<h1 class='app-header-title'>Unsloth GUI Fine-Tuning Workbench</h1>"
        "<p class='app-header-subtitle'>"
        "Train and experiment with SFT, DPO, ORPO, auto tuning, model export, "
        "and quick inference from one local web interface."
        "</p>"
        "<div class='app-meta-line'>"
        f"<span class='app-meta-chip'><span class='app-meta-chip-label'>Backend</span><span class='app-meta-chip-value'>{backend_label}</span></span>"
        f"<span class='app-meta-chip'><span class='app-meta-chip-label'>Platform</span><span class='app-meta-chip-value'>{env_info.platform_name}</span></span>"
        f"<span class='app-meta-chip'><span class='app-meta-chip-label'>GPU</span><span class='app-meta-chip-value'>{gpu_name}</span></span>"
        "</div>"
        "</div>"
    )


def _session_status_html() -> str:
    stats = session_manager.get_stats()
    online = stats["online"]
    running = stats["running"]
    queued = stats["queued"]
    max_c = stats["max_concurrent"]
    return (
        f"<div style='font-size:13px;color:var(--muted);line-height:1.8;padding:4px 0;'>"
        f"<span style='margin-right:12px;'>👥 {online} {t('status.online')}</span>"
        f"<span style='margin-right:12px;'>⚡ {running}{t('status.of')}{max_c} {t('status.training')}</span>"
        f"<span>⏳ {queued} {t('status.queued')}</span>"
        f"</div>"
    )


def build_app() -> gr.Blocks:
    env_info = get_env_info()

    with gr.Blocks(
        title="Unsloth GUI Fine-Tuning Workbench",
        fill_width=True,
    ) as app:
        with gr.Column(elem_id="app-shell", elem_classes="app-shell"):
            gr.HTML(_hero_html(env_info))

            # ── Toolbar ──────────────────────────────────────────────
            with gr.Row(equal_height=True, elem_classes="toolbar-strip"):
                with gr.Column(scale=2, elem_classes="toolbar-cell"):
                    language = gr.Dropdown(
                        choices=LANGUAGE_CHOICES,
                        value=DEFAULT_LANGUAGE,
                        label=tr("app.toolbar.language"),
                        info=tr("app.toolbar.language.help"),
                    )
                with gr.Column(scale=1, elem_classes="toolbar-cell"):
                    color_theme = gr.Dropdown(
                        choices=COLOR_THEME_CHOICES,
                        value=DEFAULT_COLOR_THEME,
                        label="Color Theme",
                        info="Switch accent colors without changing the clean layout",
                    )
                with gr.Column(scale=2, elem_classes="toolbar-cell"):
                    session_status = gr.HTML(value=_session_status_html())

            # ── Admin panel (localhost only) ──────────────────────────
            is_admin = gr.State(False)
            with gr.Column(visible=False) as admin_panel:
                admin_title = gr.Markdown(tr("admin.title"))
                register_translatable(admin_title, label_key="admin.title")
                with gr.Row():
                    admin_mode = gr.Radio(
                        choices=[
                            (ts("admin.mode.singleton", DEFAULT_LANGUAGE), "singleton"),
                            (ts("admin.mode.per_session", DEFAULT_LANGUAGE), "per_session"),
                        ],
                        value="singleton",
                        label=tr("admin.mode"),
                    )
                    register_translatable(admin_mode, label_key="admin.mode")
                    admin_max_tasks = gr.Slider(
                        minimum=1, maximum=8, value=1, step=1,
                        label=tr("admin.max_tasks"),
                        info=tr("admin.max_tasks.info"),
                    )
                    register_translatable(admin_max_tasks, label_key="admin.max_tasks", info_key="admin.max_tasks.info")

            # ── Shared State ─────────────────────────────────────────
            dataset_state = gr.State({})
            model_state = gr.State({})

            # ── Tabs (correct order) ─────────────────────────────────
            with gr.Tabs(elem_id="main-tabs", elem_classes="main-tabs"):
                build_env_tab(env_info)
                build_dataset_tab(dataset_state)
                build_model_tab(model_state, env_info)
                config_components = build_config_tab(env_info)
                build_auto_tune_tab(
                    dataset_state, model_state,
                    config_components,
                )
                reconnect_trigger = build_training_tab(
                    config_components, dataset_state, model_state,
                )
                build_export_tab()

            gr.Markdown(
                "This project is a local single-user fine-tuning GUI. See README, LICENSE, and THIRD_PARTY_NOTICES for dependency licenses and attribution details.",
                elem_classes="footer-note",
            )

            # ── Admin panel events ───────────────────────────────────
            def on_mode_change(mode):
                session_manager.set_mode(mode)

            def on_max_tasks_change(n):
                session_manager.set_max_concurrent(int(n))

            admin_mode.change(fn=on_mode_change, inputs=[admin_mode], outputs=[])
            admin_max_tasks.change(fn=on_max_tasks_change, inputs=[admin_max_tasks], outputs=[])

            # ── Color theme switching ────────────────────────────────
            color_theme.change(
                fn=None,
                inputs=[color_theme],
                outputs=[],
                js=THEME_SWITCH_JS,
                queue=False,
                preprocess=False,
                postprocess=False,
            )

            # ── Language switching ───────────────────────────────────
            _registered = get_registered_components()

            def _on_language_change(lang: str):
                lang = normalize_language(lang)
                set_current_lang(lang)
                return build_language_update(lang)

            language.change(
                fn=_on_language_change,
                inputs=[language],
                outputs=_registered,
                js="(lang) => { localStorage.setItem('gradio_lang', lang); return [lang]; }",
                queue=False,
            )

            # ── Page load: restore prefs + detect admin + reconnect ──
            def _on_load(lang: str, theme: str, request: gr.Request):
                lang = normalize_language(lang)
                set_current_lang(lang)
                label_updates = build_language_update(lang)

                host = getattr(getattr(request, "client", None), "host", "") or ""
                _is_admin = host in ("127.0.0.1", "::1", "localhost", "")

                session_id = getattr(request, "session_hash", "__singleton__")
                state = session_manager.get_or_create(session_id)
                is_running = state.monitor.status in ("loading", "running")

                status_html = _session_status_html()

                return [lang, theme, _is_admin, gr.update(visible=_is_admin), is_running, status_html] + label_updates

            app.load(
                fn=_on_load,
                inputs=[language, color_theme],
                outputs=[language, color_theme, is_admin, admin_panel, reconnect_trigger, session_status] + _registered,
                js=make_prefs_restore_js(DEFAULT_LANGUAGE),
                queue=False,
            )

            # Update session status on language change too
            language.change(
                fn=lambda _: _session_status_html(),
                inputs=[language],
                outputs=[session_status],
                queue=False,
            )

    return app, _LAUNCH_KWARGS


# Gradio 6: theme/css passed to launch() instead of Blocks()
_LAUNCH_KWARGS = {
    "theme": APP_THEME,
    "css": APP_CSS,
}
