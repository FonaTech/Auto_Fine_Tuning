"""
ui/theme.py
Centralized theme and CSS for the app shell.
"""

from __future__ import annotations

import json

import gradio as gr


DEFAULT_COLOR_THEME = "blue"

COLOR_THEME_CHOICES = [
    ("Ocean Blue", "blue"),
    ("Forest Green", "green"),
    ("Amber Sand", "amber"),
    ("Rose Clay", "rose"),
    ("Indigo Mist", "indigo"),
    ("Slate Mono", "slate"),
]


THEME_PROFILES = {
    "blue": {
        "--body-background-fill": "#f6f8fb",
        "--surface": "#ffffff",
        "--surface-2": "#f8fafc",
        "--surface-3": "#eef2f7",
        "--ink": "#0f172a",
        "--muted": "#64748b",
        "--accent": "#2563eb",
        "--accent-soft": "#eaf2ff",
        "--accent-border": "#bfdbfe",
        "--border": "#d8dee9",
        "--shadow": "0 8px 24px rgba(15, 23, 42, 0.05)",
        "--button-primary-background-fill": "#2563eb",
        "--button-primary-background-fill-hover": "#1d4ed8",
        "--button-primary-border-color": "#2563eb",
        "--button-secondary-background-fill": "#ffffff",
        "--button-secondary-background-fill-hover": "#f1f5f9",
        "--button-secondary-border-color": "#d8dee9",
        "--block-background-fill": "#ffffff",
        "--block-border-color": "#d8dee9",
        "--input-background-fill": "#ffffff",
        "--background-fill-primary": "#ffffff",
        "--background-fill-secondary": "#f8fafc",
        "--border-color-primary": "#d8dee9",
        "--body-text-color": "#0f172a",
        "--body-text-color-subdued": "#64748b",
        "--color-accent": "#2563eb",
        "--color-accent-soft": "#eaf2ff",
    },
    "green": {
        "--body-background-fill": "#f5f8f4",
        "--surface": "#ffffff",
        "--surface-2": "#f6faf6",
        "--surface-3": "#e7f1e7",
        "--ink": "#102418",
        "--muted": "#5f7468",
        "--accent": "#15803d",
        "--accent-soft": "#e7f6ec",
        "--accent-border": "#bbf7d0",
        "--border": "#d7e2d8",
        "--shadow": "0 8px 24px rgba(22, 101, 52, 0.06)",
        "--button-primary-background-fill": "#15803d",
        "--button-primary-background-fill-hover": "#166534",
        "--button-primary-border-color": "#15803d",
        "--button-secondary-background-fill": "#ffffff",
        "--button-secondary-background-fill-hover": "#f3f9f4",
        "--button-secondary-border-color": "#d7e2d8",
        "--block-background-fill": "#ffffff",
        "--block-border-color": "#d7e2d8",
        "--input-background-fill": "#ffffff",
        "--background-fill-primary": "#ffffff",
        "--background-fill-secondary": "#f6faf6",
        "--border-color-primary": "#d7e2d8",
        "--body-text-color": "#102418",
        "--body-text-color-subdued": "#5f7468",
        "--color-accent": "#15803d",
        "--color-accent-soft": "#e7f6ec",
    },
    "amber": {
        "--body-background-fill": "#fbf8f2",
        "--surface": "#ffffff",
        "--surface-2": "#fffaf2",
        "--surface-3": "#f4ead8",
        "--ink": "#2b2114",
        "--muted": "#7b6852",
        "--accent": "#c27a1a",
        "--accent-soft": "#fff1db",
        "--accent-border": "#f7d8a8",
        "--border": "#e8dcc8",
        "--shadow": "0 8px 24px rgba(146, 93, 18, 0.07)",
        "--button-primary-background-fill": "#c27a1a",
        "--button-primary-background-fill-hover": "#a86514",
        "--button-primary-border-color": "#c27a1a",
        "--button-secondary-background-fill": "#ffffff",
        "--button-secondary-background-fill-hover": "#fffbf5",
        "--button-secondary-border-color": "#e8dcc8",
        "--block-background-fill": "#ffffff",
        "--block-border-color": "#e8dcc8",
        "--input-background-fill": "#ffffff",
        "--background-fill-primary": "#ffffff",
        "--background-fill-secondary": "#fffaf2",
        "--border-color-primary": "#e8dcc8",
        "--body-text-color": "#2b2114",
        "--body-text-color-subdued": "#7b6852",
        "--color-accent": "#c27a1a",
        "--color-accent-soft": "#fff1db",
    },
    "rose": {
        "--body-background-fill": "#fbf7f8",
        "--surface": "#ffffff",
        "--surface-2": "#fff9fa",
        "--surface-3": "#f6e9ed",
        "--ink": "#2c1820",
        "--muted": "#7a6070",
        "--accent": "#be476d",
        "--accent-soft": "#fdebf1",
        "--accent-border": "#fbcfe0",
        "--border": "#ead9e0",
        "--shadow": "0 8px 24px rgba(157, 23, 77, 0.06)",
        "--button-primary-background-fill": "#be476d",
        "--button-primary-background-fill-hover": "#a53d60",
        "--button-primary-border-color": "#be476d",
        "--button-secondary-background-fill": "#ffffff",
        "--button-secondary-background-fill-hover": "#fff8fa",
        "--button-secondary-border-color": "#ead9e0",
        "--block-background-fill": "#ffffff",
        "--block-border-color": "#ead9e0",
        "--input-background-fill": "#ffffff",
        "--background-fill-primary": "#ffffff",
        "--background-fill-secondary": "#fff9fa",
        "--border-color-primary": "#ead9e0",
        "--body-text-color": "#2c1820",
        "--body-text-color-subdued": "#7a6070",
        "--color-accent": "#be476d",
        "--color-accent-soft": "#fdebf1",
    },
    "indigo": {
        "--body-background-fill": "#f6f7fc",
        "--surface": "#ffffff",
        "--surface-2": "#f8f9ff",
        "--surface-3": "#e9ecfb",
        "--ink": "#171b34",
        "--muted": "#66708e",
        "--accent": "#4f46e5",
        "--accent-soft": "#eef0ff",
        "--accent-border": "#c7d2fe",
        "--border": "#d9dff3",
        "--shadow": "0 8px 24px rgba(67, 56, 202, 0.06)",
        "--button-primary-background-fill": "#4f46e5",
        "--button-primary-background-fill-hover": "#4338ca",
        "--button-primary-border-color": "#4f46e5",
        "--button-secondary-background-fill": "#ffffff",
        "--button-secondary-background-fill-hover": "#f5f7ff",
        "--button-secondary-border-color": "#d9dff3",
        "--block-background-fill": "#ffffff",
        "--block-border-color": "#d9dff3",
        "--input-background-fill": "#ffffff",
        "--background-fill-primary": "#ffffff",
        "--background-fill-secondary": "#f8f9ff",
        "--border-color-primary": "#d9dff3",
        "--body-text-color": "#171b34",
        "--body-text-color-subdued": "#66708e",
        "--color-accent": "#4f46e5",
        "--color-accent-soft": "#eef0ff",
    },
    "slate": {
        "--body-background-fill": "#f4f6f8",
        "--surface": "#ffffff",
        "--surface-2": "#f8fafc",
        "--surface-3": "#e8edf2",
        "--ink": "#111827",
        "--muted": "#6b7280",
        "--accent": "#334155",
        "--accent-soft": "#edf2f7",
        "--accent-border": "#cbd5e1",
        "--border": "#d7dde5",
        "--shadow": "0 8px 24px rgba(17, 24, 39, 0.05)",
        "--button-primary-background-fill": "#334155",
        "--button-primary-background-fill-hover": "#1f2937",
        "--button-primary-border-color": "#334155",
        "--button-secondary-background-fill": "#ffffff",
        "--button-secondary-background-fill-hover": "#f3f6f9",
        "--button-secondary-border-color": "#d7dde5",
        "--block-background-fill": "#ffffff",
        "--block-border-color": "#d7dde5",
        "--input-background-fill": "#ffffff",
        "--background-fill-primary": "#ffffff",
        "--background-fill-secondary": "#f8fafc",
        "--border-color-primary": "#d7dde5",
        "--body-text-color": "#111827",
        "--body-text-color-subdued": "#6b7280",
        "--color-accent": "#334155",
        "--color-accent-soft": "#edf2f7",
    },
}


def _var_block(values: dict[str, str]) -> str:
    return "".join(f"{key}: {value};" for key, value in values.items())


BASE_CSS_VARS = {
    "--app-shell-width": "1380px",
    "--tab-panel-min-height": "760px",
}

DEFAULT_CSS_VARS = {
    **BASE_CSS_VARS,
    **THEME_PROFILES[DEFAULT_COLOR_THEME],
}

THEME_SWITCH_JS = f"""
(value) => {{
  const themes = {json.dumps(THEME_PROFILES, ensure_ascii=False)};
  const selected = themes[value] || themes["{DEFAULT_COLOR_THEME}"];
  const root = document.documentElement;
  root.setAttribute("data-color-theme", value || "{DEFAULT_COLOR_THEME}");
  Object.entries(selected).forEach(([key, val]) => {{
    root.style.setProperty(key, val);
  }});
  localStorage.setItem('gradio_color_theme', value);
  return [];
}}
""".strip()


def make_prefs_restore_js(default_lang: str) -> str:
    """Generate JS to restore language and theme preferences on page load."""
    themes_json = json.dumps(THEME_PROFILES, ensure_ascii=False)
    return f"""
() => {{
  const lang = localStorage.getItem('gradio_lang') || '{default_lang}';
  const theme = localStorage.getItem('gradio_color_theme') || '{DEFAULT_COLOR_THEME}';
  const themes = {themes_json};
  const selected = themes[theme] || themes['{DEFAULT_COLOR_THEME}'];
  const root = document.documentElement;
  root.setAttribute('data-color-theme', theme);
  Object.entries(selected).forEach(([key, val]) => root.style.setProperty(key, val));
  return [lang, theme];
}}
""".strip()


APP_THEME = (
    gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )
    .set(
        body_background_fill="#f6f8fb",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_border_color="#d8dee9",
        block_radius="12px",
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_border_color="#2563eb",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#f1f5f9",
        button_secondary_border_color="#d8dee9",
        input_background_fill="#ffffff",
        background_fill_secondary="#f8fafc",
    )
)


APP_CSS = (
    ":root {"
    + _var_block(DEFAULT_CSS_VARS)
    + "} "
    "body { background: var(--body-background-fill) !important; } "
    ".gradio-container { width: 100% !important; max-width: none !important; padding: 16px 20px 28px !important; box-sizing: border-box !important; color: var(--ink) !important; } "
    "#app-shell { width: min(var(--app-shell-width), 100%) !important; max-width: var(--app-shell-width) !important; margin: 0 auto !important; gap: 16px !important; } "
    ".app-header { padding: 2px 0 4px; background: transparent; border: none; box-shadow: none; } "
    ".app-header-kicker { color: var(--accent); font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 6px; } "
    ".app-header-title { margin: 0 0 8px; font-size: clamp(28px, 4vw, 34px); line-height: 1.12; letter-spacing: -0.02em; font-weight: 700; color: var(--ink); } "
    ".app-header-subtitle { margin: 0; max-width: 920px; color: var(--muted); font-size: 14px; line-height: 1.7; } "
    ".app-meta-line { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; } "
    ".app-meta-chip { display: inline-flex; align-items: center; gap: 8px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface-2); font-size: 13px; } "
    ".app-meta-chip-label { color: var(--muted); font-weight: 600; } "
    ".app-meta-chip-value { color: var(--ink); font-weight: 600; } "
    ".toolbar-strip { gap: 10px !important; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); padding: 10px 0 2px; } "
    ".toolbar-cell { padding: 0 !important; } "
    ".toolbar-note { color: var(--muted); font-size: 12px; line-height: 1.6; } "
    ".gradio-container .block { border-color: var(--border) !important; } "
    ".gradio-container .wrap, .gradio-container input, .gradio-container textarea, .gradio-container select { border-color: var(--border-color-primary) !important; } "
    ".gradio-container input, .gradio-container textarea, .gradio-container select { background: var(--input-background-fill) !important; color: var(--body-text-color) !important; } "
    ".gradio-container button.primary, .gradio-container button[variant='primary'] { background: var(--button-primary-background-fill) !important; border-color: var(--button-primary-border-color) !important; } "
    ".gradio-container button.primary:hover, .gradio-container button[variant='primary']:hover { background: var(--button-primary-background-fill-hover) !important; } "
    ".gradio-container button.secondary, .gradio-container button[variant='secondary'], .gradio-container button.stop { border-color: var(--button-secondary-border-color) !important; } "
    "#main-tabs { width: 100% !important; } "
    "#main-tabs > .tab-nav { gap: 8px; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 4px; } "
    "#main-tabs > .tab-nav button { font-size: 14px; font-weight: 600; white-space: nowrap; border-radius: 10px; border: 1px solid var(--border); background: #ffffff; box-shadow: none !important; } "
    "#main-tabs > .tab-nav button.selected { background: var(--accent-soft); color: var(--accent); border-color: var(--accent-border); } "
    "#main-tabs .workspace-tab { width: 100% !important; min-width: 100% !important; box-sizing: border-box !important; min-height: var(--tab-panel-min-height); } "
    "#main-tabs .workspace-tab > .block { width: 100% !important; } "
    "#main-tabs .gradio-row, #main-tabs .gradio-column { min-width: 0 !important; } "
    ".training-btn-row { min-height: 52px !important; } "
    ".training-btn-row button { min-height: 42px !important; height: 42px !important; } "
    ".gradio-container .checkbox-group input[type='checkbox'], .gradio-container label input[type='checkbox'] { accent-color: var(--accent) !important; width: 16px !important; height: 16px !important; cursor: pointer !important; } "
    ".gradio-container .checkbox-group input[type='checkbox']:checked, .gradio-container label input[type='checkbox']:checked { accent-color: var(--accent) !important; } "
    ".gradio-container .gradio-checkbox label, .gradio-container .gradio-checkboxgroup label { cursor: pointer !important; } "
    ".gradio-container input[type='radio'] { accent-color: var(--accent) !important; width: 16px !important; height: 16px !important; cursor: pointer !important; } "
    "#main-tabs .gradio-html, #main-tabs .gradio-dataframe, #main-tabs .gradio-plot, #main-tabs .gradio-lineplot, #main-tabs .gradio-barplot { width: 100% !important; } "
    "#main-tabs .block, .toolbar-cell .block { box-shadow: none !important; } "
    ".footer-note { color: var(--muted); font-size: 13px; line-height: 1.7; padding: 2px 2px 12px; } "
    "@media (max-width: 900px) { .gradio-container { padding: 12px !important; } :root { --tab-panel-min-height: 0px; } .app-header-title { font-size: 26px; } .toolbar-strip { padding-top: 8px; } } "
)
