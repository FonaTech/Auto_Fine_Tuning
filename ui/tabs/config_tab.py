"""
ui/tabs/config_tab.py
Tab 4: 训练超参数配置，支持预设加载/保存，返回组件字典供训练 Tab 使用。
"""

import json
import os
import gradio as gr
from core.environment import EnvironmentInfo
from ui.i18n import tr, ts, get_choices, register_translatable, t

_PRESETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "configs", "presets",
)

_PRESET_KEYS = ["quick_test", "memory_efficient", "balanced", "high_quality"]


def _load_preset(name: str) -> dict:
    path = os.path.join(_PRESETS_DIR, f"{name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_config_tab(env_info: EnvironmentInfo) -> dict:
    """
    构建训练配置 Tab。
    返回包含所有配置组件的字典，供 training_tab 作为 inputs 使用。
    """
    components = {}

    with gr.Tab(tr("tab.config"), elem_classes="workspace-tab", render_children=True) as tab:
        register_translatable(tab, label_key="tab.config")
        title_md = gr.Markdown(tr("config.title"))
        register_translatable(title_md, label_key="config.title")

        presets_hdr = gr.Markdown(tr("config.presets.title"))
        register_translatable(presets_hdr, label_key="config.presets.title")
        with gr.Row():
            preset_btns = {}
            for pname in _PRESET_KEYS:
                btn = gr.Button(tr(f"preset.{pname}"), size="sm")
                register_translatable(btn, label_key=f"preset.{pname}")
                preset_btns[pname] = btn

        # ── LoRA 配置 ────────────────────────────────────────────────
        with gr.Accordion(tr("config.accordion.lora"), open=True) as acc_lora:
            register_translatable(acc_lora, label_key="config.accordion.lora")
            with gr.Row():
                components["lora_r"] = gr.Slider(
                    minimum=4, maximum=128, value=16, step=4,
                    label=tr("config.lora_r"),
                )
                register_translatable(components["lora_r"], label_key="config.lora_r")
                components["lora_alpha"] = gr.Number(
                    value=32, label=tr("config.lora_alpha"),
                    info=tr("config.lora_alpha.info"),
                    precision=0,
                )
                register_translatable(components["lora_alpha"], label_key="config.lora_alpha", info_key="config.lora_alpha.info")
                components["lora_dropout"] = gr.Slider(
                    minimum=0.0, maximum=0.2, value=0.0, step=0.01,
                    label=tr("config.lora_dropout"),
                )
                register_translatable(components["lora_dropout"], label_key="config.lora_dropout")
            with gr.Row():
                components["target_modules"] = gr.CheckboxGroup(
                    choices=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
                    value=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
                    label=tr("config.target_modules"),
                )
                register_translatable(components["target_modules"], label_key="config.target_modules")
                components["use_rslora"] = gr.Checkbox(
                    value=False, label=tr("config.use_rslora"),
                )
                register_translatable(components["use_rslora"], label_key="config.use_rslora")

        # ── 训练参数 ─────────────────────────────────────────────────
        with gr.Accordion(tr("config.accordion.training"), open=True) as acc_training:
            register_translatable(acc_training, label_key="config.accordion.training")
            with gr.Row():
                components["training_type"] = gr.Radio(
                    choices=get_choices("config_training_type"),
                    value="sft",
                    label=tr("config.training_type"),
                )
                register_translatable(components["training_type"], label_key="config.training_type", choices_key="config_training_type")
            with gr.Row():
                components["num_epochs"] = gr.Number(
                    value=3, label=tr("config.num_epochs"), precision=0,
                )
                register_translatable(components["num_epochs"], label_key="config.num_epochs")
                components["per_device_train_batch_size"] = gr.Dropdown(
                    choices=[1, 2, 4, 8, 16], value=4,
                    label=tr("config.batch_size"),
                )
                register_translatable(components["per_device_train_batch_size"], label_key="config.batch_size")
                components["gradient_accumulation_steps"] = gr.Dropdown(
                    choices=[1, 2, 4, 8, 16], value=4,
                    label=tr("config.grad_accum"),
                )
                register_translatable(components["gradient_accumulation_steps"], label_key="config.grad_accum")
            with gr.Row():
                components["learning_rate"] = gr.Textbox(
                    value="2e-4", label=tr("config.learning_rate"),
                )
                register_translatable(components["learning_rate"], label_key="config.learning_rate")
                components["lr_scheduler_type"] = gr.Dropdown(
                    choices=["cosine", "linear", "constant", "cosine_with_restarts"],
                    value="cosine",
                    label=tr("config.lr_scheduler"),
                )
                register_translatable(components["lr_scheduler_type"], label_key="config.lr_scheduler")
                components["warmup_ratio"] = gr.Slider(
                    minimum=0.0, maximum=0.2, value=0.05, step=0.01,
                    label=tr("config.warmup_ratio"),
                )
                register_translatable(components["warmup_ratio"], label_key="config.warmup_ratio")
            with gr.Row():
                components["weight_decay"] = gr.Number(
                    value=0.01, label=tr("config.weight_decay"), precision=None,
                )
                register_translatable(components["weight_decay"], label_key="config.weight_decay")
                components["max_grad_norm"] = gr.Number(
                    value=1.0, label=tr("config.max_grad_norm"), precision=None,
                )
                register_translatable(components["max_grad_norm"], label_key="config.max_grad_norm")
                components["beta"] = gr.Number(
                    value=0.1, label=tr("config.beta"), precision=None,
                    visible=False,
                )
                register_translatable(components["beta"], label_key="config.beta")

        # ── 内存优化 ─────────────────────────────────────────────────
        with gr.Accordion(tr("config.accordion.memory"), open=True) as acc_memory:
            register_translatable(acc_memory, label_key="config.accordion.memory")
            with gr.Row():
                load_in_4bit_default = env_info.cuda_available
                components["load_in_4bit"] = gr.Checkbox(
                    value=load_in_4bit_default,
                    label=tr("config.load_in_4bit"),
                    interactive=True,
                )
                register_translatable(components["load_in_4bit"], label_key="config.load_in_4bit")
                components["use_gradient_checkpointing"] = gr.Checkbox(
                    value=True, label=tr("config.gradient_checkpointing"),
                )
                register_translatable(components["use_gradient_checkpointing"], label_key="config.gradient_checkpointing")
                components["packing"] = gr.Checkbox(
                    value=False, label=tr("config.packing"),
                )
                register_translatable(components["packing"], label_key="config.packing")
            with gr.Row():
                components["max_seq_length"] = gr.Dropdown(
                    choices=[256, 512, 1024, 2048, 4096, 8192],
                    value=2048,
                    label=tr("config.max_seq_length"),
                )
                register_translatable(components["max_seq_length"], label_key="config.max_seq_length")
                components["neftune_noise_alpha"] = gr.Slider(
                    minimum=0, maximum=15, value=0, step=1,
                    label=tr("config.neftune"),
                )
                register_translatable(components["neftune_noise_alpha"], label_key="config.neftune")

        # ── 保存与日志 ───────────────────────────────────────────────
        with gr.Accordion(tr("config.accordion.save"), open=False) as acc_save:
            register_translatable(acc_save, label_key="config.accordion.save")
            with gr.Row():
                components["output_dir"] = gr.Textbox(
                    value="outputs", label=tr("config.output_dir"),
                )
                register_translatable(components["output_dir"], label_key="config.output_dir")
                components["save_steps"] = gr.Number(
                    value=200, label=tr("config.save_steps"), precision=0,
                )
                register_translatable(components["save_steps"], label_key="config.save_steps")
                components["save_total_limit"] = gr.Number(
                    value=3, label=tr("config.save_total_limit"), precision=0,
                )
                register_translatable(components["save_total_limit"], label_key="config.save_total_limit")
            with gr.Row():
                components["logging_steps"] = gr.Number(
                    value=20, label=tr("config.logging_steps"), precision=0,
                )
                register_translatable(components["logging_steps"], label_key="config.logging_steps")
                components["report_to"] = gr.Dropdown(
                    choices=["none", "tensorboard"],
                    value="none",
                    label=tr("config.report_to"),
                )
                register_translatable(components["report_to"], label_key="config.report_to")

        # ── 配置保存/加载 ────────────────────────────────────────────
        with gr.Row():
            save_config_btn = gr.Button(tr("config.save"), size="sm")
            register_translatable(save_config_btn, label_key="config.save")
            load_config_file = gr.File(
                label=tr("config.load_file"),
                file_types=[".json"],
                visible=True,
            )
            register_translatable(load_config_file, label_key="config.load_file")
            save_status = gr.Textbox(label="", interactive=False, value="", scale=2)

        # ── 事件 ────────────────────────────────────────────────────

        # 训练类型切换 → 显示/隐藏 beta
        def on_training_type_change(t):
            return gr.update(visible=(t in ("dpo", "orpo")))

        components["training_type"].change(
            fn=on_training_type_change,
            inputs=[components["training_type"]],
            outputs=[components["beta"]],
        )

        # lora_r 同步 lora_alpha
        def on_r_change(r):
            return int(r * 2)

        components["lora_r"].change(
            fn=on_r_change,
            inputs=[components["lora_r"]],
            outputs=[components["lora_alpha"]],
        )

        # 预设按钮
        preset_output_keys = [
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

        def make_preset_fn(pname):
            def apply_preset():
                p = _load_preset(pname)
                vals = [
                    p.get("lora_r", 16),
                    p.get("lora_alpha", 32),
                    p.get("lora_dropout", 0.0),
                    p.get("use_rslora", False),
                    p.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                    p.get("training_type", "sft"),
                    p.get("num_epochs", 3),
                    p.get("per_device_train_batch_size", 4),
                    p.get("gradient_accumulation_steps", 4),
                    str(p.get("learning_rate", 2e-4)),
                    p.get("lr_scheduler_type", "cosine"),
                    p.get("warmup_ratio", 0.05),
                    p.get("weight_decay", 0.01),
                    p.get("max_grad_norm", 1.0),
                    p.get("beta", 0.1),
                    p.get("load_in_4bit", True) and env_info.cuda_available,
                    p.get("use_gradient_checkpointing", True),
                    p.get("packing", False),
                    p.get("max_seq_length", 2048),
                    p.get("neftune_noise_alpha", 0),
                    p.get("output_dir", "outputs"),
                    p.get("save_steps", 200),
                    p.get("save_total_limit", 3),
                    p.get("logging_steps", 20),
                    p.get("report_to", "none"),
                ]
                return vals
            return apply_preset

        preset_outputs = [components[k] for k in preset_output_keys]

        for pname, pbtn in preset_btns.items():
            pbtn.click(
                fn=make_preset_fn(pname),
                inputs=[],
                outputs=preset_outputs,
            )

        # 保存配置
        def save_config(*vals):
            d = dict(zip(preset_output_keys, vals))
            path = os.path.join(components["output_dir"].value or "outputs", "training_config.json")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            return f"配置已保存到: {path}"

        save_config_btn.click(
            fn=save_config,
            inputs=preset_outputs,
            outputs=[save_status],
        )

        # 加载配置文件
        def load_config(file_obj):
            if file_obj is None:
                return [gr.update()] * len(preset_output_keys) + [t("config.load.select")]
            try:
                with open(file_obj.name, "r", encoding="utf-8") as f:
                    p = json.load(f)
                vals = [
                    p.get("lora_r", 16),
                    p.get("lora_alpha", 32),
                    p.get("lora_dropout", 0.0),
                    p.get("use_rslora", False),
                    p.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                    p.get("training_type", "sft"),
                    p.get("num_epochs", 3),
                    p.get("per_device_train_batch_size", 4),
                    p.get("gradient_accumulation_steps", 4),
                    str(p.get("learning_rate", 2e-4)),
                    p.get("lr_scheduler_type", "cosine"),
                    p.get("warmup_ratio", 0.05),
                    p.get("weight_decay", 0.01),
                    p.get("max_grad_norm", 1.0),
                    p.get("beta", 0.1),
                    p.get("load_in_4bit", True) and env_info.cuda_available,
                    p.get("use_gradient_checkpointing", True),
                    p.get("packing", False),
                    p.get("max_seq_length", 2048),
                    p.get("neftune_noise_alpha", 0),
                    p.get("output_dir", "outputs"),
                    p.get("save_steps", 200),
                    p.get("save_total_limit", 3),
                    p.get("logging_steps", 20),
                    p.get("report_to", "none"),
                ]
                return vals + [t("config.load.ok").format(name=os.path.basename(file_obj.name))]
            except Exception as e:
                return [gr.update()] * len(preset_output_keys) + [t("config.load.err").format(e=e)]

        load_config_file.change(
            fn=load_config,
            inputs=[load_config_file],
            outputs=preset_outputs + [save_status],
        )

    return components
