from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def render_report(
    out_path: str | Path,
    context: dict,
    *,
    template_dir: str | Path | None = None,
) -> Path:
    from jinja2 import Environment, FileSystemLoader

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if template_dir is None:
        template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        # select_autoescape matches on filename suffix; the template is
        # "report.html.j2" (suffix ".j2"), which select_autoescape(["html"])
        # would silently fail to match, disabling autoescaping entirely.
        autoescape=True,
    )
    tmpl = env.get_template("report.html.j2")

    ctx = dict(context)
    ctx.setdefault("generated_at", datetime.now().isoformat(timespec="seconds"))
    ctx.setdefault("plugin_version", "1.0.1")
    if "config_json" not in ctx and "config_dict" in ctx:
        ctx["config_json"] = json.dumps(ctx["config_dict"], indent=2)

    out_path.write_text(tmpl.render(**ctx), encoding="utf-8")
    return out_path
