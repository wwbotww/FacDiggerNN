"""Machine-readable and small self-contained HTML factor reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_evaluation_report(metrics: dict[str, Any], output: Path) -> None:
    payload = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    raw = metrics["metrics"]["raw"]
    summary = {
        "coverage": metrics["coverage"]["coverage"],
        "mean_ic": raw["ic"]["mean"],
        "mean_rank_ic": raw["rank_ic"]["mean"],
        "rank_icir": raw["rank_ic"]["ir"],
        "gross_q_high_minus_low": raw["portfolio"].get("gross_q_high_minus_low"),
        "mean_turnover": raw["portfolio"].get("mean_turnover"),
        "research_ready": metrics["metrics"]["cross_section"]["research_ready"],
    }
    rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>FacDigger E0 Report</title>
<style>body{{font-family:system-ui;max-width:960px;margin:2rem auto;line-height:1.5}}
table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:.4rem .7rem;text-align:left}}
pre{{background:#f5f5f5;padding:1rem;overflow:auto}}</style></head>
<body><h1>FacDigger E0 Validation Report</h1><table>{rows}</table>
<h2>完整指标</h2><pre>{html.escape(payload)}</pre></body></html>"""
    output.write_text(document, encoding="utf-8")
