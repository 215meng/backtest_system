from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"无法序列化 {type(value)!r}")


def create_run_dir(root: Path, name: str) -> Path:
    token = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{token}_{name.replace(' ', '_')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def render_report(
    path: Path,
    returns: pd.Series,
    benchmark: pd.Series,
    metrics: dict[str, Any],
    provenance: dict[str, Any],
    risk_events: pd.DataFrame | None = None,
) -> None:
    equity = (1 + returns.fillna(0)).cumprod()
    benchmark_equity = (1 + benchmark.reindex(returns.index).fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("净值", "回撤"))
    figure.add_trace(go.Scatter(x=equity.index, y=equity, name="策略净值"), row=1, col=1)
    figure.add_trace(go.Scatter(x=benchmark_equity.index, y=benchmark_equity, name="等权基准"), row=1, col=1)
    figure.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="回撤", fill="tozeroy"), row=2, col=1)
    if risk_events is not None and not risk_events.empty:
        figure.add_trace(
            go.Scatter(
                x=pd.to_datetime(risk_events["trigger_time"], utc=True),
                y=risk_events["trigger_drawdown"].astype(float),
                name="回撤止损触发",
                mode="markers",
                marker={"color": "crimson", "size": 10, "symbol": "x"},
                hovertemplate="触发时间=%{x}<br>回撤=%{y:.2%}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    figure.update_layout(title=provenance.get("name", "回测报告"), template="plotly_white", height=750)
    metric_html = "".join(
        f"<li><b>{key}</b>: {value}</li>"
        for key, value in metrics.items()
        if not isinstance(value, (dict, list))
    )
    warning_html = "".join(f"<li>{item}</li>" for item in provenance.get("warnings", []))
    risk_html = ""
    if risk_events is not None and not risk_events.empty:
        risk_html = f"<h2>回撤止损事件</h2>{risk_events.to_html(index=False, border=0)}"
    path.write_text(
        "<html><meta charset='utf-8'><body>"
        f"<h1>{provenance.get('name')}</h1><h2>策略指标</h2><ul>{metric_html}</ul>"
        f"<h2>数据与复现警告</h2><ul>{warning_html}</ul>{risk_html}"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}"
        "</body></html>",
        encoding="utf-8",
    )


def render_research_report(
    path: Path,
    group_returns: pd.DataFrame,
    spread: pd.DataFrame,
    metrics: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    """Render paper-style factor evidence without presenting the spread as account equity."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("分组未来收益", "累计统计型 Top−Bottom 差值（非账户净值）"),
    )
    for bucket_name, group in group_returns.groupby("bucket_name", sort=True):
        figure.add_trace(
            go.Scatter(x=group["formation_time"], y=group["return_value"], name=str(bucket_name)),
            row=1,
            col=1,
        )
    cumulative = spread.set_index("formation_time")["spread_return"].cumsum()
    figure.add_trace(
        go.Scatter(x=cumulative.index, y=cumulative, name="累计统计差值", fill="tozeroy"), row=2, col=1
    )
    figure.update_layout(title=provenance.get("name", "因子研究报告"), template="plotly_white", height=760)
    metric_html = "".join(
        f"<li><b>{key}</b>: {value}</li>"
        for key, value in metrics.items()
        if not isinstance(value, (dict, list))
    )
    warning_html = "".join(f"<li>{item}</li>" for item in provenance.get("warnings", []))
    path.write_text(
        "<html><meta charset='utf-8'><body>"
        f"<h1>{provenance.get('name')}</h1><h2>因子研究证据</h2><ul>{metric_html}</ul>"
        f"<h2>数据与复现限制</h2><ul>{warning_html}</ul>"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}"
        "</body></html>",
        encoding="utf-8",
    )
