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


def _finite_series(series: pd.Series) -> pd.Series:
    """图表只接受有限数值，避免异常值把所有曲线压成直线。"""
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.map(lambda item: pd.notna(item) and float("-inf") < item < float("inf")))


def render_native_factor_report(
    path: Path,
    groups: pd.DataFrame,
    cumulative: pd.DataFrame,
    long_short: pd.DataFrame,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("各分组前瞻收益", "Group Cumulative Returns", "Top-Bottom 多空收益（成本后）"),
    )
    if not groups.empty:
        for group_id, item in groups.groupby("group", sort=True):
            figure.add_trace(
                go.Scatter(x=item["timestamp"], y=_finite_series(item["return"]), name=f"G{group_id}"),
                row=1,
                col=1,
            )
    if not cumulative.empty:
        for column in cumulative.columns:
            figure.add_trace(
                go.Scatter(x=cumulative.index, y=_finite_series(cumulative[column]), name=f"G{column} 累计"),
                row=2,
                col=1,
            )
    if not long_short.empty:
        figure.add_trace(
            go.Scatter(x=long_short["timestamp"], y=(1 + _finite_series(long_short["net_return"])).cumprod(), name="多空净值"),
            row=3,
            col=1,
        )
    figure.update_layout(title=manifest.get("name", "原生 Python 因子报告"), template="plotly_white", height=980)
    scalar_metrics = "".join(f"<li><b>{key}</b>: {value}</li>" for key, value in metrics.items() if not isinstance(value, (dict, list)))
    path.write_text(
        "<html><meta charset='utf-8'><body>"
        f"<h1>{manifest.get('name', '原生 Python 因子报告')}</h1>"
        "<p>该报告由原始因子表产生；分组、前瞻收益与成本口径均来自脚本的 set_factor_evaluation 声明。</p>"
        f"<h2>指标</h2><ul>{scalar_metrics}</ul>{figure.to_html(full_html=False, include_plotlyjs=True)}"
        "</body></html>",
        encoding="utf-8",
    )


def render_native_strategy_report(
    path: Path,
    returns: pd.Series,
    benchmark: pd.Series,
    equity_events: pd.DataFrame,
    orders: pd.DataFrame,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    equity = (1 + _finite_series(returns).fillna(0.0)).cumprod()
    benchmark_equity = (1 + _finite_series(benchmark.reindex(returns.index)).fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("策略净值", "基准净值", "回撤"))
    figure.add_trace(go.Scatter(x=equity.index, y=equity, name="策略净值"), row=1, col=1)
    figure.add_trace(go.Scatter(x=benchmark_equity.index, y=benchmark_equity, name="基准净值"), row=2, col=1)
    figure.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="回撤", fill="tozeroy"), row=3, col=1)
    figure.update_layout(title=manifest.get("name", "原生 Python 策略报告"), template="plotly_white", height=980)
    scalar_metrics = "".join(f"<li><b>{key}</b>: {value}</li>" for key, value in metrics.items() if not isinstance(value, (dict, list)))
    orders_html = orders.to_html(index=False, border=0) if not orders.empty else "<p>本次没有订单。</p>"
    path.write_text(
        "<html><meta charset='utf-8'><body>"
        f"<h1>{manifest.get('name', '原生 Python 策略报告')}</h1><h2>策略指标</h2><ul>{scalar_metrics}</ul>"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}<h2>订单、成交与拒单</h2>{orders_html}"
        "</body></html>",
        encoding="utf-8",
    )
