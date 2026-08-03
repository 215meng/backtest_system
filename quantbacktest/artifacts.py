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


def _finite_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.map(lambda item: pd.notna(item) and float("-inf") < item < float("inf")))


def _metric_list(metrics: dict[str, Any]) -> str:
    return "".join(
        f"<li><b>{key}</b>: {value}</li>"
        for key, value in metrics.items()
        if not isinstance(value, (dict, list))
    )


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
        subplot_titles=("各分组单期收益", "分组累计收益", "多空净值（成本后）"),
    )
    for group_id, item in groups.groupby("group", sort=True):
        figure.add_trace(
            go.Scatter(x=item["timestamp"], y=_finite_series(item["return"]), name=f"G{group_id}"),
            row=1,
            col=1,
        )
    for column in cumulative.columns:
        figure.add_trace(
            go.Scatter(x=cumulative.index, y=_finite_series(cumulative[column]), name=f"G{column} 累计"),
            row=2,
            col=1,
        )
    if not long_short.empty:
        figure.add_trace(
            go.Scatter(
                x=long_short["timestamp"],
                y=(1 + _finite_series(long_short["net_return"])).cumprod(),
                name="多空净值",
            ),
            row=3,
            col=1,
        )
    figure.update_layout(title=manifest.get("name", "原生 Python 因子报告"), template="plotly_white", height=980)
    path.write_text(
        "<html><meta charset='utf-8'><body>"
        f"<h1>{manifest.get('name', '原生 Python 因子报告')}</h1>"
        "<p>分组、前瞻收益和成本口径均来自脚本的 set_factor_evaluation 声明。</p>"
        f"<h2>指标</h2><ul>{_metric_list(metrics)}</ul>"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}"
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
    orders_html = orders.to_html(index=False, border=0) if not orders.empty else "<p>本次没有订单。</p>"
    path.write_text(
        "<html><meta charset='utf-8'><body>"
        f"<h1>{manifest.get('name', '原生 Python 策略报告')}</h1>"
        f"<h2>策略指标</h2><ul>{_metric_list(metrics)}</ul>"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}<h2>订单、成交与拒单</h2>{orders_html}"
        "</body></html>",
        encoding="utf-8",
    )
