from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from html import escape
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


def _format_metric(value: Any) -> str:
    if value is None:
        return "不适用"
    if isinstance(value, float):
        if not pd.notna(value) or not float("-inf") < value < float("inf"):
            return "不适用"
        return f"{value:.6f}"
    return str(value)


def _flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(_flatten_metrics(value, name))
        elif not isinstance(value, list):
            rows.append((name, value))
    return rows


def _metric_list(metrics: dict[str, Any]) -> str:
    return "".join(
        f"<li><b>{escape(key)}</b>: {escape(_format_metric(value))}</li>"
        for key, value in _flatten_metrics(metrics)
    )


_FACTOR_METRIC_LABELS = (
    ("方向调整 IC", "directional_ic_mean"),
    ("方向调整 Rank IC", "directional_rank_ic_mean"),
    ("方向调整 ICIR", "directional_icir"),
    ("方向调整 Rank ICIR", "directional_rank_icir"),
    ("因子输出覆盖率", "factor_output_coverage"),
    ("可评价覆盖率", "evaluable_coverage"),
    ("方向调整单调性", "directional_monotonicity"),
    ("平均换手", "mean_turnover"),
    ("多空累计收益（成本后）", "long_short_total_return"),
    ("多空年化收益（成本后）", "long_short_annual_return"),
    ("多空年化波动率（成本后）", "long_short_annual_volatility"),
    ("多空 Sharpe（成本后）", "long_short_sharpe"),
    ("多空最大回撤（成本后）", "long_short_max_drawdown"),
    ("多空累计收益（成本前）", "long_short_cost_before_total_return"),
    ("多空年化收益（成本前）", "long_short_cost_before_annual_return"),
    ("多空年化波动率（成本前）", "long_short_cost_before_annual_volatility"),
    ("多空 Sharpe（成本前）", "long_short_cost_before_sharpe"),
    ("多空最大回撤（成本前）", "long_short_cost_before_max_drawdown"),
)


def _factor_metric_value(metrics: dict[str, Any], key: str) -> Any:
    if key in metrics:
        return metrics[key]
    if key.startswith("long_short_cost_before_"):
        return metrics.get("cost_before", {}).get(key.removeprefix("long_short_cost_before_"))
    if key.startswith("long_short_"):
        return metrics.get("cost_after", {}).get(key.removeprefix("long_short_"))
    return None


def _factor_metric_table(metrics: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<th>{escape(label)}</th>"
        f"<td>{escape(_format_metric(_factor_metric_value(metrics, key)))}</td>"
        "</tr>"
        for label, key in _FACTOR_METRIC_LABELS
    )
    return (
        "<table><thead><tr><th>指标</th><th>数值</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _group_summary_table(groups: pd.DataFrame, cumulative: pd.DataFrame) -> str:
    if groups.empty or not {"group", "return"}.issubset(groups.columns):
        return "<p>单资产时序诊断不适用横截面分组收益。</p>"
    summary = groups.groupby("group", as_index=False).agg(
        平均单期收益=("return", "mean"), 平均资产数=("assets", "mean")
    )
    summary["累计收益"] = [
        cumulative[group_id].dropna().iloc[-1]
        if group_id in cumulative and not cumulative[group_id].dropna().empty
        else None
        for group_id in summary["group"]
    ]
    return summary.to_html(index=False, border=0, float_format=lambda value: f"{value:.6f}")


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
    if {"group", "timestamp", "return"}.issubset(groups.columns):
        for group_id, item in groups.groupby("group", sort=True):
            values = _finite_series(item["return"])
            if values.notna().any():
                figure.add_trace(
                    go.Scatter(x=item["timestamp"], y=values, name=f"G{group_id}"),
                    row=1,
                    col=1,
                )
    for column in cumulative.columns:
        values = _finite_series(cumulative[column])
        if values.notna().any():
            figure.add_trace(
                go.Scatter(x=cumulative.index, y=values, name=f"G{column} 累计"),
                row=2,
                col=1,
            )
    if not long_short.empty and "net_return" in long_short:
        values = _finite_series(long_short["net_return"])
        equity = (1 + values.fillna(0.0)).cumprod().where(values.notna())
        if equity.notna().any():
            figure.add_trace(
                go.Scatter(
                    x=long_short.loc[equity.index, "timestamp"],
                    y=equity,
                    name="多空净值（成本后）",
                ),
                row=3,
                col=1,
            )
    figure.update_layout(title=manifest.get("name", "原生 Python 因子报告"), template="plotly_white", height=980)
    path.write_text(
        "<html><meta charset='utf-8'><style>"
        "body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;}"
        "table{border-collapse:collapse;margin:12px 0 24px;min-width:460px;}"
        "th,td{border:1px solid #d9d9d9;padding:8px 12px;text-align:left;}"
        "th{background:#f5f7fa;}"
        "</style><body>"
        f"<h1>{escape(str(manifest.get('name', '原生 Python 因子报告')))}</h1>"
        "<p>分组、前瞻收益和成本口径均来自脚本的 set_factor_evaluation 声明。</p>"
        f"<p><b>回测区间：</b>{escape(str(metrics.get('actual_evaluation_start', '不适用')))} 至 {escape(str(metrics.get('actual_evaluation_end', '不适用')))}</p>"
        f"<p><b>平台设定区间：</b>{escape(str(metrics.get('platform_backtest_start', '不适用')))} 至 {escape(str(metrics.get('platform_backtest_end', '不适用')))}</p>"
        f"<p><b>提示：</b>{escape('；'.join(metrics.get('warnings', [])) or '无')}</p>"
        f"<h2>因子指标</h2>{_factor_metric_table(metrics)}"
        f"<h2>分组收益汇总</h2>{_group_summary_table(groups, cumulative)}"
        "<h2>Group Cumulative Returns</h2>"
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
    if equity.notna().any():
        figure.add_trace(go.Scatter(x=equity.index, y=equity, name="策略净值"), row=1, col=1)
    if benchmark_equity.notna().any():
        figure.add_trace(go.Scatter(x=benchmark_equity.index, y=benchmark_equity, name="基准净值"), row=2, col=1)
    if drawdown.notna().any():
        figure.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="回撤", fill="tozeroy"), row=3, col=1)
    figure.update_layout(title=manifest.get("name", "原生 Python 策略报告"), template="plotly_white", height=980)
    orders_html = orders.to_html(index=False, border=0) if not orders.empty else "<p>本次没有订单。</p>"
    path.write_text(
        "<html><meta charset='utf-8'><body>"
        f"<h1>{escape(str(manifest.get('name', '原生 Python 策略报告')))}</h1>"
        f"<h2>策略指标</h2><ul>{_metric_list(metrics)}</ul>"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}<h2>订单、成交与拒单</h2>{orders_html}"
        "</body></html>",
        encoding="utf-8",
    )
