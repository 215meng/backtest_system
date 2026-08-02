from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantbacktest.adapters import load_market_data
from quantbacktest.analytics import (
    equal_weight_benchmark,
    factor_diagnostics,
    performance_metrics,
    quantile_returns,
)
from quantbacktest.artifacts import create_run_dir, render_report, write_json
from quantbacktest.factors import FactorContext, execute_factor
from quantbacktest.ml import train_oos_model
from quantbacktest.schemas import DebugMode, RunSpec, StrategyMode


@dataclass
class RunResult:
    run_dir: Path
    metrics: dict[str, Any]
    warnings: list[str]


def _forward_returns(frame: pd.DataFrame, holding_bars: int) -> pd.DataFrame:
    frame = frame.copy()
    grouped = frame.groupby("symbol", group_keys=False)
    frame["entry_open"] = grouped["open"].shift(-1)
    frame["exit_open"] = grouped["open"].shift(-(holding_bars + 1))
    frame["forward_return"] = frame["exit_open"] / frame["entry_open"] - 1.0
    return frame


def _cross_sectional_positions(frame: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    strategy = spec.strategy
    active = frame.dropna(subset=["factor", "forward_return"]).copy()
    if strategy.rebalance_bars > 1:
        rebalance_times = active["timestamp"].drop_duplicates().sort_values().iloc[::strategy.rebalance_bars]
        active = active[active["timestamp"].isin(rebalance_times)]
    active["target_weight"] = 0.0
    for index in active.groupby("timestamp").groups.values():
        selected = active.loc[index].sort_values("factor")
        min_assets = (spec.data.universe.min_assets if spec.data.universe else 10)
        if len(selected) < min_assets:
            continue
        if strategy.selection == "top_k":
            n = min(strategy.top_k or 1, len(selected))
            long_index = selected.tail(n).index
            short_index = selected.head(n).index
        else:
            n = max(1, len(selected) // (strategy.quantiles or 2))
            long_index = selected.tail(n).index
            short_index = selected.head(n).index
        if strategy.weighting == "score":
            long_values = selected.loc[long_index, "factor"].abs()
            long_weights = long_values / long_values.sum() if long_values.sum() else pd.Series(1 / len(long_index), index=long_index)
        else:
            long_weights = pd.Series(1 / len(long_index), index=long_index)
        active.loc[long_index, "target_weight"] = long_weights.clip(upper=strategy.max_weight)
        if strategy.long_short == "market_neutral":
            active.loc[short_index, "target_weight"] = -1 / len(short_index)
    return active


def _single_asset_positions(frame: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    rule = spec.strategy.signal_rule
    assert rule is not None
    active = frame[frame["symbol"] == spec.strategy.symbol].dropna(subset=["factor", "forward_return"]).copy()
    if rule.kind == "sign":
        active["target_weight"] = np.sign(active["factor"]) * rule.position_size
    else:
        active["target_weight"] = np.where(active["factor"] > rule.long_above, rule.position_size, 0.0)
        if rule.short_below is not None:
            active.loc[active["factor"] < rule.short_below, "target_weight"] = -rule.position_size
    return active


def _portfolio_returns(positions: pd.DataFrame, spec: RunSpec) -> tuple[pd.Series, pd.DataFrame]:
    positions = positions.sort_values(["timestamp", "symbol"]).copy()
    positions["previous_weight"] = positions.groupby("symbol")["target_weight"].shift().fillna(0.0)
    positions["turnover"] = (positions["target_weight"] - positions["previous_weight"]).abs()
    costs = (spec.costs.fee_bps + spec.costs.slippage_bps) / 10_000
    positions["cost"] = positions["turnover"] * costs
    positions["contribution"] = positions["target_weight"] * positions["forward_return"] - positions["cost"]
    returns = positions.groupby("timestamp")["contribution"].sum().rename("strategy_return")
    return returns, positions


def _trace(frame: pd.DataFrame, positions: pd.DataFrame | None) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "data": {
            "rows": len(frame),
            "symbols": int(frame["symbol"].nunique()),
            "missing_close_rate": float(frame["close"].isna().mean()),
        },
        "factor": {
            "coverage": float(frame["factor"].notna().mean()),
            "sample": frame[["timestamp", "symbol", "close", "factor"]].dropna().head(20).to_dict("records"),
        },
    }
    if positions is not None:
        trace["portfolio"] = {
            "positions": len(positions),
            "turnover": float(positions["turnover"].sum()),
            "sample": positions[["timestamp", "symbol", "factor", "target_weight", "forward_return", "cost"]]
            .head(20)
            .to_dict("records"),
        }
    return trace


def run_backtest(spec: RunSpec, project_root: Path | None = None) -> RunResult:
    """运行统一的截面或单资产研究，所有工件落在调用项目结果目录。"""
    project_root = project_root or Path.cwd()
    data_spec = spec.data.model_copy(deep=True)
    if not data_spec.path.is_absolute():
        data_spec.path = (project_root / data_spec.path).resolve()
    factor_path = spec.factor.module_path
    if not factor_path.is_absolute():
        factor_path = (project_root / factor_path).resolve()
    output_root = spec.output.root if spec.output.root.is_absolute() else project_root / spec.output.root

    data, data_meta = load_market_data(data_spec)
    min_history = data_spec.universe.min_history_bars if data_spec.universe else 0
    if min_history:
        data = data.sort_values(["symbol", "timestamp"]).copy()
        data["history_bars"] = data.groupby("symbol").cumcount()
        data = data[data["history_bars"] >= min_history].drop(columns="history_bars")
    if spec.debug.timestamp_range:
        start, end = (pd.to_datetime(value, utc=True) for value in spec.debug.timestamp_range)
        data = data[data["timestamp"].between(start, end)]
    if spec.debug.symbols:
        data = data[data["symbol"].isin(spec.debug.symbols)]
    if data.empty:
        raise ValueError("调试时间窗口或币种过滤后没有可用数据")
    factor, factor_meta = execute_factor(factor_path, spec.factor.callable, FactorContext(data, spec.factor.parameters))
    supported_modes = factor_meta["meta"]["supported_modes"]
    if spec.strategy.mode.value not in supported_modes:
        raise ValueError(f"因子不支持策略模式 {spec.strategy.mode.value}；支持：{supported_modes}")
    frame = data.merge(factor, on=["timestamp", "symbol"], how="left", validate="one_to_one")
    frame = _forward_returns(frame, spec.strategy.execution.holding_bars)
    warnings = ["流动性筛选未验证"]
    if spec.data.adapter == "crypto_top50":
        warnings.append("静态 Top50 可能存在幸存者偏差；未验证历史上市与下市信息")
    if spec.provenance.proxy_data_note:
        warnings.append(f"代理数据：{spec.provenance.proxy_data_note}")
    if spec.costs.funding_bps_per_bar is None and spec.data.market.value == "linear_perp":
        warnings.append("线性合约未计入资金费")

    run_dir = create_run_dir(output_root, spec.name)
    shutil.copy2(factor_path, run_dir / "factor_snapshot.py")
    frame, ml_meta = train_oos_model(frame, spec.ml, run_dir / "model.pkl")
    base_meta = {"name": spec.name, "warnings": warnings, "data": data_meta, "factor": factor_meta}
    base_meta["ml"] = ml_meta
    write_json(run_dir / "run_spec.json", spec.model_dump(mode="json"))
    write_json(run_dir / "metadata.json", base_meta)
    if spec.debug.mode is DebugMode.dry_run:
        write_json(run_dir / "debug_trace.json", _trace(frame, None))
        return RunResult(run_dir, {"status": "dry_run_validated", "rows": len(frame)}, warnings)

    if spec.strategy.mode is StrategyMode.cross_sectional:
        positions = _cross_sectional_positions(frame, spec)
    else:
        positions = _single_asset_positions(frame, spec)
    if positions.empty:
        raise ValueError("没有可交易的有效信号；请检查因子覆盖、资产池和持有期")
    returns, positions = _portfolio_returns(positions, spec)
    benchmark = equal_weight_benchmark(frame)
    diagnostics = factor_diagnostics(frame)
    metrics = performance_metrics(returns, spec.data.frequency)
    metrics.update({key: value for key, value in diagnostics.items() if key != "ic_series"})
    metrics["total_turnover"] = float(positions["turnover"].sum())
    metrics["total_cost"] = float(positions["cost"].sum())
    metrics["warnings"] = warnings
    returns.to_csv(run_dir / "returns.csv", header=True)
    positions.to_csv(run_dir / "positions.csv", index=False)
    diagnostics["ic_series"].rename("ic").to_csv(run_dir / "ic.csv", header=True)
    quantile = quantile_returns(frame, spec.strategy.quantiles or 5)
    quantile.to_csv(run_dir / "quantile_returns.csv")
    if spec.debug.mode in {DebugMode.trace, DebugMode.replay}:
        write_json(run_dir / "debug_trace.json", _trace(frame, positions))
    write_json(run_dir / "metrics.json", metrics)
    render_report(run_dir / "report.html", returns, benchmark, metrics, base_meta)
    return RunResult(run_dir, metrics, warnings)
