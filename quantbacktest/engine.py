from __future__ import annotations

import shutil
import sqlite3
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
from quantbacktest.artifacts import (
    create_run_dir,
    render_report,
    render_research_report,
    write_json,
)
from quantbacktest.cuda import CudaExecutionError, create_cuda_factor_context, require_cuda
from quantbacktest.cuda_research import evaluate_factor_research_cuda
from quantbacktest.cuda_strategy import (
    cuda_cross_sectional_positions,
    cuda_factor_diagnostics,
    cuda_forward_returns,
    cuda_simulate_portfolio,
    cuda_single_asset_positions,
)
from quantbacktest.factors import FactorContext, execute_factor, execute_factor_cuda
from quantbacktest.library import LibraryError, register_completed_run
from quantbacktest.ml import train_oos_model
from quantbacktest.research import evaluate_factor_research
from quantbacktest.schemas import DebugMode, EvaluationMode, RunSpec, StrategyMode


@dataclass
class RunResult:
    run_dir: Path
    metrics: dict[str, Any]
    warnings: list[str]
    candidate: dict[str, str] | None = None
    candidate_registration_error: str | None = None


@dataclass
class SimulationResult:
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    risk_events: pd.DataFrame
    cash_bar_ratio: float
    termination_reason: str | None = None
    termination_time: pd.Timestamp | None = None


def _write_cuda_failure(
    output_root: Path, spec: RunSpec, error: CudaExecutionError
) -> Path:
    """留下结构化失败工件，避免 CUDA 请求失败后被误判为 CPU 运行。"""
    run_dir = create_run_dir(output_root, f"{spec.name}_cuda_failed")
    write_json(run_dir / "run_spec.json", spec.model_dump(mode="json"))
    write_json(run_dir / "cuda_failure.json", error.as_dict())
    (run_dir / "report.html").write_text(
        "<html><meta charset='utf-8'><body><h1>CUDA 回测失败</h1>"
        f"<p><b>错误代码：</b>{error.code}</p><p><b>原因：</b>{error}</p>"
        f"<p><b>修复建议：</b>{error.suggestion}</p></body></html>",
        encoding="utf-8",
    )
    return run_dir


def _completed_run_result(
    run_dir: Path,
    metrics: dict[str, Any],
    warnings: list[str],
    candidate_library_root: Path | None,
) -> RunResult:
    """登记完整运行为候选；登记异常不删除已写入的回测工件。"""
    try:
        candidate = register_completed_run(run_dir, candidate_library_root)
    except (LibraryError, OSError, sqlite3.Error) as exc:
        message = f"候选登记失败：{exc}"
        write_json(
            run_dir / "candidate_registration.json",
            {"status": "failed", "error": message, "run_dir": str(run_dir)},
        )
        return RunResult(run_dir, metrics, warnings, candidate_registration_error=message)
    write_json(
        run_dir / "candidate_registration.json",
        {"status": "registered", "candidate": candidate, "run_dir": str(run_dir)},
    )
    return RunResult(run_dir, metrics, warnings, candidate=candidate)


def _cuda_simulation_result(
    frame: pd.DataFrame, positions: pd.DataFrame, spec: RunSpec, device_id: int
) -> SimulationResult:
    """Convert a completed CUDA state scan into the existing labelled artifacts."""
    result = cuda_simulate_portfolio(frame, positions, spec, device_id)
    timestamps = result["timestamps"]
    symbols = result["symbols"]
    valid = result["valid"]
    returns = pd.Series(result["returns"][valid], index=timestamps[valid], name="strategy_return")
    weights = result["weights"]
    reasons = {1: "rebalance", 2: "holding_period_exit", 3: "drawdown_stop"}
    trade_rows: list[dict[str, Any]] = []
    for time_index, timestamp in enumerate(timestamps):
        reason = reasons.get(int(result["reasons"][time_index]))
        if reason is None:
            continue
        previous = weights[time_index - 1] if time_index else np.zeros(len(symbols))
        current = weights[time_index]
        changes = current - previous
        for symbol_index, change in enumerate(changes):
            if not change:
                continue
            turnover = abs(float(change))
            execution = frame.loc[
                (frame["timestamp"] == timestamp) & (frame["symbol"] == symbols[symbol_index]), "open"
            ].iloc[0]
            trade_rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbols[symbol_index],
                    "previous_weight": float(previous[symbol_index]),
                    "target_weight": float(current[symbol_index]),
                    "execution_price": float(execution),
                    "turnover": turnover,
                    "cost": turnover * (spec.costs.fee_bps + spec.costs.slippage_bps) / 10_000,
                    "reason": reason,
                }
            )
    trades = pd.DataFrame(
        trade_rows,
        columns=["timestamp", "symbol", "previous_weight", "target_weight", "execution_price", "turnover", "cost", "reason"],
    )
    risk_rows: list[dict[str, Any]] = []
    trigger_indices = np.flatnonzero(result["trigger"])
    rebalance_indices = np.flatnonzero(result["reasons"] == 1)
    for trigger_index in trigger_indices:
        liquidation_candidates = np.flatnonzero((result["liquidation"] == 1) & (np.arange(len(timestamps)) > trigger_index))
        liquidation_index = int(liquidation_candidates[0]) if len(liquidation_candidates) else None
        reentry_candidates = rebalance_indices[rebalance_indices > (liquidation_index if liquidation_index is not None else trigger_index)]
        risk_rows.append(
            {
                "trigger_time": timestamps[trigger_index],
                "liquidation_time": timestamps[liquidation_index] if liquidation_index is not None else None,
                "trigger_drawdown": float(result["trigger_drawdown"][trigger_index]),
                "peak_equity": float(result["trigger_peak"][trigger_index]),
                "trigger_equity": float(result["trigger_equity"][trigger_index]),
                "liquidation_cost": float(result["costs"][liquidation_index]) if liquidation_index is not None else None,
                "liquidation_equity": float(result["liquidation_equity"][liquidation_index]) if liquidation_index is not None else None,
                "reentry_time": timestamps[int(reentry_candidates[0])] if len(reentry_candidates) else None,
            }
        )
    risk_events = pd.DataFrame(
        risk_rows,
        columns=["trigger_time", "liquidation_time", "trigger_drawdown", "peak_equity", "trigger_equity", "liquidation_cost", "liquidation_equity", "reentry_time"],
    )
    terminated = np.flatnonzero(~valid)
    termination_time = timestamps[int(terminated[0])] if len(terminated) else None
    termination_reason = (
        "组合净值降至零或以下；CUDA 状态机已停止合成收益模拟，后续账户年化指标无经济解释"
        if termination_time is not None
        else None
    )
    positions = positions.copy()
    positions["turnover"] = 0.0
    positions["cost"] = 0.0
    return SimulationResult(
        returns=returns,
        positions=positions,
        trades=trades,
        risk_events=risk_events,
        cash_bar_ratio=float((result["reasons"] == 3).sum() / len(timestamps)),
        termination_reason=termination_reason,
        termination_time=termination_time,
    )


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
        side_gross_exposure = strategy.gross_exposure / (2 if strategy.long_short == "market_neutral" else 1)
        if strategy.weighting == "score":
            long_values = selected.loc[long_index, "factor"].abs()
            long_weights = (
                long_values / long_values.sum() * side_gross_exposure
                if long_values.sum()
                else pd.Series(side_gross_exposure / len(long_index), index=long_index)
            )
        else:
            long_weights = pd.Series(side_gross_exposure / len(long_index), index=long_index)
        active.loc[long_index, "target_weight"] = long_weights.clip(upper=strategy.max_weight)
        if strategy.long_short == "market_neutral":
            active.loc[short_index, "target_weight"] = -side_gross_exposure / len(short_index)
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


def _simulate_portfolio(frame: pd.DataFrame, positions: pd.DataFrame, spec: RunSpec) -> SimulationResult:
    """Simulate close signals, next-open execution and an optional portfolio drawdown stop."""
    positions = positions.sort_values(["timestamp", "symbol"]).copy()
    timestamps = pd.Index(frame["timestamp"].drop_duplicates().sort_values())
    if len(timestamps) < 2:
        raise ValueError("至少需要两根 K 线才能执行下一开盘成交的回测")
    bars = {
        timestamp: group.set_index("symbol")[["open", "close"]]
        for timestamp, group in frame.groupby("timestamp", sort=True)
    }
    signal_targets = {
        timestamp: group.loc[group["target_weight"] != 0, ["symbol", "target_weight"]]
        .set_index("symbol")["target_weight"]
        .to_dict()
        for timestamp, group in positions.groupby("timestamp", sort=True)
    }
    cost_rate = (spec.costs.fee_bps + spec.costs.slippage_bps) / 10_000
    stop = spec.strategy.risk_control.max_drawdown_stop

    current_weights: dict[str, float] = {}
    pending_targets: dict[pd.Timestamp, tuple[dict[str, float], int]] = {}
    active_exit_index: int | None = None
    force_exit_at: pd.Timestamp | None = None
    pending_risk_event: int | None = None
    equity = 1.0
    peak_equity = 1.0
    previous_closes: pd.Series | None = None
    returns: dict[pd.Timestamp, float] = {}
    trade_records: list[dict[str, Any]] = []
    risk_records: list[dict[str, Any]] = []
    in_stop_cooldown = False
    stop_cash_bars = 0
    termination_reason: str | None = None
    termination_time: pd.Timestamp | None = None

    def _weighted_return(weights: dict[str, float], start: pd.Series, end: pd.Series) -> float:
        result = 0.0
        for symbol, weight in weights.items():
            if symbol not in start.index or symbol not in end.index:
                raise ValueError(f"持仓 {symbol} 在 {timestamp} 缺少开盘或收盘价格")
            start_price, end_price = float(start[symbol]), float(end[symbol])
            if start_price <= 0 or end_price <= 0:
                raise ValueError(f"持仓 {symbol} 在 {timestamp} 存在非正价格")
            result += weight * (end_price / start_price - 1.0)
        return result

    def _trade_to(
        timestamp: pd.Timestamp, desired: dict[str, float], reason: str, prices: pd.Series
    ) -> float:
        nonlocal current_weights
        turnover = 0.0
        for symbol in sorted(set(current_weights) | set(desired)):
            previous_weight = current_weights.get(symbol, 0.0)
            target_weight = desired.get(symbol, 0.0)
            change = target_weight - previous_weight
            if not change:
                continue
            if symbol not in prices.index:
                raise ValueError(f"交易 {symbol} 在 {timestamp} 缺少开盘成交价格")
            item_turnover = abs(change)
            item_cost = item_turnover * cost_rate
            turnover += item_turnover
            trade_records.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "previous_weight": previous_weight,
                    "target_weight": target_weight,
                    "execution_price": float(prices[symbol]),
                    "turnover": item_turnover,
                    "cost": item_cost,
                    "reason": reason,
                }
            )
        current_weights = {symbol: weight for symbol, weight in desired.items() if weight}
        return turnover * cost_rate

    for index, timestamp in enumerate(timestamps):
        bar = bars[timestamp]
        opens, closes = bar["open"], bar["close"]
        equity_before_bar = equity
        just_liquidated = False
        if previous_closes is not None and current_weights:
            equity *= 1.0 + _weighted_return(current_weights, previous_closes, opens)

        forced_exit = force_exit_at == timestamp
        if forced_exit:
            execution_cost = _trade_to(timestamp, {}, "drawdown_stop", opens)
            equity *= 1.0 - execution_cost
            active_exit_index = None
            force_exit_at = None
            peak_equity = equity
            just_liquidated = True
            if pending_risk_event is not None:
                risk_records[pending_risk_event].update(
                    {
                        "liquidation_cost": execution_cost,
                        "liquidation_equity": equity,
                    }
                )
                pending_risk_event = None
        elif timestamp in pending_targets:
            desired, active_exit_index = pending_targets.pop(timestamp)
            equity *= 1.0 - _trade_to(timestamp, desired, "rebalance", opens)
            if desired:
                in_stop_cooldown = False
        elif active_exit_index == index:
            equity *= 1.0 - _trade_to(timestamp, {}, "holding_period_exit", opens)
            active_exit_index = None

        if current_weights:
            equity *= 1.0 + _weighted_return(current_weights, opens, closes)
        elif in_stop_cooldown:
            stop_cash_bars += 1
        returns[timestamp] = equity / equity_before_bar - 1.0
        if equity <= 0:
            termination_reason = "组合净值降至零或以下；停止合成收益模拟，后续账户年化指标无经济解释"
            termination_time = timestamp
            break
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1.0

        if (
            stop.enabled
            and not just_liquidated
            and force_exit_at is None
            and drawdown <= -(stop.threshold or 0)
            and index + 1 < len(timestamps)
        ):
            force_exit_at = timestamps[index + 1]
            in_stop_cooldown = True
            risk_records.append(
                {
                    "trigger_time": timestamp,
                    "liquidation_time": force_exit_at,
                    "trigger_drawdown": drawdown,
                    "peak_equity": peak_equity,
                    "trigger_equity": equity,
                    "liquidation_cost": None,
                    "liquidation_equity": None,
                    "reentry_time": None,
                }
            )
            pending_risk_event = len(risk_records) - 1
            previous_closes = closes
            continue

        if index + 1 < len(timestamps) and timestamp in signal_targets:
            target_weights = signal_targets[timestamp]
            pending_targets[timestamps[index + 1]] = (
                target_weights,
                min(index + spec.strategy.execution.holding_bars + 1, len(timestamps) - 1),
            )
            if target_weights and risk_records and risk_records[-1]["reentry_time"] is None:
                risk_records[-1]["reentry_time"] = timestamps[index + 1]
        previous_closes = closes

    trades = pd.DataFrame(trade_records)
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "previous_weight",
                "target_weight",
                "execution_price",
                "turnover",
                "cost",
                "reason",
            ]
        )
    risk_events = pd.DataFrame(risk_records)
    if risk_events.empty:
        risk_events = pd.DataFrame(
            columns=[
                "trigger_time",
                "liquidation_time",
                "trigger_drawdown",
                "peak_equity",
                "trigger_equity",
                "liquidation_cost",
                "liquidation_equity",
                "reentry_time",
            ]
        )
    positions["turnover"] = 0.0
    positions["cost"] = 0.0
    return SimulationResult(
        returns=pd.Series(returns, name="strategy_return"),
        positions=positions,
        trades=trades,
        risk_events=risk_events,
        cash_bar_ratio=stop_cash_bars / len(timestamps),
        termination_reason=termination_reason,
        termination_time=termination_time,
    )


def _trace(
    frame: pd.DataFrame,
    positions: pd.DataFrame | None,
    trades: pd.DataFrame | None = None,
    risk_events: pd.DataFrame | None = None,
) -> dict[str, Any]:
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
    if trades is not None:
        trace["execution"] = {"trades": len(trades), "sample": trades.head(20).to_dict("records")}
    if risk_events is not None:
        trace["risk_control"] = {
            "stop_events": len(risk_events),
            "events": risk_events.to_dict("records"),
        }
    return trace


def run_backtest(
    spec: RunSpec,
    project_root: Path | None = None,
    candidate_library_root: Path | None = None,
) -> RunResult:
    """运行统一的截面或单资产研究，所有工件落在调用项目结果目录。"""
    project_root = project_root or Path.cwd()
    data_spec = spec.data.model_copy(deep=True)
    if not data_spec.path.is_absolute():
        data_spec.path = (project_root / data_spec.path).resolve()
    if data_spec.universe and data_spec.universe.membership_path and not data_spec.universe.membership_path.is_absolute():
        data_spec.universe.membership_path = (project_root / data_spec.universe.membership_path).resolve()
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
    if spec.compute is not None and spec.compute.backend == "cuda":
        try:
            require_cuda(spec.compute.device_id)
            cuda_context = create_cuda_factor_context(data, spec.factor.parameters, spec.compute.device_id)
            factor, factor_meta = execute_factor_cuda(factor_path, spec.factor.callable, cuda_context)
            raise CudaExecutionError(
                "cuda_pipeline_stage_unsupported",
                "CUDA 因子已验证，但当前版本尚未完成研究评估与策略执行阶段的 CUDA 实现。",
                "请暂时使用 compute.backend: cpu；该运行不会被错误标记为 CUDA 回测。",
            )
        except CudaExecutionError as exc:
            failure_dir = _write_cuda_failure(output_root, spec, exc)
            raise CudaExecutionError(
                exc.code,
                str(exc),
                f"{exc.suggestion} 详细诊断与 HTML 报告已写入：{failure_dir}",
            ) from exc
    factor, factor_meta = execute_factor(factor_path, spec.factor.callable, FactorContext(data, spec.factor.parameters))
    supported_modes = factor_meta["meta"]["supported_modes"]
    is_research = spec.evaluation is not None and spec.evaluation.mode is EvaluationMode.factor_research
    required_mode = "cross_sectional" if is_research else spec.strategy.mode.value if spec.strategy else ""
    if required_mode not in supported_modes:
        raise ValueError(f"因子不支持模式 {required_mode}；支持：{supported_modes}")
    frame = data.merge(factor, on=["timestamp", "symbol"], how="left", validate="one_to_one")
    warnings = ["流动性筛选未验证"]
    if spec.data.adapter == "crypto_top50":
        warnings.append("静态 Top50 可能存在幸存者偏差；未验证历史上市与下市信息")
    if spec.provenance.proxy_data_note:
        warnings.append(f"代理数据：{spec.provenance.proxy_data_note}")
    if spec.costs.funding_bps_per_bar is None and spec.data.market.value == "linear_perp":
        warnings.append("线性合约未计入资金费")

    run_dir = create_run_dir(output_root, spec.name)
    shutil.copy2(factor_path, run_dir / "factor_snapshot.py")
    base_meta = {"name": spec.name, "warnings": warnings, "data": data_meta, "factor": factor_meta}
    write_json(run_dir / "run_spec.json", spec.model_dump(mode="json"))

    if is_research:
        assert spec.evaluation is not None and spec.evaluation.research is not None
        research = evaluate_factor_research(frame, spec.evaluation.research, data_spec.universe)
        ml_meta: dict[str, Any] = {"enabled": False}
        if spec.ml.enabled:
            ml_input = research.panel.rename(columns={"formation_time": "timestamp"})
            scored, ml_meta = train_oos_model(
                ml_input,
                spec.ml,
                run_dir / "model.pkl",
                label_end_column="return_end_time",
            )
            predictions = scored[["timestamp", "symbol", "factor"]]
            scored_frame = frame.drop(columns="factor").merge(
                predictions, on=["timestamp", "symbol"], how="left", validate="one_to_one"
            )
            research = evaluate_factor_research(scored_frame, spec.evaluation.research, data_spec.universe)
        warnings.extend(
            [
                "因子研究报告衡量统计型未来收益，不是账户净值或可执行策略。",
                "未提供时间点资产池清单时，报告按 data.symbols 的静态资产池计算。",
            ]
        )
        base_meta["evaluation"] = spec.evaluation.model_dump(mode="json")
        base_meta["ml"] = ml_meta
        write_json(run_dir / "metadata.json", base_meta)
        metrics = {**research.metrics, "warnings": warnings}
        research.panel.to_csv(run_dir / "research_panel.csv", index=False)
        research.group_returns.to_csv(run_dir / "research_group_returns.csv", index=False)
        research.spread.to_csv(run_dir / "research_spread.csv", index=False)
        research.contributions.to_csv(run_dir / "research_contributions.csv", index=False)
        research.leave_one_out.to_csv(run_dir / "research_leave_one_out.csv", index=False)
        research.ic_decay.to_csv(run_dir / "research_ic_decay.csv", index=False)
        research.periods.to_csv(run_dir / "research_periods.csv", index=False)
        research.skipped_periods.to_csv(run_dir / "research_skipped_periods.csv", index=False)
        if spec.debug.mode in {DebugMode.trace, DebugMode.replay}:
            write_json(
                run_dir / "debug_trace.json",
                {
                    "data": {"rows": len(data), "symbols": int(data["symbol"].nunique())},
                    "research": {
                        "formation_periods": research.metrics["formation_periods"],
                        "skipped_periods": research.skipped_periods.to_dict("records"),
                        "sample": research.panel.head(20).to_dict("records"),
                    },
                },
            )
        write_json(run_dir / "metrics.json", metrics)
        render_research_report(run_dir / "report.html", research.group_returns, research.spread, metrics, base_meta)
        return _completed_run_result(run_dir, metrics, warnings, candidate_library_root)

    assert spec.strategy is not None
    frame = _forward_returns(frame, spec.strategy.execution.holding_bars)
    frame, ml_meta = train_oos_model(frame, spec.ml, run_dir / "model.pkl")
    if spec.evaluation is None:
        warnings.append("legacy strategy simulation：该结果不是默认的因子研究结论")
    if spec.strategy.long_short == "market_neutral" and spec.data.market.value == "spot":
        warnings.append("现货价格构造的空头为合成收益模拟，未建模借币、保证金或强平，非可执行账户")
    base_meta["ml"] = ml_meta
    base_meta["risk_control"] = spec.strategy.risk_control.model_dump(mode="json")
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
    simulation = _simulate_portfolio(frame, positions, spec)
    returns = simulation.returns
    positions = simulation.positions
    benchmark = equal_weight_benchmark(frame)
    diagnostics = factor_diagnostics(frame)
    metrics = performance_metrics(returns, spec.data.frequency)
    metrics.update({key: value for key, value in diagnostics.items() if key != "ic_series"})
    metrics["total_turnover"] = float(simulation.trades["turnover"].sum())
    metrics["total_cost"] = float(simulation.trades["cost"].sum())
    metrics["risk_stop_trigger_count"] = len(simulation.risk_events)
    metrics["risk_stop_cost"] = float(simulation.risk_events["liquidation_cost"].fillna(0).sum())
    metrics["stop_cash_bar_ratio"] = simulation.cash_bar_ratio
    metrics["simulation_termination_reason"] = simulation.termination_reason
    metrics["simulation_termination_time"] = (
        simulation.termination_time.isoformat() if simulation.termination_time is not None else None
    )
    if simulation.termination_reason:
        warnings.append(simulation.termination_reason)
    metrics["warnings"] = warnings
    returns.to_csv(run_dir / "returns.csv", header=True)
    positions.to_csv(run_dir / "positions.csv", index=False)
    simulation.trades.to_csv(run_dir / "trades.csv", index=False)
    simulation.risk_events.to_csv(run_dir / "risk_events.csv", index=False)
    diagnostics["ic_series"].rename("ic").to_csv(run_dir / "ic.csv", header=True)
    quantile = quantile_returns(frame, spec.strategy.quantiles or 5)
    quantile.to_csv(run_dir / "quantile_returns.csv")
    if spec.debug.mode in {DebugMode.trace, DebugMode.replay}:
        write_json(run_dir / "debug_trace.json", _trace(frame, positions, simulation.trades, simulation.risk_events))
    write_json(run_dir / "metrics.json", metrics)
    render_report(run_dir / "report.html", returns, benchmark, metrics, base_meta, simulation.risk_events)
    return _completed_run_result(run_dir, metrics, warnings, candidate_library_root)
