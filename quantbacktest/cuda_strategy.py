"""CuPy/Numba implementation for the numeric strategy path.

The public engine keeps labelled pandas tables only at its boundary so CSV and HTML
artifacts remain compatible.  Matrix maths, cross-sectional selection, IC and the
stateful stop scan execute on the requested CUDA device.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantbacktest.cuda import CudaExecutionError
from quantbacktest.schemas import RunSpec


def _layout(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex, tuple[str, ...]]:
    ordered = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    timestamps = pd.DatetimeIndex(ordered["timestamp"].drop_duplicates())
    symbols = tuple(sorted(ordered["symbol"].astype(str).unique()))
    expected = len(timestamps) * len(symbols)
    if len(ordered) != expected or ordered.duplicated(["timestamp", "symbol"]).any():
        raise CudaExecutionError(
            "cuda_strategy_non_rectangular_panel",
            "策略 CUDA 路径要求规则的 timestamp × symbol OHLCV 面板。",
            "在数据导入阶段补齐或排除缺失标的；系统不会回退到 Pandas/CPU 策略模拟。",
        )
    expected_index = pd.MultiIndex.from_product([timestamps, symbols], names=["timestamp", "symbol"])
    if not pd.MultiIndex.from_frame(ordered[["timestamp", "symbol"]]).equals(expected_index):
        raise CudaExecutionError(
            "cuda_strategy_panel_alignment_error",
            "策略 CUDA 面板的时间或标的排序不连续。",
            "确保输入 timestamp 为 UTC、symbol 稳定，并且每个截面具有相同的标的集合。",
        )
    return ordered, timestamps, symbols


def _matrix(frame: pd.DataFrame, column: str, shape: tuple[int, int], cp: Any) -> Any:
    if column not in frame:
        raise CudaExecutionError(
            "cuda_strategy_field_missing",
            f"策略 CUDA 路径缺少数值字段：{column}",
            "检查数据适配器与因子输出是否提供该字段。",
        )
    return cp.asarray(frame[column].to_numpy(dtype=np.float64, copy=False)).reshape(shape)


def cuda_forward_returns(frame: pd.DataFrame, holding_bars: int, device_id: int) -> pd.DataFrame:
    """Use CuPy to produce executable next-open forward returns."""
    import cupy as cp

    ordered, timestamps, symbols = _layout(frame)
    shape = (len(timestamps), len(symbols))
    with cp.cuda.Device(device_id):
        opens = _matrix(ordered, "open", shape, cp)
        nan_row = cp.full((holding_bars + 1, shape[1]), cp.nan)
        entry = cp.concatenate((opens[1:], cp.full((1, shape[1]), cp.nan)))
        exit_ = cp.concatenate((opens[holding_bars + 1 :], nan_row))
        forward = exit_ / entry - 1.0
        ordered["entry_open"] = cp.asnumpy(entry).reshape(-1)
        ordered["exit_open"] = cp.asnumpy(exit_).reshape(-1)
        ordered["forward_return"] = cp.asnumpy(forward).reshape(-1)
    return ordered


def cuda_cross_sectional_positions(frame: pd.DataFrame, spec: RunSpec, device_id: int) -> pd.DataFrame:
    """Rank and weight every rebalance cross-section on GPU."""
    import cupy as cp

    strategy = spec.strategy
    assert strategy is not None
    ordered, timestamps, symbols = _layout(frame)
    shape = (len(timestamps), len(symbols))
    with cp.cuda.Device(device_id):
        factor = _matrix(ordered, "factor", shape, cp)
        forward = _matrix(ordered, "forward_return", shape, cp)
        valid = cp.isfinite(factor) & cp.isfinite(forward)
        scores = cp.where(valid, factor, cp.inf)
        order = cp.argsort(scores, axis=1)
        rank = cp.empty_like(order)
        rank[cp.arange(shape[0])[:, None], order] = cp.arange(shape[1])[None, :]
        count = valid.sum(axis=1)
        if strategy.selection == "top_k":
            selected_count = cp.minimum(count, strategy.top_k or 1)
        else:
            selected_count = cp.maximum(1, count // (strategy.quantiles or 2))
        enough = count >= (spec.data.universe.min_assets if spec.data.universe else 10)
        selected_count = cp.where(enough, selected_count, 0)
        long_mask = valid & (rank >= (count - selected_count)[:, None]) & (selected_count[:, None] > 0)
        short_mask = valid & (rank < selected_count[:, None]) & (selected_count[:, None] > 0)
        side_exposure = strategy.gross_exposure / (2 if strategy.long_short == "market_neutral" else 1)
        if strategy.weighting == "score":
            long_scores = cp.where(long_mask, cp.abs(factor), 0.0)
            long_total = long_scores.sum(axis=1, keepdims=True)
            equal_long = long_mask / cp.maximum(selected_count[:, None], 1)
            long_weights = cp.where(long_total > 0, long_scores / long_total, equal_long) * side_exposure
        else:
            long_weights = long_mask / cp.maximum(selected_count[:, None], 1) * side_exposure
        target = cp.minimum(long_weights, strategy.max_weight)
        if strategy.long_short == "market_neutral":
            target = target - short_mask / cp.maximum(selected_count[:, None], 1) * side_exposure
        rebalance = (cp.arange(shape[0]) % strategy.rebalance_bars) == 0
        target = cp.where(rebalance[:, None], target, 0.0)
        target_cpu = cp.asnumpy(target).reshape(-1)
        valid_cpu = cp.asnumpy(valid).reshape(-1)
    active = ordered.loc[valid_cpu].copy()
    active["target_weight"] = target_cpu[valid_cpu]
    return active


def cuda_single_asset_positions(frame: pd.DataFrame, spec: RunSpec, device_id: int) -> pd.DataFrame:
    """Evaluate the configured single-asset signal rule with CuPy."""
    import cupy as cp

    strategy = spec.strategy
    assert strategy is not None and strategy.signal_rule is not None and strategy.symbol is not None
    active = frame[frame["symbol"] == strategy.symbol].sort_values("timestamp").copy()
    with cp.cuda.Device(device_id):
        factor = cp.asarray(active["factor"].to_numpy(dtype=np.float64, copy=False))
        forward = cp.asarray(active["forward_return"].to_numpy(dtype=np.float64, copy=False))
        valid = cp.isfinite(factor) & cp.isfinite(forward)
        if strategy.signal_rule.kind == "sign":
            weights = cp.sign(factor) * strategy.signal_rule.position_size
        else:
            weights = cp.where(factor > strategy.signal_rule.long_above, strategy.signal_rule.position_size, 0.0)
            if strategy.signal_rule.short_below is not None:
                weights = cp.where(factor < strategy.signal_rule.short_below, -strategy.signal_rule.position_size, weights)
        active["target_weight"] = cp.asnumpy(weights)
        valid_cpu = cp.asnumpy(valid)
    return active.loc[valid_cpu].copy()


def _scan_kernel():
    from numba import cuda

    @cuda.jit
    def scan(opens, closes, targets, signal, holding, cost_rate, stop_enabled, stop_threshold,
             returns, equity_values, weights, turnover, costs, reasons, trigger, liquidation,
             trigger_drawdown, trigger_peak, trigger_equity, liquidation_equity):
        if cuda.grid(1) != 0:
            return
        n_times, n_symbols = opens.shape
        equity = 1.0
        peak = 1.0
        force_exit_at = -1
        active_exit_at = -1
        cooldown = 0
        for t in range(n_times):
            before = equity
            if t > 0:
                overnight = 0.0
                for s in range(n_symbols):
                    overnight += weights[t - 1, s] * (opens[t, s] / closes[t - 1, s] - 1.0)
                    weights[t, s] = weights[t - 1, s]
                equity *= 1.0 + overnight
            else:
                for s in range(n_symbols):
                    weights[t, s] = 0.0
            forced = force_exit_at == t
            turn = 0.0
            reason = 0
            if forced:
                for s in range(n_symbols):
                    turn += abs(weights[t, s])
                    weights[t, s] = 0.0
                reason = 3
                active_exit_at = -1
                force_exit_at = -1
                cooldown = 1
            elif t > 0 and signal[t - 1] != 0:
                for s in range(n_symbols):
                    turn += abs(targets[t - 1, s] - weights[t, s])
                    weights[t, s] = targets[t - 1, s]
                reason = 1
                active_exit_at = t - 1 + holding + 1
                cooldown = 0
            elif active_exit_at == t:
                for s in range(n_symbols):
                    turn += abs(weights[t, s])
                    weights[t, s] = 0.0
                reason = 2
                active_exit_at = -1
            execution_cost = turn * cost_rate
            equity *= 1.0 - execution_cost
            intrabar = 0.0
            for s in range(n_symbols):
                intrabar += weights[t, s] * (closes[t, s] / opens[t, s] - 1.0)
            equity *= 1.0 + intrabar
            turnover[t] = turn
            costs[t] = execution_cost
            reasons[t] = reason
            returns[t] = equity / before - 1.0
            equity_values[t] = equity
            if equity <= 0.0:
                return
            if equity > peak:
                peak = equity
            drawdown = equity / peak - 1.0
            if stop_enabled != 0 and not forced and force_exit_at < 0 and drawdown <= -stop_threshold and t + 1 < n_times:
                force_exit_at = t + 1
                cooldown = 1
                trigger[t] = 1
                trigger_drawdown[t] = drawdown
                trigger_peak[t] = peak
                trigger_equity[t] = equity
            if forced:
                liquidation[t] = 1
                liquidation_equity[t] = equity
    return scan


def cuda_simulate_portfolio(frame: pd.DataFrame, positions: pd.DataFrame, spec: RunSpec, device_id: int) -> dict[str, Any]:
    """Run the position/stop state machine in a Numba CUDA scan kernel."""
    import cupy as cp

    ordered, timestamps, symbols = _layout(frame)
    shape = (len(timestamps), len(symbols))
    lookup = positions.set_index(["timestamp", "symbol"])["target_weight"]
    target_values = lookup.reindex(pd.MultiIndex.from_product([timestamps, symbols]), fill_value=0.0).to_numpy(dtype=np.float64)
    target_values = target_values.reshape(shape)
    signal_values = (np.abs(target_values).sum(axis=1) > 0).astype(np.int8)
    with cp.cuda.Device(device_id):
        opens = _matrix(ordered, "open", shape, cp)
        closes = _matrix(ordered, "close", shape, cp)
        targets = cp.asarray(target_values)
        signal = cp.asarray(signal_values)
        returns = cp.full(shape[0], cp.nan)
        equity = cp.full(shape[0], cp.nan)
        weights = cp.zeros(shape)
        turnover = cp.zeros(shape[0])
        costs = cp.zeros(shape[0])
        reasons = cp.zeros(shape[0], dtype=cp.int32)
        trigger = cp.zeros(shape[0], dtype=cp.int32)
        liquidation = cp.zeros(shape[0], dtype=cp.int32)
        trigger_drawdown = cp.full(shape[0], cp.nan)
        trigger_peak = cp.full(shape[0], cp.nan)
        trigger_equity = cp.full(shape[0], cp.nan)
        liquidation_equity = cp.full(shape[0], cp.nan)
        stop = spec.strategy.risk_control.max_drawdown_stop
        _scan_kernel()[1, 1](
            opens, closes, targets, signal, spec.strategy.execution.holding_bars,
            (spec.costs.fee_bps + spec.costs.slippage_bps) / 10_000,
            int(stop.enabled), stop.threshold or 0.0, returns, equity, weights, turnover, costs,
            reasons, trigger, liquidation, trigger_drawdown, trigger_peak, trigger_equity, liquidation_equity,
        )
        values = {name: cp.asnumpy(value) for name, value in {
            "returns": returns, "equity": equity, "weights": weights, "turnover": turnover, "costs": costs,
            "reasons": reasons, "trigger": trigger, "liquidation": liquidation,
            "trigger_drawdown": trigger_drawdown, "trigger_peak": trigger_peak,
            "trigger_equity": trigger_equity, "liquidation_equity": liquidation_equity,
        }.items()}
    valid = np.isfinite(values["returns"])
    return {**values, "timestamps": timestamps, "symbols": symbols, "valid": valid}


def cuda_factor_diagnostics(frame: pd.DataFrame, device_id: int) -> dict[str, Any]:
    """Compute cross-sectional Spearman IC and descriptive statistics on GPU."""
    import cupy as cp

    ordered, timestamps, symbols = _layout(frame)
    shape = (len(timestamps), len(symbols))
    with cp.cuda.Device(device_id):
        factor = _matrix(ordered, "factor", shape, cp)
        forward = _matrix(ordered, "forward_return", shape, cp)
        valid = cp.isfinite(factor) & cp.isfinite(forward)
        rank_factor = cp.argsort(cp.argsort(cp.where(valid, factor, cp.inf), axis=1), axis=1).astype(cp.float64)
        rank_return = cp.argsort(cp.argsort(cp.where(valid, forward, cp.inf), axis=1), axis=1).astype(cp.float64)
        count = valid.sum(axis=1)
        mean_f = (rank_factor * valid).sum(axis=1) / cp.maximum(count, 1)
        mean_r = (rank_return * valid).sum(axis=1) / cp.maximum(count, 1)
        centered_f = (rank_factor - mean_f[:, None]) * valid
        centered_r = (rank_return - mean_r[:, None]) * valid
        denominator = cp.sqrt((centered_f**2).sum(axis=1) * (centered_r**2).sum(axis=1))
        ic = cp.where((count >= 2) & (denominator > 0), (centered_f * centered_r).sum(axis=1) / denominator, cp.nan)
        ic_cpu = cp.asnumpy(ic)
        factor_cpu = cp.asnumpy(factor)
        coverage = float(cp.asnumpy(cp.isfinite(factor).mean()))
    series = pd.Series(ic_cpu, index=timestamps, name="ic").dropna()
    return {
        "coverage": coverage,
        "factor_mean": float(np.nanmean(factor_cpu)),
        "factor_std": float(np.nanstd(factor_cpu, ddof=1)),
        "ic_mean": float(series.mean()) if not series.empty else None,
        "ic_std": float(series.std(ddof=1)) if len(series) > 1 else None,
        "icir": float(series.mean() / series.std(ddof=1)) if len(series) > 1 and series.std(ddof=1) else None,
        "ic_observations": len(series),
        "ic_series": series,
    }
