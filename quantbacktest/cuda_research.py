"""GPU implementation of the paper-style factor research calculations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantbacktest.cuda import CudaExecutionError
from quantbacktest.cuda_strategy import _layout, _matrix
from quantbacktest.research import (
    ResearchContractError,
    ResearchResult,
    _eligible_symbols,
    _leave_one_out,
    _load_membership,
    _research_hac_lags,
    _timedelta,
    formation_times,
)
from quantbacktest.schemas import FactorResearchSpec, UniverseSpec


def _rank_ic(cp: Any, factor: Any, returns: Any, valid: Any) -> float | None:
    count = int(cp.asnumpy(valid.sum()))
    if count < 2:
        return None
    scores_f = cp.where(valid, factor, cp.inf)
    scores_r = cp.where(valid, returns, cp.inf)
    rf = cp.argsort(cp.argsort(scores_f)).astype(cp.float64)
    rr = cp.argsort(cp.argsort(scores_r)).astype(cp.float64)
    mean_f = (rf * valid).sum() / count
    mean_r = (rr * valid).sum() / count
    centered_f = (rf - mean_f) * valid
    centered_r = (rr - mean_r) * valid
    denominator = cp.sqrt((centered_f**2).sum() * (centered_r**2).sum())
    if float(denominator) == 0:
        return None
    return float((centered_f * centered_r).sum() / denominator)


def _hac_mean_cuda(values: pd.Series, lags: int, device_id: int) -> dict[str, float | int | None]:
    import cupy as cp

    clean = values.dropna().to_numpy(dtype=np.float64)
    observations = len(clean)
    if observations == 0:
        return {"mean": None, "std": None, "t_stat_hac": None, "observations": 0, "hac_lags": lags}
    with cp.cuda.Device(device_id):
        data = cp.asarray(clean)
        mean = float(data.mean())
        if observations == 1:
            return {"mean": mean, "std": 0.0, "t_stat_hac": None, "observations": 1, "hac_lags": 0}
        centered = data - mean
        lag_limit = min(lags, observations - 1)
        long_run = (centered * centered).mean()
        for lag in range(1, lag_limit + 1):
            covariance = (centered[lag:] * centered[:-lag]).mean()
            long_run += 2 * (1 - lag / (lag_limit + 1)) * covariance
        standard_error = float(cp.sqrt(cp.maximum(long_run, 0.0) / observations))
        return {
            "mean": mean,
            "std": float(data.std(ddof=1)),
            "t_stat_hac": mean / standard_error if standard_error > 0 else None,
            "observations": observations,
            "hac_lags": lag_limit,
        }


def evaluate_factor_research_cuda(
    frame: pd.DataFrame, spec: FactorResearchSpec, universe: UniverseSpec | None, device_id: int
) -> ResearchResult:
    """Evaluate formation returns, ranks, groups and HAC statistics on CUDA.

    Calendar labels and final CSV tables deliberately remain on host; no numerical
    factor, return, rank, weight, group-return or IC calculation uses Pandas.
    """
    import cupy as cp

    ordered, timestamps, symbols = _layout(frame)
    shape = (len(timestamps), len(symbols))
    formations, skipped = formation_times(ordered, spec)
    membership = _load_membership(universe.membership_path if universe else None)
    index_of = {timestamp: index for index, timestamp in enumerate(timestamps)}
    horizon = _timedelta(spec.returns.horizon)
    direction = 1 if spec.direction == "higher_predicts_higher_return" else -1
    min_assets = universe.min_assets if universe else 10
    panel_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    period_rows: list[dict[str, Any]] = []
    decay_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    previous_weights: dict[str, np.ndarray] = {}
    decay_horizons = list(dict.fromkeys([spec.returns.horizon, *spec.ic_decay_horizons]))

    with cp.cuda.Device(device_id):
        factor_matrix = _matrix(ordered, "factor", shape, cp)
        close_matrix = _matrix(ordered, "close", shape, cp)
        open_matrix = _matrix(ordered, "open", shape, cp)
        market_cap_matrix = _matrix(ordered, "market_cap", shape, cp) if "market_cap" in ordered else None
        for formation in formations:
            end_time = formation + horizon
            if end_time not in index_of:
                skipped = pd.concat([skipped, pd.DataFrame({"timestamp": [formation], "reason": ["missing_return_endpoint"]})], ignore_index=True)
                continue
            formation_index = index_of[formation]
            start_index = formation_index + 1 if spec.returns.start_price == "next_open" else formation_index
            if start_index >= len(timestamps):
                skipped = pd.concat([skipped, pd.DataFrame({"timestamp": [formation], "reason": ["missing_next_open"]})], ignore_index=True)
                continue
            end_index = index_of[end_time]
            factor = factor_matrix[formation_index]
            start_prices = close_matrix[start_index] if spec.returns.start_price == "close" else open_matrix[start_index]
            end_prices = close_matrix[end_index] if spec.returns.end_price == "close" else open_matrix[end_index]
            forward = end_prices / start_prices - 1.0
            valid = cp.isfinite(factor) & cp.isfinite(forward) & (start_prices > 0) & (end_prices > 0)
            allowed = _eligible_symbols(membership, formation)
            if allowed is not None:
                allowed_mask = cp.asarray(np.array([symbol in allowed for symbol in symbols]))
                valid &= allowed_mask
            active_count = int(cp.asnumpy(valid.sum()))
            if active_count < min_assets:
                skipped = pd.concat([skipped, pd.DataFrame({"timestamp": [formation], "reason": ["insufficient_eligible_assets"]})], ignore_index=True)
                continue
            factor_cpu = cp.asnumpy(factor)
            forward_cpu = cp.asnumpy(forward)
            valid_cpu = cp.asnumpy(valid)
            current = pd.DataFrame({"symbol": np.array(symbols)[valid_cpu], "factor": factor_cpu[valid_cpu], "forward_return": forward_cpu[valid_cpu]})
            current["formation_time"] = formation
            current["return_end_time"] = end_time
            panel_rows.append(current)
            period_rows.append({"formation_time": formation, "active_assets": active_count, "return_end_time": end_time})
            for decay_horizon in decay_horizons:
                decay_time = formation + _timedelta(decay_horizon)
                if decay_time not in index_of:
                    continue
                decay_end_index = index_of[decay_time]
                decay_prices = close_matrix[decay_end_index] if spec.returns.end_price == "close" else open_matrix[decay_end_index]
                decay_return = decay_prices / start_prices - 1.0
                decay_valid = valid & cp.isfinite(decay_return) & (decay_prices > 0)
                raw_ic = _rank_ic(cp, factor, decay_return, decay_valid)
                if raw_ic is not None:
                    decay_rows.append({"formation_time": formation, "horizon": decay_horizon, "rank_ic": raw_ic, "directional_rank_ic": direction * raw_ic})
            scores = cp.where(valid, factor, cp.inf)
            order = cp.argsort(scores)
            if spec.portfolio.selection == "quantiles":
                bucket_count = spec.portfolio.quantiles or 2
                bucket_indices = np.array_split(cp.asnumpy(order[-active_count:]), bucket_count)
                bucket_names = [f"q{index + 1}" for index in range(bucket_count)]
            else:
                count = min(spec.portfolio.top_k or 1, active_count // 2)
                active_order = cp.asnumpy(order[-active_count:])
                bucket_indices = [active_order[:count], active_order[-count:]]
                bucket_names = ["bottom", "top"]
            for bucket_position, (bucket_name, indices) in enumerate(zip(bucket_names, bucket_indices, strict=True)):
                if len(indices) == 0:
                    continue
                values = cp.abs(factor[cp.asarray(indices)]) if spec.portfolio.weighting == "score" else None
                if spec.portfolio.weighting == "market_cap":
                    if market_cap_matrix is None:
                        raise ResearchContractError("研究权重 market_cap 需要数据提供 market_cap 字段")
                    values = cp.maximum(market_cap_matrix[formation_index, cp.asarray(indices)], 0.0)
                if values is None:
                    weights = cp.full(len(indices), 1.0 / len(indices))
                else:
                    total = values.sum()
                    weights = values / total if float(total) > 0 else cp.full(len(indices), 1.0 / len(indices))
                return_value = float((weights * forward[cp.asarray(indices)]).sum())
                weights_cpu = cp.asnumpy(weights)
                bucket_kind = "bottom" if bucket_position == 0 else "top" if bucket_position == len(bucket_indices) - 1 else "middle"
                key = bucket_name
                full_weights = np.zeros(len(symbols))
                full_weights[indices] = weights_cpu
                turnover = None if key not in previous_weights else float(np.abs(full_weights - previous_weights[key]).sum() / 2)
                previous_weights[key] = full_weights
                group_rows.append({"formation_time": formation, "bucket_name": bucket_name, "bucket": bucket_kind, "return_value": return_value, "assets": len(indices), "hhi": float((weights**2).sum()), "turnover": turnover})
                selected = pd.DataFrame({"formation_time": formation, "return_end_time": end_time, "symbol": np.array(symbols)[indices], "factor": factor_cpu[indices], "forward_return": forward_cpu[indices], "weight": weights_cpu, "bucket": bucket_kind, "bucket_name": bucket_name})
                selected_rows.append(selected)

    if not panel_rows:
        raise ResearchContractError("没有满足形成时间表、资产池和预测期价格定义的研究观测")
    panel = pd.concat(panel_rows, ignore_index=True)
    selected = pd.concat(selected_rows, ignore_index=True)
    group_returns = pd.DataFrame(group_rows)
    top = group_returns[group_returns["bucket"] == "top"].set_index("formation_time")["return_value"]
    bottom = group_returns[group_returns["bucket"] == "bottom"].set_index("formation_time")["return_value"]
    spread = pd.concat([top.rename("top_return"), bottom.rename("bottom_return")], axis=1).dropna().reset_index()
    spread["spread_return"] = direction * (spread["top_return"] - spread["bottom_return"])
    raw_ic = []
    for formation, group in panel.groupby("formation_time", sort=True):
        # The primary IC was already computed on CUDA above; this call uses the same GPU primitive for its label row.
        import cupy as cp
        with cp.cuda.Device(device_id):
            raw_ic.append((formation, _rank_ic(cp, cp.asarray(group["factor"].to_numpy()), cp.asarray(group["forward_return"].to_numpy()), cp.ones(len(group), dtype=bool))))
    ic = pd.Series(dict(raw_ic), name="rank_ic")
    market = panel.groupby("formation_time")["forward_return"].mean().rename("market_return")
    spread = spread.merge(market, on="formation_time", how="left").merge(ic, left_on="formation_time", right_index=True, how="left")
    spread["directional_rank_ic"] = direction * spread["rank_ic"]
    contributions = selected[selected["bucket"].isin(["top", "bottom"])].copy()
    contributions["signed_weight"] = np.where(contributions["bucket"] == "top", direction, -direction) * contributions["weight"]
    contributions["contribution"] = contributions["signed_weight"] * contributions["forward_return"]
    contributions = contributions[["formation_time", "symbol", "bucket", "weight", "forward_return", "contribution"]]
    leave_one_out = _leave_one_out(selected, direction)
    ic_decay = pd.DataFrame(decay_rows, columns=["formation_time", "horizon", "rank_ic", "directional_rank_ic"])
    periods = pd.DataFrame(period_rows).merge(spread, on="formation_time", how="left")
    lags = _research_hac_lags(pd.DatetimeIndex(spread["formation_time"]), horizon)
    spread_stats = _hac_mean_cuda(spread["spread_return"], lags, device_id)
    ic_stats = _hac_mean_cuda(spread["directional_rank_ic"], lags, device_id)
    usable_beta = spread.dropna(subset=["spread_return", "market_return"])
    beta = alpha = None
    if len(usable_beta) >= 2 and usable_beta["market_return"].var() > 0:
        beta = float(usable_beta["spread_return"].cov(usable_beta["market_return"]) / usable_beta["market_return"].var())
        alpha = float(usable_beta["spread_return"].mean() - beta * usable_beta["market_return"].mean())
    totals = contributions.groupby("symbol")["contribution"].sum()
    total_abs = float(totals.abs().sum())
    half = len(spread) // 2
    decay_summary = {name: _hac_mean_cuda(values, lags, device_id)["mean"] for name, values in ic_decay.groupby("horizon")["directional_rank_ic"]} if not ic_decay.empty else {}
    metrics = {
        "evaluation_mode": "factor_research", "compute_backend": "cuda", "formation_periods": len(spread),
        "formation_skipped_periods": len(skipped), "factor_coverage": float(len(panel) / max(len(formations) * len(symbols), 1)),
        "rank_ic_mean": ic_stats["mean"], "rank_icir": float(spread["directional_rank_ic"].mean() / spread["directional_rank_ic"].std(ddof=1)) if spread["directional_rank_ic"].std(ddof=1) else None,
        "rank_ic_t_stat_hac": ic_stats["t_stat_hac"], "spread_mean": spread_stats["mean"], "spread_std": spread_stats["std"],
        "spread_t_stat_hac": spread_stats["t_stat_hac"], "hac_lags": lags, "ic_decay_mean": decay_summary,
        "market_beta": beta, "market_adjusted_alpha": alpha, "mean_active_assets": float(periods["active_assets"].mean()),
        "min_active_assets": int(periods["active_assets"].min()), "mean_group_turnover": float(group_returns["turnover"].dropna().mean()) if group_returns["turnover"].notna().any() else None,
        "mean_group_hhi": float(group_returns["hhi"].mean()), "max_single_asset_contribution_share": float(totals.abs().max() / total_abs) if total_abs else 0.0,
        "first_half_spread_mean": float(spread["spread_return"].iloc[:half].mean()) if half else None,
        "second_half_spread_mean": float(spread["spread_return"].iloc[half:].mean()) if half else None,
        "research_return_note": "统计型分组未来收益，不是账户净值，不生成 CAGR、爆仓或保证金结论。",
    }
    return ResearchResult(panel, group_returns, spread, contributions, leave_one_out, ic_decay, periods, skipped, metrics)
