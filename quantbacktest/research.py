from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantbacktest.schemas import FactorResearchSpec, UniverseSpec


class ResearchContractError(ValueError):
    """可定位的因子研究配置、日历或数据契约错误。"""


@dataclass
class ResearchResult:
    panel: pd.DataFrame
    group_returns: pd.DataFrame
    spread: pd.DataFrame
    contributions: pd.DataFrame
    leave_one_out: pd.DataFrame
    ic_decay: pd.DataFrame
    periods: pd.DataFrame
    skipped_periods: pd.DataFrame
    metrics: dict[str, Any]


def _timedelta(interval: str) -> pd.Timedelta:
    return {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1), "1w": pd.Timedelta(weeks=1)}[interval]


def _time_of_day(value: str) -> time:
    hour, minute = (int(part) for part in value.split(":"))
    return time(hour=hour, minute=minute)


def formation_times(frame: pd.DataFrame, spec: FactorResearchSpec) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Return exact UTC formation timestamps and skipped calendar anchors without drifting."""
    timestamps = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    if timestamps.empty:
        raise ResearchContractError("研究数据没有可用于生成形成时点的时间戳")
    schedule = spec.formation
    if schedule.kind == "bar_interval":
        return timestamps[:: schedule.every_n_bars or 1], pd.DataFrame(columns=["timestamp", "reason"])

    assert schedule.interval is not None and schedule.time_utc is not None
    anchor_time = _time_of_day(schedule.time_utc)
    start = timestamps.min().normalize()
    end = timestamps.max().normalize()
    if schedule.interval == "1w":
        assert schedule.weekday is not None
        offset_days = (schedule.weekday - start.weekday()) % 7
        first = start + pd.Timedelta(days=offset_days, hours=anchor_time.hour, minutes=anchor_time.minute)
        candidates = pd.date_range(first, end + pd.Timedelta(days=1), freq="7D")
    elif schedule.interval == "1d":
        candidates = pd.date_range(start, end, freq="1D") + pd.Timedelta(
            hours=anchor_time.hour, minutes=anchor_time.minute
        )
    else:
        interval = _timedelta(schedule.interval)
        candidates = pd.date_range(start, end + pd.Timedelta(days=1), freq=interval)
        candidates = candidates + pd.Timedelta(hours=anchor_time.hour, minutes=anchor_time.minute)
        candidates = candidates[(candidates >= timestamps.min()) & (candidates <= timestamps.max())]
    available = set(timestamps)
    valid = pd.DatetimeIndex([candidate for candidate in candidates if candidate in available])
    skipped = pd.DataFrame(
        {"timestamp": [candidate for candidate in candidates if candidate not in available], "reason": "missing_calendar_anchor"}
    )
    return valid, skipped


def _load_membership(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise ResearchContractError(f"未找到时间点资产池清单：{path}")
    membership = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    required = {"timestamp", "symbol", "eligible"}
    missing = required - set(membership.columns)
    if missing:
        raise ResearchContractError(f"资产池清单缺少字段：{sorted(missing)}")
    membership = membership[list(required)].copy()
    membership["timestamp"] = pd.to_datetime(membership["timestamp"], utc=True)
    membership["eligible"] = membership["eligible"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"}
    )
    return membership.sort_values(["symbol", "timestamp"])


def _eligible_symbols(membership: pd.DataFrame | None, timestamp: pd.Timestamp) -> set[str] | None:
    if membership is None:
        return None
    history = membership[membership["timestamp"] <= timestamp]
    latest = history.groupby("symbol", as_index=False).tail(1)
    return set(latest.loc[latest["eligible"], "symbol"])


def _weights(group: pd.DataFrame, weighting: str) -> pd.Series:
    if weighting == "market_cap":
        if "market_cap" not in group.columns:
            raise ResearchContractError("研究权重 market_cap 需要数据提供 market_cap 字段")
        values = group["market_cap"].clip(lower=0)
    elif weighting == "score":
        values = group["factor"].abs()
    else:
        values = pd.Series(1.0, index=group.index)
    return values / values.sum() if values.sum() > 0 else pd.Series(1.0 / len(group), index=group.index)


def _hac_mean(series: pd.Series, lags: int) -> dict[str, float | int | None]:
    values = series.dropna().astype(float).to_numpy()
    observations = len(values)
    if observations == 0:
        return {"mean": None, "std": None, "t_stat_hac": None, "observations": 0, "hac_lags": lags}
    mean = float(values.mean())
    if observations == 1:
        return {"mean": mean, "std": 0.0, "t_stat_hac": None, "observations": 1, "hac_lags": lags}
    centered = values - mean
    lag_limit = min(lags, observations - 1)
    long_run_variance = float(np.mean(centered * centered))
    for lag in range(1, lag_limit + 1):
        covariance = float(np.mean(centered[lag:] * centered[:-lag]))
        long_run_variance += 2 * (1 - lag / (lag_limit + 1)) * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / observations)
    return {
        "mean": mean,
        "std": float(values.std(ddof=1)),
        "t_stat_hac": mean / standard_error if standard_error > 0 else None,
        "observations": observations,
        "hac_lags": lag_limit,
    }


def _research_hac_lags(formations: pd.DatetimeIndex, horizon: pd.Timedelta) -> int:
    if len(formations) < 2:
        return 0
    spacing = pd.Series(formations[1:] - formations[:-1]).median()
    return max(0, math.ceil(horizon / spacing) - 1) if spacing > pd.Timedelta(0) else 0


def _leave_one_out(selected: pd.DataFrame, direction: int) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for symbol in sorted(selected["symbol"].unique()):
        rows: list[float] = []
        for _, group in selected.groupby("formation_time"):
            reduced = group[group["symbol"] != symbol]
            top = reduced[reduced["bucket"] == "top"]
            bottom = reduced[reduced["bucket"] == "bottom"]
            if not top.empty and not bottom.empty:
                rows.append(direction * (top["forward_return"].mean() - bottom["forward_return"].mean()))
        result = _hac_mean(pd.Series(rows, dtype=float), 0)
        output.append({"excluded_symbol": symbol, "spread_mean_without_symbol": result["mean"], "observations": result["observations"]})
    return pd.DataFrame(output)


def _group_turnover(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (bucket_name, bucket), group in selected.groupby(["bucket_name", "bucket"]):
        previous: pd.Series | None = None
        for formation_time, current in group.groupby("formation_time", sort=True):
            weights = current.set_index("symbol")["weight"]
            turnover = None
            if previous is not None:
                aligned = weights.reindex(weights.index.union(previous.index), fill_value=0.0)
                prior = previous.reindex(aligned.index, fill_value=0.0)
                turnover = float((aligned - prior).abs().sum() / 2)
            rows.append(
                {
                    "formation_time": formation_time,
                    "bucket_name": bucket_name,
                    "bucket": bucket,
                    "turnover": turnover,
                }
            )
            previous = weights
    return pd.DataFrame(rows)


def evaluate_factor_research(
    frame: pd.DataFrame,
    spec: FactorResearchSpec,
    universe: UniverseSpec | None = None,
) -> ResearchResult:
    """Evaluate a factor as a paper-style cross-sectional prediction, not an account simulation."""
    formations, skipped = formation_times(frame, spec)
    membership = _load_membership(universe.membership_path if universe else None)
    timestamps = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    timestamp_set = set(timestamps)
    next_times = {timestamps[index]: timestamps[index + 1] for index in range(len(timestamps) - 1)}
    horizon = _timedelta(spec.returns.horizon)
    direction = 1 if spec.direction == "higher_predicts_higher_return" else -1
    panel_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    selection_rows: list[pd.DataFrame] = []
    ic_decay_rows: list[dict[str, Any]] = []
    decay_horizons = list(dict.fromkeys([spec.returns.horizon, *spec.ic_decay_horizons]))

    for formation in formations:
        end_time = formation + horizon
        if end_time not in timestamp_set:
            skipped = pd.concat(
                [skipped, pd.DataFrame({"timestamp": [formation], "reason": ["missing_return_endpoint"]})],
                ignore_index=True,
            )
            continue
        if spec.returns.start_price == "next_open":
            start_time = next_times.get(formation)
            if start_time is None:
                skipped = pd.concat(
                    [skipped, pd.DataFrame({"timestamp": [formation], "reason": ["missing_next_open"]})],
                    ignore_index=True,
                )
                continue
        else:
            start_time = formation
        current = frame[frame["timestamp"] == formation].set_index("symbol")
        start = frame[frame["timestamp"] == start_time].set_index("symbol").rename(
            columns={"close": "close_start", "open": "open_start"}
        )
        end = frame[frame["timestamp"] == end_time].set_index("symbol").rename(
            columns={"close": "close_end", "open": "open_end"}
        )
        allowed = _eligible_symbols(membership, formation)
        rows = current[["factor"]].join(start[["close_start", "open_start"]]).join(
            end[["close_end", "open_end"]]
        )
        if allowed is not None:
            rows = rows.loc[rows.index.isin(allowed)]
        start_column = "close_start" if spec.returns.start_price == "close" else "open_start"
        end_column = "close_end" if spec.returns.end_price == "close" else "open_end"
        rows["forward_return"] = rows[end_column] / rows[start_column] - 1.0
        rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=["factor", "forward_return"])
        min_assets = universe.min_assets if universe else 10
        if len(rows) < min_assets:
            skipped = pd.concat(
                [skipped, pd.DataFrame({"timestamp": [formation], "reason": ["insufficient_eligible_assets"]})],
                ignore_index=True,
            )
            continue
        rows = rows.reset_index().assign(formation_time=formation, return_end_time=end_time)
        panel_rows.extend(rows.to_dict("records"))
        for decay_horizon in decay_horizons:
            decay_end_time = formation + _timedelta(decay_horizon)
            if decay_end_time not in timestamp_set:
                continue
            decay_end = frame[frame["timestamp"] == decay_end_time].set_index("symbol").rename(
                columns={"close": "close_end", "open": "open_end"}
            )
            decay_rows = current[["factor"]].join(start[["close_start", "open_start"]]).join(
                decay_end[["close_end", "open_end"]]
            )
            if allowed is not None:
                decay_rows = decay_rows.loc[decay_rows.index.isin(allowed)]
            decay_rows["forward_return"] = decay_rows[end_column] / decay_rows[start_column] - 1.0
            decay_rows = decay_rows.replace([np.inf, -np.inf], np.nan).dropna(subset=["factor", "forward_return"])
            if len(decay_rows) >= 2:
                raw_ic = decay_rows["factor"].corr(decay_rows["forward_return"], method="spearman")
                ic_decay_rows.append(
                    {
                        "formation_time": formation,
                        "horizon": decay_horizon,
                        "rank_ic": raw_ic,
                        "directional_rank_ic": direction * raw_ic,
                    }
                )
        ordered = rows.sort_values("factor", kind="stable").copy()
        if spec.portfolio.selection == "quantiles":
            buckets = np.array_split(np.arange(len(ordered)), spec.portfolio.quantiles or 2)
        else:
            count = min(spec.portfolio.top_k or 1, len(ordered) // 2)
            buckets = [np.arange(count), np.arange(len(ordered) - count, len(ordered))]
        named_buckets: list[tuple[str, np.ndarray]] = []
        if spec.portfolio.selection == "quantiles":
            named_buckets = [(f"q{index + 1}", bucket) for index, bucket in enumerate(buckets)]
        else:
            named_buckets = [("bottom", buckets[0]), ("top", buckets[1])]
        period_rows.append({"formation_time": formation, "active_assets": len(rows), "return_end_time": end_time})
        for name, indices in named_buckets:
            group = ordered.iloc[indices].copy()
            group["weight"] = _weights(group, spec.portfolio.weighting)
            bucket_kind = "bottom" if name in {"q1", "bottom"} else "top" if name in {f"q{len(named_buckets)}", "top"} else "middle"
            group["bucket"] = bucket_kind
            group["bucket_name"] = name
            selection_rows.append(group)

    panel = pd.DataFrame(panel_rows)
    if panel.empty:
        raise ResearchContractError("没有满足形成时间表、资产池和预测期价格定义的研究观测")
    selected = pd.concat(selection_rows, ignore_index=True)
    group_returns = (
        selected.assign(weighted_return=selected["weight"] * selected["forward_return"])
        .groupby(["formation_time", "bucket_name", "bucket"], as_index=False)
        .agg(return_value=("weighted_return", "sum"), assets=("symbol", "nunique"), hhi=("weight", lambda values: float((values**2).sum())))
    )
    group_returns = group_returns.merge(
        _group_turnover(selected), on=["formation_time", "bucket_name", "bucket"], how="left"
    )
    top = group_returns[group_returns["bucket"] == "top"].set_index("formation_time")["return_value"]
    bottom = group_returns[group_returns["bucket"] == "bottom"].set_index("formation_time")["return_value"]
    spread = pd.concat([top.rename("top_return"), bottom.rename("bottom_return")], axis=1).dropna().reset_index()
    spread["spread_return"] = direction * (spread["top_return"] - spread["bottom_return"])
    market = panel.groupby("formation_time")["forward_return"].mean().rename("market_return")
    ic = panel.groupby("formation_time").apply(
        lambda group: group["factor"].corr(group["forward_return"], method="spearman"), include_groups=False
    ).rename("rank_ic")
    spread = spread.merge(market, on="formation_time", how="left").merge(ic, on="formation_time", how="left")
    spread["directional_rank_ic"] = direction * spread["rank_ic"]
    contributions = selected[selected["bucket"].isin(["top", "bottom"])].copy()
    contributions["signed_weight"] = np.where(contributions["bucket"] == "top", direction, -direction) * contributions["weight"]
    contributions["contribution"] = contributions["signed_weight"] * contributions["forward_return"]
    contributions = contributions[["formation_time", "symbol", "bucket", "weight", "forward_return", "contribution"]]
    leave_one_out = _leave_one_out(selected, direction)
    ic_decay = pd.DataFrame(
        ic_decay_rows,
        columns=["formation_time", "horizon", "rank_ic", "directional_rank_ic"],
    )
    periods = pd.DataFrame(period_rows).merge(spread, on="formation_time", how="left")
    lags = _research_hac_lags(pd.DatetimeIndex(spread["formation_time"]), horizon)
    spread_stats = _hac_mean(spread["spread_return"], lags)
    ic_stats = _hac_mean(spread["directional_rank_ic"], lags)
    beta = None
    alpha = None
    usable_beta = spread.dropna(subset=["spread_return", "market_return"])
    if len(usable_beta) >= 2 and usable_beta["market_return"].var() > 0:
        beta = float(usable_beta["spread_return"].cov(usable_beta["market_return"]) / usable_beta["market_return"].var())
        alpha = float(usable_beta["spread_return"].mean() - beta * usable_beta["market_return"].mean())
    contribution_totals = contributions.groupby("symbol")["contribution"].sum()
    total_absolute_contribution = float(contribution_totals.abs().sum())
    half = len(spread) // 2
    ic_decay_summary = {}
    if not ic_decay.empty:
        for horizon_name, values in ic_decay.groupby("horizon")["directional_rank_ic"]:
            ic_decay_summary[horizon_name] = _hac_mean(values, lags)["mean"]
    metrics = {
        "evaluation_mode": "factor_research",
        "formation_periods": len(spread),
        "formation_skipped_periods": len(skipped),
        "factor_coverage": float(len(panel) / max(len(formations) * frame["symbol"].nunique(), 1)),
        "rank_ic_mean": ic_stats["mean"],
        "rank_icir": float(spread["directional_rank_ic"].mean() / spread["directional_rank_ic"].std(ddof=1)) if spread["directional_rank_ic"].std(ddof=1) else None,
        "rank_ic_t_stat_hac": ic_stats["t_stat_hac"],
        "spread_mean": spread_stats["mean"],
        "spread_std": spread_stats["std"],
        "spread_t_stat_hac": spread_stats["t_stat_hac"],
        "hac_lags": lags,
        "ic_decay_mean": ic_decay_summary,
        "market_beta": beta,
        "market_adjusted_alpha": alpha,
        "mean_active_assets": float(periods["active_assets"].mean()),
        "min_active_assets": int(periods["active_assets"].min()),
        "mean_group_turnover": float(group_returns["turnover"].dropna().mean()) if group_returns["turnover"].notna().any() else None,
        "mean_group_hhi": float(group_returns["hhi"].mean()),
        "max_single_asset_contribution_share": float(contribution_totals.abs().max() / total_absolute_contribution) if total_absolute_contribution else 0.0,
        "first_half_spread_mean": float(spread["spread_return"].iloc[:half].mean()) if half else None,
        "second_half_spread_mean": float(spread["spread_return"].iloc[half:].mean()) if half else None,
        "research_return_note": "统计型分组未来收益，不是账户净值，不生成 CAGR、爆仓或保证金结论",
    }
    return ResearchResult(panel, group_returns, spread, contributions, leave_one_out, ic_decay, periods, skipped, metrics)
