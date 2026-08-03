from pathlib import Path

import pandas as pd
import pytest

from quantbacktest.analytics import equal_weight_benchmark, performance_metrics
from quantbacktest.engine import (
    _cross_sectional_positions,
    _forward_returns,
    _simulate_portfolio,
    run_backtest,
)
from quantbacktest.schemas import RunSpec


def test_forward_returns_use_next_open_and_holding_exit() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "symbol": ["BTCUSDT"] * 4,
            "open": [100.0, 110.0, 121.0, 133.1],
            "close": [100.0, 110.0, 121.0, 133.1],
        }
    )
    result = _forward_returns(frame, holding_bars=1)
    assert result.loc[0, "entry_open"] == 110.0
    assert result.loc[0, "exit_open"] == 121.0
    assert result.loc[0, "forward_return"] == pytest.approx(0.1)


def test_equal_weight_benchmark_uses_one_bar_returns_not_holding_period_returns() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        [
            {"timestamp": timestamp, "symbol": symbol, "close": close, "forward_return": 0.5}
            for symbol, closes in {"A": [100.0, 110.0, 121.0], "B": [200.0, 220.0, 242.0]}.items()
            for timestamp, close in zip(timestamps, closes, strict=True)
        ]
    )

    benchmark = equal_weight_benchmark(frame)

    assert benchmark.tolist() == pytest.approx([0.0, 0.1, 0.1])


def test_factor_mimicking_positions_have_one_times_gross_exposure() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        [
            {"timestamp": timestamp, "symbol": f"S{index}", "open": 100.0, "close": 100.0, "factor": float(index)}
            for timestamp in timestamps
            for index in range(12)
        ]
    )
    frame = _forward_returns(frame, holding_bars=1)
    spec = RunSpec.model_validate(
        {
            "name": "minimal_profile",
            "data": {"adapter": "crypto_top50", "path": "data/raw/crypto", "market": "spot", "frequency": "1h", "symbols": [f"S{index}" for index in range(12)]},
            "factor": {"module_path": "factor.py"},
            "strategy": {
                "profile": "factor_mimicking",
                "mode": "cross_sectional",
                "selection": "quantiles",
                "quantiles": 3,
                "long_short": "market_neutral",
                "weighting": "equal",
                "gross_exposure": 1.0,
                "rebalance_bars": 1,
                "execution": {"holding_bars": 1},
            },
        }
    )
    positions = _cross_sectional_positions(frame, spec)
    weights = positions.loc[positions["timestamp"] == timestamps[0], "target_weight"]
    assert weights[weights > 0].sum() == pytest.approx(0.5)
    assert weights[weights < 0].sum() == pytest.approx(-0.5)
    assert weights.abs().sum() == pytest.approx(1.0)


def test_trace_backtest_writes_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    data_path.mkdir()
    for symbol, shift in [("A", 0), ("B", 1), ("C", 2), ("D", 3), ("E", 4), ("F", 5), ("G", 6), ("H", 7), ("I", 8), ("J", 9)]:
        dates = pd.date_range("2024-01-01", periods=40, freq="h")
        values = [100 + shift + value for value in range(40)]
        pd.DataFrame({"timestamp": dates, "open": values, "high": values, "low": values, "close": values, "volume_from": 1, "volume_to": 1}).to_csv(data_path / f"{symbol}_1h.csv", index=False)
    factor = tmp_path / "factor.py"
    factor.write_text("FactorMeta={'required_fields':['close'], 'min_lookback_bars':1, 'supported_modes':['cross_sectional']}\ndef compute_factor(context):\n data=context.data.copy(); data['factor']=data.groupby('symbol').close.pct_change(); return data[['timestamp','symbol','factor']]\n", encoding="utf-8")
    payload = {
        "name": "artifact_case",
        "data": {"adapter": "crypto_top50", "path": str(data_path), "market": "spot", "frequency": "1h", "symbols": list("ABCDEFGHIJ")},
        "factor": {"module_path": str(factor)},
        "strategy": {"mode": "cross_sectional", "selection": "quantiles", "long_short": "market_neutral", "quantiles": 2},
        "debug": {"mode": "trace"},
        "output": {"root": str(tmp_path / "results")},
    }
    result = run_backtest(
        RunSpec.model_validate(payload), tmp_path, candidate_library_root=tmp_path / "library"
    )
    assert (result.run_dir / "report.html").exists()
    assert (result.run_dir / "debug_trace.json").exists()
    assert (result.run_dir / "risk_events.csv").exists()
    assert "risk_control" in (result.run_dir / "debug_trace.json").read_text(encoding="utf-8")
    assert result.candidate is not None
    assert result.candidate_registration_error is None
    assert (tmp_path / "library" / "artifacts" / result.candidate["factor_id"]).is_dir()

    dry_run_payload = {**payload, "name": "dry_run_case", "debug": {"mode": "dry_run"}}
    dry_run = run_backtest(
        RunSpec.model_validate(dry_run_payload),
        tmp_path,
        candidate_library_root=tmp_path / "dry_run_library",
    )
    assert dry_run.candidate is None
    assert dry_run.candidate_registration_error is None
    assert not (tmp_path / "dry_run_library").exists()

    blocked_library = tmp_path / "blocked_library"
    blocked_library.write_text("not a directory", encoding="utf-8")
    failure_payload = {**payload, "name": "candidate_failure_case"}
    registration_failure = run_backtest(
        RunSpec.model_validate(failure_payload),
        tmp_path,
        candidate_library_root=blocked_library,
    )
    assert registration_failure.candidate is None
    assert registration_failure.candidate_registration_error is not None
    assert (registration_failure.run_dir / "report.html").exists()
    assert (registration_failure.run_dir / "candidate_registration.json").exists()


def test_factor_research_writes_statistical_artifacts_without_account_metrics(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    data_path.mkdir()
    symbols = list("ABCDEFGHIJ")
    for shift, symbol in enumerate(symbols):
        dates = pd.date_range("2024-01-01", periods=24, freq="h")
        values = [100 + shift + value * (shift + 1) / 10 for value in range(24)]
        pd.DataFrame(
            {
                "timestamp": dates,
                "open": values,
                "high": values,
                "low": values,
                "close": values,
                "volume_from": 1,
                "volume_to": 1,
            }
        ).to_csv(data_path / f"{symbol}_1h.csv", index=False)
    factor = tmp_path / "factor.py"
    factor.write_text(
        "FactorMeta={'required_fields':['close'], 'min_lookback_bars':1, "
        "'supported_modes':['cross_sectional']}\n"
        "def compute_factor(context):\n"
        " data=context.data.copy(); data['factor']=data.groupby('symbol').close.pct_change(); "
        "return data[['timestamp','symbol','factor']]\n",
        encoding="utf-8",
    )
    payload = {
        "name": "research_case",
        "data": {
            "adapter": "crypto_top50",
            "path": str(data_path),
            "market": "spot",
            "frequency": "1h",
            "symbols": symbols,
        },
        "factor": {"module_path": str(factor)},
        "evaluation": {
            "mode": "factor_research",
            "research": {
                "formation": {"kind": "calendar", "interval": "1h", "time_utc": "00:00"},
                "returns": {"horizon": "1h", "start_price": "close", "end_price": "close"},
                "direction": "higher_predicts_higher_return",
                "portfolio": {"selection": "quantiles", "quantiles": 2},
                "ic_decay_horizons": ["1h", "4h"],
            },
        },
        "debug": {"mode": "trace"},
        "ml": {"enabled": True, "model": "lightgbm", "parameters": {"n_estimators": 5}},
        "output": {"root": str(tmp_path / "results")},
    }
    result = run_backtest(
        RunSpec.model_validate(payload), tmp_path, candidate_library_root=tmp_path / "library"
    )
    assert result.metrics["evaluation_mode"] == "factor_research"
    assert "annual_return" not in result.metrics
    assert (result.run_dir / "report.html").exists()
    assert (result.run_dir / "research_ic_decay.csv").exists()
    assert (result.run_dir / "research_contributions.csv").exists()
    assert (result.run_dir / "model.pkl").exists()
    assert not (result.run_dir / "returns.csv").exists()
    assert result.candidate is not None


def test_drawdown_stop_liquidates_next_open_and_reenters_next_rebalance() -> None:
    timestamps = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * len(timestamps),
            "open": [100.0, 100.0, 100.0, 70.0, 70.0],
            "close": [100.0, 100.0, 80.0, 70.0, 70.0],
        }
    )
    positions = pd.DataFrame(
        {
            "timestamp": [timestamps[0], timestamps[3]],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "target_weight": [1.0, 1.0],
            "factor": [1.0, 1.0],
            "forward_return": [0.0, 0.0],
        }
    )
    spec = RunSpec.model_validate(
        {
            "name": "drawdown_stop",
            "data": {
                "adapter": "crypto_top50",
                "path": "data/raw/crypto_top50",
                "market": "spot",
                "frequency": "1h",
                "symbols": ["BTCUSDT"],
            },
            "factor": {"module_path": "factor.py"},
            "strategy": {
                "mode": "single_asset",
                "symbol": "BTCUSDT",
                "signal_rule": {"kind": "sign"},
                "execution": {"holding_bars": 3},
                "risk_control": {"max_drawdown_stop": {"enabled": True, "threshold": 0.1}},
            },
            "costs": {"fee_bps": 100.0},
        }
    )
    result = _simulate_portfolio(frame, positions, spec)
    assert len(result.risk_events) == 1
    event = result.risk_events.iloc[0]
    assert event["trigger_time"] == timestamps[2]
    assert event["liquidation_time"] == timestamps[3]
    assert event["reentry_time"] == timestamps[4]
    assert event["liquidation_cost"] == pytest.approx(0.01)
    assert set(result.trades["reason"]) == {"rebalance", "drawdown_stop"}
    assert result.trades.loc[result.trades["reason"] == "drawdown_stop", "execution_price"].iloc[0] == 70.0
    assert result.cash_bar_ratio == pytest.approx(0.2)


def test_zero_signal_closes_existing_single_asset_position_at_next_open() -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * len(timestamps),
            "open": [100.0] * len(timestamps),
            "close": [100.0] * len(timestamps),
        }
    )
    positions = pd.DataFrame(
        {
            "timestamp": [timestamps[0], timestamps[1]],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "target_weight": [1.0, 0.0],
            "factor": [1.0, 0.0],
            "forward_return": [0.0, 0.0],
        }
    )
    spec = RunSpec.model_validate(
        {
            "name": "zero_signal",
            "data": {
                "adapter": "crypto_top50",
                "path": "data/raw/crypto_top50",
                "market": "spot",
                "frequency": "1h",
                "symbols": ["BTCUSDT"],
            },
            "factor": {"module_path": "factor.py"},
            "strategy": {
                "mode": "single_asset",
                "symbol": "BTCUSDT",
                "signal_rule": {"kind": "sign"},
                "execution": {"holding_bars": 3},
            },
        }
    )
    result = _simulate_portfolio(frame, positions, spec)
    exits = result.trades[result.trades["target_weight"] == 0.0]
    assert len(exits) == 1
    assert exits.iloc[0]["timestamp"] == timestamps[2]


def test_strategy_metrics_disable_annualization_after_non_positive_equity() -> None:
    returns = pd.Series(
        [0.1, -1.2], index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    )
    metrics = performance_metrics(returns, "1h")
    assert metrics["account_equity_non_positive"] is True
    assert metrics["annual_return"] is None
    assert metrics["calmar"] is None
    assert metrics["first_account_failure_time"] == "2024-01-01T01:00:00+00:00"
