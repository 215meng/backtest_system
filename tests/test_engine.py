from pathlib import Path

import pandas as pd
import pytest

from quantbacktest.engine import _forward_returns, _simulate_portfolio, run_backtest
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
    result = run_backtest(RunSpec.model_validate(payload), tmp_path)
    assert (result.run_dir / "report.html").exists()
    assert (result.run_dir / "debug_trace.json").exists()
    assert (result.run_dir / "risk_events.csv").exists()
    assert "risk_control" in (result.run_dir / "debug_trace.json").read_text(encoding="utf-8")


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
