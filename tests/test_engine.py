from pathlib import Path

import pandas as pd
import pytest

from quantbacktest.engine import _forward_returns, run_backtest
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
