from __future__ import annotations

import pandas as pd
import pytest

import quantbacktest
from quantbacktest.api import ScriptContext, ScriptContractError


def test_package_does_not_export_legacy_yaml_api() -> None:
    assert not hasattr(quantbacktest, "RunSpec")
    assert not hasattr(quantbacktest, "run_backtest")


def test_data_dates_are_platform_owned_and_warmup_is_enforced() -> None:
    context = ScriptContext("factor")
    with pytest.raises(ScriptContractError, match="回测区间由平台"):
        context.set_data(
            adapter="crypto_top50",
            path="data",
            market="spot",
            frequency="1h",
            symbols=["BTCUSDT"],
            start="2024-01-01",
        )

    context.set_data(
        adapter="crypto_top50",
        path="data",
        market="spot",
        frequency="1h",
        symbols=["BTCUSDT"],
        warmup_bars=2,
    )
    context.now = pd.Timestamp("2024-01-01T02:00:00Z")
    context._visible_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "symbol": ["BTCUSDT"] * 3,
            "close": [100.0, 101.0, 102.0],
        }
    )
    assert len(context.history("BTCUSDT", 3, fields=["close"])) == 3
    with pytest.raises(ScriptContractError, match="warmup_bars=2"):
        context.history("BTCUSDT", 4, fields=["close"])


def test_history_rejects_stale_or_gapped_symbol_data() -> None:
    context = ScriptContext("factor")
    context.set_data(
        adapter="crypto_top50",
        path="data",
        market="spot",
        frequency="1h",
        symbols=["BTCUSDT"],
        warmup_bars=2,
    )
    context.now = pd.Timestamp("2024-01-01T03:00:00Z")
    context._visible_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z", "2024-01-01T03:00:00Z"]),
            "symbol": ["BTCUSDT"] * 3,
            "close": [100.0, 101.0, 102.0],
        }
    )
    assert context.history("BTCUSDT", 3, fields=["close"]).empty

    context.now = pd.Timestamp("2024-01-01T04:00:00Z")
    assert context.history("BTCUSDT", 1, fields=["close"]).empty
