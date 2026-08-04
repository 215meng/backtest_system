"""用于验证原生 Python 因子链路的 Binance 现货示例。"""

import pandas as pd

from quantbacktest.api import *


def initialize(context):
    context.set_name("native_binance_momentum_smoke")
    context.set_data(
        adapter="binance_zip",
        path="data/raw/binance/spot_klines",
        market="spot",
        frequency="1h",
        symbols=["BTCUSDT", "ETHUSDT"],
        start="2026-07-01T00:00:00Z",
        end="2026-07-29T23:00:00Z",
    )
    context.set_factor_evaluation(
        formation="daily",
        horizon_bars=24,
        entry_price="next_open",
        exit_price="close",
        direction="higher_predicts_higher_return",
        groups=2,
        weighting="equal",
        fee_bps=0.0,
        slippage_bps=0.0,
    )


def main(context):
    rows = []
    for symbol in context.symbols:
        bars = context.history(symbol, 25, fields=["close"])
        if len(bars) == 25:
            rows.append(
                {"timestamp": context.now, "symbol": symbol, "factor": bars["close"].iloc[-1] / bars["close"].iloc[0] - 1}
            )
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "factor"])
