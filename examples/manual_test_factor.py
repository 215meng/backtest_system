"""可手动运行的原生 Python 因子示例：24 小时动量。"""

import pandas as pd


def initialize(context):
    """声明数据源和因子评估方式。"""
    context.set_name("manual_24h_momentum")
    context.set_data(
        adapter="binance_zip",
        path="data/raw/binance/spot_klines",
        market="spot",
        frequency="1h",
        symbols=["BTCUSDT", "ETHUSDT"],
        warmup_bars=24,
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
    """返回每个可交易标的在当前时点的 24 小时收益率因子值。"""
    factor_rows = []
    for symbol in context.symbols:
        history = context.history(symbol, 25, fields=["close"])
        if len(history) < 25:
            continue
        factor_rows.append(
            {
                "timestamp": context.now,
                "symbol": symbol,
                "factor": history["close"].iloc[-1] / history["close"].iloc[0] - 1.0,
            }
        )
    return pd.DataFrame(factor_rows, columns=["timestamp", "symbol", "factor"])
