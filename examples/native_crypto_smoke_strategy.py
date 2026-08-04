"""用于验证原生 Python 事件策略链路的本地加密数据示例。"""

from quantbacktest.api import *


def initialize(context):
    context.set_name("native_crypto_strategy_smoke")
    context.set_data(
        adapter="crypto_top50",
        path="data/raw/crypto_top50",
        market="spot",
        frequency="1h",
        symbols=["BTCUSDT", "ETHUSDT"],
        start="2024-01-01T00:00:00Z",
        end="2024-01-07T23:00:00Z",
    )
    context.set_account(initial_cash=100_000.0, benchmark="BTCUSDT", fee_bps=5.0, slippage_bps=2.0)
    run_daily(rebalance, when="close")


def rebalance(context):
    weights = {}
    for symbol in context.symbols:
        bars = context.history(symbol, 24, fields=["close"])
        if len(bars) == 24 and bars["close"].iloc[-1] > bars["close"].mean():
            weights[symbol] = 0.5
    order_target_weights(weights)
