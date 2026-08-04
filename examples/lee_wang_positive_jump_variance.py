"""Lee 和 Wang（2025）正向跳跃方差因子的本地平台验收脚本。

论文使用 Coinbase/Kaiko 中间价和约 100 个币种。本脚本仅为回测平台功能
验收：使用 Binance 15 分钟 OHLC 收盘价与本地 12 币种样本，不能视为论文的
严格复现。
"""

import math

import numpy as np
import pandas as pd

SYMBOLS = [
    "ADAUSDT",
    "BCHUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "EOSUSDT",
    "ETHUSDT",
    "LTCUSDT",
    "SOLUSDT",
    "TRXUSDT",
    "XLMUSDT",
    "XRPUSDT",
]
BAR_MINUTES = 15
LOCAL_WINDOW_BARS = 156
LOCAL_BIPOWER_TERMS = LOCAL_WINDOW_BARS - 2
LOOKBACK_DAYS = 28
LOOKBACK_BARS = LOOKBACK_DAYS * 24 * 60 // BAR_MINUTES
PRE_ESTIMATION_RETURNS = LOCAL_WINDOW_BARS - 1
HISTORY_RETURNS = PRE_ESTIMATION_RETURNS + LOOKBACK_BARS
HISTORY_BARS = HISTORY_RETURNS + 1
JUMP_SIGNIFICANCE = 0.05


def _lee_mykland_threshold(observations: int, significance: float) -> float:
    """返回两侧 Lee-Mykland 风格跳跃检验的 Gumbel 近似阈值。"""
    if observations < 3:
        raise ValueError("跳跃阈值至少需要三根收益观测")
    leading = math.sqrt(2.0 * math.log(observations))
    centering = leading - (math.log(math.pi) + math.log(math.log(observations))) / (2.0 * leading)
    scale = 1.0 / leading
    gumbel_quantile = -math.log(-math.log(1.0 - significance))
    return centering + scale * gumbel_quantile


def decompose_variances(close: pd.Series) -> dict[str, float] | None:
    """将可见收盘价分解为总、正负跳跃及跳跃稳健方差。

    局部双乘积方差在每根收益发生前计算，因此当前收益不参与自己的跳跃阈值。
    这是与 ``context.history`` 配合避免未来函数的关键约束。
    """
    close = pd.to_numeric(close, errors="coerce").dropna()
    if len(close) < HISTORY_BARS:
        return None
    if (close <= 0).any():
        raise ValueError("跳跃方差需要严格为正的收盘价")

    returns = np.log(close).diff().dropna().iloc[-HISTORY_RETURNS:]
    target_returns = returns.iloc[-LOOKBACK_BARS:]
    absolute_returns = returns.abs()
    # 使用 t-1 及更早的收益估计 t 时点的局部波动率，严格不查看当前收益。
    lagged_bipower_terms = absolute_returns.shift(1) * absolute_returns.shift(2)
    local_variance = math.pi / 2.0 * lagged_bipower_terms.rolling(
        LOCAL_BIPOWER_TERMS, min_periods=LOCAL_BIPOWER_TERMS
    ).mean()
    standardized = target_returns / np.sqrt(local_variance.reindex(target_returns.index))
    threshold = _lee_mykland_threshold(LOOKBACK_BARS, JUMP_SIGNIFICANCE)
    is_jump = standardized.abs() > threshold

    squared = target_returns.pow(2)
    positive_jump = squared[is_jump & (target_returns > 0)].sum()
    negative_jump = squared[is_jump & (target_returns < 0)].sum()
    jump_robust = squared[~is_jump].sum()
    total = squared.sum()
    return {
        "total_variance": float(total),
        "positive_jump_variance": float(positive_jump),
        "negative_jump_variance": float(negative_jump),
        "jump_robust_variance": float(jump_robust),
    }


def initialize(context):
    context.set_name("lee_wang_positive_jump_variance_platform_test")
    context.set_data(
        adapter="binance_zip",
        path="data/raw/binance/spot_klines",
        market="spot",
        frequency="15m",
        symbols=SYMBOLS,
        warmup_bars=HISTORY_RETURNS,
    )
    context.set_factor_evaluation(
        formation="weekly",
        horizon_bars=7 * 24 * 60 // BAR_MINUTES,
        entry_price="next_open",
        exit_price="close",
        direction="higher_predicts_lower_return",
        groups=3,
        weighting="equal",
        fee_bps=0.0,
        slippage_bps=0.0,
    )


def main(context):
    rows = []
    for symbol in context.symbols:
        history = context.history(symbol, HISTORY_BARS, fields=["close"])
        components = decompose_variances(history["close"])
        if components is None:
            continue
        rows.append(
            {
                "timestamp": context.now,
                "symbol": symbol,
                "factor": components["positive_jump_variance"],
            }
        )
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "factor"])
