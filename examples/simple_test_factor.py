"""仅用于验证因子接口与回测流程的最小截面因子。"""

FactorMeta = {
    "required_fields": ["timestamp", "symbol", "close"],
    "min_lookback_bars": 1,
    "supported_modes": ["cross_sectional"],
    "output_columns": ["factor"],
}


def compute_factor(context):
    data = context.data.sort_values(["symbol", "timestamp"]).copy()
    data["factor"] = data.groupby("symbol")["close"].pct_change()
    return data[["timestamp", "symbol", "factor"]]
