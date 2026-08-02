"""可复制到其他研究项目的外部因子示例。"""

FactorMeta = {
    "required_fields": ["timestamp", "symbol", "close"],
    "min_lookback_bars": 24,
    "supported_modes": ["cross_sectional", "single_asset"],
    "output_columns": ["factor"],
}


def compute_factor(context):
    data = context.data.sort_values(["symbol", "timestamp"]).copy()
    lookback = int(context.parameters.get("lookback_bars", 24))
    data["factor"] = data.groupby("symbol")["close"].pct_change(lookback)
    return data[["timestamp", "symbol", "factor"]]
