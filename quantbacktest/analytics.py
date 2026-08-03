from __future__ import annotations

import math

import pandas as pd


def bars_per_year(frequency: str) -> int:
    return {"1m": 365 * 24 * 60, "15m": 365 * 24 * 4, "1h": 365 * 24, "4h": 365 * 6, "1d": 365}[frequency]


def performance_metrics(returns: pd.Series, frequency: str) -> dict[str, object]:
    returns = returns.dropna()
    if returns.empty:
        return {"observations": 0, "total_return": None, "sharpe": None, "max_drawdown": None}
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annual_factor = bars_per_year(frequency)
    volatility = float(returns.std(ddof=1) * math.sqrt(annual_factor)) if len(returns) > 1 else 0.0
    invalid_equity = equity <= 0
    first_invalid_time = equity.index[invalid_equity][0] if invalid_equity.any() else None
    annual_return = (
        None
        if invalid_equity.any()
        else float(equity.iloc[-1] ** (annual_factor / len(returns)) - 1.0)
    )
    sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(annual_factor)) if returns.std(ddof=1) else None
    max_drawdown = float(drawdown.min())
    return {
        "observations": len(returns),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if annual_return is not None and max_drawdown < 0 else None,
        "account_equity_non_positive": bool(invalid_equity.any()),
        "first_account_failure_time": first_invalid_time.isoformat() if first_invalid_time is not None else None,
    }
