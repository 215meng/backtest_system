from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).parents[1] / "examples" / "lee_wang_positive_jump_variance.py"


def _module():
    spec = importlib.util.spec_from_file_location("lee_wang_factor", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _close_from_returns(returns: list[float]) -> pd.Series:
    return pd.Series(100.0 * np.exp(np.r_[0.0, np.cumsum(returns)]))


def _stable_returns(length: int) -> list[float]:
    return [0.001 if index % 2 else -0.001 for index in range(length)]


def test_decomposition_separates_positive_and_negative_jumps() -> None:
    module = _module()
    returns = _stable_returns(module.HISTORY_RETURNS - 1)
    positive = module.decompose_variances(_close_from_returns([*returns, 0.12]))
    negative = module.decompose_variances(_close_from_returns([*returns, -0.12]))

    assert positive is not None and negative is not None
    assert positive["positive_jump_variance"] > 0.01
    assert positive["negative_jump_variance"] == 0.0
    assert negative["positive_jump_variance"] == 0.0
    assert negative["negative_jump_variance"] > 0.01
    assert np.isclose(
        positive["total_variance"],
        positive["positive_jump_variance"]
        + positive["negative_jump_variance"]
        + positive["jump_robust_variance"],
    )


def test_decomposition_requires_complete_history() -> None:
    module = _module()

    assert module.decompose_variances(_close_from_returns(_stable_returns(module.HISTORY_RETURNS - 1))) is None


def test_pre_estimation_returns_are_not_aggregated_into_factor() -> None:
    module = _module()
    returns = _stable_returns(module.HISTORY_RETURNS)
    returns[module.PRE_ESTIMATION_RETURNS - 1] = 0.25

    result = module.decompose_variances(_close_from_returns(returns))

    assert result is not None
    assert result["positive_jump_variance"] == 0.0
    assert np.isclose(result["total_variance"], module.LOOKBACK_BARS * 0.001**2)


def test_main_uses_history_only_and_returns_factor_contract() -> None:
    module = _module()
    close = _close_from_returns([*_stable_returns(module.HISTORY_RETURNS - 1), 0.12])

    class Context:
        symbols = ("TESTUSDT",)
        now = pd.Timestamp("2024-02-04T23:45:00Z")

        def history(self, symbol: str, bars: int, fields: list[str]) -> pd.DataFrame:
            assert symbol == "TESTUSDT"
            assert bars == module.HISTORY_BARS
            assert fields == ["close"]
            return pd.DataFrame({"close": close})

    result = module.main(Context())

    assert list(result.columns) == ["timestamp", "symbol", "factor"]
    assert result.loc[0, "timestamp"] == Context.now
    assert result.loc[0, "symbol"] == "TESTUSDT"
    assert result.loc[0, "factor"] > 0.01
