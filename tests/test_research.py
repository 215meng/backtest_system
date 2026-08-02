import pandas as pd

from quantbacktest.research import evaluate_factor_research, formation_times
from quantbacktest.schemas import FactorResearchSpec, UniverseSpec


def _research_spec() -> FactorResearchSpec:
    return FactorResearchSpec.model_validate(
        {
            "formation": {"kind": "bar_interval", "every_n_bars": 2},
            "returns": {"horizon": "1h", "start_price": "close", "end_price": "close"},
            "direction": "higher_predicts_higher_return",
            "portfolio": {"selection": "quantiles", "quantiles": 2},
        }
    )


def test_research_evaluation_reports_directional_spread_without_account_metrics() -> None:
    timestamps = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    rows = []
    for symbol, rate, factor in [("A", 0.01, 1.0), ("B", 0.02, 2.0), ("C", 0.03, 3.0), ("D", 0.04, 4.0)]:
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": 100 * (1 + rate) ** index,
                    "close": 100 * (1 + rate) ** index,
                    "factor": factor,
                }
            )
    result = evaluate_factor_research(
        pd.DataFrame(rows), _research_spec(), UniverseSpec(min_assets=4)
    )
    assert result.metrics["spread_mean"] > 0
    assert result.metrics["rank_ic_mean"] > 0
    assert "CAGR" not in result.metrics
    assert {"top", "bottom"}.issubset(set(result.contributions["bucket"]))
    assert len(result.leave_one_out) == 4


def test_calendar_schedule_skips_missing_anchor_instead_of_drifting() -> None:
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01T00:00:00Z"),
            pd.Timestamp("2024-01-08T00:15:00Z"),
        ]
    )
    frame = pd.DataFrame({"timestamp": timestamps, "symbol": ["A", "A"], "close": [1.0, 1.0]})
    spec = FactorResearchSpec.model_validate(
        {
            "formation": {"kind": "calendar", "interval": "1w", "weekday": 0, "time_utc": "00:00"},
            "returns": {"horizon": "1w", "start_price": "close", "end_price": "close"},
            "direction": "higher_predicts_higher_return",
            "portfolio": {"selection": "top_k", "top_k": 1},
        }
    )
    valid, skipped = formation_times(frame, spec)
    assert valid.tolist() == [pd.Timestamp("2024-01-01T00:00:00Z")]
    assert skipped["timestamp"].tolist() == [pd.Timestamp("2024-01-08T00:00:00Z")]


def test_time_point_membership_limits_assets_at_each_formation(tmp_path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    rows = []
    for factor, symbol in enumerate(["A", "B", "C", "D"], start=1):
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": 100 + index * factor,
                    "close": 100 + index * factor,
                    "factor": float(factor),
                }
            )
    membership_path = tmp_path / "membership.csv"
    pd.DataFrame(
        {
            "timestamp": [timestamps[0]] * 4,
            "symbol": ["A", "B", "C", "D"],
            "eligible": ["true", "true", "true", "false"],
        }
    ).to_csv(membership_path, index=False)
    result = evaluate_factor_research(
        pd.DataFrame(rows),
        _research_spec(),
        UniverseSpec(min_assets=3, membership_path=membership_path),
    )
    first_period = result.panel[result.panel["formation_time"] == timestamps[0]]
    assert set(first_period["symbol"]) == {"A", "B", "C"}
