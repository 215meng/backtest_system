import pytest
from pydantic import ValidationError

from quantbacktest.schemas import RunSpec


def base_payload() -> dict:
    return {
        "name": "schema_case",
        "data": {
            "adapter": "crypto_top50",
            "path": "data/raw/crypto_top50",
            "market": "spot",
            "frequency": "1h",
            "symbols": [f"S{i}" for i in range(10)],
        },
        "factor": {"module_path": "factor.py"},
        "strategy": {"mode": "cross_sectional", "selection": "quantiles", "long_short": "long_only", "quantiles": 2},
    }


def test_cross_section_requires_enough_assets() -> None:
    payload = base_payload()
    payload["data"]["symbols"] = ["BTCUSDT"]
    with pytest.raises(ValidationError, match="至少需要"):
        RunSpec.model_validate(payload)


def test_unknown_field_is_rejected() -> None:
    payload = base_payload()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        RunSpec.model_validate(payload)


def test_single_asset_requires_signal_rule() -> None:
    payload = base_payload()
    payload["data"]["symbols"] = ["BTCUSDT"]
    payload["strategy"] = {"mode": "single_asset", "symbol": "BTCUSDT"}
    with pytest.raises(ValidationError, match="signal_rule"):
        RunSpec.model_validate(payload)


def test_max_drawdown_stop_defaults_to_disabled() -> None:
    spec = RunSpec.model_validate(base_payload())
    stop = spec.strategy.risk_control.max_drawdown_stop
    assert stop.enabled is False
    assert stop.threshold is None


def test_max_drawdown_stop_requires_valid_threshold() -> None:
    payload = base_payload()
    payload["strategy"]["risk_control"] = {"max_drawdown_stop": {"enabled": True, "threshold": 0.15}}
    assert RunSpec.model_validate(payload).strategy.risk_control.max_drawdown_stop.threshold == 0.15

    payload["strategy"]["risk_control"] = {"max_drawdown_stop": {"enabled": True}}
    with pytest.raises(ValidationError, match="threshold"):
        RunSpec.model_validate(payload)

    payload["strategy"]["risk_control"] = {"max_drawdown_stop": {"enabled": True, "threshold": 1.0}}
    with pytest.raises(ValidationError):
        RunSpec.model_validate(payload)


def test_factor_mimicking_profile_requires_minimal_strategy_rules() -> None:
    payload = base_payload()
    payload["strategy"].update(
        {
            "profile": "factor_mimicking",
            "long_short": "market_neutral",
            "quantiles": 3,
            "gross_exposure": 1.0,
            "rebalance_bars": 24,
            "execution": {"holding_bars": 24},
        }
    )
    spec = RunSpec.model_validate(payload)
    assert spec.strategy.profile == "factor_mimicking"
    assert spec.strategy.gross_exposure == 1.0

    payload["costs"] = {"fee_bps": 1.0}
    with pytest.raises(ValidationError, match="zero fees"):
        RunSpec.model_validate(payload)


def test_factor_research_requires_explicit_research_contract() -> None:
    payload = base_payload()
    payload.pop("strategy")
    payload["evaluation"] = {"mode": "factor_research"}
    with pytest.raises(ValidationError, match="evaluation.research"):
        RunSpec.model_validate(payload)

    payload["evaluation"] = {
        "mode": "factor_research",
        "research": {
            "formation": {"kind": "calendar", "interval": "1w", "weekday": 6, "time_utc": "23:45"},
            "returns": {"horizon": "1w", "start_price": "close", "end_price": "close"},
            "direction": "higher_predicts_lower_return",
            "portfolio": {"selection": "quantiles", "quantiles": 3},
        },
    }
    spec = RunSpec.model_validate(payload)
    assert spec.strategy is None
    assert spec.evaluation is not None


def test_debug_mode_rejects_boolean_with_actionable_message() -> None:
    payload = base_payload()
    payload["debug"] = {"mode": False}

    with pytest.raises(ValidationError, match="debug.mode 必须是字符串"):
        RunSpec.model_validate(payload)
