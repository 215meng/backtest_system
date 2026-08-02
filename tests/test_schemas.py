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
