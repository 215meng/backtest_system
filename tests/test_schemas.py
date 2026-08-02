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
