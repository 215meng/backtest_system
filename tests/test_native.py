from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quantbacktest.library import attach_strategy_evidence, list_factors
from quantbacktest.native import (
    NativeScriptError,
    run_factor_script,
    run_strategy_script,
    validate_factor_script,
)


def _data(root: Path, symbols: int = 2) -> Path:
    data = root / "data"
    data.mkdir()
    timestamps = pd.date_range("2024-01-01", periods=32, freq="h", tz="UTC")
    for index in range(symbols):
        values = [100 + index * 10 + step * (1 + index) for step in range(len(timestamps))]
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": values,
                "high": [value + 1 for value in values],
                "low": [value - 1 for value in values],
                "close": values,
                "volume_from": 1.0,
                "volume_to": values,
            }
        ).to_csv(data / f"S{index}_1h.csv", index=False)
    return data


def _factor_script(path: Path, data: Path, *, invalid: bool = False) -> None:
    script = f'''from quantbacktest.api import *
import pandas as pd
def initialize(context):
    context.set_name("native_factor")
    context.set_data(adapter="crypto_top50", path=r"{data}", market="spot", frequency="1h", symbols=["S0", "S1"], start="2024-01-01T00:00:00Z", end="2024-01-02T07:00:00Z")
    context.set_factor_evaluation(formation="daily", horizon_bars=1, entry_price="close", exit_price="close", direction="higher_predicts_higher_return", groups=2, weighting="equal", fee_bps=1.0, slippage_bps=1.0)
def main(context):
    return {"pd.DataFrame()" if invalid else "pd.DataFrame([{'timestamp': context.now, 'symbol': 'S0', 'factor': 1.0}, {'timestamp': context.now, 'symbol': 'S1', 'factor': 2.0}])"}
'''
    path.write_text(script, encoding="utf-8")


def _strategy_script(path: Path, data: Path) -> None:
    path.write_text(
        f'''from quantbacktest.api import *
def initialize(context):
    context.set_name("native_strategy")
    context.set_data(adapter="crypto_top50", path=r"{data}", market="spot", frequency="1h", symbols=["S0", "S1"], start="2024-01-01T00:00:00Z", end="2024-01-02T07:00:00Z")
    context.set_account(initial_cash=10000.0, benchmark="S0", fee_bps=1.0, slippage_bps=1.0)
    run_every_bars(rebalance, when="close")
def rebalance(context):
    order_target_weights({{"S0": 0.5, "S1": 0.5}})
''',
        encoding="utf-8",
    )


def test_native_factor_generates_report_and_candidate(tmp_path: Path) -> None:
    data = _data(tmp_path)
    script = tmp_path / "factor.py"
    _factor_script(script, data)

    result = run_factor_script(script, tmp_path, tmp_path / "library")

    assert result.candidate is not None
    assert (result.run_dir / "factor_values.csv").exists()
    assert (result.run_dir / "group_cumulative_returns.csv").exists()
    assert (result.run_dir / "report.html").exists()
    assert json.loads((result.run_dir / "run_spec.json").read_text(encoding="utf-8"))["run_kind"] == "factor"
    assert list_factors(tmp_path / "library", status="candidate")[0]["source_run"] == str(
        result.run_dir.resolve()
    )


def test_invalid_native_factor_does_not_create_candidate(tmp_path: Path) -> None:
    data = _data(tmp_path)
    script = tmp_path / "factor.py"
    _factor_script(script, data, invalid=True)

    with pytest.raises(NativeScriptError, match="缺少字段|为空"):
        run_factor_script(script, tmp_path, tmp_path / "library")
    assert list_factors(tmp_path / "library") == []


def test_native_strategy_is_separate_and_can_be_attached_as_evidence(tmp_path: Path) -> None:
    data = _data(tmp_path)
    factor_script = tmp_path / "factor.py"
    strategy_script = tmp_path / "strategy.py"
    _factor_script(factor_script, data)
    _strategy_script(strategy_script, data)
    factor = run_factor_script(factor_script, tmp_path, tmp_path / "library")
    strategy = run_strategy_script(strategy_script, tmp_path)

    assert strategy.candidate is None
    orders = pd.read_csv(strategy.run_dir / "orders.csv")
    assert {"filled", "rejected"}.issubset(set(orders["status"]))
    assert orders.loc[orders["status"] == "rejected", "reason"].notna().all()
    attached = attach_strategy_evidence(factor.candidate["factor_id"], strategy.run_dir, tmp_path / "library")
    assert attached["status"] == "evidence_attached"
    assert (tmp_path / "library" / "artifacts" / factor.candidate["factor_id"] / "strategy_evidence" / "orders.csv").exists()


def test_factor_context_rejects_future_or_wrong_timestamp_output(tmp_path: Path) -> None:
    data = _data(tmp_path)
    script = tmp_path / "factor.py"
    _factor_script(script, data)
    script.write_text(script.read_text(encoding="utf-8").replace("context.now, 'symbol': 'S0'", "context.now + pd.Timedelta(hours=1), 'symbol': 'S0'"), encoding="utf-8")

    with pytest.raises(NativeScriptError, match="timestamp"):
        run_factor_script(script, tmp_path, tmp_path / "library")


def test_static_validation_rejects_obvious_future_shift(tmp_path: Path) -> None:
    data = _data(tmp_path)
    script = tmp_path / "factor.py"
    _factor_script(script, data)
    script.write_text(script.read_text(encoding="utf-8") + "\nfuture = pd.Series([1]).shift(-1)\n", encoding="utf-8")

    with pytest.raises(NativeScriptError, match="未来数据"):
        validate_factor_script(script, tmp_path, check_data=False)
