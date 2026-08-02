from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from quantbacktest.factors import FactorContext, execute_factor, inspect_factor
from quantbacktest.schemas import RunSpec

DOCUMENT_PATH = Path(__file__).parents[1] / "docs" / "跨项目调用指南.md"


def _snippet(name: str, language: str) -> str:
    content = DOCUMENT_PATH.read_text(encoding="utf-8")
    pattern = rf"<!-- snippet:{re.escape(name)} -->\s*```{language}\n(.*?)```"
    match = re.search(pattern, content, flags=re.DOTALL)
    assert match is not None, f"未找到文档样例 {name}"
    return match.group(1)


def test_document_yaml_examples_match_run_spec_schema() -> None:
    examples = {
        "factor_research_yaml": "factor_research",
        "strategy_simulation_yaml": "strategy_simulation",
        "ml_factor_research_yaml": "factor_research",
    }

    for name, expected_mode in examples.items():
        payload = yaml.safe_load(_snippet(name, "yaml"))
        spec = RunSpec.model_validate(payload)
        assert spec.evaluation is not None
        assert spec.evaluation.mode.value == expected_mode


def test_document_factor_example_can_be_inspected_and_executed(tmp_path: Path) -> None:
    factor_path = tmp_path / "momentum_factor.py"
    factor_path.write_text(_snippet("factor_python", "python"), encoding="utf-8")

    meta = inspect_factor(factor_path)
    assert meta["required_fields"] == ["timestamp", "symbol", "close"]
    assert meta["min_lookback_bars"] == 24
    assert meta["supported_modes"] == ["cross_sectional"]

    timestamps = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    rows = [
        {"timestamp": timestamp, "symbol": f"ASSET{symbol}", "close": 100 + bar + symbol}
        for symbol in range(10)
        for bar, timestamp in enumerate(timestamps)
    ]
    result, provenance = execute_factor(
        factor_path,
        "compute_factor",
        FactorContext(data=pd.DataFrame(rows), parameters={"lookback_bars": 24}),
    )

    assert list(result.columns) == ["timestamp", "symbol", "factor"]
    assert result["timestamp"].dt.tz is not None
    assert result.duplicated(["timestamp", "symbol"]).sum() == 0
    assert provenance["meta"] == meta
