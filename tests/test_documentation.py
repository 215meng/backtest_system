from __future__ import annotations

from pathlib import Path

DOCUMENT_PATH = Path(__file__).parents[1] / "docs" / "跨项目调用指南.md"
README_PATH = Path(__file__).parents[1] / "README.md"


def test_cross_project_documentation_describes_native_python_only() -> None:
    content = DOCUMENT_PATH.read_text(encoding="utf-8")

    assert "create_factor_script" in content
    assert "def initialize(context):" in content
    assert "def main(context):" in content
    assert "run_daily(rebalance" in content
    assert "不需要也不接受 YAML" in content
    assert "FactorMeta" not in content
    assert "preflight_id" not in content


def test_readme_exposes_native_python_commands() -> None:
    content = README_PATH.read_text(encoding="utf-8")

    assert "quantbacktest factor-run" in content
    assert "quantbacktest strategy-run" in content
    assert "quantbacktest run examples" not in content
