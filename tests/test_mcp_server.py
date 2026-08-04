from __future__ import annotations

from pathlib import Path

from quantbacktest.mcp_server import server


def _write_crypto_data(root: Path) -> None:
    data = root / "data" / "raw" / "binance" / "spot_klines"
    data.mkdir(parents=True)
    # 生成器默认路径在此测试中故意不可用；校验应返回下载清单而不是执行 YAML。


def test_mcp_generates_native_python_skeleton_without_yaml(tmp_path: Path) -> None:
    created = server.create_factor_script(str(tmp_path), "factors/momentum.py", "momentum")
    script = Path(created["script_path"])

    assert script.is_file()
    content = script.read_text(encoding="utf-8")
    assert "def initialize(context)" in content
    assert "def main(context)" in content
    assert "warmup_bars=24" in content
    assert "start=" not in content
    assert "end=" not in content
    assert ".yaml" not in content
    assert created["next"] == "调用 validate_factor_script"


def test_mcp_validation_blocks_missing_local_data_with_manifest(tmp_path: Path) -> None:
    created = server.create_strategy_script(str(tmp_path), "strategies/trend.py", "trend")
    content = Path(created["script_path"]).read_text(encoding="utf-8")

    assert "warmup_bars=20" in content
    assert "start=" not in content
    assert "end=" not in content

    validation = server.validate_strategy_script(created["script_path"], str(tmp_path))

    assert validation["valid"] is False
    assert "download_manifest" in validation
    blocked = server.run_backtest_tool(created["script_path"], str(tmp_path))
    assert blocked["status"] == "blocked"


def test_mcp_contract_has_no_yaml_or_paper_preflight_chain() -> None:
    contract = server.get_external_project_contract()

    assert contract["version"] == "native-python-v1"
    assert "YAML 新运行入口" in contract["unsupported"]
    assert contract["workflow"] == ["create_*_script", "validate_*_script", "run_*_tool"]
