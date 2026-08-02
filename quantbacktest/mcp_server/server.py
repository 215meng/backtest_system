from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from quantbacktest.adapters import available_assets
from quantbacktest.engine import run_backtest
from quantbacktest.factors import inspect_factor
from quantbacktest.library import list_factors, promote
from quantbacktest.schemas import RunSpec

mcp = MCPServer("QuantBacktest", instructions="先检查数据和因子，再校验 RunSpec，最后运行回测。")


def _parse_spec(run_spec: dict[str, Any]) -> RunSpec:
    return RunSpec.model_validate(run_spec)


@mcp.tool()
def list_data_assets(adapter: str, path: str) -> dict[str, Any]:
    """列出数据源可用资产和标准字段。"""
    return available_assets(adapter, Path(path))


@mcp.tool()
def inspect_factor_tool(module_path: str) -> dict[str, Any]:
    """读取 FactorMeta，返回因子依赖、回看长度和支持模式。"""
    return inspect_factor(Path(module_path))


@mcp.tool()
def validate_run_spec(run_spec: dict[str, Any]) -> dict[str, Any]:
    """验证完整配置；未知字段和条件缺失字段都会被拒绝。"""
    spec = _parse_spec(run_spec)
    return {"valid": True, "normalized_spec": spec.model_dump(mode="json")}


@mcp.tool()
def run_backtest_tool(run_spec: dict[str, Any], project_root: str) -> dict[str, Any]:
    """执行正常或调试回测，返回结果目录、关键指标和警告。"""
    result = run_backtest(_parse_spec(run_spec), Path(project_root))
    return {"run_dir": str(result.run_dir), "metrics": result.metrics, "warnings": result.warnings}


@mcp.tool()
def get_run_status(run_dir: str) -> dict[str, Any]:
    """读取已完成运行的指标、警告和可用工件。"""
    directory = Path(run_dir)
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        return {"status": "pending_or_dry_run", "run_dir": str(directory), "exists": directory.exists()}
    return {
        "status": "completed",
        "run_dir": str(directory),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "artifacts": sorted(path.name for path in directory.iterdir()),
    }


@mcp.tool()
def get_debug_trace(run_dir: str) -> dict[str, Any]:
    """读取 trace/replay 运行的结构化调试轨迹。"""
    path = Path(run_dir) / "debug_trace.json"
    if not path.exists():
        raise FileNotFoundError("该运行没有调试轨迹；请使用 trace 或 replay 模式")
    return json.loads(path.read_text(encoding="utf-8"))


@mcp.tool()
def promote_factor(run_dir: str, library_root: str) -> dict[str, str]:
    """人工确认后，将运行快照提升到集中因子库。"""
    return promote(Path(run_dir), Path(library_root))


@mcp.tool()
def list_factor_library(library_root: str) -> list[dict[str, str]]:
    """列出已经人工审核通过的因子版本。"""
    return list_factors(Path(library_root))


def main() -> None:
    mcp.run(transport="stdio")
