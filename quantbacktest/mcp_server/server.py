"""跨项目调用的原生 Python MCP 契约。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from quantbacktest.adapters import available_assets
from quantbacktest.library import attach_strategy_evidence, default_library_root, list_factors
from quantbacktest.native import (
    DownloadManifestError,
    NativeScriptError,
    run_factor_script,
    run_strategy_script,
)
from quantbacktest.native import (
    validate_factor_script as _validate_factor_script,
)
from quantbacktest.native import (
    validate_strategy_script as _validate_strategy_script,
)

mcp = MCPServer(
    "QuantBacktest",
    instructions=(
        "QuantBacktest 只接受原生 Python 因子或事件策略脚本，绝不要求 YAML。"
        "外部项目先调用 create_factor_script 或 create_strategy_script，让平台在项目目录写入正确骨架；"
        "再调用相应 validate 工具。校验失败或能力缺失时，必须把下载清单和修复建议报告给用户，"
        "不得构造代理 YAML 或静默替换数据。仅校验通过后调用 run_factor_tool 或 run_backtest_tool。"
        "因子 main(context) 返回 timestamp/date、symbol/instrument、factor；策略使用 initialize、调度和订单 API。"
    ),
)


def _safe_target(project_root: str, relative_path: str) -> Path:
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("project_root 必须是已存在的外部项目绝对目录")
    target = (root / relative_path).resolve()
    if root not in target.parents:
        raise ValueError("脚本路径必须位于 project_root 内")
    if target.suffix.lower() != ".py":
        raise ValueError("脚本文件必须以 .py 结尾")
    return target


def _factor_template(name: str) -> str:
    return f'''"""{name}：原生 Python 因子。"""
from quantbacktest.api import *
import pandas as pd


def initialize(context):
    context.set_name("{name}")
    context.set_data(
        adapter="binance_zip",
        path="data/raw/binance/spot_klines",
        market="spot",
        frequency="1h",
        symbols=["BTCUSDT", "ETHUSDT"],
        warmup_bars=24,
    )
    context.set_factor_evaluation(
        formation="daily", horizon_bars=24,
        entry_price="next_open", exit_price="close",
        direction="higher_predicts_higher_return", groups=5,
        weighting="equal", fee_bps=0.0, slippage_bps=0.0,
    )


def main(context):
    rows = []
    for symbol in context.symbols:
        bars = context.history(symbol, 25, fields=["close"])
        if len(bars) == 25:
            rows.append({{"timestamp": context.now, "symbol": symbol,
                         "factor": bars["close"].iloc[-1] / bars["close"].iloc[0] - 1}})
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "factor"])
'''


def _strategy_template(name: str) -> str:
    return f'''"""{name}：原生 Python 事件策略。"""
from quantbacktest.api import *


def initialize(context):
    context.set_name("{name}")
    context.set_data(
        adapter="binance_zip",
        path="data/raw/binance/spot_klines",
        market="spot",
        frequency="1h",
        symbols=["BTCUSDT", "ETHUSDT"],
        warmup_bars=20,
    )
    context.set_account(
        initial_cash=100_000.0,
        benchmark="BTCUSDT",
        fee_bps=5.0,
        slippage_bps=2.0,
    )
    run_daily(rebalance, when="close")


def rebalance(context):
    # 策略本身决定信号、选币、仓位、止损和调仓；平台不添加默认风控。
    weights = {{}}
    for symbol in context.symbols:
        bars = context.history(symbol, 21, fields=["close"])
        if len(bars) == 21 and bars["close"].iloc[-1] > bars["close"].mean():
            weights[symbol] = 1.0 / len(context.symbols)
    order_target_weights(weights)
'''


def _result(result: Any) -> dict[str, Any]:
    return {
        "status": "candidate_registration_failed" if result.candidate_registration_error else "completed",
        "run_kind": result.run_kind,
        "run_dir": str(result.run_dir),
        "metrics": result.metrics,
        "candidate": result.candidate,
        "candidate_registration_error": result.candidate_registration_error,
    }


def _validation_error(exc: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {"valid": False, "error": str(exc), "repair": "请按错误位置修改原生 Python 脚本后重新校验。"}
    if isinstance(exc, DownloadManifestError):
        payload["download_manifest"] = exc.manifest
    return payload


@mcp.tool()
def get_external_project_contract() -> dict[str, Any]:
    """返回无需 YAML 的因子、策略、数据与执行契约。"""
    return {
        "version": "native-python-v1",
        "workflow": ["create_*_script", "validate_*_script", "run_*_tool"],
        "factor_contract": "initialize(context) 声明数据和评价口径；main(context) 返回 timestamp/date、symbol/instrument、factor。",
        "strategy_contract": "initialize(context) 声明数据、账户、成本和调度；回调中用 history/get_bars 与 order_target_weights/order_value/order_target。",
        "data": {"adapters": ["binance_zip", "bybit_parquet", "crypto_top50"], "fields": ["timestamp", "symbol", "open", "high", "low", "close", "volume", "turnover"]},
        "unsupported": ["YAML 新运行入口", "A股", "订单簿", "链上数据", "Fama–MacBeth", "自动联网下载"],
    }


@mcp.tool()
def create_factor_script(project_root: str, relative_path: str = "factors/my_factor.py", name: str = "my_factor") -> dict[str, str]:
    """在外部项目中写入可校验的因子 Python 骨架；不会覆盖既有文件。"""
    target = _safe_target(project_root, relative_path)
    if target.exists():
        raise FileExistsError(f"不会覆盖已有脚本：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_factor_template(name), encoding="utf-8")
    return {"script_path": str(target), "next": "调用 validate_factor_script"}


@mcp.tool()
def create_strategy_script(project_root: str, relative_path: str = "strategies/my_strategy.py", name: str = "my_strategy") -> dict[str, str]:
    """在外部项目中写入可校验的事件策略 Python 骨架；不会覆盖既有文件。"""
    target = _safe_target(project_root, relative_path)
    if target.exists():
        raise FileExistsError(f"不会覆盖已有脚本：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_strategy_template(name), encoding="utf-8")
    return {"script_path": str(target), "next": "调用 validate_strategy_script"}


@mcp.tool()
def validate_factor_script(script_path: str, project_root: str) -> dict[str, Any]:
    """校验实际保存的因子脚本、数据声明与因子评价能力；不产生回测结果。"""
    try:
        return _validate_factor_script(Path(script_path), Path(project_root))
    except (NativeScriptError, DownloadManifestError, OSError, ValueError) as exc:
        return _validation_error(exc)


@mcp.tool()
def validate_strategy_script(script_path: str, project_root: str) -> dict[str, Any]:
    """校验实际保存的策略脚本、账户、事件和本地交易能力；不产生回测结果。"""
    try:
        return _validate_strategy_script(Path(script_path), Path(project_root))
    except (NativeScriptError, DownloadManifestError, OSError, ValueError) as exc:
        return _validation_error(exc)


@mcp.tool()
def run_factor_tool(script_path: str, project_root: str) -> dict[str, Any]:
    """重复校验后运行因子，成功时自动进入候选审核区。"""
    validation = validate_factor_script(script_path, project_root)
    if not validation.get("valid"):
        return {"status": "blocked", "validation": validation}
    return _result(run_factor_script(Path(script_path), Path(project_root)))


@mcp.tool()
def run_backtest_tool(script_path: str, project_root: str) -> dict[str, Any]:
    """重复校验后运行事件策略；策略运行不会自动创建因子候选。"""
    validation = validate_strategy_script(script_path, project_root)
    if not validation.get("valid"):
        return {"status": "blocked", "validation": validation}
    return _result(run_strategy_script(Path(script_path), Path(project_root)))


@mcp.tool()
def list_data_assets(adapter: str, path: str) -> dict[str, Any]:
    """列出适配器可用资产与标准字段。"""
    return available_assets(adapter, Path(path))


@mcp.tool()
def attach_strategy_evidence_tool(factor_id: str, strategy_run_dir: str) -> dict[str, str]:
    """将完成的策略运行手工关联为因子候选的交易证据。"""
    return attach_strategy_evidence(factor_id, Path(strategy_run_dir), default_library_root())


@mcp.tool()
def list_factor_library() -> list[dict[str, str | None]]:
    """列出候选审核区，候选批准仍由 Web 人工完成。"""
    return list_factors(default_library_root())


@mcp.tool()
def get_run_status(run_dir: str) -> dict[str, Any]:
    """读取完成运行的指标与工件清单。"""
    directory = Path(run_dir)
    metrics = directory / "metrics.json"
    if not metrics.exists():
        return {"status": "missing_or_incomplete", "run_dir": str(directory)}
    return {"status": "completed", "run_dir": str(directory), "metrics": json.loads(metrics.read_text(encoding="utf-8")), "artifacts": sorted(item.name for item in directory.iterdir())}


def main() -> None:
    mcp.run(transport="stdio")
