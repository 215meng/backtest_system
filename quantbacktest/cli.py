"""原生 Python 回测命令行入口；YAML 仅保留为历史工件。"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quantbacktest.adapters import available_assets
from quantbacktest.library import attach_strategy_evidence, default_library_root, list_factors
from quantbacktest.native import (
    DownloadManifestError,
    run_factor_script,
    run_strategy_script,
    validate_factor_script,
    validate_strategy_script,
)

app = typer.Typer(no_args_is_help=True, help="QuantBacktest 原生 Python 因子与策略回测平台")
DEFAULT_PROJECT_ROOT = Path.cwd()
DEFAULT_LIBRARY_ROOT = default_library_root()


def _emit(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _run_payload(result: object) -> dict[str, object]:
    return {
        "status": "candidate_registration_failed" if result.candidate_registration_error else "completed",
        "run_kind": result.run_kind,
        "run_dir": str(result.run_dir),
        "metrics": result.metrics,
        "candidate": result.candidate,
        "candidate_registration_error": result.candidate_registration_error,
    }


@app.command("assets")
def assets(adapter: str, path: Path) -> None:
    """列出本地适配器可用交易对与标准字段。"""
    _emit(available_assets(adapter, path))


@app.command("factor-validate")
def factor_validate(script: Path, project_root: Path = DEFAULT_PROJECT_ROOT) -> None:
    """校验原生 Python 因子脚本及其本地数据能力，不创建运行。"""
    try:
        _emit(validate_factor_script(script, project_root))
    except DownloadManifestError as exc:
        _emit({"valid": False, "error": str(exc), "download_manifest": exc.manifest})
        raise typer.Exit(code=1) from exc


@app.command("strategy-validate")
def strategy_validate(script: Path, project_root: Path = DEFAULT_PROJECT_ROOT) -> None:
    """校验原生 Python 事件策略及其交易能力，不创建运行。"""
    try:
        _emit(validate_strategy_script(script, project_root))
    except DownloadManifestError as exc:
        _emit({"valid": False, "error": str(exc), "download_manifest": exc.manifest})
        raise typer.Exit(code=1) from exc


@app.command("factor-run")
def factor_run(script: Path, project_root: Path = DEFAULT_PROJECT_ROOT) -> None:
    """运行一个原生 Python 因子，并自动登记候选。"""
    _emit(_run_payload(run_factor_script(script, project_root)))


@app.command("strategy-run")
def strategy_run(script: Path, project_root: Path = DEFAULT_PROJECT_ROOT) -> None:
    """运行一个原生 Python 事件策略；策略不会自动成为因子候选。"""
    _emit(_run_payload(run_strategy_script(script, project_root)))


@app.command("attach-strategy-evidence")
def attach_strategy_evidence_command(
    factor_id: str,
    strategy_run_dir: Path,
    library_root: Path = DEFAULT_LIBRARY_ROOT,
) -> None:
    """手工把策略运行关联为某个因子候选的交易证据。"""
    _emit(attach_strategy_evidence(factor_id, strategy_run_dir, library_root))


@app.command("library")
def library_command(library_root: Path = DEFAULT_LIBRARY_ROOT) -> None:
    """查看候选审核区；批准动作仍需在 Web 界面人工完成。"""
    _emit(list_factors(library_root))
