from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from quantbacktest.adapters import available_assets
from quantbacktest.engine import run_backtest
from quantbacktest.factors import inspect_factor
from quantbacktest.library import list_factors, promote
from quantbacktest.schemas import RunSpec

app = typer.Typer(no_args_is_help=True, help="加密货币因子研究与回测")


def _spec(path: Path) -> RunSpec:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunSpec.model_validate(payload)


@app.command("schema")
def schema() -> None:
    """打印 Codex/MCP 可使用的 JSON Schema。"""
    typer.echo(json.dumps(RunSpec.model_json_schema(), ensure_ascii=False, indent=2))


@app.command("validate")
def validate(path: Path) -> None:
    """验证配置，不计算因子或生成交易。"""
    spec = _spec(path)
    typer.echo(json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str))


@app.command("assets")
def assets(adapter: str, path: Path) -> None:
    typer.echo(json.dumps(available_assets(adapter, path), ensure_ascii=False, indent=2))


@app.command("inspect-factor")
def inspect_factor_command(path: Path) -> None:
    typer.echo(json.dumps(inspect_factor(path), ensure_ascii=False, indent=2))


@app.command("run")
def run(path: Path) -> None:
    result = run_backtest(_spec(path), Path.cwd())
    typer.echo(json.dumps({"run_dir": str(result.run_dir), "metrics": result.metrics}, ensure_ascii=False, indent=2, default=str))


@app.command("promote")
def promote_command(run_dir: Path, library_root: Path = Path("factor_library")) -> None:
    typer.echo(json.dumps(promote(run_dir, library_root), ensure_ascii=False, indent=2))


@app.command("library")
def library_command(library_root: Path = Path("factor_library")) -> None:
    typer.echo(json.dumps(list_factors(library_root), ensure_ascii=False, indent=2))
