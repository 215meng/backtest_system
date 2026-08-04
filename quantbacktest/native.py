"""不依赖 YAML 的原生 Python 因子、策略运行时。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantbacktest.adapters import load_market_data
from quantbacktest.adapters.market import DataContractError
from quantbacktest.analytics import performance_metrics
from quantbacktest.api import (
    _ACTIVE_CONTEXT,
    DataDeclaration,
    FactorEvaluation,
    Order,
    Position,
    ScriptContext,
    ScriptContractError,
)
from quantbacktest.artifacts import (
    create_run_dir,
    render_native_factor_report,
    render_native_strategy_report,
    write_json,
)
from quantbacktest.library import LibraryError, register_completed_run


class NativeScriptError(ValueError):
    """原生 Python 脚本的可定位校验或执行错误。"""


class DownloadManifestError(NativeScriptError):
    """本地数据无法满足脚本声明时携带明确下载清单。"""

    def __init__(self, message: str, manifest: dict[str, Any]) -> None:
        super().__init__(message)
        self.manifest = manifest


DEFAULT_BACKTEST_START = "2024-01-01T00:00:00+00:00"
DEFAULT_BACKTEST_END = "2025-01-01T23:59:59.999999+00:00"
METRIC_SCHEMA_VERSION = 2


@dataclass
class NativeRunResult:
    run_dir: Path
    metrics: dict[str, Any]
    warnings: list[str]
    run_kind: Literal["factor", "strategy"]
    candidate: dict[str, str] | None = None
    candidate_registration_error: str | None = None


def _module_name(path: Path) -> str:
    return f"quantbacktest_native_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"


def _load_module(path: Path) -> ModuleType:
    if path.suffix.lower() != ".py" or not path.is_file():
        raise NativeScriptError(f"未找到可执行的 Python 脚本：{path}")
    spec = importlib.util.spec_from_file_location(_module_name(path), path)
    if spec is None or spec.loader is None:
        raise NativeScriptError(f"无法加载脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # 用户脚本的导入异常必须保留原始原因。
        raise NativeScriptError(f"导入脚本失败：{type(exc).__name__}: {exc}") from exc
    return module


def _source_functions(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise NativeScriptError(f"Python 语法错误，第 {exc.lineno} 行：{exc.msg}") from exc
    blocked_imports = {"dai", "jqdata", "bigmodule", "requests", "urllib", "httpx"}
    blocked_calls = {"open", "read_csv", "read_parquet", "read_excel", "read_sql", "read_pickle"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {alias.name.split(".")[0] for alias in node.names}
            if names & blocked_imports:
                raise NativeScriptError(
                    f"第 {node.lineno} 行导入了平台不支持的外部数据接口：{sorted(names & blocked_imports)}；请使用 context.set_data/history/get_bars"
                )
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in blocked_calls:
                raise NativeScriptError(
                    f"第 {node.lineno} 行直接读取文件或数据表：{name}；原生回测脚本只能通过 context 提供的数据访问市场数据"
                )
            if name == "shift" and node.args:
                first = node.args[0]
                is_negative = isinstance(first, ast.UnaryOp) and isinstance(first.op, ast.USub)
                if is_negative:
                    raise NativeScriptError(f"第 {node.lineno} 行使用 shift(-n)，这会制造未来数据引用")
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _initialise(path: Path, kind: Literal["factor", "strategy"]) -> tuple[ModuleType, ScriptContext]:
    functions = _source_functions(path)
    required = {"initialize", "main"} if kind == "factor" else {"initialize"}
    missing = sorted(required - functions)
    if missing:
        raise NativeScriptError(f"脚本缺少公开回调：{', '.join(missing)}")
    module = _load_module(path)
    initialize = getattr(module, "initialize", None)
    if not callable(initialize):
        raise NativeScriptError("initialize(context) 必须是可调用函数")
    context = ScriptContext(kind)
    token = _ACTIVE_CONTEXT.set(context)
    try:
        initialize(context)
    except ScriptContractError:
        raise
    except Exception as exc:
        raise NativeScriptError(f"initialize(context) 执行失败：{type(exc).__name__}: {exc}") from exc
    finally:
        _ACTIVE_CONTEXT.reset(token)
    if context.data_declaration is None:
        raise NativeScriptError("initialize(context) 必须调用 context.set_data(...)")
    if kind == "factor" and context.factor_evaluation is None:
        raise NativeScriptError("因子脚本必须调用 context.set_factor_evaluation(...)")
    if kind == "strategy":
        if context.account_declaration is None:
            raise NativeScriptError("策略脚本必须调用 context.set_account(...)")
        if not context.schedules:
            raise NativeScriptError("策略脚本必须使用 run_daily/run_weekly/run_every_bars 注册至少一个回调")
    context.name = context.name or path.stem
    return module, context


def _resolved_declaration(declaration: DataDeclaration, project_root: Path) -> DataDeclaration:
    source = Path(declaration.path)
    return replace(
        declaration,
        path=str(source if source.is_absolute() else (project_root / source).resolve()),
    )


def _end_of_day_bar_start(timestamp: pd.Timestamp, frequency: str) -> pd.Timestamp:
    """将 Web 的全天结束时刻映射到该频率最后一根 K 线的开盘时刻。"""
    end_of_day = timestamp.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if timestamp != end_of_day:
        return timestamp
    minutes = {"1m": 1, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}[frequency]
    return timestamp.normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=minutes)


def _platform_range(
    context: ScriptContext,
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, str]:
    """解析平台唯一控制的评价区间，并计算预热行情加载起点。"""
    declaration = context.data_declaration
    assert declaration is not None
    try:
        parsed_start = pd.to_datetime(start or DEFAULT_BACKTEST_START, utc=True)
        requested_end = pd.to_datetime(end or DEFAULT_BACKTEST_END, utc=True)
    except (TypeError, ValueError) as exc:
        raise NativeScriptError("平台回测区间必须是可解析的 ISO 8601 时间") from exc
    effective_end = _end_of_day_bar_start(requested_end, declaration.frequency)
    if parsed_start > effective_end:
        raise NativeScriptError("平台回测开始时间不能晚于结束时间")
    bar_delta = pd.Timedelta({"1m": "1min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}[declaration.frequency])
    load_start = parsed_start - declaration.warmup_bars * bar_delta
    return {
        "start": parsed_start.isoformat(),
        "end": requested_end.isoformat(),
        "effective_end": effective_end.isoformat(),
        "load_start": load_start.isoformat(),
        "load_end": effective_end.isoformat(),
    }


def _availability_manifest(
    declaration: DataDeclaration,
    data_range: dict[str, str],
    actual: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return {
        "adapter": declaration.adapter,
        "market": declaration.market,
        "frequency": declaration.frequency,
        "symbols": declaration.symbols,
        "requested_start": data_range["load_start"],
        "requested_end": data_range["load_end"],
        "required_fields": ["timestamp", "symbol", "open", "high", "low", "close", "volume", "turnover"],
        "available_start": actual["timestamp"].min().isoformat() if actual is not None and not actual.empty else None,
        "available_end": actual["timestamp"].max().isoformat() if actual is not None and not actual.empty else None,
        "action": "请下载缺失资产/频率/日期/字段后重试；平台不会自动联网下载或替换数据。",
    }


def _load_declared_data(
    context: ScriptContext,
    project_root: Path,
    data_range: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    declaration = context.data_declaration
    assert declaration is not None
    try:
        frame, metadata = load_market_data(
            _resolved_declaration(declaration, project_root),
            start=data_range["load_start"],
            end=data_range["load_end"],
        )
    except (DataContractError, ValueError) as exc:
        raise DownloadManifestError(str(exc), _availability_manifest(declaration, data_range)) from exc
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True), metadata


def validate_factor_script(
    path: Path,
    project_root: Path | None = None,
    *,
    check_data: bool = True,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    _, context = _initialise(path.resolve(), "factor")
    data_range = _platform_range(context, start=start, end=end)
    payload: dict[str, Any] = {"valid": True, "kind": "factor", "manifest": context.manifest(), "platform_backtest_range": data_range}
    if check_data:
        frame, data = _load_declared_data(context, project_root or path.parent, data_range)
        payload["data"] = {**data, "rows": len(frame)}
    return payload


def validate_strategy_script(
    path: Path,
    project_root: Path | None = None,
    *,
    check_data: bool = True,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    _, context = _initialise(path.resolve(), "strategy")
    data_range = _platform_range(context, start=start, end=end)
    account = context.account_declaration
    assert account is not None and context.data_declaration is not None
    if context.data_declaration.market == "spot" and account.leverage is not None:
        raise NativeScriptError("现货账户不支持 leverage；现货仅允许现金多头")
    if context.data_declaration.market == "linear_perp":
        if account.leverage is None or account.margin_mode != "isolated" or account.funding_bps_per_bar is None:
            raise DownloadManifestError(
                "线性合约必须明确逐仓、杠杆和资金费率，且本地数据还需合约规格",
                {**_availability_manifest(context.data_declaration, data_range), "required_contract_fields": ["funding_rate", "contract_spec"]},
            )
        # 当前适配器仅有 K 线；即使脚本声明费率，也不能伪造合约规格。
        raise DownloadManifestError(
            "当前 Bybit 本地适配器没有合约规格与资金费率时间序列",
            {**_availability_manifest(context.data_declaration, data_range), "required_contract_fields": ["funding_rate", "contract_spec"]},
        )
    payload: dict[str, Any] = {"valid": True, "kind": "strategy", "manifest": context.manifest(), "platform_backtest_range": data_range}
    if check_data:
        frame, data = _load_declared_data(context, project_root or path.parent, data_range)
        payload["data"] = {**data, "rows": len(frame)}
    return payload


def _formations(frame: pd.DataFrame, formation: str) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    if formation == "bar":
        return times
    series = pd.Series(times, index=times)
    if formation == "daily":
        return pd.DatetimeIndex(series.groupby(times.normalize()).max().to_list())
    iso = times.isocalendar()
    keys = pd.MultiIndex.from_arrays([iso.year, iso.week])
    return pd.DatetimeIndex(series.groupby(keys).max().to_list())


def _normalise_factor_output(value: Any, now: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise NativeScriptError("main(context) 必须返回 Pandas DataFrame")
    frame = value.copy().rename(columns={"date": "timestamp", "instrument": "symbol"})
    required = {"timestamp", "symbol", "factor"}
    missing = required - set(frame.columns)
    if missing:
        raise NativeScriptError(f"因子输出缺少字段：{sorted(missing)}")
    frame = frame[["timestamp", "symbol", "factor"]]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
    # 暖启动期间允许某一个形成时点没有信号；整次运行结束后仍会拒绝全空因子。
    if frame.empty:
        return frame
    if frame["factor"].notna().sum() == 0:
        return frame
    if frame["timestamp"].isna().any() or frame["symbol"].isna().any() or not frame["timestamp"].eq(now).all():
        raise NativeScriptError("因子输出 timestamp 必须全部等于当前形成时点，且不能缺失")
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise NativeScriptError("因子输出存在重复 timestamp-symbol 行")
    if not np.isfinite(frame.loc[frame["factor"].notna(), "factor"]).all():
        raise NativeScriptError("因子输出包含非有限数值")
    return frame


def _entry_exit(frame: pd.DataFrame, factors: pd.DataFrame, evaluation: FactorEvaluation) -> pd.DataFrame:
    times = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    locations = {timestamp: index for index, timestamp in enumerate(times)}
    price = frame.set_index(["timestamp", "symbol"])[["open", "close"]]
    rows: list[pd.DataFrame] = []
    for now, group in factors.groupby("timestamp", sort=True):
        i = locations.get(now)
        if i is None:
            continue
        entry_i = i if evaluation.entry_price == "close" else i + 1
        # horizon_bars 从形成时点开始计数：close 信号持有 1 根到下一根；
        # next_open 信号则从下一根开盘进入并在同一根收盘（horizon=1）退出。
        exit_i = i + evaluation.horizon_bars
        if exit_i >= len(times):
            continue
        entry_field = "close" if evaluation.entry_price == "close" else "open"
        entry = price.reindex(pd.MultiIndex.from_product([[times[entry_i]], group["symbol"]], names=["timestamp", "symbol"]))[entry_field].to_numpy()
        exit_ = price.reindex(pd.MultiIndex.from_product([[times[exit_i]], group["symbol"]], names=["timestamp", "symbol"]))[evaluation.exit_price].to_numpy()
        item = group.copy()
        item["entry_price"] = entry
        item["exit_price"] = exit_
        item["forward_return"] = item["exit_price"] / item["entry_price"] - 1.0
        item["return_end_time"] = times[exit_i]
        rows.append(item)
    if not rows:
        raise NativeScriptError("形成时点后没有足够的入场/出场 K 线，请缩短样本或持有期")
    return pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def _weight_group(group: pd.DataFrame, weighting: str) -> pd.Series:
    if weighting == "equal":
        return pd.Series(1.0 / len(group), index=group.index)
    values = group["factor"].abs().astype(float)
    return values / values.sum() if values.sum() > 0 else pd.Series(1.0 / len(group), index=group.index)


def _factor_evaluation(
    panel: pd.DataFrame,
    evaluation: FactorEvaluation,
    frequency: str,
    opportunity_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valid = panel.dropna(subset=["factor", "forward_return"]).copy()
    if valid.empty:
        raise NativeScriptError("没有同时具有有效因子与前瞻收益的观测")
    direction = 1 if evaluation.direction == "higher_predicts_higher_return" else -1
    groups_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    previous = pd.Series(dtype=float)
    for timestamp, item in valid.groupby("timestamp", sort=True):
        item = item.sort_values("factor", kind="stable").copy()
        if len(item) < evaluation.groups:
            continue
        item["group"] = np.repeat(np.arange(1, evaluation.groups + 1), np.diff(np.linspace(0, len(item), evaluation.groups + 1, dtype=int)))
        # np.repeat 的四舍五入可能有轻微偏差，使用 array_split 保证完整覆盖。
        item["group"] = 0
        for index, indexes in enumerate(np.array_split(np.arange(len(item)), evaluation.groups), start=1):
            item.iloc[indexes, item.columns.get_loc("group")] = index
        weights: list[pd.DataFrame] = []
        for group_id, bucket in item.groupby("group"):
            bucket = bucket.copy()
            bucket["weight"] = _weight_group(bucket, evaluation.weighting)
            weights.append(bucket)
            groups_rows.append({"timestamp": timestamp, "group": int(group_id), "return": float((bucket["weight"] * bucket["forward_return"]).sum()), "assets": len(bucket)})
        selected = pd.concat(weights)
        high = selected[selected["group"] == evaluation.groups].copy()
        low = selected[selected["group"] == 1].copy()
        long, short = (high, low) if direction == 1 else (low, high)
        target = pd.concat([
            long.assign(signed_weight=0.5 * long["weight"]),
            short.assign(signed_weight=-0.5 * short["weight"]),
        ]).set_index("symbol")["signed_weight"]
        aligned = target.reindex(target.index.union(previous.index), fill_value=0.0)
        turnover = float((aligned - previous.reindex(aligned.index, fill_value=0.0)).abs().sum() / 2)
        pre_cost = float((target * selected.set_index("symbol").loc[target.index, "forward_return"]).sum())
        cost = turnover * (evaluation.fee_bps + evaluation.slippage_bps) / 10_000
        portfolio_rows.append({"timestamp": timestamp, "gross_return": pre_cost, "cost": cost, "net_return": pre_cost - cost, "turnover": turnover})
        previous = target
    groups = pd.DataFrame(groups_rows)
    returns = pd.DataFrame(portfolio_rows)
    if groups.empty or returns.empty:
        raise NativeScriptError("每个形成时点的有效资产少于分组数，无法进行截面因子评价")
    group_curve = groups.pivot(index="timestamp", columns="group", values="return").sort_index()
    cumulative = (1.0 + group_curve.fillna(0.0)).cumprod() - 1.0
    returns = returns.set_index("timestamp").sort_index()
    ic = valid.groupby("timestamp").apply(lambda group: group["factor"].corr(group["forward_return"]), include_groups=False)
    rank_ic = valid.groupby("timestamp").apply(lambda group: group["factor"].corr(group["forward_return"], method="spearman"), include_groups=False)
    annual = {
        "weekly": 52,
        "daily": 365,
        "bar": {"1m": 525600, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}[frequency],
    }[evaluation.formation]
    def stats(series: pd.Series) -> dict[str, float | None]:
        series = series.dropna()
        if series.empty:
            return {"total_return": None, "annual_return": None, "annual_volatility": None, "sharpe": None, "max_drawdown": None}
        equity = pd.concat([pd.Series([1.0]), (1 + series).cumprod().reset_index(drop=True)], ignore_index=True)
        drawdown = equity / equity.cummax() - 1
        vol = float(series.std(ddof=1) * math.sqrt(annual)) if len(series) > 1 else 0.0
        sharpe = float(series.mean() / series.std(ddof=1) * math.sqrt(annual)) if len(series) > 1 and series.std(ddof=1) else None
        return {"total_return": float(equity.iloc[-1] - 1), "annual_return": float(equity.iloc[-1] ** (annual / len(series)) - 1), "annual_volatility": vol, "sharpe": sharpe, "max_drawdown": float(drawdown.min())}
    gross_stats = stats(returns["gross_return"])
    net_stats = stats(returns["net_return"])
    metrics: dict[str, Any] = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "evaluation_mode": "native_factor",
        "annualization_periods": annual,
        "factor_output_coverage": float(panel["factor"].notna().sum() / opportunity_count) if opportunity_count else 0.0,
        "evaluable_coverage": float(len(valid) / opportunity_count) if opportunity_count else 0.0,
        "factor_coverage": float(len(valid) / opportunity_count) if opportunity_count else 0.0,
        "formation_periods": len(returns),
        "ic_mean": float(ic.mean()) if not ic.dropna().empty else None,
        "rank_ic_mean": float(rank_ic.mean()) if not rank_ic.dropna().empty else None,
        "icir": float(ic.mean() / ic.std(ddof=1)) if len(ic.dropna()) > 1 and ic.std(ddof=1) else None,
        "rank_icir": float(rank_ic.mean() / rank_ic.std(ddof=1)) if len(rank_ic.dropna()) > 1 and rank_ic.std(ddof=1) else None,
        "monotonicity": float(np.corrcoef(np.arange(1, evaluation.groups + 1), groups.groupby("group")["return"].mean().reindex(range(1, evaluation.groups + 1)))[0, 1]) if evaluation.groups > 1 else None,
        "mean_turnover": float(returns["turnover"].mean()),
        # 顶层字段是候选审核、因子库和外部消费者的稳定展示契约；
        # 嵌套字段保留，确保既有运行工件仍可读取。
        **{f"long_short_{key}": value for key, value in net_stats.items()},
        **{f"long_short_cost_before_{key}": value for key, value in gross_stats.items()},
        **{f"long_short_cost_after_{key}": value for key, value in net_stats.items()},
        "cost_before": gross_stats,
        "cost_after": net_stats,
    }
    for key in ("ic_mean", "rank_ic_mean", "icir", "rank_icir", "monotonicity"):
        value = metrics.get(key)
        metrics[f"directional_{key}"] = value * direction if isinstance(value, (int, float)) else value
    return groups, cumulative, returns.reset_index(), metrics


def _single_asset_diagnostics(panel: pd.DataFrame, opportunity_count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valid = panel.dropna(subset=["factor", "forward_return"]).sort_values("timestamp")
    correlation = valid["factor"].corr(valid["forward_return"])
    rank = valid["factor"].corr(valid["forward_return"], method="spearman")
    valid["quantile"] = pd.qcut(valid["factor"].rank(method="first"), q=min(5, len(valid)), labels=False, duplicates="drop")
    quantiles = valid.groupby("quantile", as_index=False)["forward_return"].mean().rename(columns={"forward_return": "return"})
    coverage = float(len(valid) / opportunity_count) if opportunity_count else 0.0
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"metric_schema_version": METRIC_SCHEMA_VERSION, "evaluation_mode": "native_factor_single_asset", "cross_sectional_ic": "不适用", "group_returns": "不适用", "time_series_ic": float(correlation) if pd.notna(correlation) else None, "time_series_rank_ic": float(rank) if pd.notna(rank) else None, "factor_output_coverage": float(panel["factor"].notna().sum() / opportunity_count) if opportunity_count else 0.0, "evaluable_coverage": coverage, "factor_coverage": coverage, "quantile_returns": quantiles.to_dict("records")}


def _runtime_fingerprint(root: Path, script: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.run(args, cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    tracked = [
        "quantbacktest/api.py", "quantbacktest/native.py", "quantbacktest/adapters/market.py",
        "quantbacktest/library.py", "quantbacktest/web.py", "quantbacktest/artifacts.py",
    ]
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in tracked if (root / name).is_file()
    }
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "streamlit": __import__("streamlit").__version__,
        "git_commit": command("git", "rev-parse", "HEAD"),
        "workspace_dirty": bool(command("git", "status", "--porcelain")),
        "platform_file_hashes": hashes,
        "factor_script_hash": hashlib.sha256(script.read_bytes()).hexdigest(),
        "executable": sys.executable,
    }


def _universe_diagnostics(
    formations: pd.DatetimeIndex,
    factor: pd.DataFrame,
    panel: pd.DataFrame,
    symbols: list[str],
    rejections: dict[tuple[str, str], str],
    data_metadata: dict[str, Any],
    groups: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timestamp in formations:
        key = timestamp.isoformat()
        output_symbols = set(factor.loc[(factor["timestamp"] == timestamp) & factor["factor"].notna(), "symbol"])
        evaluable_symbols = set(panel.loc[(panel["timestamp"] == timestamp) & panel["factor"].notna() & panel["forward_return"].notna(), "symbol"])
        missing = [symbol for symbol in symbols if symbol not in output_symbols]
        symbol_diagnostics = data_metadata.get("symbol_diagnostics", {})
        rows.append({
            "timestamp": timestamp,
            "declared_assets": len(symbols),
            "factor_output_assets": len(output_symbols),
            "evaluable_assets": len(evaluable_symbols),
            "missing_asset_count": len(missing),
            "missing_assets": "|".join(missing),
            "missing_reasons": "|".join(f"{symbol}:{rejections.get((key, symbol), '无有效因子')}" for symbol in missing),
            "missing_bar_counts": "|".join(
                f"{symbol}:{symbol_diagnostics.get(symbol, {}).get('missing_bars')}" for symbol in missing
            ),
            "last_market_times": "|".join(
                f"{symbol}:{symbol_diagnostics.get(symbol, {}).get('last_market_time')}" for symbol in missing
            ),
            "period_status": "evaluated" if len(evaluable_symbols) >= groups else "skipped",
            "skip_reason": "" if len(evaluable_symbols) >= groups else "有效标的少于分组数",
        })
    return pd.DataFrame(rows)


def run_factor_script(
    path: Path,
    project_root: Path | None = None,
    candidate_library_root: Path | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> NativeRunResult:
    path = path.resolve()
    root = (project_root or path.parent).resolve()
    module, context = _initialise(path, "factor")
    script_data_declaration = context.manifest()["data"]
    data_range = _platform_range(context, start=start, end=end)
    data, data_meta = _load_declared_data(context, root, data_range)
    evaluation = context.factor_evaluation
    assert evaluation is not None and context.data_declaration is not None
    main = module.main
    outputs: list[pd.DataFrame] = []
    evaluation_start = pd.to_datetime(data_range["start"], utc=True)
    evaluation_end = pd.to_datetime(data_range["effective_end"], utc=True)
    formation_times = _formations(data, evaluation.formation)
    formation_times = formation_times[(formation_times >= evaluation_start) & (formation_times <= evaluation_end)]
    all_times = pd.DatetimeIndex(data["timestamp"].drop_duplicates().sort_values())
    positions = {timestamp: index for index, timestamp in enumerate(all_times)}
    evaluable_formations = pd.DatetimeIndex([
        timestamp for timestamp in formation_times
        if positions[timestamp] + evaluation.horizon_bars < len(all_times)
    ])
    for timestamp in formation_times:
        context.now, context.event = timestamp, "close"
        context._visible_data = data[data["timestamp"] <= timestamp].copy()
        try:
            output = main(context)
        except Exception as exc:
            raise NativeScriptError(f"main(context) 在 {timestamp.isoformat()} 失败：{type(exc).__name__}: {exc}") from exc
        outputs.append(_normalise_factor_output(output, timestamp))
    factor = pd.concat(outputs, ignore_index=True)
    if factor.empty or factor["factor"].notna().sum() == 0:
        raise NativeScriptError("整个样本期的因子输出为空或全为空值")
    panel = _entry_exit(data, factor, evaluation)
    opportunity_count = len(evaluable_formations) * len(context.data_declaration.symbols)
    is_single = len(context.data_declaration.symbols) == 1
    if is_single:
        groups, curves, long_short, metrics = _single_asset_diagnostics(panel, opportunity_count)
    else:
        groups, curves, long_short, metrics = _factor_evaluation(panel, evaluation, context.data_declaration.frequency, opportunity_count)
    valid_panel = panel.dropna(subset=["factor", "forward_return"])
    if not is_single:
        used_times = pd.to_datetime(long_short["timestamp"], utc=True)
        valid_panel = valid_panel[valid_panel["timestamp"].isin(used_times)]
    actual_evaluation_range = {
        "start": valid_panel["timestamp"].min().isoformat(),
        "end": valid_panel["return_end_time"].max().isoformat(),
    }
    universe = _universe_diagnostics(
        evaluable_formations,
        factor,
        panel,
        context.data_declaration.symbols,
        context._history_rejections,
        data_meta,
        evaluation.groups,
    )
    warnings: list[str] = []
    if len(context.data_declaration.symbols) < 10:
        warnings.append("研究样本过小：截面资产少于 10 个，IC 与分组结果可能高度离散。")
    if evaluation.fee_bps == 0 and evaluation.slippage_bps == 0:
        warnings.append("交易成本未启用：成本前后结果相同。")
    metrics.update({
        "platform_backtest_start": data_range["start"],
        "platform_backtest_end": data_range["end"],
        "market_data_start": data_meta["start"],
        "market_data_end": data_meta["end"],
        "actual_evaluation_start": actual_evaluation_range["start"],
        "actual_evaluation_end": actual_evaluation_range["end"],
        "warnings": warnings,
    })
    output_root = root / "results" / "backtests"
    run_dir = create_run_dir(output_root, context.name or path.stem)
    shutil.copy2(path, run_dir / "factor_snapshot.py")
    manifest = {
        "run_kind": "factor",
        **context.manifest(),
        "script_path": str(path),
        "script_data_declaration": script_data_declaration,
        "platform_backtest_range": data_range,
        "runtime_fingerprint": _runtime_fingerprint(root, path),
        "data": data_meta,
    }
    write_json(run_dir / "run_spec.json", manifest)
    write_json(
        run_dir / "metadata.json",
        {
            "run_kind": "factor",
            "data": data_meta,
            "script_data_declaration": script_data_declaration,
            "platform_backtest_range": data_range,
            "market_data_range": {"start": data_meta["start"], "end": data_meta["end"]},
            "actual_evaluation_range": actual_evaluation_range,
            "runtime_fingerprint": manifest["runtime_fingerprint"],
        },
    )
    factor.to_csv(run_dir / "factor_values.csv", index=False)
    panel.to_csv(run_dir / "factor_panel.csv", index=False)
    groups.to_csv(run_dir / "group_returns.csv", index=False)
    curves.to_csv(run_dir / "group_cumulative_returns.csv")
    long_short.to_csv(run_dir / "long_short_returns.csv", index=False)
    universe.to_csv(run_dir / "universe_diagnostics.csv", index=False)
    write_json(run_dir / "metrics.json", metrics)
    render_native_factor_report(run_dir / "report.html", groups, curves, long_short, metrics, manifest)
    candidate: dict[str, str] | None = None
    registration_error: str | None = None
    try:
        candidate = register_completed_run(run_dir, candidate_library_root)
    except (LibraryError, OSError, ValueError) as exc:
        registration_error = str(exc)
    return NativeRunResult(run_dir, metrics, warnings, "factor", candidate, registration_error)


def _strategy_visible_data(frame: pd.DataFrame, timestamp: pd.Timestamp, event: str) -> pd.DataFrame:
    visible = frame[frame["timestamp"] <= timestamp].copy()
    if event == "open":
        current = visible["timestamp"] == timestamp
        visible.loc[current, ["high", "low", "close", "volume", "turnover"]] = np.nan
    return visible


def _is_due(schedule: Any, timestamp: pd.Timestamp, bar_number: int, seen_days: set[tuple[str, str]]) -> bool:
    if schedule.frequency == "bars":
        return bar_number % schedule.every == 0
    key = (schedule.when, timestamp.date().isoformat())
    if schedule.frequency == "daily":
        if key in seen_days:
            return False
        seen_days.add(key)
        return True
    if timestamp.weekday() != schedule.weekday or key in seen_days:
        return False
    seen_days.add(key)
    return True


def run_strategy_script(
    path: Path,
    project_root: Path | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> NativeRunResult:
    path = path.resolve()
    root = (project_root or path.parent).resolve()
    _, context = _initialise(path, "strategy")
    script_data_declaration = context.manifest()["data"]
    data_range = _platform_range(context, start=start, end=end)
    validate_strategy_script(path, root, check_data=False, start=start, end=end)
    data, data_meta = _load_declared_data(context, root, data_range)
    declaration, account = context.data_declaration, context.account_declaration
    assert declaration is not None and account is not None
    prices = data.set_index(["timestamp", "symbol"])[["open", "close"]]
    order_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    def marked_value(timestamp: pd.Timestamp) -> float:
        value = context.portfolio.cash
        for symbol, position in context.portfolio.positions.items():
            close = prices.loc[(timestamp, symbol), "close"] if (timestamp, symbol) in prices.index else position.average_cost
            position.market_value = position.quantity * float(close)
            value += position.market_value
        context.portfolio.total_value = float(value)
        return float(value)

    def submit(timestamp: pd.Timestamp, event: str, symbol: str, requested: float, kind: str) -> Order:
        def record(order: Order) -> Order:
            order_rows.append(
                {
                    "timestamp": timestamp,
                    "event": event,
                    "symbol": symbol,
                    "requested_value": requested,
                    "status": order.status,
                    "reason": order.reason,
                    "quantity": order.quantity,
                    "price": order.price,
                    "fee": order.fee,
                }
            )
            return order

        if symbol not in declaration.symbols:
            return record(Order(timestamp.to_pydatetime(), symbol, requested, "rejected", "交易对不在 set_data 声明的资产池中"))
        field = "open" if event == "open" else "close"
        price = float(prices.loc[(timestamp, symbol), field])
        if not np.isfinite(price):
            return record(Order(timestamp.to_pydatetime(), symbol, requested, "rejected", f"当前 {field} 价格不可用"))
        position = context.portfolio.positions.get(symbol, Position(symbol, 0.0, 0.0, 0.0))
        if kind == "target_quantity":
            desired_quantity = requested
        else:
            desired_value = requested if kind == "target_value" else position.quantity * price + requested
            desired_quantity = desired_value / price
        delta = desired_quantity - position.quantity
        if declaration.market == "spot" and desired_quantity < -1e-12:
            return record(Order(timestamp.to_pydatetime(), symbol, requested, "rejected", "现货账户不允许做空"))
        fill_price = price * (1 + (1 if delta > 0 else -1) * account.slippage_bps / 10_000)
        gross = abs(delta) * fill_price
        fee = gross * account.fee_bps / 10_000
        if delta > 0 and gross + fee > context.portfolio.cash + 1e-9:
            return record(Order(timestamp.to_pydatetime(), symbol, requested, "rejected", "可用现金不足；平台不会自动缩放或透支"))
        if declaration.market == "spot" and delta < 0 and abs(delta) > position.quantity + 1e-9:
            return record(Order(timestamp.to_pydatetime(), symbol, requested, "rejected", "卖出数量超过现货持仓"))
        context.portfolio.cash -= delta * fill_price + fee
        new_quantity = position.quantity + delta
        if abs(new_quantity) < 1e-12:
            context.portfolio.positions.pop(symbol, None)
        else:
            average_cost = fill_price if delta > 0 and position.quantity <= 0 else position.average_cost
            context.portfolio.positions[symbol] = Position(symbol, new_quantity, average_cost, new_quantity * price)
        order = Order(timestamp.to_pydatetime(), symbol, requested, "filled", quantity=delta, price=fill_price, fee=fee)
        return record(order)

    timestamps = pd.DatetimeIndex(data["timestamp"].drop_duplicates().sort_values())
    timestamps = timestamps[
        (timestamps >= pd.to_datetime(data_range["start"], utc=True))
        & (timestamps <= pd.to_datetime(data_range["effective_end"], utc=True))
    ]
    daily_seen: dict[int, set[tuple[str, str]]] = {index: set() for index in range(len(context.schedules))}
    for bar_number, timestamp in enumerate(timestamps):
        for event in ("open", "close"):
            context.now, context.event = timestamp, event
            context._visible_data = _strategy_visible_data(data, timestamp, event)
            context._order_handler = lambda symbol, value, kind, ts=timestamp, ev=event: submit(ts, ev, symbol, value, kind)
            marked_value(timestamp)
            for index, schedule in enumerate(context.schedules):
                if schedule.when == event and _is_due(schedule, timestamp, bar_number, daily_seen[index]):
                    token = _ACTIVE_CONTEXT.set(context)
                    try:
                        schedule.callback(context)
                    except Exception as exc:
                        raise NativeScriptError(f"策略回调 {schedule.callback.__name__} 在 {timestamp.isoformat()} 失败：{type(exc).__name__}: {exc}") from exc
                    finally:
                        _ACTIVE_CONTEXT.reset(token)
            equity = marked_value(timestamp)
            equity_rows.append({"timestamp": timestamp, "event": event, "equity": equity, "cash": context.portfolio.cash})
    equity_frame = pd.DataFrame(equity_rows)
    close_equity = equity_frame[equity_frame["event"] == "close"].drop_duplicates("timestamp", keep="last").set_index("timestamp")
    returns = close_equity["equity"].pct_change().fillna(0.0).rename("strategy_return")
    if account.benchmark == "equal_weight_universe":
        benchmark = data.sort_values(["symbol", "timestamp"]).groupby("symbol")["close"].pct_change().groupby(data.sort_values(["symbol", "timestamp"])["timestamp"]).mean().reindex(returns.index).fillna(0.0)
    else:
        benchmark = data[data["symbol"] == account.benchmark].set_index("timestamp")["close"].pct_change().reindex(returns.index).fillna(0.0)
    metrics = performance_metrics(returns, declaration.frequency)
    metrics.update({"run_kind": "strategy", "orders": len(context.orders()), "filled_orders": sum(order.status == "filled" for order in context.orders()), "rejected_orders": sum(order.status == "rejected" for order in context.orders()), "total_cost": float(sum(order.fee for order in context.orders())), "actual_data_start": data_meta["start"], "actual_data_end": data_meta["end"]})
    output_root = root / "results" / "backtests"
    run_dir = create_run_dir(output_root, context.name or path.stem)
    shutil.copy2(path, run_dir / "strategy_snapshot.py")
    manifest = {
        "run_kind": "strategy",
        **context.manifest(),
        "script_path": str(path),
        "script_data_declaration": script_data_declaration,
        "platform_backtest_range": data_range,
        "runtime_fingerprint": _runtime_fingerprint(root, path),
        "data": data_meta,
    }
    write_json(run_dir / "run_spec.json", manifest)
    write_json(
        run_dir / "metadata.json",
        {
            "run_kind": "strategy",
            "data": data_meta,
            "script_data_declaration": script_data_declaration,
            "platform_backtest_range": data_range,
            "market_data_range": {"start": data_meta["start"], "end": data_meta["end"]},
            "actual_evaluation_range": {"start": timestamps.min().isoformat(), "end": timestamps.max().isoformat()},
            "runtime_fingerprint": manifest["runtime_fingerprint"],
        },
    )
    returns.to_csv(run_dir / "returns.csv", header=True)
    pd.DataFrame(order_rows, columns=["timestamp", "event", "symbol", "requested_value", "status", "reason", "quantity", "price", "fee"]).to_csv(run_dir / "orders.csv", index=False)
    pd.DataFrame([{"timestamp": timestamp, "symbol": symbol, "quantity": position.quantity, "market_value": position.market_value} for timestamp in [timestamps[-1]] for symbol, position in context.portfolio.positions.items()]).to_csv(run_dir / "positions.csv", index=False)
    write_json(run_dir / "metrics.json", metrics)
    render_native_strategy_report(run_dir / "report.html", returns, benchmark, equity_frame, pd.DataFrame(order_rows), metrics, manifest)
    return NativeRunResult(run_dir, metrics, [], "strategy")
