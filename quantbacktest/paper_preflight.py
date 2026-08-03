from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from quantbacktest.adapters import load_market_data
from quantbacktest.config_io import load_yaml_file
from quantbacktest.factors import FactorContext, execute_factor, inspect_factor
from quantbacktest.schemas import EvaluationMode, PaperBacktestRequest, RunSpec, StrictModel

CAPABILITIES = {
    "adapters": {"crypto_top50", "bybit_parquet", "binance_zip"},
    "markets": {"spot", "linear_perp"},
    "frequencies": {"1m", "15m", "1h", "4h", "1d"},
    "factor_inputs": {"timestamp", "symbol", "open", "high", "low", "close", "volume", "turnover"},
    "universe": {"static_symbols", "membership_path", "min_history_bars", "liquidity_rule"},
    "formation": {"calendar", "bar_interval"},
    "portfolio": {"quantiles", "top_k", "equal", "score", "market_neutral", "long_only", "max_weight", "factor_mimicking"},
    "execution": {"close_signal", "next_open", "holding_bars", "fee_slippage", "drawdown_stop"},
    "research": {"factor_research", "quantile_returns", "rank_ic", "ic_decay", "strategy_simulation", "minimal_factor_strategy"},
}


class PaperAssessment(StrictModel):
    decision: str
    assessment_id: str | None = None
    paper_title: str
    supported_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    blockers: list[dict[str, str]] = Field(default_factory=list)


@dataclass(frozen=True)
class _AssessmentRecord:
    request_fingerprint: str
    request: PaperBacktestRequest


@dataclass(frozen=True)
class _PreflightRecord:
    assessment_id: str
    project_root: Path
    config_path: Path
    factor_path: Path
    config_fingerprint: str
    factor_fingerprint: str
    spec: RunSpec


_ASSESSMENTS: dict[str, _AssessmentRecord] = {}
_PREFLIGHTS: dict[str, _PreflightRecord] = {}


def _fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unsupported(category: str, values: list[str]) -> list[dict[str, str]]:
    supported = CAPABILITIES[category]
    return [
        {
            "category": category,
            "paper_requirement": value,
            "platform_status": "不支持",
            "blocker": f"QuantBacktest 当前不支持 {category}={value}",
            "required_capability": value,
        }
        for value in values
        if value not in supported
    ]


def assess_paper_request(request: PaperBacktestRequest) -> PaperAssessment:
    """将论文声明与当前平台能力清单比较；任何差异都会阻断后续流程。"""
    blockers: list[dict[str, str]] = []
    if request.adapter not in CAPABILITIES["adapters"]:
        blockers.extend(_unsupported("adapters", [request.adapter]))
    if request.market not in CAPABILITIES["markets"]:
        blockers.extend(_unsupported("markets", [request.market]))
    if request.frequency not in CAPABILITIES["frequencies"]:
        blockers.extend(_unsupported("frequencies", [request.frequency]))
    for category, values in (
        ("factor_inputs", request.factor_inputs),
        ("universe", request.universe_features),
        ("formation", request.formation_features),
        ("portfolio", request.portfolio_features),
        ("execution", request.execution_features),
        ("research", request.research_features),
    ):
        blockers.extend(_unsupported(category, values))
    blockers.extend(_unsupported("research", [request.run_kind]))
    if blockers:
        return PaperAssessment(
            decision="blocked",
            paper_title=request.paper_title,
            blockers=blockers,
        )

    request_json = request.model_dump_json()
    assessment_id = secrets.token_urlsafe(24)
    _ASSESSMENTS[assessment_id] = _AssessmentRecord(
        request_fingerprint=_fingerprint(request_json.encode("utf-8")), request=request
    )
    supported_capabilities = {
        "adapter": [request.adapter],
        "market": [request.market],
        "frequency": [request.frequency],
        "factor_inputs": request.factor_inputs,
        "universe": request.universe_features,
        "formation": request.formation_features,
        "portfolio": request.portfolio_features,
        "execution": request.execution_features,
        "research": request.research_features,
        "run_kind": [request.run_kind],
    }
    return PaperAssessment(
        decision="ready",
        assessment_id=assessment_id,
        paper_title=request.paper_title,
        supported_capabilities=supported_capabilities,
    )


def _project_file(project_root: Path, candidate: str | Path, label: str) -> Path:
    root = project_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"project_root 必须是存在的目录：{root}")
    path = Path(candidate).expanduser()
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} 必须位于 project_root 内：{resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"未找到{label}：{resolved}")
    return resolved


def _validate_assessment_compatibility(spec: RunSpec, meta: dict[str, Any], request: PaperBacktestRequest) -> None:
    errors: list[str] = []
    if spec.data.adapter != request.adapter:
        errors.append("data.adapter 与通过评估的论文声明不一致")
    if spec.data.market.value != request.market:
        errors.append("data.market 与通过评估的论文声明不一致")
    if spec.data.frequency != request.frequency:
        errors.append("data.frequency 与通过评估的论文声明不一致")
    missing_inputs = set(meta["required_fields"]) - set(request.factor_inputs)
    if missing_inputs:
        errors.append(f"FactorMeta.required_fields 未包含在论文声明中：{sorted(missing_inputs)}")
    strategy = spec.strategy
    if strategy is None or strategy.profile != "factor_mimicking":
        errors.append("external paper runs require strategy.profile: factor_mimicking")
    elif spec.evaluation is None or spec.evaluation.mode.value != "strategy_simulation":
        errors.append("factor_mimicking requires evaluation.mode: strategy_simulation")
    if strategy is not None:
        expected = {strategy.selection, strategy.weighting, strategy.long_short}
        if not expected - set(request.portfolio_features):
            pass
        else:
            errors.append("strategy 的选币、权重或多空规则未包含在论文声明中")
        if strategy.execution.entry_price not in request.execution_features:
            errors.append("strategy.execution.entry_price 未包含在论文声明中")
    if spec.evaluation is not None and spec.evaluation.mode.value not in request.research_features:
        errors.append("evaluation.mode 未包含在论文声明中")
    required_mode = "cross_sectional" if (
        spec.evaluation is not None and spec.evaluation.mode is EvaluationMode.factor_research
    ) else strategy.mode.value if strategy is not None else ""
    if required_mode not in meta["supported_modes"]:
        errors.append(f"因子不支持当前所需模式 {required_mode}")
    if errors:
        raise ValueError("预检阻断：" + "；".join(errors))


def preflight_external_project(
    *, config_path: str | Path, factor_path: str | Path, project_root: str | Path, assessment_id: str
) -> dict[str, Any]:
    """验证真实的外部项目文件，并绑定其哈希与已经通过的论文能力评估。"""
    assessment = _ASSESSMENTS.get(assessment_id)
    if assessment is None:
        raise ValueError("assessment_id 无效、已过期或对应论文需求未通过评估")
    root = Path(project_root)
    config = _project_file(root, config_path, "YAML 配置")
    factor = _project_file(root, factor_path, "因子脚本")
    payload = load_yaml_file(config)
    payload.setdefault("factor", {})
    if not isinstance(payload["factor"], dict):
        raise TypeError("YAML 的 factor 必须是对象")
    payload["factor"]["module_path"] = str(factor)
    spec = RunSpec.model_validate(payload)
    meta = inspect_factor(factor)
    _validate_assessment_compatibility(spec, meta, assessment.request)
    data_spec = spec.data.model_copy(deep=True)
    if not data_spec.path.is_absolute():
        data_spec.path = (root / data_spec.path).resolve()
    if (
        data_spec.universe
        and data_spec.universe.membership_path
        and not data_spec.universe.membership_path.is_absolute()
    ):
        data_spec.universe.membership_path = (root / data_spec.universe.membership_path).resolve()
    data, _ = load_market_data(data_spec)
    factor_output, _ = execute_factor(
        factor, spec.factor.callable, FactorContext(data=data, parameters=spec.factor.parameters)
    )
    preflight_id = secrets.token_urlsafe(24)
    _PREFLIGHTS[preflight_id] = _PreflightRecord(
        assessment_id=assessment_id,
        project_root=root.expanduser().resolve(),
        config_path=config,
        factor_path=factor,
        config_fingerprint=_fingerprint(config.read_bytes()),
        factor_fingerprint=_fingerprint(factor.read_bytes()),
        spec=spec,
    )
    return {
        "status": "ready",
        "preflight_id": preflight_id,
        "normalized_spec": spec.model_dump(mode="json"),
        "factor_meta": meta,
        "factor_output_rows": len(factor_output),
        "config_fingerprint": _PREFLIGHTS[preflight_id].config_fingerprint,
        "factor_fingerprint": _PREFLIGHTS[preflight_id].factor_fingerprint,
    }


def consume_preflight(preflight_id: str) -> tuple[RunSpec, Path]:
    """在执行前复核文件指纹；预检后任何文件变动都必须重新预检。"""
    record = _PREFLIGHTS.get(preflight_id)
    if record is None:
        raise ValueError("preflight_id 无效、已过期或尚未通过外部项目预检")
    if _fingerprint(record.config_path.read_bytes()) != record.config_fingerprint:
        raise ValueError("YAML 配置在预检后已修改；请重新预检")
    if _fingerprint(record.factor_path.read_bytes()) != record.factor_fingerprint:
        raise ValueError("因子脚本在预检后已修改；请重新预检")
    return record.spec, record.project_root


def external_project_contract() -> dict[str, Any]:
    """给 MCP 客户端的紧凑、可执行调用契约。"""
    return {
        "required_order": [
            "assess_paper_backtest_request",
            "write_factor_and_yaml_only_when_ready",
            "preflight_external_project",
            "run_backtest_tool",
        ],
        "factor_contract": {
            "required_meta": ["required_fields", "min_lookback_bars", "supported_modes"],
            "required_output": ["timestamp", "symbol", "factor"],
            "data_source": "只使用 context.data 和 context.parameters，禁止未来数据或自行读取数据文件",
        },
        "yaml_rules": {
            "quote_strings": True,
            "debug_modes": ["off", "dry_run", "trace", "replay"],
            "debug_example": 'debug:\n  mode: "off"',
            "mcp_json_example": {"debug": {"mode": "off"}},
            "minimal_factor_strategy": {
                "required": True,
                "profile": "factor_mimicking",
                "fixed_rules": [
                    "terciles, equal weighting, market neutral",
                    "gross_exposure: 1.0 (long +0.5, short -0.5)",
                    "rebalance_bars must equal holding_bars",
                    "zero fees, slippage, and funding; no stop-loss or ML",
                ],
            },
        },
        "capabilities": {key: sorted(value) for key, value in CAPABILITIES.items()},
    }
