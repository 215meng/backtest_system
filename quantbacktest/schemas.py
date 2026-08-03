from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Market(str, Enum):
    spot = "spot"
    linear_perp = "linear_perp"


class StrategyMode(str, Enum):
    cross_sectional = "cross_sectional"
    single_asset = "single_asset"


class DebugMode(str, Enum):
    off = "off"
    dry_run = "dry_run"
    trace = "trace"
    replay = "replay"


class EvaluationMode(str, Enum):
    factor_research = "factor_research"
    strategy_simulation = "strategy_simulation"


class UniverseSpec(StrictModel):
    min_assets: int = Field(default=10, ge=2)
    min_history_bars: int = Field(default=0, ge=0)
    liquidity_rule: dict[str, Any] | None = None
    membership_path: Path | None = None


class DataSpec(StrictModel):
    adapter: Literal["crypto_top50", "bybit_parquet", "binance_zip"]
    path: Path = Field(description="适配器数据根目录，可使用相对调用项目的路径")
    market: Market
    frequency: str = Field(pattern=r"^(1m|15m|1h|4h|1d)$")
    symbols: list[str] = Field(min_length=1)
    start: datetime | None = None
    end: datetime | None = None
    universe: UniverseSpec | None = None


class FactorSpec(StrictModel):
    module_path: Path
    callable: str = "compute_factor"
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExecutionSpec(StrictModel):
    signal_price: Literal["close"] = "close"
    entry_price: Literal["next_open"] = "next_open"
    holding_bars: int = Field(default=1, ge=1)


class SignalRule(StrictModel):
    kind: Literal["threshold", "sign"] = "threshold"
    long_above: float = 0.0
    short_below: float | None = None
    position_size: float = Field(default=1.0, gt=0, le=1.0)


class MaxDrawdownStopSpec(StrictModel):
    enabled: bool = False
    threshold: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_threshold(self) -> MaxDrawdownStopSpec:
        if self.enabled and self.threshold is None:
            raise ValueError("启用组合最大回撤止损时必须提供 threshold")
        return self


class RiskControlSpec(StrictModel):
    max_drawdown_stop: MaxDrawdownStopSpec = Field(default_factory=MaxDrawdownStopSpec)


class StrategySpec(StrictModel):
    profile: Literal["factor_mimicking"] | None = None
    mode: StrategyMode
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    rebalance_bars: int = Field(default=1, ge=1)
    selection: Literal["quantiles", "top_k"] | None = None
    long_short: Literal["long_only", "market_neutral"] | None = None
    quantiles: int | None = Field(default=None, ge=2)
    top_k: int | None = Field(default=None, ge=1)
    weighting: Literal["equal", "score"] = "equal"
    max_weight: float = Field(default=1.0, gt=0, le=1.0)
    gross_exposure: float = Field(default=2.0, gt=0, le=2.0)
    symbol: str | None = None
    signal_rule: SignalRule | None = None
    hook_callable: str | None = None
    risk_control: RiskControlSpec = Field(default_factory=RiskControlSpec)

    @model_validator(mode="after")
    def validate_mode(self) -> StrategySpec:
        if self.mode is StrategyMode.cross_sectional:
            if self.long_short is None or self.selection is None:
                raise ValueError("截面模式必须提供 selection 与 long_short")
            if self.selection == "quantiles" and self.quantiles is None:
                raise ValueError("selection=quantiles 时必须提供 quantiles")
            if self.selection == "top_k" and self.top_k is None:
                raise ValueError("selection=top_k 时必须提供 top_k")
        else:
            if self.symbol is None or self.signal_rule is None:
                raise ValueError("单资产模式必须提供 symbol 与 signal_rule")
        if self.profile == "factor_mimicking":
            if self.mode is not StrategyMode.cross_sectional:
                raise ValueError("factor_mimicking only supports cross-sectional strategies")
            if self.selection != "quantiles" or self.quantiles != 3:
                raise ValueError("factor_mimicking requires tercile selection")
            if self.long_short != "market_neutral" or self.weighting != "equal":
                raise ValueError("factor_mimicking requires equal-weight market neutrality")
            if self.gross_exposure != 1.0:
                raise ValueError("factor_mimicking requires gross_exposure: 1.0")
            if self.rebalance_bars != self.execution.holding_bars:
                raise ValueError("factor_mimicking requires rebalance_bars to equal holding_bars")
            if self.risk_control.max_drawdown_stop.enabled:
                raise ValueError("factor_mimicking disables stop-losses to isolate factor performance")
        return self


class FormationScheduleSpec(StrictModel):
    kind: Literal["calendar", "bar_interval"]
    interval: Literal["1h", "4h", "1d", "1w"] | None = None
    weekday: int | None = Field(default=None, ge=0, le=6)
    time_utc: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    every_n_bars: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_schedule(self) -> FormationScheduleSpec:
        if self.kind == "calendar":
            if self.interval is None or self.time_utc is None:
                raise ValueError("日历形成时间表必须提供 interval 和 time_utc")
            if self.interval == "1w" and self.weekday is None:
                raise ValueError("周频日历形成时间表必须提供 weekday（0=周一，6=周日）")
            if self.every_n_bars is not None:
                raise ValueError("calendar 时间表不能提供 every_n_bars")
        elif self.every_n_bars is None:
            raise ValueError("bar_interval 时间表必须提供 every_n_bars")
        return self


class ResearchReturnSpec(StrictModel):
    horizon: Literal["1h", "4h", "1d", "1w"]
    start_price: Literal["close", "next_open"]
    end_price: Literal["close", "open"]


class ResearchPortfolioSpec(StrictModel):
    selection: Literal["quantiles", "top_k"]
    quantiles: int | None = Field(default=None, ge=2)
    top_k: int | None = Field(default=None, ge=1)
    weighting: Literal["equal", "score", "market_cap"] = "equal"

    @model_validator(mode="after")
    def validate_selection(self) -> ResearchPortfolioSpec:
        if self.selection == "quantiles" and self.quantiles is None:
            raise ValueError("研究分组 selection=quantiles 时必须提供 quantiles")
        if self.selection == "top_k" and self.top_k is None:
            raise ValueError("研究分组 selection=top_k 时必须提供 top_k")
        return self


class FactorResearchSpec(StrictModel):
    formation: FormationScheduleSpec
    returns: ResearchReturnSpec
    direction: Literal["higher_predicts_higher_return", "higher_predicts_lower_return"]
    portfolio: ResearchPortfolioSpec
    ic_decay_horizons: list[Literal["1h", "4h", "1d", "1w"]] = Field(default_factory=list)


class EvaluationSpec(StrictModel):
    mode: EvaluationMode
    research: FactorResearchSpec | None = None

    @model_validator(mode="after")
    def validate_research(self) -> EvaluationSpec:
        if self.mode is EvaluationMode.factor_research and self.research is None:
            raise ValueError("factor_research 模式必须提供 evaluation.research")
        if self.mode is EvaluationMode.strategy_simulation and self.research is not None:
            raise ValueError("strategy_simulation 模式不能提供 evaluation.research")
        return self


class CostSpec(StrictModel):
    fee_bps: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=0.0, ge=0)
    funding_bps_per_bar: float | None = None


class BenchmarkSpec(StrictModel):
    type: Literal["equal_weight_universe"] = "equal_weight_universe"


class ComputeSpec(StrictModel):
    backend: Literal["cuda", "cpu"] = "cuda"
    device_id: int = Field(default=0, ge=0)
    on_unavailable: Literal["error", "cpu_explicit"] = "error"
    profile: bool = True


class MLSpec(StrictModel):
    enabled: bool = False
    model: Literal["lightgbm", "xgboost"] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_model(self) -> MLSpec:
        if self.enabled and self.model is None:
            raise ValueError("启用 ML 时必须提供 model")
        return self


class DebugSpec(StrictModel):
    mode: DebugMode = DebugMode.off
    timestamp_range: tuple[datetime, datetime] | None = None
    symbols: list[str] | None = None
    stages: list[Literal["data", "factor", "signal", "portfolio", "execution"]] = Field(
        default_factory=lambda: ["data", "factor", "signal", "portfolio", "execution"]
    )

    @field_validator("mode", mode="before")
    @classmethod
    def reject_boolean_mode(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004 - Pydantic must convert this into a ValidationError.
                'debug.mode 必须是字符串；YAML 请写 mode: "off"，MCP JSON 请写 '
                '{"mode": "off"}。如无需调试，可省略整个 debug 块。'
            )
        return value


class OutputSpec(StrictModel):
    root: Path = Path("results/backtests")


class ProvenanceSpec(StrictModel):
    paper_title: str | None = None
    paper_reference: str | None = None
    notes: str | None = None
    proxy_data_note: str | None = None


class PaperBacktestRequest(StrictModel):
    """外部论文复现提交给 MCP 的能力评估声明。"""

    paper_title: str = Field(min_length=1)
    paper_reference: str | None = None
    data_source: str | None = None
    adapter: str = Field(min_length=1)
    market: str = Field(min_length=1)
    frequency: str = Field(min_length=1)
    asset_universe_note: str | None = None
    factor_inputs: list[str] = Field(default_factory=list)
    universe_features: list[str] = Field(default_factory=list)
    formation_features: list[str] = Field(default_factory=list)
    portfolio_features: list[str] = Field(default_factory=list)
    execution_features: list[str] = Field(default_factory=list)
    research_features: list[str] = Field(default_factory=list)
    run_kind: Literal["minimal_factor_strategy"] = "minimal_factor_strategy"


class RunSpec(StrictModel):
    name: str = Field(min_length=1)
    data: DataSpec
    factor: FactorSpec
    strategy: StrategySpec | None = None
    evaluation: EvaluationSpec | None = None
    costs: CostSpec = Field(default_factory=CostSpec)
    benchmark: BenchmarkSpec = Field(default_factory=BenchmarkSpec)
    compute: ComputeSpec | None = None
    ml: MLSpec = Field(default_factory=MLSpec)
    debug: DebugSpec = Field(default_factory=DebugSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    provenance: ProvenanceSpec = Field(default_factory=ProvenanceSpec)

    @model_validator(mode="after")
    def validate_cross_section(self) -> RunSpec:
        is_research = self.evaluation is not None and self.evaluation.mode is EvaluationMode.factor_research
        if not is_research and self.strategy is None:
            raise ValueError("strategy_simulation 或旧配置必须提供 strategy")
        if is_research:
            return self
        assert self.strategy is not None
        if self.strategy.profile == "factor_mimicking":
            if self.costs.fee_bps != 0 or self.costs.slippage_bps != 0 or self.costs.funding_bps_per_bar not in (None, 0):
                raise ValueError("factor_mimicking requires zero fees, slippage, and funding")
            if self.ml.enabled:
                raise ValueError("factor_mimicking disables ML to isolate factor performance")
        if self.strategy.mode is StrategyMode.cross_sectional:
            universe = self.data.universe or UniverseSpec()
            if len(self.data.symbols) < universe.min_assets:
                raise ValueError(
                    f"截面模式至少需要 {universe.min_assets} 个资产，当前只提供 {len(self.data.symbols)} 个"
                )
        if self.strategy.mode is StrategyMode.single_asset and self.strategy.symbol not in self.data.symbols:
            raise ValueError("单资产 strategy.symbol 必须存在于 data.symbols")
        if self.debug.mode is DebugMode.replay and self.debug.timestamp_range is None:
            raise ValueError("replay 调试模式必须提供 timestamp_range")
        return self
