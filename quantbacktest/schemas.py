from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class UniverseSpec(StrictModel):
    min_assets: int = Field(default=10, ge=2)
    min_history_bars: int = Field(default=0, ge=0)
    liquidity_rule: dict[str, Any] | None = None


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
    mode: StrategyMode
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    rebalance_bars: int = Field(default=1, ge=1)
    selection: Literal["quantiles", "top_k"] | None = None
    long_short: Literal["long_only", "market_neutral"] | None = None
    quantiles: int | None = Field(default=None, ge=2)
    top_k: int | None = Field(default=None, ge=1)
    weighting: Literal["equal", "score"] = "equal"
    max_weight: float = Field(default=1.0, gt=0, le=1.0)
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
        return self


class CostSpec(StrictModel):
    fee_bps: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=0.0, ge=0)
    funding_bps_per_bar: float | None = None


class BenchmarkSpec(StrictModel):
    type: Literal["equal_weight_universe"] = "equal_weight_universe"


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


class OutputSpec(StrictModel):
    root: Path = Path("results/backtests")


class ProvenanceSpec(StrictModel):
    paper_title: str | None = None
    paper_reference: str | None = None
    notes: str | None = None
    proxy_data_note: str | None = None


class RunSpec(StrictModel):
    name: str = Field(min_length=1)
    data: DataSpec
    factor: FactorSpec
    strategy: StrategySpec
    costs: CostSpec = Field(default_factory=CostSpec)
    benchmark: BenchmarkSpec = Field(default_factory=BenchmarkSpec)
    ml: MLSpec = Field(default_factory=MLSpec)
    debug: DebugSpec = Field(default_factory=DebugSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    provenance: ProvenanceSpec = Field(default_factory=ProvenanceSpec)

    @model_validator(mode="after")
    def validate_cross_section(self) -> RunSpec:
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
