"""QuantBacktest 原生 Python 脚本 API。

该模块是因子和策略脚本唯一需要导入的公开接口。配置直接写在脚本的
``initialize(context)`` 中；运行时不会读取 YAML，也不会为用户补充选币、
调仓或风险控制规则。
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd


class ScriptContractError(ValueError):
    """用户脚本不符合公开 API 契约时抛出。"""


@dataclass
class DataDeclaration:
    adapter: Literal["crypto_top50", "binance_zip", "bybit_parquet"]
    path: str
    market: Literal["spot", "linear_perp"]
    frequency: Literal["1m", "15m", "1h", "4h", "1d"]
    symbols: list[str]
    start: str | None = None
    end: str | None = None


@dataclass
class FactorEvaluation:
    formation: Literal["bar", "daily", "weekly"]
    horizon_bars: int
    entry_price: Literal["close", "next_open"]
    exit_price: Literal["close", "open"]
    direction: Literal["higher_predicts_higher_return", "higher_predicts_lower_return"]
    groups: int
    weighting: Literal["equal", "factor_value"]
    fee_bps: float
    slippage_bps: float


@dataclass
class AccountDeclaration:
    initial_cash: float
    benchmark: str
    fee_bps: float
    slippage_bps: float
    leverage: float | None = None
    margin_mode: Literal["isolated"] | None = None
    funding_bps_per_bar: float | None = None


@dataclass
class Schedule:
    callback: Callable[[ScriptContext], None]
    frequency: Literal["daily", "weekly", "bars"]
    when: Literal["open", "close"]
    every: int = 1
    weekday: int = 0


@dataclass
class Position:
    symbol: str
    quantity: float
    average_cost: float
    market_value: float


@dataclass
class Order:
    timestamp: datetime
    symbol: str
    requested_value: float
    status: Literal["filled", "rejected"]
    reason: str | None = None
    quantity: float = 0.0
    price: float | None = None
    fee: float = 0.0


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0


_ACTIVE_CONTEXT: ContextVar[ScriptContext | None] = ContextVar("quantbacktest_context", default=None)


class ScriptContext:
    """传给用户脚本的上下文。

    ``history`` 和 ``get_bars`` 永远只返回当前事件时点可见的数据。开盘事件
    的当前 bar 仅暴露 ``open``，避免读取尚未形成的 high/low/close。
    """

    def __init__(self, kind: Literal["factor", "strategy"]) -> None:
        self.kind = kind
        self.name: str | None = None
        self.data_declaration: DataDeclaration | None = None
        self.factor_evaluation: FactorEvaluation | None = None
        self.account_declaration: AccountDeclaration | None = None
        self.schedules: list[Schedule] = []
        self.now: pd.Timestamp | None = None
        self.event: Literal["open", "close"] = "close"
        self._visible_data = pd.DataFrame()
        self._orders: list[Order] = []
        self.portfolio = Portfolio(cash=0.0)
        self._order_handler: Callable[[str, float, str], Order] | None = None

    def set_name(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ScriptContractError("set_name(name) 的 name 必须是非空字符串")
        self.name = name.strip()

    @property
    def symbols(self) -> list[str]:
        """脚本当前声明的交易对副本。"""
        return list(self.data_declaration.symbols) if self.data_declaration else []

    def set_data(
        self,
        *,
        adapter: Literal["crypto_top50", "binance_zip", "bybit_parquet"],
        path: str,
        market: Literal["spot", "linear_perp"],
        frequency: Literal["1m", "15m", "1h", "4h", "1d"],
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        if not isinstance(symbols, list) or not symbols or not all(isinstance(item, str) and item for item in symbols):
            raise ScriptContractError("set_data(..., symbols=...) 必须提供非空交易对列表")
        self.data_declaration = DataDeclaration(adapter, path, market, frequency, symbols, start, end)

    def set_factor_evaluation(
        self,
        *,
        formation: Literal["bar", "daily", "weekly"],
        horizon_bars: int,
        entry_price: Literal["close", "next_open"],
        exit_price: Literal["close", "open"],
        direction: Literal["higher_predicts_higher_return", "higher_predicts_lower_return"],
        groups: int,
        weighting: Literal["equal", "factor_value"],
        fee_bps: float,
        slippage_bps: float,
    ) -> None:
        if self.kind != "factor":
            raise ScriptContractError("set_factor_evaluation 只能在因子脚本中调用")
        if horizon_bars < 1 or groups < 2 or fee_bps < 0 or slippage_bps < 0:
            raise ScriptContractError("因子评价的 horizon_bars/groups/成本参数无效")
        self.factor_evaluation = FactorEvaluation(
            formation, horizon_bars, entry_price, exit_price, direction, groups, weighting, fee_bps, slippage_bps
        )

    def set_account(
        self,
        *,
        initial_cash: float,
        benchmark: str,
        fee_bps: float,
        slippage_bps: float,
        leverage: float | None = None,
        margin_mode: Literal["isolated"] | None = None,
        funding_bps_per_bar: float | None = None,
    ) -> None:
        if self.kind != "strategy":
            raise ScriptContractError("set_account 只能在策略脚本中调用")
        if initial_cash <= 0 or fee_bps < 0 or slippage_bps < 0:
            raise ScriptContractError("账户本金和成本参数无效")
        self.account_declaration = AccountDeclaration(
            initial_cash, benchmark, fee_bps, slippage_bps, leverage, margin_mode, funding_bps_per_bar
        )
        self.portfolio.cash = float(initial_cash)
        self.portfolio.total_value = float(initial_cash)

    def run_daily(self, callback: Callable[[ScriptContext], None], *, when: Literal["open", "close"] = "close") -> None:
        self.schedules.append(Schedule(callback, "daily", when))

    def run_weekly(
        self,
        callback: Callable[[ScriptContext], None],
        *,
        weekday: int = 0,
        when: Literal["open", "close"] = "close",
    ) -> None:
        if weekday not in range(7):
            raise ScriptContractError("run_weekly 的 weekday 必须为 0 到 6")
        self.schedules.append(Schedule(callback, "weekly", when, weekday=weekday))

    def run_every_bars(
        self,
        callback: Callable[[ScriptContext], None],
        *,
        every: int = 1,
        when: Literal["open", "close"] = "close",
    ) -> None:
        if every < 1:
            raise ScriptContractError("run_every_bars 的 every 必须不小于 1")
        self.schedules.append(Schedule(callback, "bars", when, every=every))

    def history(self, symbol: str, bars: int, fields: list[str] | None = None) -> pd.DataFrame:
        if bars < 1:
            raise ScriptContractError("history 的 bars 必须不小于 1")
        result = self._visible_data[self._visible_data["symbol"] == symbol].sort_values("timestamp").tail(bars)
        columns = ["timestamp", "symbol", *(fields or ["open", "high", "low", "close", "volume", "turnover"])]
        unknown = set(columns) - set(result.columns)
        if unknown:
            raise ScriptContractError(f"history 请求了不可用字段：{sorted(unknown)}")
        return result[columns].copy().reset_index(drop=True)

    def get_bars(self, symbol: str, bars: int, fields: list[str] | None = None) -> pd.DataFrame:
        return self.history(symbol, bars, fields)

    def current(self, symbol: str, field: Literal["open", "close"] = "close") -> float:
        row = self._visible_data[self._visible_data["symbol"] == symbol].sort_values("timestamp").tail(1)
        if row.empty or pd.isna(row.iloc[0][field]):
            raise ScriptContractError(f"当前事件没有可用的 {symbol}.{field} 价格")
        return float(row.iloc[0][field])

    def get_account_positions(self) -> dict[str, Position]:
        return dict(self.portfolio.positions)

    def orders(self) -> list[Order]:
        return list(self._orders)

    def order_target_weights(self, weights: dict[str, float]) -> list[Order]:
        if not isinstance(weights, dict):
            raise ScriptContractError("order_target_weights 需要 {交易对: 目标权重} 字典")
        result: list[Order] = []
        targets = {**{symbol: 0.0 for symbol in self.portfolio.positions}, **weights}
        for symbol, weight in targets.items():
            if not isinstance(weight, (int, float)):
                raise ScriptContractError(f"{symbol} 的目标权重必须是数值")
            result.append(self._submit(symbol, float(weight) * self.portfolio.total_value, "target_value"))
        return result

    def order_value(self, symbol: str, value: float) -> Order:
        return self._submit(symbol, float(value), "delta_value")

    def order_target(self, symbol: str, quantity: float) -> Order:
        return self._submit(symbol, float(quantity), "target_quantity")

    def _submit(self, symbol: str, value: float, kind: str) -> Order:
        if self._order_handler is None:
            raise ScriptContractError("订单只能在策略回测事件中提交")
        order = self._order_handler(symbol, value, kind)
        self._orders.append(order)
        return order

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "data": asdict(self.data_declaration) if self.data_declaration else None,
            "factor_evaluation": asdict(self.factor_evaluation) if self.factor_evaluation else None,
            "account": asdict(self.account_declaration) if self.account_declaration else None,
            "schedules": [
                {"callback": schedule.callback.__name__, "frequency": schedule.frequency, "when": schedule.when,
                 "every": schedule.every, "weekday": schedule.weekday}
                for schedule in self.schedules
            ],
        }


def _active() -> ScriptContext:
    context = _ACTIVE_CONTEXT.get()
    if context is None:
        raise ScriptContractError("run_daily/run_weekly/run_every_bars 只能在 initialize(context) 中调用")
    return context


def run_daily(callback: Callable[[ScriptContext], None], *, when: Literal["open", "close"] = "close") -> None:
    _active().run_daily(callback, when=when)


def run_weekly(
    callback: Callable[[ScriptContext], None], *, weekday: int = 0, when: Literal["open", "close"] = "close"
) -> None:
    _active().run_weekly(callback, weekday=weekday, when=when)


def run_every_bars(
    callback: Callable[[ScriptContext], None], *, every: int = 1, when: Literal["open", "close"] = "close"
) -> None:
    _active().run_every_bars(callback, every=every, when=when)


def order_target_weights(weights: dict[str, float]) -> list[Order]:
    """当前策略事件的全局下单快捷方式，等价于 context.order_target_weights。"""
    return _active().order_target_weights(weights)


def order_value(symbol: str, value: float) -> Order:
    """当前策略事件的按金额下单快捷方式。"""
    return _active().order_value(symbol, value)


def order_target(symbol: str, quantity: float) -> Order:
    """当前策略事件的目标数量下单快捷方式。"""
    return _active().order_target(symbol, quantity)
