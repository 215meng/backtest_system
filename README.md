# QuantBacktest

QuantBacktest 是本地加密数据的原生 Python 因子与事件策略回测平台。用户只维护 Python 文件；平台提供数据、严格历史时点约束、因子评价、交易仿真、运行工件和候选审核。

```text
因子 Python → IC / 分组 / 多空绩效 → 自动候选
策略 Python → 回调 / 订单 / 净值绩效 → 手工关联为交易证据
```

## 快速开始

```powershell
conda activate backtest-system
pip install -e .[dev]
quantbacktest factor-validate examples\native_binance_smoke_factor.py
quantbacktest factor-run examples\native_binance_smoke_factor.py
quantbacktest strategy-validate examples\native_crypto_smoke_strategy.py
quantbacktest strategy-run examples\native_crypto_smoke_strategy.py
streamlit run quantbacktest\web.py
```

新运行不需要 YAML。因子脚本使用 `initialize(context)` 声明数据和评价口径，并由 `main(context)` 返回 `timestamp`、`symbol`、`factor`；策略脚本使用 `initialize(context)` 声明账户和调度，在回调中调用历史数据与下单 API。

外部项目请通过 MCP 的 `create_factor_script` / `create_strategy_script` 生成骨架，再调用相应 `validate_*_script` 与运行工具。详细契约见 [跨项目调用指南](docs/跨项目调用指南.md)。

本项目执行外部 Python 脚本时将其视为本地可信代码，不把独立进程当作安全沙箱。历史 YAML 运行目录仅保留审计用途，不能作为新的公开运行入口。
