# QuantBacktest

本项目是面向加密货币论文因子复现的本机回测系统。它统一提供 Python API、CLI、Streamlit Web 与 MCP 调用入口，并将每次运行保存为可审计的结果目录。

## 快速开始

```powershell
conda activate backtest-system
pip install -e .[dev]
quantbacktest validate examples/cross_sectional.yaml
quantbacktest run examples/cross_sectional.yaml
streamlit run quantbacktest/web.py
```

调用前先运行 `quantbacktest schema` 或 MCP 的 `validate_run_spec`。完整输入由 Pydantic JSON Schema 定义；未知字段会被拒绝，条件字段会给出业务化错误。

## 外部因子约定

外部脚本需定义 `FactorMeta` 和 `compute_factor(context)`。`context.data` 是标准长表，包含 `timestamp`、`symbol`、`open`、`high`、`low`、`close`、`volume` 与 `turnover`（可用时）。函数返回带 `timestamp`、`symbol`、`factor` 的表，或相同索引的 `Series`。

所有外部脚本均视作本地可信代码。回测系统会检查其输入输出与数据覆盖，不会把独立进程当作安全沙箱。
