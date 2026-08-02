from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from quantbacktest.engine import RunResult, run_backtest
from quantbacktest.imports import prepare_imported_run
from quantbacktest.library import list_factors
from quantbacktest.schemas import RunSpec


def _show_result(result: RunResult) -> None:
    st.success(f"完成：{result.run_dir}")
    st.json(result.metrics)
    st.link_button("打开 HTML 报告", (result.run_dir / "report.html").as_uri())
    for file_name, title in (("risk_events.csv", "回撤止损事件"), ("trades.csv", "交易明细")):
        path = result.run_dir / file_name
        if path.exists():
            st.download_button(title, path.read_bytes(), file_name=file_name, mime="text/csv")
    for file_name, title in (
        ("research_group_returns.csv", "因子研究分组收益"),
        ("research_spread.csv", "因子研究 Top-Bottom 差值"),
        ("research_ic_decay.csv", "因子研究 IC 衰减"),
        ("research_contributions.csv", "因子研究单币贡献"),
        ("research_leave_one_out.csv", "因子研究逐币剔除"),
    ):
        path = result.run_dir / file_name
        if path.exists():
            st.download_button(title, path.read_bytes(), file_name=file_name, mime="text/csv")
    trace = result.run_dir / "debug_trace.json"
    if trace.exists():
        st.subheader("调试轨迹")
        st.json(json.loads(trace.read_text(encoding="utf-8")))


def _date_from_payload(payload: dict[str, Any], field: str) -> date:
    raw_value = payload.get("data", {}).get(field)
    if raw_value:
        return date.fromisoformat(str(raw_value)[:10])
    return datetime.now(UTC).date()


def _load_yaml(uploaded: Any) -> dict[str, Any]:
    payload = yaml.safe_load(uploaded.getvalue().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("YAML 配置根节点必须是对象")
    return payload


def _show_factor_research_contract() -> None:
    with st.expander("论文因子研究配置说明（新研究默认使用）", expanded=False):
        st.markdown(
            "`evaluation.mode: factor_research` 用于检验因子对未来横截面收益的预测力，"
            "不会生成账户净值、CAGR 或爆仓结论。每篇论文都必须明确形成时间表、收益口径、"
            "因子方向和分组规则。所有时间均为 UTC。"
        )
        st.code(
            """evaluation:
  mode: factor_research
  research:
    formation:
      kind: calendar        # calendar 或 bar_interval
      interval: 1w          # 1h / 4h / 1d / 1w
      weekday: 6            # 周频必填；0=周一，6=周日
      time_utc: \"00:00\"    # 必填，不存在锚点将跳过而非漂移
    returns:
      horizon: 1w
      start_price: close    # close 或 next_open
      end_price: close      # close 或 open
    direction: higher_predicts_higher_return
    portfolio:
      selection: quantiles
      quantiles: 3
      weighting: equal      # equal / score / market_cap
    ic_decay_horizons: [1d, 1w]
""",
            language="yaml",
        )


st.set_page_config(page_title="QuantBacktest", layout="wide")
st.title("QuantBacktest 加密货币因子研究台")
st.caption("信号在收盘生成、默认于下一根开盘成交；回测结果保存在调用项目的 results/backtests。")

st.header("外部项目导入回测")
st.info("仅上传你自己编写或已审核的可信 Python 因子脚本；该脚本会在本机执行。")
_show_factor_research_contract()
source_root = st.text_input("源项目绝对目录", placeholder=r"E:\py\my_factor_project")
factor_upload = st.file_uploader("因子脚本（.py）", type=["py"], key="import_factor")
config_upload = st.file_uploader("回测配置（.yaml / .yml）", type=["yaml", "yml"], key="import_config")

if config_upload is not None:
    try:
        import_payload = _load_yaml(config_upload)
        default_start = _date_from_payload(import_payload, "start")
        default_end = _date_from_payload(import_payload, "end")
        date_left, date_right = st.columns(2)
        start_date = date_left.date_input("回测开始日期（UTC）", value=default_start, key="import_start")
        end_date = date_right.date_input("回测结束日期（UTC）", value=default_end, key="import_end")
        if end_date < start_date:
            st.error("回测结束日期不能早于开始日期")
        elif factor_upload is not None:
            factor_hash = hashlib.sha256(factor_upload.getvalue()).hexdigest()
            st.caption(f"因子 SHA-256：{factor_hash}；结果目录：<源项目>/results/backtests")
            trusted = st.checkbox("我确认该 Python 脚本可信，并允许本机执行", key="trusted_factor")
            if st.button("导入并运行回测", type="primary", disabled=not trusted):
                prepared = prepare_imported_run(
                    factor_name=factor_upload.name,
                    factor_content=factor_upload.getvalue(),
                    config_content=config_upload.getvalue(),
                    source_root=source_root,
                    start_date=start_date,
                    end_date=end_date,
                )
                st.subheader("最终标准化配置")
                st.code(prepared.normalized_yaml, language="yaml")
                result = run_backtest(prepared.spec, prepared.source_root)
                (result.run_dir / "imported_run_spec.yaml").write_text(
                    prepared.normalized_yaml, encoding="utf-8"
                )
                _show_result(result)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        st.error(str(exc))

st.divider()
st.header("当前项目 YAML 回测")
uploaded = st.file_uploader("回测配置（YAML）", type=["yaml", "yml"], key="local_config")
if uploaded:
    try:
        spec = RunSpec.model_validate(_load_yaml(uploaded))
        st.success("输入规范校验通过")
        st.json(spec.model_dump(mode="json"))
        if st.button("运行当前项目回测", type="primary"):
            _show_result(run_backtest(spec, Path.cwd()))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        st.error(str(exc))

st.subheader("已批准因子库")
try:
    st.dataframe(list_factors(Path("factor_library")), use_container_width=True)
except (OSError, ValueError) as exc:
    st.info(f"尚无已批准因子：{exc}")
