from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import yaml

from quantbacktest.engine import run_backtest
from quantbacktest.library import list_factors
from quantbacktest.schemas import RunSpec

st.set_page_config(page_title="QuantBacktest", layout="wide")
st.title("QuantBacktest 加密货币因子研究台")
st.caption("选择 YAML 配置后运行；报告与调试轨迹保存在调用项目的 results/backtests 目录。")

uploaded = st.file_uploader("回测配置（YAML）", type=["yaml", "yml"])
if uploaded:
    payload = yaml.safe_load(uploaded.getvalue().decode("utf-8"))
    try:
        spec = RunSpec.model_validate(payload)
        st.success("输入规范校验通过")
        st.json(spec.model_dump(mode="json"))
        if st.button("运行回测", type="primary"):
            result = run_backtest(spec, Path.cwd())
            st.success(f"完成：{result.run_dir}")
            st.json(result.metrics)
            st.link_button("打开 HTML 报告", (result.run_dir / "report.html").as_uri())
            trace = result.run_dir / "debug_trace.json"
            if trace.exists():
                st.subheader("调试轨迹")
                st.json(json.loads(trace.read_text(encoding="utf-8")))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        st.error(str(exc))

st.subheader("已批准因子库")
try:
    st.dataframe(list_factors(Path("factor_library")), use_container_width=True)
except (OSError, ValueError) as exc:
    st.info(f"尚无已批准因子：{exc}")
