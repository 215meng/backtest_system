"""QuantBacktest Web：仅提供原生 Python 因子、策略与候选审核入口。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from quantbacktest.library import (
    LibraryError,
    approve_candidate,
    attach_strategy_evidence,
    default_library_root,
    list_factors,
)
from quantbacktest.native import (
    DownloadManifestError,
    NativeScriptError,
    run_factor_script,
    run_strategy_script,
    validate_factor_script,
    validate_strategy_script,
)

RESULTS_ROOT = Path.cwd() / "results" / "backtests"
LIBRARY_ROOT = default_library_root()

st.set_page_config(page_title="QuantBacktest", layout="wide")
st.title("QuantBacktest")
st.caption("原生 Python 因子与事件策略回测平台：新运行不读取 YAML。")


def _show_exception(exc: Exception) -> None:
    st.error(str(exc))
    if isinstance(exc, DownloadManifestError):
        st.subheader("数据下载清单")
        st.json(exc.manifest)


def _show_run(result: object) -> None:
    with st.container(border=True):
        st.success(f"{result.run_kind} 运行已完成：{result.run_dir}")
        metrics = result.metrics
        with st.container(horizontal=True):
            for key in ("total_return", "sharpe", "max_drawdown", "factor_coverage", "rank_ic_mean"):
                if key in metrics:
                    st.metric(key, metrics[key], border=True)
        st.json(metrics)
        report = result.run_dir / "report.html"
        if report.exists():
            st.download_button("下载 HTML 报告", report.read_bytes(), file_name="report.html", mime="text/html")
        if result.candidate:
            st.info(f"因子已自动进入候选审核：{result.candidate['factor_id']}")
        if result.candidate_registration_error:
            st.warning(f"回测工件已完成，但候选未入库：{result.candidate_registration_error}")


def _run_page(kind: str) -> None:
    st.header("因子 Python 运行" if kind == "factor" else "策略 Python 运行")
    st.caption(
        "因子脚本必须定义 initialize(context)、main(context)；策略脚本必须在 initialize(context) 中声明数据、账户、成本和调度。"
        if kind == "factor"
        else "策略只执行用户回调与订单；平台不添加默认止损、杠杆、选币或调仓规则。"
    )
    with st.form(f"{kind}_run_form", border=True):
        script = st.text_input("Python 脚本绝对路径", key=f"{kind}_script")
        root = st.text_input("项目根目录（绝对路径）", value=str(Path.cwd()), key=f"{kind}_root")
        validate = st.form_submit_button("先校验", icon=":material/fact_check:")
        run = st.form_submit_button("校验并运行", type="primary", icon=":material/play_arrow:")
    if not script:
        return
    try:
        validator = validate_factor_script if kind == "factor" else validate_strategy_script
        runner = run_factor_script if kind == "factor" else run_strategy_script
        if validate:
            st.success("脚本、数据与能力校验通过")
            st.json(validator(Path(script), Path(root)))
        if run:
            _show_run(runner(Path(script), Path(root)))
    except (NativeScriptError, DownloadManifestError, OSError, ValueError) as exc:
        _show_exception(exc)


def _review_page() -> None:
    st.header("候选审核")
    st.caption("完整因子运行自动进入候选区；策略运行仅可在这里手工关联为交易证据。批准仍需人工执行。")
    try:
        factors = list_factors(LIBRARY_ROOT)
    except (LibraryError, OSError, ValueError) as exc:
        _show_exception(exc)
        return
    if not factors:
        st.info("暂无候选。先运行一个原生 Python 因子。")
        return
    table = pd.DataFrame(factors)
    st.dataframe(table, hide_index=True)
    candidates = [item for item in factors if item["status"] == "candidate"]
    if not candidates:
        return
    labels = {item["factor_id"]: f"{item['name']} · {item['factor_id']}" for item in candidates}
    factor_id = st.selectbox("候选因子", list(labels), format_func=labels.get)
    selected = next(item for item in candidates if item["factor_id"] == factor_id)
    artifact = LIBRARY_ROOT / "artifacts" / factor_id
    report = artifact / "candidate_report.html"
    if report.exists():
        st.download_button("下载因子候选报告", report.read_bytes(), file_name=f"{factor_id}_report.html", mime="text/html")
    with st.form("attach_strategy_evidence", border=True):
        strategy_dir = st.text_input("完成的策略运行目录（可选）")
        attach = st.form_submit_button("关联策略交易证据", icon=":material/link:")
    if attach:
        try:
            st.success(str(attach_strategy_evidence(factor_id, Path(strategy_dir), LIBRARY_ROOT)))
        except (LibraryError, OSError, ValueError) as exc:
            _show_exception(exc)
    if st.button("人工批准此因子候选", type="primary", icon=":material/verified:"):
        try:
            st.success(str(approve_candidate(factor_id, Path(str(selected["source_run"])), LIBRARY_ROOT)))
        except (LibraryError, OSError, ValueError) as exc:
            _show_exception(exc)


page = st.segmented_control(
    "功能区",
    ["因子 Python 运行", "策略 Python 运行", "候选审核"],
    default="因子 Python 运行",
    label_visibility="collapsed",
)
if page == "因子 Python 运行":
    _run_page("factor")
elif page == "策略 Python 运行":
    _run_page("strategy")
else:
    _review_page()
