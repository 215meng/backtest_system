"""QuantBacktest Web：原生 Python 因子、策略与候选审核入口。"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from quantbacktest.library import (
    LibraryError,
    approve_candidate,
    attach_strategy_evidence,
    default_library_root,
    delete_factor_permanently,
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

LIBRARY_ROOT = default_library_root()

_REPORT_FRAME = st.components.v2.component(
    "quantbacktest_report_frame",
    html="<iframe id='report-frame' title='QuantBacktest HTML 报告'></iframe>",
    css="""
    :host { display: block; width: 100%; }
    #report-frame { border: 1px solid var(--st-border-color); border-radius: 6px; width: 100%; }
    """,
    js="""
    export default function(component) {
        const { data, parentElement } = component;
        const frame = parentElement.querySelector('#report-frame');
        if (!frame) return;
        const source = data?.html ?? '';
        if (frame.srcdoc !== source) frame.srcdoc = source;
        frame.style.height = `${data?.height ?? 1600}px`;
        frame.setAttribute('sandbox', 'allow-scripts');
    }
    """,
)

st.set_page_config(page_title="QuantBacktest", layout="wide")
st.title("QuantBacktest")
st.caption("原生 Python 因子与事件策略回测平台：新运行不读取 YAML。")


_FACTOR_METRICS = (
    ("方向调整 IC", "directional_ic_mean"),
    ("方向调整 Rank IC", "directional_rank_ic_mean"),
    ("方向调整 ICIR", "directional_icir"),
    ("方向调整 Rank ICIR", "directional_rank_icir"),
    ("因子输出覆盖率", "factor_output_coverage"),
    ("可评价覆盖率", "evaluable_coverage"),
    ("方向调整单调性", "directional_monotonicity"),
    ("平均换手", "mean_turnover"),
    ("多空年化收益（成本后）", "long_short_annual_return"),
    ("多空年化波动率（成本后）", "long_short_annual_volatility"),
    ("多空 Sharpe（成本后）", "long_short_sharpe"),
    ("多空最大回撤（成本后）", "long_short_max_drawdown"),
    ("多空累计收益（成本前）", "long_short_cost_before_total_return"),
    ("多空 Sharpe（成本前）", "long_short_cost_before_sharpe"),
)
_PERCENT_METRICS = {
    "factor_coverage",
    "factor_output_coverage",
    "evaluable_coverage",
    "mean_turnover",
    "long_short_total_return",
    "long_short_annual_return",
    "long_short_annual_volatility",
    "long_short_max_drawdown",
    "long_short_cost_before_total_return",
}


def _show_exception(exc: Exception) -> None:
    st.error(str(exc))
    if isinstance(exc, DownloadManifestError):
        st.subheader("数据下载清单")
        st.json(exc.manifest)


def _factor_metric_value(metrics: dict[str, Any], key: str) -> Any:
    if key in metrics:
        return metrics[key]
    if key.startswith("long_short_cost_before_"):
        return metrics.get("cost_before", {}).get(key.removeprefix("long_short_cost_before_"))
    if key.startswith("long_short_"):
        return metrics.get("cost_after", {}).get(key.removeprefix("long_short_"))
    return None


def _item_metrics(item: dict[str, Any]) -> dict[str, Any]:
    """兼容尚未携带 metrics 字段的历史候选和热重载中的旧库函数。"""
    metrics = item.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    factor_id = item.get("factor_id")
    paths: list[Path] = []
    if isinstance(factor_id, str):
        paths.append(LIBRARY_ROOT / "artifacts" / factor_id / "candidate_metrics.json")
    source_run = item.get("source_run")
    if isinstance(source_run, str):
        paths.append(Path(source_run) / "metrics.json")
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _format_metric(key: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and not pd.notna(value)):
        return "不适用"
    if isinstance(value, (int, float)):
        return f"{float(value):.2%}" if key in _PERCENT_METRICS else f"{float(value):.4f}"
    return str(value)


def _render_factor_metrics(metrics: dict[str, Any]) -> None:
    for first in range(0, len(_FACTOR_METRICS), 4):
        row = _FACTOR_METRICS[first : first + 4]
        columns = st.columns(len(row))
        for column, (label, key) in zip(columns, row, strict=True):
            with column:
                st.metric(label, _format_metric(key, _factor_metric_value(metrics, key)), border=True)


def _render_html_report(report: Path, *, key: str) -> None:
    if not report.is_file():
        st.warning("未找到本次运行的 HTML 报告。")
        return
    with st.expander("HTML 报告（页面内预览）", expanded=True):
        # 报告在 sandbox iframe 中运行 Plotly，避免 st.html 的 DOM 净化阻断图表脚本。
        _REPORT_FRAME(
            data={"html": report.read_text(encoding="utf-8"), "height": 1600},
            key=f"preview_report_{key}",
            height=1600,
        )
    st.download_button(
        "下载 HTML 报告",
        report.read_bytes(),
        file_name="report.html",
        mime="text/html",
        key=f"download_report_{key}",
    )


def _render_factor_data(run_dir: Path, *, key: str) -> None:
    group_returns = run_dir / "group_returns.csv"
    group_cumulative = run_dir / "group_cumulative_returns.csv"
    long_short = run_dir / "long_short_returns.csv"
    if group_returns.is_file():
        with st.expander("分组收益", expanded=False):
            st.dataframe(pd.read_csv(group_returns), hide_index=True)
    if group_cumulative.is_file():
        with st.expander("Group Cumulative Returns", expanded=False):
            cumulative = pd.read_csv(group_cumulative)
            if "timestamp" in cumulative:
                cumulative = cumulative.set_index("timestamp")
            st.line_chart(cumulative)
    if long_short.is_file():
        with st.expander("多空收益与换手", expanded=False):
            st.dataframe(pd.read_csv(long_short), hide_index=True)
    for filename in (
        "factor_values.csv",
        "factor_panel.csv",
        "group_returns.csv",
        "group_cumulative_returns.csv",
        "long_short_returns.csv",
        "metrics.json",
    ):
        artifact = run_dir / filename
        if artifact.is_file():
            st.download_button(
                f"下载 {filename}",
                artifact.read_bytes(),
                file_name=filename,
                key=f"download_{key}_{filename}",
            )


def _render_strategy_data(run_dir: Path, *, key: str) -> None:
    for filename in ("returns.csv", "orders.csv", "positions.csv", "metrics.json"):
        artifact = run_dir / filename
        if artifact.is_file():
            st.download_button(
                f"下载 {filename}",
                artifact.read_bytes(),
                file_name=filename,
                key=f"download_{key}_{filename}",
            )


def _show_run(result: Any) -> None:
    with st.container(border=True):
        st.success(f"{result.run_kind} 运行已完成：{result.run_dir}")
        if result.run_kind == "factor":
            st.info(
                f"回测区间：{result.metrics.get('actual_evaluation_start', '不适用')} 至 "
                f"{result.metrics.get('actual_evaluation_end', '不适用')}"
            )
            for warning in result.warnings:
                st.warning(warning)
            _render_factor_metrics(result.metrics)
            _render_factor_data(result.run_dir, key=result.run_dir.name)
        else:
            columns = st.columns(3)
            for column, key in zip(columns, ("total_return", "sharpe", "max_drawdown"), strict=True):
                with column:
                    st.metric(key, _format_metric(key, result.metrics.get(key)), border=True)
            _render_strategy_data(result.run_dir, key=result.run_dir.name)
        _render_html_report(result.run_dir / "report.html", key=result.run_dir.name)
        if result.candidate:
            st.info(f"因子已自动进入候选审核：{result.candidate['factor_id']}")
        if result.candidate_registration_error:
            st.warning(f"回测工件已完成，但候选未入库：{result.candidate_registration_error}")
        with st.expander("运行审计详情", expanded=False):
            st.json({"run_dir": str(result.run_dir), "metrics": result.metrics})


def _platform_date_range(value: object) -> dict[str, str]:
    if not isinstance(value, tuple) or len(value) != 2 or not all(isinstance(item, date) for item in value):
        raise NativeScriptError("必须选择完整的平台回测开始和结束日期")
    start, end = value
    return {
        "start": datetime.combine(start, time.min, tzinfo=UTC).isoformat(),
        "end": datetime.combine(end, time.max, tzinfo=UTC).isoformat(),
    }


def _run_page(kind: str) -> None:
    st.header("因子 Python 运行" if kind == "factor" else "策略 Python 运行")
    st.caption(
        "因子脚本必须定义 initialize(context)、main(context)；评价口径由脚本显式声明。"
        if kind == "factor"
        else "策略只执行用户回调与订单；平台不添加默认止损、杠杆、选币或调仓规则。"
    )
    with st.form(f"{kind}_run_form", border=True):
        script = st.text_input("Python 脚本绝对路径", key=f"{kind}_script")
        root = st.text_input("项目根目录（绝对路径）", value=str(Path.cwd()), key=f"{kind}_root")
        date_range = st.date_input(
            "平台回测区间（UTC，结束日包含全天）",
            value=(date(2024, 1, 1), date(2025, 1, 1)),
            key=f"{kind}_platform_range",
        )
        validate = st.form_submit_button("先校验", icon=":material/fact_check:")
        run = st.form_submit_button("校验并运行", type="primary", icon=":material/play_arrow:")
    if not script:
        return
    try:
        overrides = _platform_date_range(date_range)
        validator = validate_factor_script if kind == "factor" else validate_strategy_script
        runner = run_factor_script if kind == "factor" else run_strategy_script
        if validate:
            st.success("脚本、数据与能力校验通过")
            st.json(validator(Path(script), Path(root), **overrides))
        if run:
            _show_run(runner(Path(script), Path(root), **overrides))
    except (NativeScriptError, DownloadManifestError, OSError, ValueError) as exc:
        _show_exception(exc)


def _factor_table(items: list[dict[str, Any]], *, approved: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        metrics = _item_metrics(item)
        rows.append(
            {
                "因子": item["name"],
                "创建时间": item["created_at"],
                "批准时间": item["approved_at"] if approved else None,
                "回测区间": f"{item.get('actual_evaluation_start') or metrics.get('actual_evaluation_start', '不适用')} 至 {item.get('actual_evaluation_end') or metrics.get('actual_evaluation_end', '不适用')}",
                "IC": _factor_metric_value(metrics, "ic_mean"),
                "Rank IC": _factor_metric_value(metrics, "rank_ic_mean"),
                "ICIR": _factor_metric_value(metrics, "icir"),
                "覆盖率": _factor_metric_value(metrics, "factor_coverage"),
                "多空年化收益": _factor_metric_value(metrics, "long_short_annual_return"),
                "多空波动率": _factor_metric_value(metrics, "long_short_annual_volatility"),
                "多空 Sharpe": _factor_metric_value(metrics, "long_short_sharpe"),
                "最大回撤": _factor_metric_value(metrics, "long_short_max_drawdown"),
                "换手": _factor_metric_value(metrics, "mean_turnover"),
            }
        )
    return pd.DataFrame(rows)


def _delete_controls(item: dict[str, Any]) -> None:
    factor_id = str(item["factor_id"])
    targets = [LIBRARY_ROOT / "artifacts" / factor_id]
    targets.extend(Path(value) for value in (item.get("source_run"), item.get("evidence_run")) if value)
    with st.expander("永久删除", expanded=False):
        st.error("此操作会永久删除数据库记录、因子库工件和自有因子回测目录；原始 Python 源脚本与独立策略证据保留。")
        st.code("\n".join(str(path.resolve()) for path in dict.fromkeys(targets)))
        with st.form(f"delete_{factor_id}", border=True):
            typed = st.text_input("输入完整 factor_id", key=f"delete_id_{factor_id}")
            confirmed = st.checkbox("我确认永久删除以上目标", key=f"delete_confirm_{factor_id}")
            submit = st.form_submit_button("永久删除", type="primary")
        if submit:
            if typed != factor_id or not confirmed:
                st.error("factor_id 不匹配或尚未勾选永久删除确认。")
            else:
                try:
                    delete_factor_permanently(
                        factor_id,
                        LIBRARY_ROOT,
                        Path(__file__).resolve().parents[1] / "results" / "backtests",
                    )
                    st.success("因子及其自有工件已永久删除；原始源脚本和独立策略证据未删除。")
                    st.rerun()
                except (LibraryError, OSError, ValueError, json.JSONDecodeError) as exc:
                    _show_exception(exc)


def _review_details(item: dict[str, Any], *, allow_approval: bool) -> None:
    factor_id = str(item["factor_id"])
    artifact = LIBRARY_ROOT / "artifacts" / factor_id
    st.subheader(item["name"])
    metrics = _item_metrics(item)
    if not metrics:
        st.warning("该历史候选未找到可读取的指标工件；可查看其审计详情和报告。")
    _render_factor_metrics(metrics)
    report = artifact / ("candidate_report.html" if allow_approval else "report.html")
    _render_html_report(report, key=f"candidate_{factor_id}")
    if allow_approval:
        with st.form(f"attach_strategy_evidence_{factor_id}", border=True):
            strategy_dir = st.text_input("完整策略运行目录（可选）")
            attach = st.form_submit_button("关联策略交易证据", icon=":material/link:")
        if attach:
            try:
                st.success(str(attach_strategy_evidence(factor_id, Path(strategy_dir), LIBRARY_ROOT)))
            except (LibraryError, OSError, ValueError) as exc:
                _show_exception(exc)
        if st.button("人工批准此因子候选", type="primary", icon=":material/verified:", key=f"approve_{factor_id}"):
            try:
                approve_candidate(factor_id, Path(str(item["source_run"])), LIBRARY_ROOT)
                st.success("已人工批准，因子已进入正式因子库。")
                st.rerun()
            except (LibraryError, OSError, ValueError) as exc:
                _show_exception(exc)
    with st.expander("审计详情", expanded=False):
        st.json(
            {
                "factor_id": factor_id,
                "source_run": item["source_run"],
                "evidence_run": item["evidence_run"],
                "strategy_evidence_run": item["strategy_evidence_run"],
                "script_hash": item["script_hash"],
                "validation_status": item.get("validation_status"),
                "invalid_reason": item.get("invalid_reason"),
                "replacement_factor_id": item.get("replacement_factor_id"),
                "platform_backtest_range": {
                    "start": metrics.get("platform_backtest_start"),
                    "end": metrics.get("platform_backtest_end"),
                },
                "market_data_range": {
                    "start": metrics.get("market_data_start"),
                    "end": metrics.get("market_data_end"),
                },
                "actual_evaluation_range": {
                    "start": item.get("actual_evaluation_start") or metrics.get("actual_evaluation_start"),
                    "end": item.get("actual_evaluation_end") or metrics.get("actual_evaluation_end"),
                },
                "metrics": metrics,
            }
        )
    _delete_controls(item)


def _review_page() -> None:
    st.header("候选审核")
    st.caption("完整因子运行自动进入候选区；策略运行可手工关联为交易证据，批准必须人工执行。")
    try:
        factors = list_factors(LIBRARY_ROOT)
    except (LibraryError, OSError, ValueError) as exc:
        _show_exception(exc)
        return
    candidates = [item for item in factors if item["status"] == "candidate" and item.get("validation_status") == "valid"]
    approved = [item for item in factors if item["status"] == "approved" and item.get("validation_status") == "valid"]
    historical = [item for item in factors if item.get("validation_status") != "valid"]
    candidate_tab, approved_tab, historical_tab = st.tabs(("待审核候选", "已批准因子库", "历史失效/待重跑"))
    with candidate_tab:
        if not candidates:
            st.info("暂无待审核候选。")
        else:
            st.dataframe(_factor_table(candidates, approved=False), hide_index=True)
            labels = {
                item["factor_id"]: f"{item['name']}（{item['created_at']}）" for item in candidates
            }
            selected_id = st.selectbox("选择待审核因子", list(labels), format_func=labels.get)
            _review_details(next(item for item in candidates if item["factor_id"] == selected_id), allow_approval=True)
    with approved_tab:
        if not approved:
            st.info("暂无已批准因子。")
        else:
            st.dataframe(_factor_table(approved, approved=True), hide_index=True)
            labels = {
                item["factor_id"]: f"{item['name']}（{item['approved_at']}）" for item in approved
            }
            selected_id = st.selectbox("选择已批准因子", list(labels), format_func=labels.get)
            _review_details(next(item for item in approved if item["factor_id"] == selected_id), allow_approval=False)
    with historical_tab:
        if not historical:
            st.info("暂无历史失效或待重跑记录。")
        else:
            table = _factor_table(historical, approved=False)
            table.insert(1, "可信状态", [item.get("validation_status") for item in historical])
            table.insert(2, "失效原因", [item.get("invalid_reason") for item in historical])
            st.dataframe(table, hide_index=True)
            labels = {item["factor_id"]: f"{item['name']}（{item.get('validation_status')}）" for item in historical}
            selected_id = st.selectbox("选择历史记录", list(labels), format_func=labels.get)
            _review_details(next(item for item in historical if item["factor_id"] == selected_id), allow_approval=False)


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
