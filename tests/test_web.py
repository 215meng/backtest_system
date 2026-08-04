from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_web_has_mandatory_platform_range_and_three_review_tabs() -> None:
    app = AppTest.from_file(str(Path(__file__).parents[1] / "quantbacktest" / "web.py"))
    app.run(timeout=20)

    assert not app.exception
    assert any(item.label == "平台回测区间（UTC，结束日包含全天）" for item in app.date_input)

    app.segmented_control[0].set_value("候选审核").run(timeout=20)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["待审核候选", "已批准因子库", "历史失效/待重跑"]
