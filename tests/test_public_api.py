from __future__ import annotations

import quantbacktest


def test_package_does_not_export_legacy_yaml_api() -> None:
    assert not hasattr(quantbacktest, "RunSpec")
    assert not hasattr(quantbacktest, "run_backtest")
