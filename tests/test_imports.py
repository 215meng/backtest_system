from datetime import date
from pathlib import Path

import yaml

from quantbacktest.imports import prepare_imported_run


def test_prepare_imported_run_snapshots_factor_and_resolves_paths(tmp_path: Path) -> None:
    source_root = tmp_path / "factor_project"
    source_root.mkdir()
    payload = {
        "name": "import_case",
        "data": {
            "adapter": "crypto_top50",
            "path": "data/raw/crypto",
            "market": "spot",
            "frequency": "1h",
            "symbols": [f"S{index}" for index in range(10)],
        },
        "factor": {"module_path": "ignored.py"},
        "strategy": {
            "mode": "cross_sectional",
            "selection": "quantiles",
            "long_short": "long_only",
            "quantiles": 2,
        },
    }
    factor = b"FactorMeta={'required_fields':['close'], 'min_lookback_bars':1, 'supported_modes':['cross_sectional']}\ndef compute_factor(context):\n return context.data[['timestamp', 'symbol']].assign(factor=1.0)\n"
    prepared = prepare_imported_run(
        factor_name="external_factor.py",
        factor_content=factor,
        config_content=yaml.safe_dump(payload).encode("utf-8"),
        source_root=source_root,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )
    assert prepared.spec.data.path == (source_root / "data/raw/crypto").resolve()
    assert prepared.spec.factor.module_path.read_bytes() == factor
    assert prepared.spec.output.root == source_root / "results/backtests"
    assert prepared.spec.data.start.isoformat().startswith("2024-01-01T00:00:00")
    assert prepared.spec.data.end.isoformat().startswith("2024-01-31T23:59:59")
    assert "external_factor.py" in prepared.normalized_yaml
