from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantbacktest.library import (
    LibraryError,
    approve_candidate,
    default_library_root,
    list_factors,
    register_completed_run,
)


def _completed_run(tmp_path: Path, script: bytes, name: str, *, run_kind: str = "factor") -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    snapshot_name = "factor_snapshot.py" if run_kind == "factor" else "strategy_snapshot.py"
    (run_dir / snapshot_name).write_bytes(script)
    (run_dir / "run_spec.json").write_text(
        json.dumps({"run_kind": run_kind, "name": name}), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(json.dumps({"total_return": 0.1}), encoding="utf-8")
    (run_dir / "report.html").write_text("<html>report</html>", encoding="utf-8")
    if run_kind == "factor":
        (run_dir / "factor_panel.csv").write_text("timestamp,symbol,factor\n", encoding="utf-8")
    else:
        (run_dir / "orders.csv").write_text("timestamp,symbol,status\n", encoding="utf-8")
    return run_dir


def test_default_library_root_isolated_from_legacy_library() -> None:
    assert default_library_root().name == "factor_library_v2"


def test_automatic_candidates_are_per_run_and_idempotent(tmp_path: Path) -> None:
    script = b"def main(context):\n    return context.data\n"
    first_run = _completed_run(tmp_path, script, "first_run")
    second_run = _completed_run(tmp_path, script, "second_run")
    library_root = tmp_path / "library"

    first = register_completed_run(first_run, library_root)
    repeated = register_completed_run(first_run, library_root)
    second = register_completed_run(second_run, library_root)

    assert first["factor_id"] == repeated["factor_id"]
    assert first["factor_id"] != second["factor_id"]
    assert len(list_factors(library_root, status="candidate")) == 2
    assert (Path(first["artifact_path"]) / "backtest_results" / "factor_panel.csv").exists()


def test_candidate_approval_requires_matching_native_factor_run(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    script = b"def main(context):\n    return context.data\n"
    evidence = _completed_run(tmp_path, script, "evidence")
    candidate = register_completed_run(evidence, library_root)

    approved = approve_candidate(candidate["factor_id"], evidence, library_root)

    assert approved["status"] == "approved"
    assert list_factors(library_root, status="approved")[0]["evidence_run"] == str(evidence.resolve())


def test_strategy_run_cannot_create_factor_candidate(tmp_path: Path) -> None:
    strategy_run = _completed_run(tmp_path, b"def rebalance(context):\n    pass\n", "strategy", run_kind="strategy")

    with pytest.raises(LibraryError, match="策略运行"):
        register_completed_run(strategy_run, tmp_path / "library")
