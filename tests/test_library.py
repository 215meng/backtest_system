from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantbacktest.library import (
    LibraryError,
    approve_candidate,
    default_library_root,
    delete_factor_permanently,
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
    (run_dir / "metrics.json").write_text(
        json.dumps({"metric_schema_version": 2, "total_return": 0.1}), encoding="utf-8"
    )
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


def test_candidate_and_approved_factor_expose_metrics_for_review(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    evidence = _completed_run(tmp_path, b"def main(context):\n    return context.data\n", "evidence")
    candidate = register_completed_run(evidence, library_root)

    pending = list_factors(library_root, status="candidate")[0]
    assert pending["metrics"] == {"metric_schema_version": 2, "total_return": 0.1}
    approve_candidate(candidate["factor_id"], evidence, library_root)
    approved = list_factors(library_root, status="approved")[0]
    assert approved["metrics"] == {"metric_schema_version": 2, "total_return": 0.1}


def test_strategy_run_cannot_create_factor_candidate(tmp_path: Path) -> None:
    strategy_run = _completed_run(tmp_path, b"def rebalance(context):\n    pass\n", "strategy", run_kind="strategy")

    with pytest.raises(LibraryError, match="策略运行"):
        register_completed_run(strategy_run, tmp_path / "library")


def test_permanent_delete_removes_owned_artifacts_and_run_but_keeps_source_script(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    backtest_root = tmp_path / "results" / "backtests"
    backtest_root.mkdir(parents=True)
    source_script = tmp_path / "original_factor.py"
    source_script.write_text("def main(context): pass\n", encoding="utf-8")
    run = _completed_run(backtest_root, b"def main(context): pass\n", "owned_run")
    spec = {"run_kind": "factor", "name": "owned_run", "script_path": str(source_script)}
    (run / "run_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    candidate = register_completed_run(run, library_root)

    result = delete_factor_permanently(candidate["factor_id"], library_root, backtest_root)

    assert source_script.exists()
    assert not run.exists()
    assert not Path(candidate["artifact_path"]).exists()
    assert list_factors(library_root) == []
    assert str(run.resolve()) in result["deleted_paths"]


def test_permanent_delete_rejects_shared_run_directory(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    backtest_root = tmp_path / "results" / "backtests"
    backtest_root.mkdir(parents=True)
    run = _completed_run(backtest_root, b"def main(context): pass\n", "shared_run")
    candidate = register_completed_run(run, library_root)
    connection = __import__("sqlite3").connect(library_root / "factor_library.sqlite")
    row = connection.execute("SELECT * FROM factors WHERE factor_id = ?", (candidate["factor_id"],)).fetchone()
    columns = [item[1] for item in connection.execute("PRAGMA table_info(factors)")]
    values = dict(zip(columns, row, strict=True))
    values["factor_id"] = "shared_reference"
    placeholders = ",".join("?" for _ in columns)
    connection.execute(f"INSERT INTO factors ({','.join(columns)}) VALUES ({placeholders})", tuple(values[name] for name in columns))
    connection.commit()
    connection.close()

    with pytest.raises(LibraryError, match="共享"):
        delete_factor_permanently(candidate["factor_id"], library_root, backtest_root)

    assert run.exists()
