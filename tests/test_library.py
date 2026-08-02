import hashlib
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from quantbacktest.library import (
    LibraryError,
    approve_candidate,
    create_candidate_from_run,
    create_candidate_from_upload,
    list_completed_runs,
    list_factors,
    open_html_report,
    promote,
)


def _payload(name: str = "library_case") -> dict:
    return {
        "name": name,
        "data": {
            "adapter": "crypto_top50",
            "path": "data/raw/crypto_top50",
            "market": "spot",
            "frequency": "1h",
            "symbols": [f"S{index}" for index in range(10)],
        },
        "factor": {"module_path": "factor.py"},
        "strategy": {
            "mode": "cross_sectional",
            "selection": "quantiles",
            "long_short": "long_only",
            "quantiles": 2,
        },
    }


def _completed_run(tmp_path: Path, script: bytes, name: str = "library_case") -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "factor_snapshot.py").write_bytes(script)
    (run_dir / "run_spec.json").write_text(json.dumps(_payload(name)), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"total_return": 0.1}), encoding="utf-8")
    (run_dir / "report.html").write_text("<html>report</html>", encoding="utf-8")
    (run_dir / "returns.csv").write_text("timestamp,strategy_return\n", encoding="utf-8")
    (run_dir / "positions.csv").write_text("timestamp,symbol,target_weight\n", encoding="utf-8")
    (run_dir / "trades.csv").write_text("timestamp,symbol,cost\n", encoding="utf-8")
    (run_dir / "debug_trace.json").write_text("{}", encoding="utf-8")
    (run_dir / "model.pkl").write_bytes(b"model")
    return run_dir


def test_create_candidate_from_completed_run_keeps_preview_evidence(tmp_path: Path) -> None:
    script = b"def compute_factor(context):\n    return context.data\n"
    run_dir = _completed_run(tmp_path, script)
    library_root = tmp_path / "library"

    result = create_candidate_from_run(run_dir, library_root)

    artifact = Path(result["artifact_path"])
    assert result["status"] == "candidate"
    assert (artifact / "factor_snapshot.py").read_bytes() == script
    assert (artifact / "candidate_metrics.json").exists()
    assert (artifact / "candidate_report.html").exists()
    assert (artifact / "backtest_results" / "positions.csv").exists()
    assert (artifact / "backtest_results" / "debug_trace.json").exists()
    assert (artifact / "backtest_results" / "model.pkl").read_bytes() == b"model"
    assert list_factors(library_root, status="candidate")[0]["source_type"] == "completed_run"
    assert list_completed_runs(tmp_path) == [
        {"run_dir": str(run_dir.resolve()), "name": "library_case", "script_hash": hashlib.sha256(script).hexdigest()}
    ]


def test_direct_upload_candidate_is_not_executed_or_given_metrics(tmp_path: Path) -> None:
    script = b"raise RuntimeError('must not execute')\n"
    library_root = tmp_path / "library"

    result = create_candidate_from_upload(
        script,
        yaml.safe_dump(_payload("uploaded_case")).encode("utf-8"),
        "uploaded_factor.py",
        library_root,
    )

    artifact = Path(result["artifact_path"])
    assert result["status"] == "candidate"
    assert (artifact / "factor_snapshot.py").read_bytes() == script
    assert not (artifact / "candidate_metrics.json").exists()
    assert list_factors(library_root, status="candidate")[0]["source_type"] == "direct_upload"


def test_candidate_requires_matching_completed_run_to_be_approved(tmp_path: Path) -> None:
    script = b"def compute_factor(context):\n    return context.data\n"
    library_root = tmp_path / "library"
    candidate = create_candidate_from_upload(
        script,
        yaml.safe_dump(_payload("approval_case")).encode("utf-8"),
        "approval_factor.py",
        library_root,
    )
    evidence = _completed_run(tmp_path, script, "evidence")

    approved = approve_candidate(candidate["factor_id"], evidence, library_root)

    artifact = Path(approved["artifact_path"])
    assert approved["status"] == "approved"
    assert (artifact / "metrics.json").exists()
    assert (artifact / "report.html").exists()
    assert (artifact / "backtest_results" / "trades.csv").exists()
    record = list_factors(library_root, status="approved")[0]
    assert record["evidence_run"] == str(evidence.resolve())


def test_candidate_rejects_mismatched_or_incomplete_evidence(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    candidate = create_candidate_from_upload(
        b"factor = 1\n",
        yaml.safe_dump(_payload("mismatch_case")).encode("utf-8"),
        "mismatch_factor.py",
        library_root,
    )
    mismatched = _completed_run(tmp_path, b"factor = 2\n", "mismatch_evidence")
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()

    with pytest.raises(LibraryError, match="哈希不一致"):
        approve_candidate(candidate["factor_id"], mismatched, library_root)
    with pytest.raises(LibraryError, match="缺少"):
        approve_candidate(candidate["factor_id"], incomplete, library_root)


def test_duplicate_candidate_and_legacy_database_migration_are_safe(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    script = b"factor = 1\n"
    config = yaml.safe_dump(_payload("duplicate_case")).encode("utf-8")
    create_candidate_from_upload(script, config, "duplicate.py", library_root)
    with pytest.raises(LibraryError, match="已在因子库中"):
        create_candidate_from_upload(script, config, "duplicate.py", library_root)

    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    connection = sqlite3.connect(legacy_root / "factor_library.sqlite")
    connection.execute(
        """
        CREATE TABLE factors (
            factor_id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, source_run TEXT NOT NULL, metrics_json TEXT NOT NULL,
            script_hash TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO factors VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("legacy_id", "legacy", "approved", "2024-01-01T00:00:00+00:00", "run", "{}", "hash"),
    )
    connection.commit()
    connection.close()

    first_read = list_factors(legacy_root)
    second_read = list_factors(legacy_root)
    assert first_read == second_read
    assert first_read[0]["status"] == "approved"
    assert first_read[0]["source_type"] == "legacy_run"


def test_legacy_promote_still_creates_an_approved_factor(tmp_path: Path) -> None:
    script = b"def compute_factor(context):\n    return context.data\n"
    run_dir = _completed_run(tmp_path, script, "legacy_promote")

    result = promote(run_dir, tmp_path / "library")

    assert result["status"] == "approved"


def test_open_html_report_uses_the_local_default_browser(tmp_path: Path) -> None:
    report = tmp_path / "report.html"
    report.write_text("<html>report</html>", encoding="utf-8")

    with patch("quantbacktest.library.webbrowser.open_new_tab", return_value=True) as open_browser:
        opened = open_html_report(report)

    assert opened == str(report.resolve())
    open_browser.assert_called_once_with(report.resolve().as_uri())
    with pytest.raises(LibraryError, match="HTML 报告"):
        open_html_report(tmp_path / "missing.html")
