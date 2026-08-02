from __future__ import annotations

import hashlib
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def _connection(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "factor_library.sqlite")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS factors (
            factor_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source_run TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            script_hash TEXT NOT NULL
        )
        """
    )
    return connection


def promote(run_dir: Path, library_root: Path) -> dict[str, str]:
    """人工调用后复制可复现快照并登记 approved 因子。"""
    spec = (run_dir / "run_spec.json").read_text(encoding="utf-8")
    metrics = (run_dir / "metrics.json").read_text(encoding="utf-8")
    import json

    run_spec = json.loads(spec)
    script_path = run_dir / "factor_snapshot.py"
    # 只从运行快照提升，避免外部项目的后续修改污染已审核版本。
    if not script_path.exists():
        raise FileNotFoundError(f"无法提升：因子脚本已不可访问：{script_path}")
    script_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
    factor_id = f"{run_spec['name']}_{script_hash[:12]}"
    artifact = library_root / "artifacts" / factor_id
    artifact.mkdir(parents=True, exist_ok=False)
    shutil.copy2(script_path, artifact / script_path.name)
    shutil.copy2(run_dir / "run_spec.json", artifact / "run_spec.json")
    shutil.copy2(run_dir / "metrics.json", artifact / "metrics.json")
    if (run_dir / "report.html").exists():
        shutil.copy2(run_dir / "report.html", artifact / "report.html")
    connection = _connection(library_root)
    try:
        connection.execute(
            "INSERT INTO factors VALUES (?, ?, 'approved', ?, ?, ?, ?)",
            (factor_id, run_spec["name"], datetime.now(UTC).isoformat(), str(run_dir), metrics, script_hash),
        )
        connection.commit()
    finally:
        connection.close()
    return {"factor_id": factor_id, "artifact_path": str(artifact), "status": "approved"}


def list_factors(library_root: Path) -> list[dict[str, str]]:
    connection = _connection(library_root)
    try:
        rows = connection.execute(
            "SELECT factor_id, name, status, created_at, source_run, script_hash FROM factors ORDER BY created_at DESC"
        ).fetchall()
    finally:
        connection.close()
    fields = ["factor_id", "name", "status", "created_at", "source_run", "script_hash"]
    return [dict(zip(fields, row)) for row in rows]
