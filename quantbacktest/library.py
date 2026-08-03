from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

CORE_LIBRARY_ROOT = Path(__file__).resolve().parents[1] / "factor_library_v2"


class LibraryError(ValueError):
    """候选库运行证据或审核状态不符合原生运行时契约。"""


@dataclass(frozen=True)
class CompletedRun:
    directory: Path
    name: str
    script_path: Path
    script_hash: str
    run_spec_json: str
    metrics_json: str
    run_kind: Literal["factor", "strategy"]


def _connection(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "factor_library.sqlite")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS factors (
            factor_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source_run TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            script_hash TEXT NOT NULL,
            approved_at TEXT,
            evidence_run TEXT,
            strategy_evidence_run TEXT
        )
        """
    )
    connection.commit()
    return connection


def default_library_root() -> Path:
    """返回第二版原生运行专用的候选审核库。"""
    return CORE_LIBRARY_ROOT


def _factor_id(name: str, script_hash: str, run_identity: str) -> str:
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip(". ") or "factor"
    suffix = hashlib.sha256(run_identity.encode("utf-8")).hexdigest()[:12]
    return f"{safe_name}_{script_hash[:12]}_{suffix}"


def _read_completed_run(run_dir: Path) -> CompletedRun:
    directory = run_dir.expanduser().resolve()
    required = {
        "run_spec.json": directory / "run_spec.json",
        "metrics.json": directory / "metrics.json",
        "report.html": directory / "report.html",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise LibraryError(f"运行目录不是完整的原生回测工件，缺少：{', '.join(missing)}")
    try:
        run_spec = json.loads(required["run_spec.json"].read_text(encoding="utf-8"))
        metrics = json.loads(required["metrics.json"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise LibraryError(f"运行目录中的 JSON 工件无效：{exc}") from exc
    if not isinstance(run_spec, dict) or run_spec.get("run_kind") not in {"factor", "strategy"}:
        raise LibraryError("候选库仅接受第二版原生 Python 运行工件")
    if not isinstance(metrics, dict):
        raise LibraryError("metrics.json 必须是对象")
    run_kind = run_spec["run_kind"]
    snapshot_name = "factor_snapshot.py" if run_kind == "factor" else "strategy_snapshot.py"
    evidence_name = "factor_panel.csv" if run_kind == "factor" else "orders.csv"
    snapshot = directory / snapshot_name
    evidence = directory / evidence_name
    missing = [name for name, path in ((snapshot_name, snapshot), (evidence_name, evidence)) if not path.is_file()]
    if missing:
        raise LibraryError(f"运行目录不是完整的原生回测工件，缺少：{', '.join(missing)}")
    return CompletedRun(
        directory=directory,
        name=str(run_spec.get("name") or directory.name),
        script_path=snapshot,
        script_hash=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        run_spec_json=json.dumps(run_spec, ensure_ascii=False, indent=2),
        metrics_json=json.dumps(metrics, ensure_ascii=False, indent=2),
        run_kind=run_kind,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_backtest_results(run_dir: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    excluded = {"factor_snapshot.py", "strategy_snapshot.py", "run_spec.json"}
    for source in run_dir.iterdir():
        if source.is_file() and source.name not in excluded:
            shutil.copy2(source, destination / source.name)


def register_completed_run(run_dir: Path, library_root: Path | None = None) -> dict[str, str]:
    """将成功的原生因子运行幂等登记为待审核候选。"""
    run = _read_completed_run(run_dir)
    if run.run_kind != "factor":
        raise LibraryError("策略运行不能自动创建因子候选；请手工关联为交易证据")
    root = library_root or default_library_root()
    factor_id = _factor_id(run.name, run.script_hash, str(run.directory))
    artifact = root / "artifacts" / factor_id
    connection = _connection(root)
    try:
        if connection.execute("SELECT 1 FROM factors WHERE factor_id = ?", (factor_id,)).fetchone():
            return {"factor_id": factor_id, "artifact_path": str(artifact), "status": "candidate"}
        if artifact.exists():
            raise LibraryError(f"候选工件目录已存在：{artifact}")
        artifact.mkdir(parents=True)
        try:
            (artifact / "factor_snapshot.py").write_bytes(run.script_path.read_bytes())
            (artifact / "run_spec.json").write_text(run.run_spec_json, encoding="utf-8")
            (artifact / "candidate_metrics.json").write_text(run.metrics_json, encoding="utf-8")
            shutil.copy2(run.directory / "report.html", artifact / "candidate_report.html")
            _copy_backtest_results(run.directory, artifact / "backtest_results")
            now = datetime.now(UTC).isoformat()
            _write_json(
                artifact / "candidate_metadata.json",
                {
                    "factor_id": factor_id,
                    "name": run.name,
                    "status": "candidate",
                    "created_at": now,
                    "source_run": str(run.directory),
                    "script_hash": run.script_hash,
                },
            )
            connection.execute(
                """
                INSERT INTO factors (
                    factor_id, name, status, created_at, source_run, metrics_json, script_hash,
                    approved_at, evidence_run, strategy_evidence_run
                ) VALUES (?, ?, 'candidate', ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (factor_id, run.name, now, str(run.directory), run.metrics_json, run.script_hash),
            )
            connection.commit()
        except Exception:
            shutil.rmtree(artifact, ignore_errors=True)
            raise
    finally:
        connection.close()
    return {"factor_id": factor_id, "artifact_path": str(artifact), "status": "candidate"}


def approve_candidate(factor_id: str, evidence_run_dir: Path, library_root: Path) -> dict[str, str]:
    """用同一原生因子快照的成功运行批准候选。"""
    run = _read_completed_run(evidence_run_dir)
    if run.run_kind != "factor":
        raise LibraryError("批准候选必须提供原生因子运行证据")
    connection = _connection(library_root)
    try:
        row = connection.execute("SELECT * FROM factors WHERE factor_id = ?", (factor_id,)).fetchone()
        if row is None:
            raise LibraryError(f"未找到候选因子：{factor_id}")
        if row["status"] != "candidate":
            raise LibraryError(f"因子 {factor_id} 当前状态为 {row['status']}，不能重复批准")
        if row["script_hash"] != run.script_hash:
            raise LibraryError("候选脚本与因子运行快照哈希不一致")
        artifact = library_root / "artifacts" / factor_id
        shutil.copy2(run.directory / "metrics.json", artifact / "metrics.json")
        shutil.copy2(run.directory / "report.html", artifact / "report.html")
        shutil.copy2(run.directory / "run_spec.json", artifact / "evidence_run_spec.json")
        _copy_backtest_results(run.directory, artifact / "backtest_results")
        now = datetime.now(UTC).isoformat()
        _write_json(
            artifact / "approval_metadata.json",
            {"status": "approved", "approved_at": now, "evidence_run": str(run.directory)},
        )
        connection.execute(
            """
            UPDATE factors
            SET status = 'approved', approved_at = ?, evidence_run = ?, metrics_json = ?
            WHERE factor_id = ?
            """,
            (now, str(run.directory), run.metrics_json, factor_id),
        )
        connection.commit()
    finally:
        connection.close()
    return {"factor_id": factor_id, "artifact_path": str(artifact), "status": "approved"}


def attach_strategy_evidence(factor_id: str, strategy_run_dir: Path, library_root: Path) -> dict[str, str]:
    """关联原生事件策略运行作为候选的交易证据。"""
    run = _read_completed_run(strategy_run_dir)
    if run.run_kind != "strategy":
        raise LibraryError("只能关联原生 Python 策略运行作为交易证据")
    connection = _connection(library_root)
    try:
        row = connection.execute("SELECT 1 FROM factors WHERE factor_id = ?", (factor_id,)).fetchone()
        if row is None:
            raise LibraryError(f"未找到候选因子：{factor_id}")
        artifact = library_root / "artifacts" / factor_id
        evidence_dir = artifact / "strategy_evidence"
        _copy_backtest_results(run.directory, evidence_dir)
        shutil.copy2(run.script_path, evidence_dir / "strategy_snapshot.py")
        _write_json(
            evidence_dir / "metadata.json",
            {"strategy_run": str(run.directory), "strategy_script_hash": run.script_hash},
        )
        connection.execute(
            "UPDATE factors SET strategy_evidence_run = ? WHERE factor_id = ?",
            (str(run.directory), factor_id),
        )
        connection.commit()
    finally:
        connection.close()
    return {"factor_id": factor_id, "strategy_run": str(run.directory), "status": "evidence_attached"}


def list_factors(library_root: Path, status: str | None = None) -> list[dict[str, str | None]]:
    connection = _connection(library_root)
    try:
        query = """
            SELECT factor_id, name, status, created_at, approved_at, source_run,
                   evidence_run, strategy_evidence_run, script_hash
            FROM factors
        """
        parameters: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status,)
        rows = connection.execute(query + " ORDER BY created_at DESC", parameters).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]
