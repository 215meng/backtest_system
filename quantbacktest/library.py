from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import webbrowser
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from quantbacktest.schemas import RunSpec


class LibraryError(ValueError):
    """因子库候选创建或人工批准时可定位的业务错误。"""


@dataclass(frozen=True)
class CompletedRun:
    directory: Path
    name: str
    script_path: Path
    script_hash: str
    run_spec_json: str
    metrics_json: str


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
            script_hash TEXT NOT NULL
        )
        """
    )
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(factors)")}
    migrations = {
        "source_type": "TEXT NOT NULL DEFAULT 'legacy_run'",
        "config_hash": "TEXT NOT NULL DEFAULT ''",
        "approved_at": "TEXT",
        "evidence_run": "TEXT",
    }
    for column, definition in migrations.items():
        if column not in columns:
            connection.execute(f"ALTER TABLE factors ADD COLUMN {column} {definition}")
    connection.execute(
        "UPDATE factors SET source_type = 'legacy_run' WHERE source_type IS NULL OR source_type = ''"
    )
    connection.commit()
    return connection


def _factor_id(name: str, script_hash: str) -> str:
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip(". ") or "factor"
    return f"{safe_name}_{script_hash[:12]}"


def _read_completed_run(run_dir: Path) -> CompletedRun:
    directory = run_dir.expanduser().resolve()
    if not directory.is_dir():
        raise LibraryError(f"回测运行目录不存在：{directory}")
    required = {
        "run_spec.json": directory / "run_spec.json",
        "metrics.json": directory / "metrics.json",
        "factor_snapshot.py": directory / "factor_snapshot.py",
        "report.html": directory / "report.html",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise LibraryError(f"运行目录不是完整的成功回测，缺少：{', '.join(missing)}")
    if not ((directory / "returns.csv").exists() or (directory / "research_panel.csv").exists()):
        raise LibraryError("运行目录缺少策略或因子研究结果，不能作为批准证据")
    try:
        run_spec = json.loads(required["run_spec.json"].read_text(encoding="utf-8"))
        spec = RunSpec.model_validate(run_spec)
        metrics = json.loads(required["metrics.json"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise LibraryError(f"运行目录中的配置或指标无效：{exc}") from exc
    if not isinstance(metrics, dict):
        raise LibraryError("运行目录中的 metrics.json 必须是对象")
    return CompletedRun(
        directory=directory,
        name=spec.name,
        script_path=required["factor_snapshot.py"],
        script_hash=hashlib.sha256(required["factor_snapshot.py"].read_bytes()).hexdigest(),
        run_spec_json=json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2),
        metrics_json=json.dumps(metrics, ensure_ascii=False, indent=2),
    )


def list_completed_runs(results_root: Path) -> list[dict[str, str]]:
    """列出本项目结果目录中可作为候选或批准证据的完整运行。"""
    root = results_root.expanduser()
    if not root.is_dir():
        return []
    runs: list[dict[str, str]] = []
    for directory in sorted((path for path in root.iterdir() if path.is_dir()), reverse=True):
        try:
            run = _read_completed_run(directory)
        except LibraryError:
            continue
        runs.append({"run_dir": str(run.directory), "name": run.name, "script_hash": run.script_hash})
    return runs


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_backtest_results(run_dir: Path, destination: Path) -> None:
    """封存运行目录中的全部结果工件，不复制因子脚本和主配置快照。"""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    excluded = {"factor_snapshot.py", "run_spec.json"}
    for source in run_dir.iterdir():
        if source.is_file() and source.name not in excluded:
            shutil.copy2(source, destination / source.name)


def open_html_report(report_path: Path) -> str:
    """在本机默认浏览器中打开因子库或运行目录的离线 HTML 报告。"""
    path = report_path.expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".html":
        raise LibraryError(f"未找到可打开的 HTML 报告：{path}")
    if not webbrowser.open_new_tab(path.as_uri()):
        raise LibraryError("系统默认浏览器未能打开 HTML 报告")
    return str(path)


def _create_candidate(
    *,
    name: str,
    script_content: bytes,
    run_spec_json: str,
    source_type: str,
    source_run: str,
    metrics_json: str,
    report_source: Path | None,
    run_source: Path | None,
    library_root: Path,
) -> dict[str, str]:
    script_hash = hashlib.sha256(script_content).hexdigest()
    config_hash = hashlib.sha256(run_spec_json.encode("utf-8")).hexdigest()
    factor_id = _factor_id(name, script_hash)
    artifact = library_root / "artifacts" / factor_id
    connection = _connection(library_root)
    try:
        if connection.execute("SELECT 1 FROM factors WHERE factor_id = ?", (factor_id,)).fetchone():
            raise LibraryError(f"同一因子快照已在因子库中：{factor_id}")
        if artifact.exists():
            raise LibraryError(f"因子库工件目录已存在：{artifact}")
        artifact.mkdir(parents=True)
        try:
            (artifact / "factor_snapshot.py").write_bytes(script_content)
            (artifact / "run_spec.json").write_text(run_spec_json, encoding="utf-8")
            if metrics_json != "{}":
                (artifact / "candidate_metrics.json").write_text(metrics_json, encoding="utf-8")
            if report_source is not None:
                shutil.copy2(report_source, artifact / "candidate_report.html")
            if run_source is not None:
                _copy_backtest_results(run_source, artifact / "backtest_results")
            now = datetime.now(UTC).isoformat()
            _write_json(
                artifact / "candidate_metadata.json",
                {
                    "factor_id": factor_id,
                    "name": name,
                    "status": "candidate",
                    "created_at": now,
                    "source_type": source_type,
                    "source_run": source_run or None,
                    "script_hash": script_hash,
                    "config_hash": config_hash,
                    "has_reproducible_evidence": False,
                },
            )
            connection.execute(
                """
                INSERT INTO factors (
                    factor_id, name, status, created_at, source_run, metrics_json, script_hash,
                    source_type, config_hash, approved_at, evidence_run
                ) VALUES (?, ?, 'candidate', ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (factor_id, name, now, source_run, metrics_json, script_hash, source_type, config_hash),
            )
            connection.commit()
        except Exception:
            shutil.rmtree(artifact, ignore_errors=True)
            raise
    finally:
        connection.close()
    return {"factor_id": factor_id, "artifact_path": str(artifact), "status": "candidate"}


def create_candidate_from_run(run_dir: Path, library_root: Path) -> dict[str, str]:
    """从完整运行快照创建待人工审核的候选因子。"""
    run = _read_completed_run(run_dir)
    return _create_candidate(
        name=run.name,
        script_content=run.script_path.read_bytes(),
        run_spec_json=run.run_spec_json,
        source_type="completed_run",
        source_run=str(run.directory),
        metrics_json=run.metrics_json,
        report_source=run.directory / "report.html",
        run_source=run.directory,
        library_root=library_root,
    )


def create_candidate_from_upload(
    factor_content: bytes,
    config_content: bytes,
    factor_name: str,
    library_root: Path,
) -> dict[str, str]:
    """保存未执行的上传脚本和标准化配置，创建候选因子。"""
    if Path(factor_name).suffix.lower() != ".py":
        raise LibraryError("候选因子脚本必须是 .py 文件")
    try:
        payload = yaml.safe_load(config_content.decode("utf-8"))
        if not isinstance(payload, dict):
            raise LibraryError("候选因子 YAML 根节点必须是对象")
        spec = RunSpec.model_validate(payload)
    except (UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise LibraryError(f"候选因子 YAML 不符合 RunSpec：{exc}") from exc
    run_spec_json = json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2)
    return _create_candidate(
        name=spec.name,
        script_content=factor_content,
        run_spec_json=run_spec_json,
        source_type="direct_upload",
        source_run="",
        metrics_json="{}",
        report_source=None,
        run_source=None,
        library_root=library_root,
    )


def approve_candidate(factor_id: str, evidence_run_dir: Path, library_root: Path) -> dict[str, str]:
    """用脚本哈希一致的成功回测证据批准候选因子。"""
    run = _read_completed_run(evidence_run_dir)
    connection = _connection(library_root)
    try:
        row = connection.execute("SELECT * FROM factors WHERE factor_id = ?", (factor_id,)).fetchone()
        if row is None:
            raise LibraryError(f"未找到候选因子：{factor_id}")
        if row["status"] != "candidate":
            raise LibraryError(f"因子 {factor_id} 当前状态为 {row['status']}，不能重复批准")
        if row["script_hash"] != run.script_hash:
            raise LibraryError("候选脚本与回测运行的 factor_snapshot.py 哈希不一致；请使用同一快照重新回测")
        artifact = library_root / "artifacts" / factor_id
        if not artifact.is_dir():
            raise LibraryError(f"候选因子工件目录不存在：{artifact}")
        shutil.copy2(run.directory / "metrics.json", artifact / "metrics.json")
        shutil.copy2(run.directory / "report.html", artifact / "report.html")
        shutil.copy2(run.directory / "run_spec.json", artifact / "evidence_run_spec.json")
        _copy_backtest_results(run.directory, artifact / "backtest_results")
        now = datetime.now(UTC).isoformat()
        _write_json(
            artifact / "approval_metadata.json",
            {
                "status": "approved",
                "approved_at": now,
                "evidence_run": str(run.directory),
                "evidence_script_hash": run.script_hash,
            },
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


def promote(run_dir: Path, library_root: Path) -> dict[str, str]:
    """兼容旧 CLI/MCP：从运行创建候选并以该运行证据立即批准。"""
    candidate = create_candidate_from_run(run_dir, library_root)
    return approve_candidate(candidate["factor_id"], run_dir, library_root)


def list_factors(library_root: Path, status: str | None = None) -> list[dict[str, str | None]]:
    connection = _connection(library_root)
    try:
        query = """
            SELECT factor_id, name, status, created_at, approved_at, source_type, source_run,
                   evidence_run, script_hash, config_hash
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
