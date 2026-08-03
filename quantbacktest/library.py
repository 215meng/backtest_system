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

from quantbacktest.config_io import ConfigLoadError, load_yaml_mapping
from quantbacktest.schemas import RunSpec

CORE_LIBRARY_ROOT = Path(__file__).resolve().parents[1] / "factor_library"


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
    run_kind: str = "legacy"


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
        "strategy_evidence_run": "TEXT",
    }
    for column, definition in migrations.items():
        if column not in columns:
            connection.execute(f"ALTER TABLE factors ADD COLUMN {column} {definition}")
    connection.execute(
        "UPDATE factors SET source_type = 'legacy_run' WHERE source_type IS NULL OR source_type = ''"
    )
    connection.commit()
    return connection


def default_library_root() -> Path:
    """返回 Web 候选审核区使用的核心统一因子库。"""
    return CORE_LIBRARY_ROOT


def _factor_id(name: str, script_hash: str, run_identity: str = "") -> str:
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip(". ") or "factor"
    suffix = f"_{hashlib.sha256(run_identity.encode('utf-8')).hexdigest()[:12]}" if run_identity else ""
    return f"{safe_name}_{script_hash[:12]}{suffix}"


def _read_completed_run(run_dir: Path) -> CompletedRun:
    directory = run_dir.expanduser().resolve()
    if not directory.is_dir():
        raise LibraryError(f"回测运行目录不存在：{directory}")
    snapshot = directory / "factor_snapshot.py"
    if not snapshot.is_file():
        snapshot = directory / "strategy_snapshot.py"
    required = {
        "run_spec.json": directory / "run_spec.json",
        "metrics.json": directory / "metrics.json",
        "report.html": directory / "report.html",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise LibraryError(f"运行目录不是完整的成功回测，缺少：{', '.join(missing)}")
    if not snapshot.is_file():
        missing.append("factor_snapshot.py 或 strategy_snapshot.py")
    if missing:
        raise LibraryError(f"运行目录不是完整的成功回测，缺少：{', '.join(missing)}")
    if not ((directory / "returns.csv").exists() or (directory / "research_panel.csv").exists() or (directory / "factor_panel.csv").exists()):
        raise LibraryError("运行目录缺少策略或因子研究结果，不能作为批准证据")
    try:
        run_spec = json.loads(required["run_spec.json"].read_text(encoding="utf-8"))
        metrics = json.loads(required["metrics.json"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise LibraryError(f"运行目录中的配置或指标无效：{exc}") from exc
    if not isinstance(metrics, dict):
        raise LibraryError("运行目录中的 metrics.json 必须是对象")
    run_kind = str(run_spec.get("run_kind", "legacy")) if isinstance(run_spec, dict) else "legacy"
    if run_kind in {"factor", "strategy"}:
        name = str(run_spec.get("name") or directory.name)
        normalized_spec = json.dumps(run_spec, ensure_ascii=False, indent=2)
    else:
        try:
            spec = RunSpec.model_validate(run_spec)
        except ValueError as exc:
            raise LibraryError(f"历史 YAML 运行记录无效：{exc}") from exc
        name = spec.name
        normalized_spec = json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2)
    return CompletedRun(
        directory=directory,
        name=name,
        script_path=snapshot,
        script_hash=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        run_spec_json=normalized_spec,
        metrics_json=json.dumps(metrics, ensure_ascii=False, indent=2),
        run_kind=run_kind,
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
    idempotent: bool = False,
) -> dict[str, str]:
    script_hash = hashlib.sha256(script_content).hexdigest()
    config_hash = hashlib.sha256(run_spec_json.encode("utf-8")).hexdigest()
    factor_id = _factor_id(name, script_hash, source_run)
    artifact = library_root / "artifacts" / factor_id
    connection = _connection(library_root)
    try:
        if connection.execute("SELECT 1 FROM factors WHERE factor_id = ?", (factor_id,)).fetchone():
            if idempotent:
                return {
                    "factor_id": factor_id,
                    "artifact_path": str(artifact),
                    "status": "candidate",
                }
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
                    "has_reproducible_evidence": run_source is not None,
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


def create_candidate_from_run(
    run_dir: Path,
    library_root: Path,
    *,
    source_type: str = "completed_run",
    idempotent: bool = True,
) -> dict[str, str]:
    """从完整运行快照创建待人工审核的候选因子。"""
    run = _read_completed_run(run_dir)
    return _create_candidate(
        name=run.name,
        script_content=run.script_path.read_bytes(),
        run_spec_json=run.run_spec_json,
        source_type=source_type,
        source_run=str(run.directory),
        metrics_json=run.metrics_json,
        report_source=run.directory / "report.html",
        run_source=run.directory,
        library_root=library_root,
        idempotent=idempotent,
    )


def register_completed_run(run_dir: Path, library_root: Path | None = None) -> dict[str, str]:
    """将一条完整成功运行自动登记到核心候选审核区；对同一运行幂等。"""
    run = _read_completed_run(run_dir)
    if run.run_kind == "strategy":
        raise LibraryError("策略运行不会自动创建因子候选；请在候选审核区手工关联为交易证据")
    return create_candidate_from_run(
        run_dir,
        library_root or default_library_root(),
        source_type="automatic_run",
        idempotent=True,
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
        payload = load_yaml_mapping(config_content.decode("utf-8"), source="候选因子 YAML")
        spec = RunSpec.model_validate(payload)
    except (UnicodeDecodeError, ConfigLoadError, ValueError) as exc:
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


def attach_strategy_evidence(factor_id: str, strategy_run_dir: Path, library_root: Path) -> dict[str, str]:
    """把已完成策略运行作为候选因子的交易证据，不改变候选审批状态。"""
    run = _read_completed_run(strategy_run_dir)
    if run.run_kind != "strategy":
        raise LibraryError("只能关联原生 Python 策略运行作为交易证据")
    connection = _connection(library_root)
    try:
        row = connection.execute("SELECT * FROM factors WHERE factor_id = ?", (factor_id,)).fetchone()
        if row is None:
            raise LibraryError(f"未找到候选因子：{factor_id}")
        artifact = library_root / "artifacts" / factor_id
        evidence_dir = artifact / "strategy_evidence"
        _copy_backtest_results(run.directory, evidence_dir)
        shutil.copy2(run.script_path, evidence_dir / "strategy_snapshot.py")
        _write_json(
            evidence_dir / "metadata.json",
            {"strategy_run": str(run.directory), "strategy_script_hash": run.script_hash, "attached_at": datetime.now(UTC).isoformat()},
        )
        connection.execute("UPDATE factors SET strategy_evidence_run = ? WHERE factor_id = ?", (str(run.directory), factor_id))
        connection.commit()
    finally:
        connection.close()
    return {"factor_id": factor_id, "strategy_run": str(run.directory), "status": "evidence_attached"}


def promote(run_dir: Path, library_root: Path) -> dict[str, str]:
    """兼容旧 CLI/MCP：从运行创建候选并以该运行证据立即批准。"""
    candidate = create_candidate_from_run(run_dir, library_root)
    return approve_candidate(candidate["factor_id"], run_dir, library_root)


def list_factors(library_root: Path, status: str | None = None) -> list[dict[str, str | None]]:
    connection = _connection(library_root)
    try:
        query = """
            SELECT factor_id, name, status, created_at, approved_at, source_type, source_run,
                   evidence_run, strategy_evidence_run, script_hash, config_hash
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
