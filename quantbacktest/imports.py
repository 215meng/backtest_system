from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import yaml

from quantbacktest.schemas import RunSpec


@dataclass(frozen=True)
class PreparedImportedRun:
    spec: RunSpec
    source_root: Path
    factor_sha256: str
    normalized_yaml: str


def _as_utc_datetime(value: date, *, end_of_day: bool) -> datetime:
    value_time = time.max if end_of_day else time.min
    return datetime.combine(value, value_time, tzinfo=UTC)


def prepare_imported_run(
    *,
    factor_name: str,
    factor_content: bytes,
    config_content: bytes,
    source_root: str | Path,
    start_date: date | None = None,
    end_date: date | None = None,
) -> PreparedImportedRun:
    """Create a validated run specification from trusted files uploaded in the Web UI."""
    root_input = Path(source_root).expanduser()
    if not root_input.is_absolute() or not root_input.is_dir():
        raise ValueError("源项目目录必须是存在的绝对本地目录")
    root = root_input.resolve()
    if Path(factor_name).suffix.lower() != ".py":
        raise ValueError("因子文件必须是 .py Python 脚本")
    try:
        payload: Any = yaml.safe_load(config_content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取 YAML 配置：{exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("YAML 配置的根节点必须是对象")

    data = payload.get("data")
    if not isinstance(data, dict) or not data.get("path"):
        raise ValueError("YAML 配置必须提供 data.path")
    data_path = Path(data["path"])
    data["path"] = str(data_path if data_path.is_absolute() else (root / data_path).resolve())

    cache_dir = root / "results" / "backtests" / ".import_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    factor_sha256 = hashlib.sha256(factor_content).hexdigest()
    safe_name = Path(factor_name).name
    cached_factor = cache_dir / f"{factor_sha256}_{safe_name}"
    if not cached_factor.exists():
        cached_factor.write_bytes(factor_content)

    factor = payload.setdefault("factor", {})
    if not isinstance(factor, dict):
        raise TypeError("YAML 的 factor 必须是对象")
    factor["module_path"] = str(cached_factor)
    payload["output"] = {"root": str(root / "results" / "backtests")}
    if start_date is not None:
        data["start"] = _as_utc_datetime(start_date, end_of_day=False).isoformat()
    if end_date is not None:
        data["end"] = _as_utc_datetime(end_date, end_of_day=True).isoformat()

    spec = RunSpec.model_validate(payload)
    normalized_yaml = yaml.safe_dump(
        spec.model_dump(mode="json"), allow_unicode=True, sort_keys=False
    )
    return PreparedImportedRun(
        spec=spec,
        source_root=root,
        factor_sha256=factor_sha256,
        normalized_yaml=normalized_yaml,
    )
