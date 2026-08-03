from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd

from quantbacktest.api import DataDeclaration

RAW_COLUMNS = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "turnover",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


class DataContractError(ValueError):
    """可定位的数据输入错误。"""


def _digest(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        hasher.update(str(path.resolve()).encode())
        hasher.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return hasher.hexdigest()


def _canonicalize(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise DataContractError(f"数据缺少标准字段：{sorted(missing)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = symbol
    for column in ["open", "high", "low", "close", "volume", "turnover"]:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["timestamp", "symbol", "open", "high", "low", "close", "volume", "turnover"]]


def _load_crypto_top50(spec: DataDeclaration) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for symbol in spec.symbols:
        path = Path(spec.path) / f"{symbol}_1h.csv"
        if not path.exists():
            raise DataContractError(f"crypto_top50 未找到 {symbol}：{path}")
        raw = pd.read_csv(path)
        raw = raw.rename(columns={"volume_from": "volume", "volume_to": "turnover"})
        frames.append(_canonicalize(raw, symbol))
        paths.append(path)
    return pd.concat(frames, ignore_index=True), paths


def _load_bybit_parquet(spec: DataDeclaration) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for symbol in spec.symbols:
        candidates = sorted(Path(spec.path).glob(f"{symbol}_linear_1m/*.parquet"))
        if not candidates:
            raise DataContractError(f"Bybit 未找到 {symbol} 的 Parquet 文件")
        for path in candidates:
            raw = pd.read_parquet(path)
            if "timestamp" not in raw and "timestamp_ms" in raw:
                raw["timestamp"] = pd.to_datetime(raw["timestamp_ms"], unit="ms", utc=True)
            raw = raw.rename(columns={"turnover": "turnover"})
            frames.append(_canonicalize(raw, symbol))
            paths.append(path)
    return pd.concat(frames, ignore_index=True), paths


def _load_binance_zip(spec: DataDeclaration) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for symbol in spec.symbols:
        candidates = sorted((Path(spec.path) / symbol / spec.frequency).glob("*.zip"))
        if not candidates:
            raise DataContractError(f"Binance 未找到 {symbol}/{spec.frequency} 的压缩 K 线")
        for path in candidates:
            with zipfile.ZipFile(path) as archive:
                member = archive.namelist()[0]
                with archive.open(member) as handle:
                    raw = pd.read_csv(handle, names=RAW_COLUMNS, header=None)
            # Binance 压缩包既可能无表头，也可能带表头；表头若被当作数据会被
            # pandas 误解析成远未来时间，必须先将毫秒时间戳强制转为数值。
            raw["open_time_ms"] = pd.to_numeric(raw["open_time_ms"], errors="coerce")
            raw = raw.dropna(subset=["open_time_ms"])
            # 历史归档中同时存在毫秒和微秒时间戳；按数量级识别，避免把微秒
            # 误当毫秒而生成数万年后的伪造样本。
            unit = "us" if raw["open_time_ms"].median() >= 100_000_000_000_000 else "ms"
            raw["timestamp"] = pd.to_datetime(raw["open_time_ms"], unit=unit, utc=True)
            frames.append(_canonicalize(raw, symbol))
            paths.append(path)
    return pd.concat(frames, ignore_index=True), paths


def load_market_data(spec: DataDeclaration) -> tuple[pd.DataFrame, dict[str, object]]:
    """加载、规范化并过滤市场数据。"""
    if spec.adapter == "crypto_top50":
        frame, paths = _load_crypto_top50(spec)
    elif spec.adapter == "bybit_parquet":
        frame, paths = _load_bybit_parquet(spec)
    else:
        frame, paths = _load_binance_zip(spec)

    frame = frame.dropna(subset=["timestamp", "open", "close"])
    frame = frame.sort_values(["timestamp", "symbol"]).drop_duplicates(["timestamp", "symbol"])
    if spec.start:
        frame = frame[frame["timestamp"] >= pd.to_datetime(spec.start, utc=True)]
    if spec.end:
        frame = frame[frame["timestamp"] <= pd.to_datetime(spec.end, utc=True)]
    if frame.empty:
        raise DataContractError("日期过滤后没有可用于回测的数据")
    metadata = {
        "adapter": spec.adapter,
        "source_paths": [str(path) for path in paths],
        "data_fingerprint": _digest(paths),
        "rows": len(frame),
        "symbols": sorted(frame["symbol"].unique().tolist()),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
    }
    return frame.reset_index(drop=True), metadata


def available_assets(adapter: str, path: Path) -> dict[str, object]:
    """列出调用前可查询的数据资产和标准字段。"""
    if adapter == "crypto_top50":
        symbols = sorted(p.stem.replace("_1h", "") for p in path.glob("*_1h.csv"))
    elif adapter == "bybit_parquet":
        symbols = sorted(p.name.replace("_linear_1m", "") for p in path.glob("*_linear_1m"))
    elif adapter == "binance_zip":
        symbols = sorted(p.name for p in path.iterdir() if p.is_dir())
    else:
        raise DataContractError(f"未知 adapter：{adapter}")
    return {
        "adapter": adapter,
        "path": str(path),
        "symbols": symbols,
        "standard_fields": ["timestamp", "symbol", "open", "high", "low", "close", "volume", "turnover"],
    }
