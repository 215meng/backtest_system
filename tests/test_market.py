from __future__ import annotations

import zipfile
from pathlib import Path

from quantbacktest.adapters import load_market_data
from quantbacktest.api import DataDeclaration


def test_binance_zip_ignores_optional_header_row(tmp_path: Path) -> None:
    directory = tmp_path / "BTCUSDT" / "1h"
    directory.mkdir(parents=True)
    archive = directory / "BTCUSDT-1h-2024-01.zip"
    content = (
        "open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,"
        "taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore\n"
        "1704067200000,100,101,99,100,1,1704070799999,100,1,1,100,0\n"
    )
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("BTCUSDT-1h-2024-01.csv", content)
    spec = DataDeclaration(
        adapter="binance_zip",
        path=str(tmp_path),
        market="spot",
        frequency="1h",
        symbols=["BTCUSDT"],
    )

    frame, metadata = load_market_data(spec)

    assert len(frame) == 1
    assert metadata["end"].startswith("2024-01-01")


def test_binance_zip_accepts_microsecond_archive_timestamps(tmp_path: Path) -> None:
    directory = tmp_path / "BTCUSDT" / "1h"
    directory.mkdir(parents=True)
    with zipfile.ZipFile(directory / "BTCUSDT-1h-2024-01.zip", "w") as output:
        output.writestr("rows.csv", "1704067200000000,100,101,99,100,1,1704070799999999,100,1,1,100,0\n")
    spec = DataDeclaration(
        adapter="binance_zip",
        path=str(tmp_path),
        market="spot",
        frequency="1h",
        symbols=["BTCUSDT"],
    )

    _, metadata = load_market_data(spec)

    assert metadata["start"].startswith("2024-01-01")
