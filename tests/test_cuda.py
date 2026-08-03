from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantbacktest.cuda import CudaExecutionError, create_cuda_factor_context, cuda_doctor
from quantbacktest.engine import run_backtest
from quantbacktest.factors import execute_factor_cuda
from quantbacktest.schemas import RunSpec


def test_compute_defaults_to_legacy_cpu_compatibility() -> None:
    spec = RunSpec.model_validate(
        {
            "name": "legacy_cpu",
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
    )
    assert spec.compute is None


def test_cuda_factor_contract_requires_declaration(tmp_path: Path) -> None:
    report = cuda_doctor()
    if not report["ready"]:
        pytest.skip("本机没有可用 CUDA，跳过 GPU 因子契约测试")
    factor_path = tmp_path / "cpu_only_factor.py"
    factor_path.write_text(
        "FactorMeta = {'required_fields': ['close'], 'min_lookback_bars': 1, "
        "'supported_modes': ['cross_sectional']}\n",
        encoding="utf-8",
    )
    context = create_cuda_factor_context(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
                "symbol": ["A", "A"],
                "close": [1.0, 2.0],
            }
        ),
        {},
        0,
    )
    with pytest.raises(CudaExecutionError, match="未声明 CUDA"):
        execute_factor_cuda(factor_path, "compute_factor", context)


def test_cuda_factor_contract_executes_cupy_output(tmp_path: Path) -> None:
    report = cuda_doctor()
    if not report["ready"]:
        pytest.skip("本机没有可用 CUDA，跳过 GPU 因子执行测试")
    factor_path = tmp_path / "cuda_factor.py"
    factor_path.write_text(
        "FactorMeta = {'required_fields': ['close'], 'min_lookback_bars': 1, "
        "'supported_modes': ['cross_sectional'], 'supported_backends': ['cuda']}\n"
        "def compute_factor_cuda(context):\n"
        "    return context.arrays['close'] * 2\n",
        encoding="utf-8",
    )
    context = create_cuda_factor_context(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
                "symbol": ["A", "A"],
                "close": [1.0, 2.0],
            }
        ),
        {},
        0,
    )

    output, metadata = execute_factor_cuda(factor_path, "compute_factor", context)

    assert output["factor"].tolist() == [2.0, 4.0]
    assert metadata["backend"] == "cuda"


def test_cuda_request_writes_failure_report_instead_of_running_cpu(tmp_path: Path) -> None:
    report = cuda_doctor()
    if not report["ready"]:
        pytest.skip("本机没有可用 CUDA，跳过 GPU 回测失败工件测试")
    data_path = tmp_path / "data"
    data_path.mkdir()
    for symbol_index in range(10):
        prices = [100.0 + symbol_index, 101.0 + symbol_index]
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume_from": [1.0, 1.0],
                "volume_to": [1.0, 1.0],
            }
        ).to_csv(data_path / f"S{symbol_index}_1h.csv", index=False)
    factor_path = tmp_path / "cuda_factor.py"
    factor_path.write_text(
        "FactorMeta = {'required_fields': ['close'], 'min_lookback_bars': 1, "
        "'supported_modes': ['cross_sectional'], 'supported_backends': ['cuda']}\n"
        "def compute_factor_cuda(context):\n"
        "    return context.arrays['close']\n",
        encoding="utf-8",
    )
    spec = RunSpec.model_validate(
        {
            "name": "cuda_strict_failure",
            "data": {
                "adapter": "crypto_top50",
                "path": str(data_path),
                "market": "spot",
                "frequency": "1h",
                "symbols": [f"S{index}" for index in range(10)],
            },
            "factor": {"module_path": str(factor_path)},
            "strategy": {
                "mode": "cross_sectional",
                "selection": "quantiles",
                "long_short": "long_only",
                "quantiles": 2,
            },
            "compute": {"backend": "cuda"},
            "output": {"root": str(tmp_path / "results")},
        }
    )

    with pytest.raises(CudaExecutionError, match="尚未完成"):
        run_backtest(spec, tmp_path)

    failure_dirs = list((tmp_path / "results").glob("*_cuda_strict_failure_cuda_failed"))
    assert len(failure_dirs) == 1
    assert (failure_dirs[0] / "cuda_failure.json").exists()
    assert (failure_dirs[0] / "report.html").exists()
