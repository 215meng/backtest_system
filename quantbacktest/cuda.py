from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class CudaExecutionError(RuntimeError):
    """CUDA 请求无法满足时提供可报告的业务错误。"""

    def __init__(self, code: str, message: str, suggestion: str) -> None:
        super().__init__(message)
        self.code = code
        self.suggestion = suggestion

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self), "suggestion": self.suggestion}


@dataclass(frozen=True)
class CudaFactorContext:
    """提供给 ``compute_factor_cuda`` 的规则 GPU 面板。

    ``arrays`` 中每个字段均为 ``(time, symbol)`` 形状的 CuPy 数组；``rows``
    是同一面板按 timestamp、symbol 展平后的标签表。这样因子脚本可在 GPU 上
    沿时间轴计算滚动统计，而无需把 Pandas 对象传回 CPU。
    """

    arrays: dict[str, Any]
    rows: pd.DataFrame
    parameters: dict[str, Any]
    device_id: int
    timestamps: pd.DatetimeIndex
    symbols: tuple[str, ...]
    shape: tuple[int, int]


def cuda_doctor(device_id: int = 0) -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "backend": "cuda",
        "device_id": device_id,
        "ready": False,
        "checks": {},
    }
    try:
        import cupy as cp
        import numba
    except ImportError as exc:
        report["checks"]["python_cuda_packages"] = {"ok": False, "error": str(exc)}
        return report
    report["checks"]["python_cuda_packages"] = {
        "ok": True,
        "cupy": cp.__version__,
        "numba": numba.__version__,
    }
    try:
        count = cp.cuda.runtime.getDeviceCount()
        if device_id >= count:
            raise CudaExecutionError(
                "cuda_device_not_found",
                f"请求 GPU {device_id}，但系统只检测到 {count} 个 CUDA 设备",
                "检查 compute.device_id，或运行 quantbacktest cuda-doctor 查看可用设备。",
            )
        with cp.cuda.Device(device_id):
            properties = cp.cuda.runtime.getDeviceProperties(device_id)
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            probe = cp.arange(1024, dtype=cp.float64)
            float(probe.sum())
        report["checks"]["device"] = {
            "ok": True,
            "name": properties["name"].decode("utf-8"),
            "total_memory_bytes": int(total_bytes),
            "free_memory_bytes": int(free_bytes),
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        }
        try:
            import xgboost as xgb

            report["checks"]["xgboost"] = {
                "ok": bool(xgb.build_info().get("USE_CUDA")),
                "version": xgb.__version__,
            }
        except ImportError:
            report["checks"]["xgboost"] = {"ok": False, "error": "未安装 xgboost"}
    except CudaExecutionError as exc:
        report["checks"]["device"] = {"ok": False, **exc.as_dict()}
        return report
    except Exception as exc:  # noqa: BLE001 - CUDA runtime errors vary by driver and DLL installation.
        report["checks"]["device"] = {"ok": False, "error": str(exc)}
        return report
    report["ready"] = True
    return report


def require_cuda(device_id: int) -> dict[str, Any]:
    report = cuda_doctor(device_id)
    if not report["ready"]:
        detail = report["checks"].get("device") or report["checks"].get("python_cuda_packages", {})
        raise CudaExecutionError(
            detail.get("code", "cuda_unavailable"),
            detail.get("message", detail.get("error", "CUDA 环境不可用")),
            detail.get("suggestion", "安装 cupy-cuda12x 与 numba，并确认 NVIDIA 驱动及 GPU 可用。"),
        )
    return report


def create_cuda_factor_context(
    frame: pd.DataFrame, parameters: dict[str, Any], device_id: int
) -> CudaFactorContext:
    import cupy as cp

    required = {"timestamp", "symbol"}
    if missing := required - set(frame.columns):
        raise CudaExecutionError(
            "cuda_panel_labels_missing",
            f"CUDA 因子输入缺少标签字段：{sorted(missing)}",
            "数据适配器必须输出 timestamp 和 symbol 字段。",
        )
    ordered = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if ordered.duplicated(["timestamp", "symbol"]).any():
        raise CudaExecutionError(
            "cuda_panel_duplicate_row",
            "CUDA 面板存在重复的 timestamp × symbol 行。",
            "先在数据适配层去重，再运行 CUDA 回测。",
        )
    timestamps = pd.DatetimeIndex(ordered["timestamp"].drop_duplicates().sort_values())
    symbols = tuple(sorted(ordered["symbol"].astype(str).unique()))
    expected_rows = len(timestamps) * len(symbols)
    if len(ordered) != expected_rows:
        raise CudaExecutionError(
            "cuda_non_rectangular_panel",
            "CUDA 回测要求每个时间点都有同一组标的，当前 OHLCV 面板存在缺失行。",
            "补齐缺失 K 线，或在数据/资产池配置中先排除不完整标的；系统不会隐式用 CPU 处理该缺口。",
        )
    expected_index = pd.MultiIndex.from_product([timestamps, symbols], names=["timestamp", "symbol"])
    actual_index = pd.MultiIndex.from_frame(ordered[["timestamp", "symbol"]])
    if not actual_index.equals(expected_index):
        raise CudaExecutionError(
            "cuda_panel_alignment_error",
            "CUDA 面板无法按 UTC 时间与标的构成连续矩阵。",
            "确认 symbol 名称一致、timestamp 为 UTC，并对每个标的提供完整时间轴。",
        )
    numeric_columns = ordered.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        raise CudaExecutionError(
            "cuda_no_numeric_input",
            "因子输入没有可传输到 GPU 的数值字段",
            "在 FactorMeta.required_fields 中声明 close、open、volume 等数值字段。",
        )
    with cp.cuda.Device(device_id):
        arrays = {
            column: cp.asarray(ordered[column].to_numpy(dtype=np.float64, copy=False)).reshape(
                len(timestamps), len(symbols)
            )
            for column in numeric_columns
        }
    rows = ordered[["timestamp", "symbol"]].copy()
    return CudaFactorContext(
        arrays=arrays,
        rows=rows,
        parameters=parameters,
        device_id=device_id,
        timestamps=timestamps,
        symbols=symbols,
        shape=(len(timestamps), len(symbols)),
    )


def cuda_factor_output(values: Any, context: CudaFactorContext) -> pd.DataFrame:
    import cupy as cp

    with cp.cuda.Device(context.device_id):
        output = cp.asarray(values)
        if tuple(output.shape) not in {context.shape, (len(context.rows),)}:
            raise CudaExecutionError(
                "cuda_factor_shape_mismatch",
                f"compute_factor_cuda 输出形状为 {tuple(output.shape)}，应为 {context.shape} 或 {(len(context.rows),)}",
                "返回与 context.arrays['close'] 同形状的二维 CuPy 数组，或按 context.rows 顺序展平的一维数组。",
            )
        output = output.reshape(-1)
        if output.size != len(context.rows):
            raise CudaExecutionError(
                "cuda_factor_shape_mismatch",
                f"compute_factor_cuda 输出长度为 {output.size}，应为 {len(context.rows)}",
                "返回与 context.rows 行序一一对应的一维 CuPy 因子数组。",
            )
        factor = cp.asnumpy(output)
    result = context.rows.copy()
    result["factor"] = factor
    return result
