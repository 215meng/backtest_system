from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from quantbacktest.cuda import CudaExecutionError, CudaFactorContext, cuda_factor_output


@dataclass(frozen=True)
class FactorContext:
    data: pd.DataFrame
    parameters: dict[str, Any]


class FactorContractError(ValueError):
    """可定位的因子脚本契约错误。"""


def load_factor_module(path: Path) -> ModuleType:
    resolved = path.resolve()
    if not resolved.exists():
        raise FactorContractError(f"未找到因子脚本：{resolved}")
    spec = importlib.util.spec_from_file_location(f"quantbacktest_factor_{hash(resolved)}", resolved)
    if spec is None or spec.loader is None:
        raise FactorContractError(f"无法加载因子脚本：{resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_factor(path: Path) -> dict[str, Any]:
    module = load_factor_module(path)
    meta = getattr(module, "FactorMeta", None)
    if meta is None:
        raise FactorContractError("因子脚本必须定义 FactorMeta")
    if not isinstance(meta, dict):
        raise FactorContractError("FactorMeta 必须是字典")
    required = {"required_fields", "min_lookback_bars", "supported_modes"}
    missing = required - set(meta)
    if missing:
        raise FactorContractError(f"FactorMeta 缺少字段：{sorted(missing)}")
    return meta


def execute_factor_cuda(
    path: Path, callable_name: str, context: CudaFactorContext
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """执行明确声明的 GPU 因子；不允许回退到 compute_factor。"""
    module = load_factor_module(path)
    meta = inspect_factor(path)
    supported_backends = meta.get("supported_backends", [])
    if "cuda" not in supported_backends:
        raise CudaExecutionError(
            "cuda_factor_not_declared",
            f"因子脚本未声明 CUDA 支持：{path}",
            '在 FactorMeta 中添加 supported_backends: ["cuda"]，并实现 compute_factor_cuda(context)。',
        )
    cuda_callable = f"{callable_name}_cuda"
    function = getattr(module, cuda_callable, None)
    if not callable(function):
        raise CudaExecutionError(
            "cuda_factor_callable_missing",
            f"因子脚本缺少 {cuda_callable}(context)",
            "实现该函数并返回与输入行一一对应的一维 CuPy 因子数组。",
        )
    output = cuda_factor_output(function(context), context)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return output, {
        "meta": meta,
        "factor_fingerprint": digest,
        "module_path": str(path.resolve()),
        "backend": "cuda",
        "callable": cuda_callable,
    }


def execute_factor(path: Path, callable_name: str, context: FactorContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    module = load_factor_module(path)
    meta = inspect_factor(path)
    required_fields = set(meta["required_fields"])
    missing = required_fields - set(context.data.columns)
    if missing:
        raise FactorContractError(f"因子依赖的数据字段不存在：{sorted(missing)}")
    function = getattr(module, callable_name, None)
    if not callable(function):
        raise FactorContractError(f"因子脚本没有可调用函数：{callable_name}")
    output = function(context)
    if isinstance(output, pd.Series):
        output = output.rename("factor").reset_index()
    if not isinstance(output, pd.DataFrame):
        raise FactorContractError("compute_factor 必须返回 pandas DataFrame 或 Series")
    required_output = {"timestamp", "symbol", "factor"}
    missing_output = required_output - set(output.columns)
    if missing_output:
        raise FactorContractError(f"因子输出缺少字段：{sorted(missing_output)}")
    output = output[["timestamp", "symbol", "factor"]].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    output["factor"] = pd.to_numeric(output["factor"], errors="coerce")
    output = output.drop_duplicates(["timestamp", "symbol"]).sort_values(["timestamp", "symbol"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return output, {"meta": meta, "factor_fingerprint": digest, "module_path": str(path.resolve())}
