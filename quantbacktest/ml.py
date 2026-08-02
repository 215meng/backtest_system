from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from quantbacktest.schemas import MLSpec


class MLContractError(ValueError):
    """时间切分或模型训练不满足复现要求。"""


def train_oos_model(frame: pd.DataFrame, spec: MLSpec, artifact_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """仅用早期时间段训练，在后续样本外窗口产生替代因子分数。"""
    if not spec.enabled or spec.model is None:
        return frame, {"enabled": False}
    usable = frame.dropna(subset=["factor", "forward_return"]).sort_values("timestamp").copy()
    timestamps = usable["timestamp"].drop_duplicates().sort_values().tolist()
    if len(timestamps) < 20:
        raise MLContractError("ML 训练至少需要 20 个有效时间截面")
    split = int(len(timestamps) * 0.7)
    train_end = timestamps[split - 1]
    train = usable[usable["timestamp"] <= train_end]
    test = usable[usable["timestamp"] > train_end]
    if train.empty or test.empty:
        raise MLContractError("时间训练/测试切分为空")
    parameters = {"n_estimators": 100, "random_state": 42, "n_jobs": 1, **spec.parameters}
    if spec.model == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(**parameters)
    else:
        from xgboost import XGBRegressor

        model = XGBRegressor(**parameters)
    model.fit(train[["factor"]], train["forward_return"])
    result = frame.copy()
    result["raw_factor"] = result["factor"]
    result["factor"] = pd.NA
    test_index = result["timestamp"] > train_end
    result.loc[test_index, "factor"] = model.predict(result.loc[test_index, ["raw_factor"]].fillna(0.0))
    with artifact_path.open("wb") as handle:
        pickle.dump(model, handle)
    return result, {
        "enabled": True,
        "model": spec.model,
        "train_end": train_end.isoformat(),
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_columns": ["factor"],
        "random_state": parameters["random_state"],
        "artifact": artifact_path.name,
    }
