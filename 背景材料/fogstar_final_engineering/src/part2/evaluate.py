"""评价：单一阈值、按数据集×配置的等权宏平均、配对差值。

阈值规则（13.2 已确认）：0–1、步长 0.001，最小化「各单元等权平均敏感度」与
「各单元等权平均特异度」之差的绝对值，并列取最小阈值。
单类单元不参与阈值目标，但要在目标表里记 undefined（不静默换 0.5）。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, roc_auc_score)

from . import schema


def unit_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y = y.astype(int)
    z = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, z, labels=[0, 1]).ravel()
    out = {"n": int(len(y)), "pos": int(y.sum()),
           "sensitivity": float(tp / max(1, tp + fn)),
           "specificity": float(tn / max(1, tn + fp)),
           "precision": float(precision_score(y, z, zero_division=0)),
           "f1": float(f1_score(y, z, zero_division=0)),
           "accuracy": float((tp + tn) / len(y)),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
           "threshold": float(thr)}
    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = None
        out["auprc"] = None
    return out


def select_threshold(units: list[tuple[str, np.ndarray, np.ndarray]]) -> tuple[float, list[dict]]:
    """units: [(unit_name, y, p)]；返回阈值与逐阈值目标表（只记最优行与单元可用性）。"""
    usable, table = [], []
    for name, y, p in units:
        two = len(np.unique(y)) > 1
        table.append({"unit": name, "n": int(len(y)), "pos": int(y.sum()),
                      "two_class": bool(two)})
        if two:
            usable.append((y, p))
    if not usable:
        raise ValueError("no usable unit for threshold selection")
    best = (9e9, 0.5, None)
    for t in np.linspace(0.0, 1.0, 1001):
        sens, spec = [], []
        for y, p in usable:
            z = p >= t
            tp = int((z & (y == 1)).sum()); fn = int((~z & (y == 1)).sum())
            tn = int((~z & (y == 0)).sum()); fp = int((z & (y == 0)).sum())
            sens.append(tp / max(1, tp + fn)); spec.append(tn / max(1, tn + fp))
        d = abs(float(np.mean(sens)) - float(np.mean(spec)))
        if d < best[0] - 1e-12:
            best = (d, float(t), (float(np.mean(sens)), float(np.mean(spec))))
    return best[1], table


def macro(unit_results: dict[str, dict], keys=("auroc", "auprc", "f1", "sensitivity",
                                               "specificity", "accuracy")) -> dict:
    """对可定义单元等权平均；不可定义单元记入 missing 列表。"""
    out = {"n_units": len(unit_results)}
    for k in keys:
        vals = [v[k] for v in unit_results.values() if v.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
        out[f"{k}_n"] = len(vals)
    out["undefined_units"] = [u for u, v in unit_results.items() if v.get("auroc") is None]
    return out


def delta(new: dict, ref: dict, keys=("auroc", "auprc", "f1")) -> dict:
    return {k: (None if new.get(k) is None or ref.get(k) is None else new[k] - ref[k])
            for k in keys}
