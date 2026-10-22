"""Shared evaluation: one threshold rule, one smoother, one metric function.

No model may invent its own threshold or post-processing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, roc_auc_score)

from . import schema

CONTEXT_KEYS = ["subject", "session", "task"]


def eer_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Minimise |sensitivity - specificity| on a 0..1 grid of step 0.001.

    Ties keep the smallest threshold, as in the reference implementation.
    """
    if len(np.unique(y)) < 2:
        raise ValueError("validation set has a single class; threshold undefined")
    best = (9.0, 0.5)
    for t in np.linspace(0.0, 1.0, 1001):
        z = p >= t
        tp = int((z & (y == 1)).sum()); fn = int((~z & (y == 1)).sum())
        tn = int((~z & (y == 0)).sum()); fp = int((z & (y == 0)).sum())
        d = abs(tp / max(1, tp + fn) - tn / max(1, tn + fp))
        if d < best[0]:
            best = (d, float(t))
    return best[1]


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y = y.astype(int)
    z = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, z, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "threshold": float(thr),
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "precision": float(precision_score(y, z, zero_division=0)),
        "f1": float(f1_score(y, z, zero_division=0)),
        "accuracy": float((tp + tn) / len(y)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = None
        out["auprc"] = None
    return out


def smooth_segments(p: np.ndarray, windows: pd.DataFrame, k: int = 7) -> np.ndarray:
    """Centered moving average inside real segments only.

    Segments are broken at patient/session/task boundaries and at any window
    gap (start_sample step different from the protocol stride). Edges use the
    actually available neighbours, so segments shorter than k stay well defined.
    """
    out = p.copy()
    half = k // 2
    win = windows.reset_index(drop=True)
    for _, idx in win.groupby(CONTEXT_KEYS, sort=False).indices.items():
        idx = np.asarray(idx)
        idx = idx[np.argsort(win.start_sample.values[idx])]
        st = win.start_sample.values[idx]
        brk = np.r_[0, np.flatnonzero(np.diff(st) != schema.STRIDE) + 1, len(idx)]
        for a, b in zip(brk[:-1], brk[1:]):
            seg = idx[a:b]
            q = p[seg]
            n = len(q)
            cs = np.r_[0.0, np.cumsum(q)]
            r = np.empty(n)
            for i in range(n):
                lo, hi = max(0, i - half), min(n, i + half + 1)
                r[i] = (cs[hi] - cs[lo]) / (hi - lo)
            out[seg] = r
    return out


def prediction_frame(windows: pd.DataFrame, mask: np.ndarray, p_raw: np.ndarray,
                     p_eval: np.ndarray, thr: float, meta: dict) -> pd.DataFrame:
    w = windows.loc[mask].reset_index(drop=True)
    df = pd.DataFrame({
        "window_id": w.window_id,
        "segment_id": w.segment_id,
        "subject": w.subject, "session": w.session, "task": w.task,
        "start_sample": w.start_sample, "end_sample": w.end_sample,
        "split": w.split, "y_true": w.y, "activity": w.activity,
        "p_raw": p_raw, "p_eval": p_eval, "threshold": thr,
        "y_pred": (p_eval >= thr).astype(int),
    })
    for k, v in meta.items():
        df[k] = v
    return df


def delta(ref: dict, new: dict, keys=("auroc", "auprc", "f1", "sensitivity", "specificity")) -> dict:
    out = {}
    for k in keys:
        a, b = ref.get(k), new.get(k)
        out[k] = None if (a is None or b is None) else float(b - a)
    return out
