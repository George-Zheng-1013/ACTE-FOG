"""Two lossless views of the very same tensor.

CNN gets (N, T, C); ExtraTrees gets the row-major flatten (N, T*C). The flatten
is a pure rearrangement, so the inverse reshape must reproduce T exactly.
"""
from __future__ import annotations

import numpy as np


def cnn_view(T: np.ndarray, ch_idx: list[int]) -> np.ndarray:
    return np.ascontiguousarray(T[:, :, ch_idx], dtype=np.float32)


def et_view(T: np.ndarray, ch_idx: list[int]) -> np.ndarray:
    x = np.ascontiguousarray(T[:, :, ch_idx], dtype=np.float32)
    return x.reshape(len(x), -1)


def roundtrip_check(T: np.ndarray, ch_idx: list[int], tol: float = 0.0) -> dict:
    """Prove that the ET flatten carries exactly the CNN information."""
    a = cnn_view(T, ch_idx)
    b = et_view(T, ch_idx)
    back = b.reshape(a.shape)
    diff = float(np.max(np.abs(back.astype(np.float64) - a.astype(np.float64))))
    return {
        "cnn_shape": list(a.shape),
        "et_shape": list(b.shape),
        "flatten_order": "row-major (time-major, channel-minor)",
        "max_abs_roundtrip_diff": diff,
        "exact": bool(diff <= tol),
    }
