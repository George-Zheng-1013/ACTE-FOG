"""Lossless views of the very same tensor.

CNN gets (N, T, C); ExtraTrees gets the row-major flatten (N, T*C). The flatten
is a pure rearrangement, so the inverse reshape must reproduce the input exactly.

If the missing-indicator module (P5) is enabled, both views additionally receive
the raw observation mask M for the selected channels, appended as a channel
block:  [signal channels ..., mask channels ...].  Both models get the same
array, so the extra information is identical by construction.
"""
from __future__ import annotations

import numpy as np


def _stack(T: np.ndarray, M: np.ndarray, ch_idx: list[int], with_mask: bool) -> np.ndarray:
    sig = np.ascontiguousarray(T[:, :, ch_idx], dtype=np.float32)
    if not with_mask:
        return sig
    mask = np.ascontiguousarray(M[:, :, ch_idx].astype(np.float32))
    return np.concatenate([sig, mask], axis=2)


def cnn_view(T: np.ndarray, M: np.ndarray, ch_idx: list[int], with_mask: bool = False) -> np.ndarray:
    return _stack(T, M, ch_idx, with_mask)


def et_view(T: np.ndarray, M: np.ndarray, ch_idx: list[int], with_mask: bool = False) -> np.ndarray:
    x = _stack(T, M, ch_idx, with_mask)
    return x.reshape(len(x), -1)


def roundtrip_check(T: np.ndarray, M: np.ndarray, ch_idx: list[int],
                    with_mask: bool = False, tol: float = 0.0) -> dict:
    """Prove that the ET flatten carries exactly the CNN information."""
    a = cnn_view(T, M, ch_idx, with_mask)
    b = et_view(T, M, ch_idx, with_mask)
    back = b.reshape(a.shape)
    diff = float(np.max(np.abs(back.astype(np.float64) - a.astype(np.float64))))
    return {
        "cnn_shape": list(a.shape),
        "et_shape": list(b.shape),
        "missing_indicator": bool(with_mask),
        "channel_order": "selected signal channels, then the same channels' observation mask",
        "flatten_order": "row-major (time-major, channel-minor)",
        "max_abs_roundtrip_diff": diff,
        "exact": bool(diff <= tol),
    }
