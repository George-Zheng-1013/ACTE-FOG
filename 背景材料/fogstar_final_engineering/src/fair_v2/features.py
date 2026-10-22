"""Hand-crafted 132-dim representation, derived once from the shared T and M.

Block layout (protocol 2.6.2):
    std 24 | rms 24 | mad 24 | freeze-index 24 | missing-ratio 24 | coordination 12
`mad` is the mean absolute first difference; `freeze-index` is the log ratio of
3-8 Hz power to 0.5-3 Hz power. All six blocks come from the same demeaned,
zero-filled tensor, so no branch redoes filtering or demeaning.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq

from . import schema

LOCo = (0.5, 3.0)
FREEZE = (3.0, 8.0)

BLOCKS = {
    "std": (0, 24),
    "rms": (24, 48),
    "mad": (48, 72),
    "freeze": (72, 96),
    "missing": (96, 120),
    "coord": (120, 132),
}

FEATURE_NAMES = (
    [f"std_{c}" for c in schema.CHANNELS]
    + [f"rms_{c}" for c in schema.CHANNELS]
    + [f"mad_{c}" for c in schema.CHANNELS]
    + [f"freeze_{c}" for c in schema.CHANNELS]
    + [f"missing_{c}" for c in schema.CHANNELS]
    + [f"coord_{t}_{i}_{j}" for t in schema.TYPES
       for i in range(4) for j in range(i + 1, 4)]
)
assert len(FEATURE_NAMES) == 132


def _coord(z: np.ndarray) -> np.ndarray:
    """Pearson correlation between whole-body magnitudes, per acc/gyro type.

    z: (N, T, 24). Magnitudes are taken per body node over its three axes,
    exactly as the reference implementation does.
    """
    n = len(z)
    out = np.empty((n, 12), np.float32)
    for ti in range(2):
        mags = np.empty((n, 4, z.shape[1]), np.float32)
        for pi in range(4):
            base = pi * 6 + ti * 3
            seg = z[:, :, base:base + 3].astype(np.float64)
            mags[:, pi] = np.sqrt((seg ** 2).sum(2))
        m = mags - mags.mean(2, keepdims=True)
        sd = m.std(2, keepdims=True)
        sd[sd == 0] = 1.0                       # constant magnitude -> zero corr
        m = m / sd
        c = np.einsum("nat,nbt->nab", m, m) / z.shape[1]
        iu = np.triu_indices(4, 1)
        out[:, ti * 6:(ti + 1) * 6] = c[:, iu[0], iu[1]]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_features(T: np.ndarray, M: np.ndarray, fs: float = schema.FS) -> np.ndarray:
    z = T.astype(np.float32)
    sd = z.std(1)
    rms = np.sqrt((z ** 2).mean(1))
    mad = np.abs(np.diff(z, axis=1)).mean(1)
    spec = np.abs(rfft(z, axis=1)) ** 2
    f = rfftfreq(z.shape[1], d=1.0 / fs)
    loco = (f >= LOCo[0]) & (f < LOCo[1])
    fre = (f >= FREEZE[0]) & (f < FREEZE[1])
    pl = spec[:, loco, :].sum(1)
    pf = spec[:, fre, :].sum(1)
    fi = np.log1p(pf / (pl + 1e-6))
    miss = 1.0 - M.mean(1)
    F = np.concatenate([sd, rms, mad, fi, miss.astype(np.float32), _coord(z)], axis=1)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    assert F.shape[1] == 132 and np.isfinite(F).all()
    return F


def indices_without(*drop: str) -> list[int]:
    keep = [i for name, (a, b) in BLOCKS.items() if name not in drop
            for i in range(a, b)]
    return keep


def acte72_indices() -> list[int]:
    """The original ACTE selection: 12 ankle channels of every block + coord."""
    idx = []
    for off in (0, 24, 48, 72, 96):
        idx += [off + i for i in range(12)]
    idx += list(range(120, 132))
    return idx
