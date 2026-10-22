"""Fixed vocabulary of the fair_ctrl_v2 protocol.

Anything that defines *what the data is* lives here so that the shared
processor, both model branches and the feature derivation cannot drift apart.
"""
from __future__ import annotations

import hashlib
import json

POS = ["ankleL", "ankleR", "back", "wrist"]
TYPES = ["acc", "gyro"]
AX = list("xyz")

# 24 candidate channels, canonical order: position -> signal type -> axis
CHANNELS = [f"{p}_{t}_{a}" for p in POS for t in TYPES for a in AX]
CH_INDEX = {c: i for i, c in enumerate(CHANNELS)}

# Patient-level split, fixed for every experiment (protocol 2.3.1)
SPLITS = {
    "train": list(range(1, 12)),
    "val": list(range(12, 17)),
    "test": list(range(17, 23)),
}

# Sensor input sets of protocol 2.6.1, expressed as channel names
def _ch(pos, types=("acc", "gyro")):
    return [f"{p}_{t}_{a}" for p in pos for t in types for a in AX]


SENSOR_SETS = {
    "S0": _ch(["ankleL"], ("gyro",)),                      # 3
    "S1": _ch(["ankleL", "ankleR"], ("gyro",)),            # 6
    "S2": _ch(["ankleL", "ankleR"]),                       # 12
    "S3": _ch(["ankleL", "ankleR", "back"]),               # 18
    "S4": _ch(["ankleL", "ankleR", "wrist"]),              # 18
    "S5": _ch(["ankleL", "ankleR", "back", "wrist"]),      # 24
}

WINDOW = 120            # samples per window (2 s at 60 Hz)
STRIDE = 30             # hop
FS = 60.0
POS_TIE_RULE = "pos>60 -> 1 else 0"   # protocol 2.3.1


def channel_indices(sensor_set: str) -> list[int]:
    return [CH_INDEX[c] for c in SENSOR_SETS[sensor_set]]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def short_hash(obj, n: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]


def array_hash(arr) -> str:
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(memoryview(arr.tobytes()))
    return h.hexdigest()[:16]


P_REF = {
    "fs": FS,
    "window": WINDOW,
    "stride": STRIDE,
    "filter": {
        "enabled": True,
        "design": "butterworth-bandpass-sos",
        "order": 4,
        "band_hz": [0.5, 20.0],
        "phase": "zero-phase (sosfiltfilt)",
        "scope": "per channel, per maximal finite run inside one record group",
        "min_run_samples": 31,
        "short_run_policy": "keep raw samples unfiltered",
    },
    "demean": {
        "enabled": True,
        "scope": "per window, per channel",
        "uses": "valid observations only",
        "all_missing_channel_default": 0.0,
    },
    "standardize": {"enabled": False, "fit_on": "train patients only"},
    "fill": {"method": "constant", "value": 0.0, "position": "after demean/standardize"},
    "missing_indicator": {"enabled": False},
    "smoothing": {"enabled": False},
    "label": {"rule": POS_TIE_RULE, "source": "fog column"},
}
