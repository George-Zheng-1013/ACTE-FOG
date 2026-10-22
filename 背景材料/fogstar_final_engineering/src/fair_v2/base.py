"""Shared data base: audit, fixed window population W, and the single processor.

Only this module is allowed to touch the raw CSV. Both model branches and the
hand-crafted feature derivation consume its output (T, M, W).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from . import schema

GROUP_KEYS = ["subjectID", "sessionID", "taskID"]


# --------------------------------------------------------------------------- #
# load + audit
# --------------------------------------------------------------------------- #
def load_sensor(csv: str | Path) -> pd.DataFrame:
    use = schema.CHANNELS + ["timestamp", "fog", "activity"] + GROUP_KEYS
    df = pd.read_csv(csv, usecols=use)
    return df.reset_index(drop=True)


def audit(df: pd.DataFrame) -> dict:
    """Facts used to freeze W. Nothing here may be chosen from test scores."""
    ts = df["timestamp"].to_numpy(np.float64)
    same = (
        (df.subjectID.values[1:] == df.subjectID.values[:-1])
        & (df.sessionID.values[1:] == df.sessionID.values[:-1])
        & (df.taskID.values[1:] == df.taskID.values[:-1])
    )
    step_all = np.diff(ts)
    step_in = step_all[same]
    step_edge = step_all[~same]
    med = float(np.median(step_in)) if len(step_in) else float("nan")
    vals = df[schema.CHANNELS].to_numpy(np.float64)
    fin = np.isfinite(vals)
    return {
        "rows": int(len(df)),
        "columns_used": len(schema.CHANNELS) + 5,
        "subjects": sorted(int(x) for x in df.subjectID.unique()),
        "record_groups": int(df.groupby(GROUP_KEYS, sort=True).ngroups),
        "timestamp": {
            "unit_inferred": "seconds",
            "inferred_from_median_in_group_step": med,
            "implied_fs_hz": float(round(1.0 / med, 6)) if med else None,
            "readme_claim": "milliseconds (contradicted by the median step)",
            "in_group_steps_unique_values": int(np.unique(np.round(step_in, 9)).size),
            "in_group_max_step": float(step_in.max()) if len(step_in) else None,
            "in_group_min_step": float(step_in.min()) if len(step_in) else None,
            "group_boundary_steps": int(len(step_edge)),
            "gaps_inside_group": int((step_in > 1.5 * med).sum()) if len(step_in) else 0,
            "conclusion": "no internal gap; each (subject,session,task) is one continuous segment",
        },
        "channel_missing_fraction": {
            c: float(1.0 - fin[:, i].mean()) for i, c in enumerate(schema.CHANNELS)
        },
        "rows_all_channels_missing": int((~fin).all(1).sum()),
        "label": {
            "fog_values": sorted(int(v) for v in pd.unique(df.fog.dropna())),
            "fog_nan": int(df.fog.isna().sum()),
            "activity_values": sorted(int(v) for v in pd.unique(df.activity.dropna())),
        },
        "group_length": {
            "min": int(df.groupby(GROUP_KEYS, sort=True).size().min()),
            "median": float(df.groupby(GROUP_KEYS, sort=True).size().median()),
            "max": int(df.groupby(GROUP_KEYS, sort=True).size().max()),
            "groups_shorter_than_window": int(
                (df.groupby(GROUP_KEYS, sort=True).size() < schema.WINDOW).sum()
            ),
        },
    }


# --------------------------------------------------------------------------- #
# window population W (identity only, independent of any processing config)
# --------------------------------------------------------------------------- #
def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    w, stride = schema.WINDOW, schema.STRIDE
    for seg_id, ((sub, ses, task), g) in enumerate(df.groupby(GROUP_KEYS, sort=True)):
        pos = g.index.to_numpy()
        n = len(g)
        if n < w:
            continue
        fog = df.fog.values
        act = df.activity.values
        ts = df.timestamp.values
        for st in range(0, n - w + 1, stride):
            win_rows = pos[st : st + w]
            n_pos = int(fog[win_rows].sum())
            av = act[win_rows]
            av = av[av > 0]
            rows.append(
                {
                    "window_id": f"sub{sub:02d}_ses{ses}_task{task}_st{st:05d}",
                    "segment_id": seg_id,
                    "subject": int(sub),
                    "session": int(ses),
                    "task": int(task),
                    "start_sample": int(st),
                    "end_sample": int(st + w - 1),
                    "start_row": int(win_rows[0]),
                    "end_row_exclusive": int(win_rows[-1]) + 1,
                    "start_ts": float(ts[win_rows[0]]),
                    "end_ts": float(ts[win_rows[-1]]),
                    "y": int(n_pos > 60),
                    "n_pos": n_pos,
                    "activity": int(np.bincount(av).argmax()) if len(av) else 0,
                }
            )
    win = pd.DataFrame(rows)
    win["split"] = np.select(
        [win.subject.isin(schema.SPLITS["train"]),
         win.subject.isin(schema.SPLITS["val"]),
         win.subject.isin(schema.SPLITS["test"])],
        ["train", "val", "test"], default="unassigned")
    return win


def annotate_validity(df: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """Rule 3 of protocol 2.3.3, decided on the raw observation mask only.

    A window is excluded when none of the 24 candidate channels has a single
    valid observation. Partial missingness is kept and handled by the processor.
    """
    raw = df[schema.CHANNELS].to_numpy(np.float64)
    fin = np.isfinite(raw)                      # Inf counts as missing
    w = schema.WINDOW
    nobs, cov = [], []
    for r in win.itertuples(index=False):
        m = fin[r.start_row : r.end_row_exclusive]
        nobs.append(int(m.sum()))
        cov.append(float(m.mean()))
    win = win.copy()
    win["n_valid_obs"] = nobs
    win["valid_obs_fraction"] = cov
    win["eligible"] = win.n_valid_obs > 0
    win["exclude_reason"] = np.where(win.eligible, "", "all_channels_missing_in_window")
    return win


# --------------------------------------------------------------------------- #
# the single processor
# --------------------------------------------------------------------------- #
def _filter_channel(x: np.ndarray, sos, min_run: int, stats: dict) -> np.ndarray:
    """Zero-phase band-pass inside every maximal finite run of one channel."""
    out = x.copy()
    idx = np.flatnonzero(np.isfinite(x))
    if len(idx) == 0:
        return out
    cuts = np.r_[0, np.where(np.diff(idx) > 1)[0] + 1, len(idx)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        run = idx[a:b]
        if len(run) < min_run:
            stats["short_runs_kept_raw"] += 1
            continue
        try:
            out[run] = sosfiltfilt(sos, x[run])
            stats["runs_filtered"] += 1
        except ValueError:
            stats["runs_failed_kept_raw"] += 1
    return out


def _train_scaling(Ttr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std fitted on train patients only, valid values only."""
    flat = Ttr.reshape(-1, Ttr.shape[-1])
    mu = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    mu = np.nan_to_num(mu, nan=0.0)
    bad = ~np.isfinite(sd) | (sd == 0)
    sd = np.where(bad, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def _fill_linear(T: np.ndarray, M: np.ndarray) -> tuple[int, int]:
    """Per-window, per-channel linear interpolation over time.

    Uses only the window's own valid samples; the edges extend the nearest valid
    value; a channel with no valid sample in the window stays NaN (filled with 0
    by the caller). Missing positions never become observations: M is untouched.
    """
    filled = 0
    hopeless = 0
    idx = np.arange(T.shape[1])
    for i in range(len(T)):
        m = M[i]
        for j in range(T.shape[2]):
            ok = m[:, j]
            if ok.all():
                continue
            t = T[i, :, j]
            if not ok.any():
                hopeless += int(len(ok))     # whole window missing: falls back to 0
                continue
            t[~ok] = np.interp(idx[~ok], idx[ok], t[ok].astype(np.float64))
            filled += int((~ok).sum())
    T[np.isnan(T)] = 0.0
    return filled, hopeless


def process(df: pd.DataFrame, windows: pd.DataFrame, p: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return T (N,120,24) float32, M (N,120,24) bool and processing stats."""
    t0 = time.time()
    vals = df[schema.CHANNELS].to_numpy(np.float64)
    vals[~np.isfinite(vals)] = np.nan          # Inf is treated as missing
    mask_all = np.isfinite(vals)

    stats = {"runs_filtered": 0, "short_runs_kept_raw": 0, "runs_failed_kept_raw": 0,
             "windows_all_missing_channel_cells": 0, "channels_all_missing_in_window": 0}

    filt = p["filter"]
    if filt["enabled"]:
        sos = butter(filt["order"], filt["band_hz"], btype="bandpass", fs=p["fs"], output="sos")
        out = np.empty_like(vals)
        for _, g in df.groupby(GROUP_KEYS, sort=True):
            pos = g.index.to_numpy()
            block = vals[pos]
            oblock = out[pos]
            for j in range(block.shape[1]):
                oblock[:, j] = _filter_channel(block[:, j], sos, filt["min_run_samples"], stats)
            out[pos] = oblock
        vals = out

    w = p["window"]
    n = len(windows)
    nch = len(schema.CHANNELS)
    # keep NaN as "missing" until the very last step so that neither demean nor
    # standardisation can mistake a filled value for an observation
    T = np.full((n, w, nch), np.nan, np.float32)
    M = np.zeros((n, w, nch), bool)

    for i, r in enumerate(windows.itertuples(index=False)):
        z = vals[r.start_row : r.end_row_exclusive].copy()
        m = mask_all[r.start_row : r.end_row_exclusive]
        M[i] = m
        if p["demean"]["enabled"]:
            cnt = m.sum(0)
            mu = np.zeros(nch)
            ok = cnt > 0
            mu[ok] = np.nanmean(z[:, ok], axis=0)
            stats["channels_all_missing_in_window"] += int((~ok).sum())
            z = z - mu
        T[i] = z

    if p["standardize"]["enabled"]:
        tr = (windows.split == "train").to_numpy()
        mu, sd = _train_scaling(T[tr])
        T = (T - mu) / sd
        stats["standardize"] = {
            "fit_on": "train patients (1-11), valid observations only",
            "mu_first_channels": [round(float(x), 6) for x in mu[:6]],
            "sd_first_channels": [round(float(x), 6) for x in sd[:6]],
            "channels_with_zero_sd": int((sd == 1.0).sum()),
        }

    fill = p["fill"]
    if fill["method"] == "linear":
        filled, hopeless = _fill_linear(T, M)
        stats["interpolated_cells"] = int(filled)
        stats["cells_in_all_missing_windows_filled_with_0"] = int(hopeless)
    else:
        T[np.isnan(T)] = fill["value"]

    stats["elapsed_s"] = round(time.time() - t0, 2)
    stats["T_finite"] = bool(np.isfinite(T).all())
    return T, M, stats


# --------------------------------------------------------------------------- #
# dataset assembly + cache
# --------------------------------------------------------------------------- #
def get_dataset(csv: str | Path, p: dict, cache_dir: str | Path) -> dict:
    """Load (or rebuild) the shared tensor for one processing config."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # the hash covers the signal chain only: smoothing is a post-process on
    # probabilities and must not fork the cached tensor
    phash = schema.short_hash({k: v for k, v in p.items() if k != "smoothing"})
    tpath, mpath = cache_dir / f"T_{phash}.npy", cache_dir / f"M_{phash}.npy"
    wpath, apath = cache_dir / "windows.csv", cache_dir / "data_audit.json"

    df = load_sensor(csv)
    if wpath.exists():
        windows = pd.read_csv(wpath)
        info = json.loads(apath.read_text(encoding="utf-8"))
    else:
        info = audit(df)
        win_all = annotate_validity(df, build_windows(df))
        excluded = win_all[~win_all.eligible]
        windows = win_all[win_all.eligible].reset_index(drop=True)
        windows.to_csv(wpath, index=False)
        excluded.to_csv(cache_dir / "exclusions.csv", index=False)
        apath.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    if tpath.exists() and mpath.exists():
        T = np.load(tpath)
        M = np.load(mpath)
        pstats = json.loads((cache_dir / f"procstats_{phash}.json").read_text(encoding="utf-8"))
    else:
        T, M, pstats = process(df, windows, p)
        np.save(tpath, T)
        np.save(mpath, M)
        (cache_dir / f"procstats_{phash}.json").write_text(
            json.dumps(pstats, indent=2), encoding="utf-8")

    return {"T": T, "M": M, "windows": windows, "audit": info,
            "proc_stats": pstats, "processing_hash": phash}


def sample_weights(y: np.ndarray) -> np.ndarray:
    """w_c = N_train / (2 * N_train,c); one vector shared by both models."""
    n = len(y)
    w = np.ones(n, np.float64)
    for c in (0, 1):
        m = y == c
        w[m] = n / (2.0 * max(1, int(m.sum())))
    return w
