"""共享数据底座：适配、审计、质量判定、唯一处理器。

与 Part 1 的差别（有意为之，不改旧函数）：
  * 允许缺列/缺组：映射到固定 24 槽位，不压缩列；
  * 缺失不再填 0：不可用组整体记 NaN，原始观测掩码单独保存；
  * 窗口总体由「至少一侧踝 gyro 组质量合格」决定，而不是「24 通道任一有值」。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from . import schema

GROUP_KEYS = ["subject", "session", "task"]

# FoG-STAR 的列名 -> Part 2 固定槽位
FOGSTAR_MAP = {}
for _n in ("ankleL", "ankleR", "back", "wrist"):
    for _s in ("acc", "gyro"):
        for _a in ("x", "y", "z"):
            _dst = "wrist_affected" if _n == "wrist" else _n
            FOGSTAR_MAP[f"{_n}_{_s}_{_a}"] = f"{_dst}_{_s}_{_a}"


def load_fogstar(csv: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(csv, usecols=list(FOGSTAR_MAP) + ["timestamp", "fog", "activity",
                                                        "subjectID", "sessionID", "taskID"])
    df = pd.DataFrame(index=raw.index)
    for src, dst in FOGSTAR_MAP.items():
        df[dst] = raw[src]
    df["timestamp"] = raw["timestamp"]
    df["fog"] = raw["fog"]
    df["activity"] = raw["activity"]
    df["subject"] = raw["subjectID"].astype(int)
    df["session"] = raw["sessionID"].astype(int)
    df["task"] = raw["taskID"].astype(int)
    # 全部 24 槽位到齐（FoG-STAR 四部位齐全），顺序按 schema
    return df[schema.CHANNELS + ["timestamp", "fog", "activity"] + GROUP_KEYS].reset_index(drop=True)


def audit(df: pd.DataFrame, dataset_id: str = "fogstar") -> dict:
    ts = df.timestamp.to_numpy(float)
    same = ((df.subject.values[1:] == df.subject.values[:-1])
            & (df.session.values[1:] == df.session.values[:-1])
            & (df.task.values[1:] == df.task.values[:-1]))
    step = np.diff(ts)
    in_step = step[same]
    vals = df[schema.CHANNELS].to_numpy(np.float64)
    fin = np.isfinite(vals)
    out = {
        "dataset_id": dataset_id, "protocol_version": schema.PROTOCOL_VERSION,
        "rows": int(len(df)),
        "subjects": sorted(int(x) for x in df.subject.unique()),
        "groups": int(df.groupby(GROUP_KEYS, sort=True).ngroups),
        "columns_present": int(fin.shape[1]),
        "timestamp": {
            "unit": "seconds",
            "median_in_group_step": float(np.median(in_step)) if len(in_step) else None,
            "implied_fs": float(round(1.0 / np.median(in_step), 6)) if len(in_step) else None,
            "in_group_gaps": int((in_step > 1.5 * np.median(in_step)).sum()) if len(in_step) else 0,
        },
        "channel_missing_fraction": {c: float(1 - fin[:, i].mean())
                                     for i, c in enumerate(schema.CHANNELS)},
        "group_missing_fraction": {
            f"{n}_{s}": float(1 - fin[:, schema.group_slice(n, s)].min(1).mean())
            for n, s in schema.GROUPS},
        "label": {
            "fog_values": sorted(int(v) for v in pd.unique(df.fog.dropna())),
            "fog_nan": int(df.fog.isna().sum()),
            "note": "FoG-STAR 无 unknown 区域；WearGait 的 unlabeled 处理待接入",
        },
    }
    return out


def build_windows(df: pd.DataFrame, dataset_id: str = "fogstar") -> pd.DataFrame:
    rows = []
    w, stride = schema.WINDOW, schema.STRIDE
    fog = df.fog.to_numpy()
    ts = df.timestamp.to_numpy()
    for seg_id, ((sub, ses, task), g) in enumerate(df.groupby(GROUP_KEYS, sort=True)):
        pos = g.index.to_numpy()
        if len(g) < w:
            continue
        for st in range(0, len(g) - w + 1, stride):
            idx = pos[st:st + w]
            n_pos = int(np.nansum(fog[idx]))
            n_known = int(np.isfinite(fog[idx]).sum())
            rows.append({
                "window_id": f"{dataset_id}_s{sub:02d}_e{ses}_t{task}_st{st:05d}",
                "dataset_id": dataset_id, "subject": int(sub), "session": int(ses),
                "task": int(task), "segment_id": seg_id,
                "start_sample": int(st), "end_sample": int(st + w - 1),
                "start_row": int(idx[0]), "end_row_exclusive": int(idx[-1]) + 1,
                "start_ts": float(ts[idx[0]]), "end_ts": float(ts[idx[-1]]),
                "n_pos": n_pos, "label_valid": bool(n_known == w),
            })
    win = pd.DataFrame(rows)
    win["y"] = np.where(win.label_valid, (win.n_pos > 60).astype(int), -1)
    win["split"] = np.select(
        [win.subject.isin(schema.SPLITS["train"]), win.subject.isin(schema.SPLITS["val"]),
         win.subject.isin(schema.SPLITS["test"])],
        ["train", "val", "test"], default="unassigned")
    return win


def _filter_channel(x: np.ndarray, sos, min_run: int, stats: dict) -> np.ndarray:
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
            out[run] = sosfiltfilt(sos, x[run]); stats["runs_filtered"] += 1
        except ValueError:
            stats["runs_failed_kept_raw"] += 1
    return out


def _group_quality(m: np.ndarray) -> tuple[bool, dict]:
    """m: (120,3) 原始观测掩码。按 Q10 判定该三轴组是否可用。"""
    cov = m.mean(0)
    min_cov = float(cov.min())
    gaps, edges_ok = 0, bool(m[0].all() and m[-1].all())
    for j in range(3):
        miss = ~m[:, j]
        if not miss.any():
            continue
        d = np.diff(np.r_[0, miss.astype(int), 0])
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(starts):
            gaps = max(gaps, int((ends - starts).max()))
    ok = (min_cov >= schema.QUALITY["min_coverage"]
          and gaps <= schema.QUALITY["max_internal_gap"]
          and edges_ok)
    return ok, {"min_coverage": min_cov, "max_gap": gaps, "edges_ok": edges_ok}


def _repair_short_gaps(z: np.ndarray, m: np.ndarray, max_gap: int) -> np.ndarray:
    """在已去均值的窗口内，对内部短缺口做线性插值；不改 m。"""
    out = z.copy()
    t = np.arange(len(out))
    for j in range(out.shape[1]):
        ok = m[:, j]
        if ok.all() or not ok.any():
            continue
        miss = ~ok
        d = np.diff(np.r_[0, miss.astype(int), 0])
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        for a, b in zip(starts, ends):
            if (b - a) <= max_gap:
                out[a:b, j] = np.interp(t[a:b], t[ok], out[ok, j])
    return out


def process(df: pd.DataFrame, win: pd.DataFrame) -> dict:
    t0 = time.time()
    vals = df[schema.CHANNELS].to_numpy(np.float64)
    vals[~np.isfinite(vals)] = np.nan
    mask_raw_all = np.isfinite(vals)

    stats = {"runs_filtered": 0, "short_runs_kept_raw": 0, "runs_failed_kept_raw": 0,
             "windows_total": int(len(win)), "windows_excluded_insufficient": 0,
             "groups_unusable": 0, "channels_repaired": 0}
    f = schema.FILTER
    sos = butter(f["order"], f["band_hz"], btype="bandpass", fs=schema.FS, output="sos")
    filt = np.full_like(vals, np.nan)
    for _, g in df.groupby(GROUP_KEYS, sort=True):
        pos = g.index.to_numpy()
        block = vals[pos]
        oblock = filt[pos]
        for j in range(block.shape[1]):
            oblock[:, j] = _filter_channel(block[:, j], sos, f["min_run"], stats)
        filt[pos] = oblock

    n = len(win)
    T = np.full((n, schema.WINDOW, len(schema.CHANNELS)), np.nan, np.float32)
    M = np.zeros((n, schema.WINDOW, len(schema.CHANNELS)), bool)
    G = np.zeros((n, len(schema.GROUPS)), bool)
    cov_mat = np.zeros((n, len(schema.GROUPS)), np.float32)
    gap_mat = np.zeros((n, len(schema.GROUPS)), np.int16)
    eligible, reasons = [], []

    for i, r in enumerate(win.itertuples(index=False)):
        z = filt[r.start_row:r.end_row_exclusive].copy()
        m = mask_raw_all[r.start_row:r.end_row_exclusive]
        M[i] = m
        # 1) 质量判定（先于任何填补，也先于人工遮蔽）
        for gi, (node, sig) in enumerate(schema.GROUPS):
            sl = schema.group_slice(node, sig)
            ok, info = _group_quality(m[:, sl])
            G[i, gi] = ok
            cov_mat[i, gi] = info["min_coverage"]
            gap_mat[i, gi] = info["max_gap"]
            if not ok:
                stats["groups_unusable"] += 1
        # 2) 窗内去均值（逐通道、仅有效观测）——对所有通道做，后续不可用组会被置 NaN
        cnt = m.sum(0)
        mu = np.zeros(z.shape[1])
        has = cnt > 0
        mu[has] = np.nanmean(z[:, has], axis=0)
        z = z - mu
        # 3) 短缺口修复（仅限合格组）
        for gi, (node, sig) in enumerate(schema.GROUPS):
            if not G[i, gi]:
                continue
            sl = schema.group_slice(node, sig)
            before = np.isnan(z[:, sl]).sum()
            z[:, sl] = _repair_short_gaps(z[:, sl], m[:, sl],
                                          schema.QUALITY["max_internal_gap"])
            stats["channels_repaired"] += int(before - np.isnan(z[:, sl]).sum())
        # 4) 不可用组整体记 NaN（不提供任何部分信息）
        for gi, (node, sig) in enumerate(schema.GROUPS):
            if not G[i, gi]:
                z[:, schema.group_slice(node, sig)] = np.nan
        T[i] = z
        # 5) 可预测总体：至少一侧踝 gyro 合格
        ankle_ok = G[i, schema.GROUP_INDEX[("ankleL", "gyro")]] or \
            G[i, schema.GROUP_INDEX[("ankleR", "gyro")]]
        eligible.append(bool(ankle_ok))
        reasons.append("" if ankle_ok else "no_qualified_ankle_gyro")

    win = win.copy()
    win["eligible"] = eligible
    win["exclude_reason"] = reasons
    stats["windows_excluded_insufficient"] = int((~np.asarray(eligible)).sum())
    stats["elapsed_s"] = round(time.time() - t0, 2)
    return {"T": T, "M": M, "G": G, "coverage": cov_mat, "max_gap": gap_mat,
            "windows": win, "stats": stats}


def get_dataset(csv: str | Path, cache_dir: str | Path, dataset_id: str = "fogstar") -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tpath, mpath, gpath = (cache_dir / f"T_{dataset_id}.npy", cache_dir / f"M_{dataset_id}.npy",
                           cache_dir / f"G_{dataset_id}.npy")
    wpath, apath = cache_dir / f"windows_{dataset_id}.csv", cache_dir / f"audit_{dataset_id}.json"

    df = load_fogstar(csv)
    if wpath.exists() and tpath.exists():
        win = pd.read_csv(wpath)
        info = json.loads(apath.read_text(encoding="utf-8"))
        T, M, G = np.load(tpath), np.load(mpath), np.load(gpath)
        stats = json.loads((cache_dir / f"procstats_{dataset_id}.json").read_text(encoding="utf-8"))
        cov = np.load(cache_dir / f"cov_{dataset_id}.npy")
        gap = np.load(cache_dir / f"gap_{dataset_id}.npy")
    else:
        info = audit(df, dataset_id)
        w0 = build_windows(df, dataset_id)
        out = process(df, w0)
        win, T, M, G, cov, gap, stats = (out["windows"], out["T"], out["M"], out["G"],
                                         out["coverage"], out["max_gap"], out["stats"])
        np.save(tpath, T); np.save(mpath, M); np.save(gpath, G)
        np.save(cache_dir / f"cov_{dataset_id}.npy", cov)
        np.save(cache_dir / f"gap_{dataset_id}.npy", gap)
        win.to_csv(wpath, index=False)
        apath.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        (cache_dir / f"procstats_{dataset_id}.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8")
    return {"T": T, "M": M, "G": G, "coverage": cov, "max_gap": gap,
            "windows": win, "audit": info, "stats": stats}


def sample_weights(y: np.ndarray, dataset: np.ndarray | None = None) -> np.ndarray:
    """w = N / (2·D·N[d,c])：联合训练时每库每类总权重相同。"""
    n = len(y)
    w = np.ones(n, np.float64)
    ds = np.zeros(n, int) if dataset is None else np.asarray(dataset)
    D = len(np.unique(ds))
    for d in np.unique(ds):
        for c in (0, 1):
            m = (ds == d) & (y == c)
            if m.any():
                w[m] = n / (2.0 * D * m.sum())
    return w
