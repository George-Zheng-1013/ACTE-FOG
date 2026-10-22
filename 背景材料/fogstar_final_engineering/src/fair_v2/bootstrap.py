"""Patient-level paired bootstrap (protocol 2.11.2).

Design (pre-registered):
  * resample whole test patients with replacement, 2000 replicates, seed 20261022;
  * every contrast in a replicate uses the same patient draw (paired);
  * each seed's metric is computed separately; the cross-seed mean is taken only
    after that, so probabilities are never averaged into an ensemble;
  * a replicate where a resample contains a single class is skipped for AUROC and
    the valid replicate count is reported;
  * interval: 2.5-97.5 percentile.

    python -m src.fair_v2.bootstrap --root results/fair_ctrl_v2_batch2 \
        --seeds 1,2,3,4,5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE_ROOT = Path(__file__).resolve().parents[2]

# (family, reference, variant, contrast label)
PAIRS = [
    ("P", "M", "P1", "filter on - off"),
    ("P", "M", "P2", "demean on - off"),
    ("P", "M", "P3", "z-score on - off"),
    ("P", "M", "P4", "zero fill - linear interpolation"),
    ("P", "M", "P5", "mask appended - not appended"),
    ("P", "M", "P6", "smoothing on - off"),
    ("P", "M", "XF", "filter x demean both off - reference"),
    ("W", "M", "W0", "uniform weight - balanced weight"),
    ("S", "M", "S1", "S1 - S0"),
    ("S", "S1", "S2", "S2 - S1"),
    ("S", "S2", "S3", "S3 - S2"),
    ("S", "S2", "S4", "S4 - S2"),
    ("S", "M", "S5", "S5 - S0"),
    ("F", "F0", "F1", "F1 - F0"),
    ("F", "F1", "F2", "F1 - F2"),
    ("F", "F1", "F3", "F1 - F3"),
    ("F", "F1", "F4", "F1 - F4"),
    ("F", "F1", "F5", "F1 - F5"),
    ("F", "F1", "F6", "F1 - F6"),
    ("F", "F1", "F7", "F1 - F7"),
]

def model_dir(seed_dir: Path, model: str) -> Path | None:
    """CNN always uses raw__cnn; ET uses whatever single *__et view the row has."""
    if model == "cnn":
        d = seed_dir / "raw__cnn"
        return d if d.exists() else None
    cands = [q for q in seed_dir.iterdir() if q.is_dir() and q.name.endswith("__et")]
    return cands[0] if len(cands) == 1 else None


def fast_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """Mann-Whitney U with average ranks for ties (equals roc_auc_score)."""
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    _, inv, counts = np.unique(sp, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    # average rank of each tie group: (first_rank + last_rank) / 2
    avg = (cum - counts + 1 + cum) / 2.0
    ranks_sorted = avg[inv]
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = ranks_sorted
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def boot_p(d: np.ndarray) -> float:
    """Two-sided bootstrap p-value: 2 x the smaller tail mass at zero."""
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return float(min(1.0, max(p, 1.0 / len(d))))


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj


def f1_at(y: np.ndarray, p: np.ndarray, thr: float) -> float | None:
    z = p >= thr
    tp = int((z & (y == 1)).sum()); fp = int((z & (y == 0)).sum())
    fn = int((~z & (y == 1)).sum())
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def load_combo(root: Path, exp: str, model: str, seeds: list[int]):
    out = {}
    for s in seeds:
        sd = root / exp / f"seed{s}"
        if not sd.exists():
            continue
        md = model_dir(sd, model)
        if md is None:
            continue
        d = md / "test_predictions.npz"
        if not d.exists():
            continue
        z = np.load(d, allow_pickle=False)
        out[s] = {"id": z["window_id"].astype(str), "y": z["y_true"],
                  "p": z["p_eval"], "thr": float(z["threshold"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ENGINE_ROOT / "results" / "fair_ctrl_v2_batch2"))
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=20261022)
    a = ap.parse_args()
    root = Path(a.root)
    seeds = [int(s) for s in a.seeds.split(",")]

    exps = sorted({e for f, r, v, _ in PAIRS for e in (r, v)})
    combos = {}
    for e in exps:
        for m in ("cnn", "et"):
            c = load_combo(root, e, m, seeds)
            if c:
                combos[(e, m)] = c
    if not combos:
        raise SystemExit(f"no predictions under {root}")

    first = next(iter(combos.values()))
    ids = first[list(first)[0]]["id"]
    y = first[list(first)[0]]["y"]
    subj = np.array([w.split("_")[0] for w in ids])
    uniq = sorted(set(subj))
    per_subject = {s: np.flatnonzero(subj == s) for s in uniq}
    print(f"test windows={len(ids)} patients={len(uniq)} combos={len(combos)} seeds={seeds}")

    rng = np.random.default_rng(a.boot_seed)
    draws = [rng.choice(uniq, size=len(uniq), replace=True) for _ in range(a.n_boot)]

    point, reps = {}, {k: {"auroc": [], "f1": []} for k in combos}
    for k, per_seed in combos.items():
        aus, f1s = [], []
        for s, d in per_seed.items():
            assert (d["id"] == ids).all(), f"{k} seed{s}: window order differs"
            auc = fast_auc(y, d["p"])
            if auc is not None:
                aus.append(auc)
            f1s.append(f1_at(y, d["p"], d["thr"]))
        point[k] = {"auroc": float(np.mean(aus)) if aus else None,
                    "f1": float(np.mean(f1s))}

    for draw in draws:
        idx = np.concatenate([per_subject[s] for s in draw])
        yb = y[idx]
        for k, per_seed in combos.items():
            aus, f1s = [], []
            for s, d in per_seed.items():
                auc = fast_auc(yb, d["p"][idx])
                if auc is not None:
                    aus.append(auc)
                f1s.append(f1_at(yb, d["p"][idx], d["thr"]))
            reps[k]["auroc"].append(float(np.mean(aus)) if aus else np.nan)
            reps[k]["f1"].append(float(np.mean(f1s)))

    rows = []
    for fam, ref, var, label in PAIRS:
        for m in ("cnn", "et"):
            if (ref, m) not in combos or (var, m) not in combos:
                continue
            for metric in ("auroc", "f1"):
                a_ = point[(ref, m)][metric]
                b_ = point[(var, m)][metric]
                if a_ is None or b_ is None:
                    continue
                d = np.asarray(reps[(var, m)][metric]) - np.asarray(reps[(ref, m)][metric])
                d = d[np.isfinite(d)]
                rows.append({"family": fam, "model": m, "contrast": label,
                             "metric": metric, "reference": a_, "variant": b_,
                             "delta": b_ - a_,
                             "ci_low": float(np.percentile(d, 2.5)),
                             "ci_high": float(np.percentile(d, 97.5)),
                             "p_boot": boot_p(d),
                             "n_boot_valid": int(len(d))})
    # model contrast inside each experiment
    for e in exps:
        if ("cnn" in {m for x, m in combos if x == e} and
                "et" in {m for x, m in combos if x == e}):
            r_, v_ = "cnn", "et"
            for metric in ("auroc", "f1"):
                a_ = point[(e, r_)][metric]
                b_ = point[(e, v_)][metric]
                d = np.asarray(reps[(e, r_)][metric]) - np.asarray(reps[(e, v_)][metric])
                d = d[np.isfinite(d)]
                rows.append({"family": "M" if e == "M" else "model",
                             "model": "cnn-et", "contrast": f"{e}: CNN - ET",
                             "metric": metric, "reference": a_, "variant": b_,
                             "delta": a_ - b_,
                             "ci_low": float(np.percentile(d, 2.5)),
                             "ci_high": float(np.percentile(d, 97.5)),
                             "p_boot": boot_p(d),
                             "n_boot_valid": int(len(d))})

    out = pd.DataFrame(rows)
    # multiplicity: 36 AUROC contrasts were tested, so raw p-values overstate
    # evidence. Holm is applied both within each question (group) and across all.
    out["group"] = out.family.where(~out.family.isin(["M", "model"]), "model")
    out["p_holm_all"] = holm(out.p_boot.values)
    out["p_holm_group"] = out.groupby("group").p_boot.transform(
        lambda col: pd.Series(holm(col.values), index=col.index))
    out.to_csv(root / "bootstrap.csv", index=False)
    (root / "bootstrap_protocol.json").write_text(json.dumps({
        "n_boot": a.n_boot, "boot_seed": a.boot_seed,
        "resampling": "patients resampled with replacement, len = number of test patients",
        "pairing": "same patient draw for every contrast within a replicate",
        "seeds": seeds,
        "seed_aggregation": "metric per seed, then mean across seeds; probabilities never averaged",
        "interval": "2.5-97.5 percentile",
        "metrics": ["auroc", "f1"],
        "auprc_note": "AUPRC interval omitted: sklearn AP is too slow for 2000x5x~20 calls; reported as mean/sd across seeds instead",
    }, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    au = out[out.metric == "auroc"].sort_values("p_boot")
    print(au[["group", "model", "contrast", "delta", "ci_low", "ci_high",
              "p_boot", "p_holm_group", "p_holm_all"]].to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nAUROC contrasts: {len(au)}；"
          f"未校正 p<0.05: {(au.p_boot < 0.05).sum()}；"
          f"组内 Holm<0.05: {(au.p_holm_group < 0.05).sum()}；"
          f"全体 Holm<0.05: {(au.p_holm_all < 0.05).sum()}")


if __name__ == "__main__":
    main()
