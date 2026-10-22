"""Equal-budget validation search (protocol 2.7.3) for both model families.

Pre-registration (written before any run):
  * CNN space   : lr in {8e-4, 2e-3, 8e-3} x batch in {64, 128}      -> 6 configs
  * ET  space   : max_features in {8, 16, 32} x min_samples_leaf in {1, 3} -> 6 configs
  * identical attempt count per family (6), identical seeds (1, 2, 3)
  * selection metric: mean validation AUROC over the three seeds (higher better)
    tie-break: mean validation AUPRC, then fewer mean epochs / cheaper fit
  * the test split is never read during selection
  * tuning runs on the reference condition (S0 / P_ref); the selected config is
    then frozen for the whole matrix, so no experiment gets its own tuning

    python -m src.fair_v2.tune --out results/fair_ctrl_v2_tuned
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import base, models, schema, views

ENGINE_ROOT = Path(__file__).resolve().parents[2]
SEEDS = [1, 2, 3]

CNN_SPACE = [{"lr": lr, "batch_size": bs} for lr in (8e-4, 2e-3, 8e-3) for bs in (64, 128)]
ET_SPACE = [{"max_features": mf, "min_samples_leaf": ml}
            for mf in (8, 16, 32) for ml in (1, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ENGINE_ROOT / "results" / "fair_ctrl_v2_tuned"))
    ap.add_argument("--csv", default=str(ENGINE_ROOT / "data" / "sensor_data.csv"))
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args()
    out = Path(a.out)
    tdir = out / "tuning"
    tdir.mkdir(parents=True, exist_ok=True)

    ds = base.get_dataset(a.csv, schema.P_REF, cache_dir=out / "_cache")
    win = ds["windows"]
    y = win.y.to_numpy(np.int64)
    tr = (win.split == "train").to_numpy()
    va = (win.split == "val").to_numpy()
    te = (win.split == "test").to_numpy()
    w = np.ones(len(y))
    w[tr] = base.sample_weights(y[tr])

    ch_idx = schema.channel_indices("S0")
    Xc = views.cnn_view(ds["T"], ch_idx)
    Xe = views.et_view(ds["T"], ch_idx)

    (tdir / "protocol.json").write_text(json.dumps({
        "purpose": "equal-budget validation search, protocol 2.7.3",
        "reference_condition": {"sensor_set": "S0", "processing": "P_ref"},
        "cnn_space": CNN_SPACE, "et_space": ET_SPACE,
        "attempts_per_family": [len(CNN_SPACE), len(ET_SPACE)],
        "seeds": SEEDS,
        "selection_metric": "mean val AUROC over seeds; tie -> mean val AUPRC; tie -> fewer epochs",
        "test_split_used_for_selection": False,
        "frozen_after_selection": True,
    }, indent=2), encoding="utf-8")

    rows = []
    t0 = time.time()

    for i, cfg in enumerate(CNN_SPACE):
        for s in SEEDS:
            r = models.train_cnn(Xc, y, w, tr, va, te, seed=s, device=a.device, cfg=cfg)
            v = r["history"][r["best_epoch"] - 1]
            rows.append({"model": "cnn", "config_id": f"cnn{i}", **cfg, "seed": s,
                         "val_auroc": v["val_auroc"], "val_auprc": v["val_auprc"],
                         "val_loss": v["val_loss"], "best_epoch": r["best_epoch"],
                         "epochs_run": r["epochs_run"], "seconds": r["train_seconds"]})
            print(f"[cnn{i}] lr={cfg['lr']:<7} bs={cfg['batch_size']:<4} seed={s} "
                  f"val_auc={v['val_auroc']:.4f} ep={r['best_epoch']}/{r['epochs_run']} "
                  f"({r['train_seconds']}s)", flush=True)

    for i, cfg in enumerate(ET_SPACE):
        for s in SEEDS:
            r = models.train_et(Xe, y, w, tr, va, te, seed=s, cfg=cfg)
            from sklearn.metrics import average_precision_score, roc_auc_score
            auc = float(roc_auc_score(y[va], r["p_val"]))
            ap_ = float(average_precision_score(y[va], r["p_val"]))
            rows.append({"model": "et", "config_id": f"et{i}", **cfg, "seed": s,
                         "val_auroc": auc, "val_auprc": ap_, "val_loss": np.nan,
                         "best_epoch": np.nan, "epochs_run": np.nan,
                         "seconds": r["train_seconds"]})
            print(f"[et{i}] mf={cfg['max_features']:<3} leaf={cfg['min_samples_leaf']} "
                  f"seed={s} val_auc={auc:.4f} ({r['train_seconds']}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(tdir / "tuning_results.csv", index=False)

    sel = {}
    agg = (df.groupby(["model", "config_id"], as_index=False)
             .agg(val_auroc=("val_auroc", "mean"), val_auprc=("val_auprc", "mean"),
                  val_auroc_sd=("val_auroc", "std"),
                  epochs=("epochs_run", "mean"), seconds=("seconds", "mean")))
    agg.to_csv(tdir / "tuning_aggregate.csv", index=False)
    for fam, space in (("cnn", CNN_SPACE), ("et", ET_SPACE)):
        g = agg[agg.model == fam].sort_values(
            ["val_auroc", "val_auprc", "epochs"], ascending=[False, False, True])
        best = g.iloc[0]
        idx = int(best.config_id[len(fam):])
        chosen = dict(space[idx])
        chosen.update({"selected_on": "mean val AUROC over seeds 1-3",
                       "mean_val_auroc": float(best.val_auroc),
                       "mean_val_auprc": float(best.val_auprc),
                       "val_auroc_sd": None if pd.isna(best.val_auroc_sd) else float(best.val_auroc_sd),
                       "attempts": len(space)})
        sel[fam] = chosen

    sel["baseline_for_reference"] = {
        "cnn": {"lr": 8e-3, "batch_size": 64}, "et": {"max_features": 16, "min_samples_leaf": 3}}
    sel["seeds"] = SEEDS
    sel["elapsed_s"] = round(time.time() - t0, 1)
    (tdir / "selected_config.json").write_text(
        json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(sel, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
