"""Fast batch-level consistency audit (no model reload, no tensor rebuild).

Checks, for every experiment/seed/model of a batch root:
  * the val/test window_id order equals the reference (M, seed 1);
  * the per-window weight vector follows the registered rule;
  * the declared threshold equals the one re-derived from the saved val p_eval;
  * test AUROC / F1 recomputed from the saved npz equal metrics.json.

Model-reload reproducibility (acceptance 8) is still checked by src.fair_v2.verify
on a per-experiment basis.

    python -m src.fair_v2.verify_batch --root results/fair_ctrl_v2_batch2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import base, evaluate

ENGINE_ROOT = Path(__file__).resolve().parents[2]
TAGS = ("raw__cnn", "raw__et", "features_full__et", "features_nocoord__et",
        "features_no_stdrms__et", "features_nomad__et", "features_nofreeze__et",
        "features_nomissing__et", "features_acte72__et")


def _npz(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as z:
        return pd.DataFrame({"window_id": z["window_id"].astype(str),
                             "y_true": z["y_true"], "p_raw": z["p_raw"],
                             "p_eval": z["p_eval"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ENGINE_ROOT / "results" / "fair_ctrl_v2_batch2"))
    a = ap.parse_args()
    root = Path(a.root)

    ref = {}
    ref_dir = root / "M" / "seed1" / "raw__cnn"
    for split in ("val", "test"):
        ref[split] = np.load(ref_dir / f"{split}_predictions.npz")["window_id"].astype(str)

    rows = []
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "_cache"):
        for seed_dir in sorted(exp_dir.glob("seed*")):
            seed = int(seed_dir.name[4:])
            wcsv = pd.read_csv(seed_dir / "weights.csv")
            tr = wcsv[wcsv.split == "train"]
            expect = base.sample_weights(tr.y.to_numpy())
            w_ok = bool(np.allclose(tr.weight.to_numpy(), expect) or
                        np.allclose(tr.weight.to_numpy(), 1.0))
            for tag in TAGS:
                mfile = seed_dir / tag / "metrics.json"
                if not mfile.exists():
                    continue
                mj = json.loads(mfile.read_text(encoding="utf-8"))
                dv = _npz(seed_dir / tag / "val_predictions.npz")
                dt = _npz(seed_dir / tag / "test_predictions.npz")
                order_ok = (dv.window_id.astype(str).tolist() == ref["val"].tolist()
                            and dt.window_id.astype(str).tolist() == ref["test"].tolist())
                thr = evaluate.eer_threshold(dv.y_true.to_numpy(), dv.p_eval.to_numpy())
                mv = evaluate.binary_metrics(dv.y_true.to_numpy(), dv.p_eval.to_numpy(), thr)
                mt = evaluate.binary_metrics(dt.y_true.to_numpy(), dt.p_eval.to_numpy(), thr)
                rows.append({
                    "experiment": exp_dir.name, "seed": seed, "tag": tag,
                    "windows_aligned": order_ok, "weights_ok": w_ok,
                    "threshold_recomputed": round(float(thr), 6),
                    "threshold_declared": round(float(mj["threshold"]), 6),
                    "thr_ok": abs(thr - mj["threshold"]) < 1e-12,
                    "auroc_ok": abs(mt["auroc"] - mj["test"]["auroc"]) < 1e-9,
                    "f1_ok": abs(mt["f1"] - mj["test"]["f1"]) < 1e-9,
                    "test_auroc": mt["auroc"],
                })
    d = pd.DataFrame(rows)
    d.to_csv(root / "audit_batch.csv", index=False)
    ok_cols = ["windows_aligned", "weights_ok", "thr_ok", "auroc_ok", "f1_ok"]
    bad = d[~d[ok_cols].all(axis=1)]
    print(f"checked {len(d)} runs ({d.experiment.nunique()} experiments × "
          f"{d.seed.nunique()} seeds × models)")
    print(f"all checks passed: {len(bad) == 0}   failing rows: {len(bad)}")
    if len(bad):
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
