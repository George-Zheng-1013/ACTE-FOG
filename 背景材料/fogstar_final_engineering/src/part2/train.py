"""ET 四条表示行 + 桥接行的训练与评价（单库 FoG-STAR）。

    python -m src.part2.train --rows E00,E01,E10,E11,BRIDGE --seed 1

E00  ACTE72 原生 NaN，无人工屏蔽
E01  同上，加入 1 自然 + 3 人工屏蔽版本
E10  占位 0 + 72 个有效性标记（144 列），无人工屏蔽
E11  同上，加入屏蔽版本
BRIDGE 72 维零填补、无标记、无屏蔽（只为与 Part 1 的 F7 对齐核对）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from . import dataset, evaluate, features, masking, schema

ENGINE_ROOT = Path(__file__).resolve().parents[2]


def _et() -> ExtraTreesClassifier:
    c = schema.TRAIN["et"]
    return ExtraTreesClassifier(n_estimators=c["n_estimators"],
                                min_samples_leaf=c["min_samples_leaf"],
                                max_features=c["max_features"],
                                class_weight=c["class_weight"], n_jobs=-1)


def _build_training(row: str, T, M, G, win_tr, w_tr, seed):
    if row in ("E00", "BRIDGE"):
        keep = G[win_tr]
        F, V = features.build(T[win_tr], M[win_tr], keep)
        X = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0) if row == "BRIDGE" else F
        return X, w_tr, np.arange(int(win_tr.sum()))
    if row == "E10":
        F, V = features.build(T[win_tr], M[win_tr], G[win_tr])
        return features.to_finite(F, V), w_tr, np.where(win_tr)[0]
    # E01 / E11：屏蔽增强
    man = masking.build(pd.DataFrame({"window_id": win_ids[win_tr]}), G[win_tr], seed)
    exp = masking.expand(pd.DataFrame({"window_id": win_ids[win_tr],
                                       "y": y[win_tr]}), G[win_tr], w_tr, man)
    F, V = features.build(T[win_tr][exp["parent"]], M[win_tr][exp["parent"]], exp["keep"])
    X = features.to_finite(F, V) if row == "E11" else F
    return X, exp["weight"], exp["parent"], man


def _predict_configs(model, T, M, G, rows_mask, y, want_finite: bool) -> dict:
    """在 12 个部位保留请求下重算特征并预测。"""
    out = {}
    for cfg in schema.ANKLE_CONFIGS:
        keep = masking.requested_keep(G[rows_mask], cfg)
        F, V = features.build(T[rows_mask], M[rows_mask], keep)
        X = features.to_finite(F, V) if want_finite else F
        out[cfg] = model.predict_proba(X)[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="E00,E01,E10,E11,BRIDGE")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csv", default=str(ENGINE_ROOT / "data" / "sensor_data.csv"))
    ap.add_argument("--out", default=str(ENGINE_ROOT / "results" / "part2_v1"))
    ap.add_argument("--cache", default=str(ENGINE_ROOT / "results" / "part2_v1" / "_cache"))
    a = ap.parse_args()
    out_root = Path(a.out)

    t0 = time.time()
    global win_ids, y
    ds = dataset.get_dataset(a.csv, Path(a.cache))
    win = ds["windows"]
    elig = win.eligible.to_numpy()
    T, M, G = ds["T"][elig], ds["M"][elig], ds["G"][elig]
    win = win[elig].reset_index(drop=True)
    win_ids = win.window_id.to_numpy()
    y = win.y.to_numpy(np.int64)
    tr = (win.split == "train").to_numpy()
    va = (win.split == "val").to_numpy()
    te = (win.split == "test").to_numpy()
    w_tr = dataset.sample_weights(y[tr])

    print(f"[data] eligible={len(win)} train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"pos={int(y.sum())}", flush=True)

    rows = [r.strip() for r in a.rows.split(",") if r.strip()]
    summary = {}
    for row in rows:
        t1 = time.time()
        built = _build_training(row, T, M, G, tr, w_tr, a.seed)
        X, w, parent = built[0], built[1], built[2]
        finite = row in ("E10", "E11")   # 预测时的列数必须与训练一致
        model = _et()
        model.fit(X, y[tr][parent], sample_weight=w)
        nan_note = None
        if row in ("E00", "E01"):
            nan_note = int(np.isnan(X).sum())

        pv = _predict_configs(model, T, M, G, va, y, finite)
        pt = _predict_configs(model, T, M, G, te, y, finite)
        yv, yt = y[va], y[te]
        units = [(f"fogstar|{c}", yv, pv[c]) for c in schema.ANKLE_CONFIGS]
        thr, ttab = evaluate.select_threshold(units)

        res = {c: evaluate.unit_metrics(yt, pt[c], thr) for c in schema.ANKLE_CONFIGS}
        macro_te = evaluate.macro(res)
        macro_va = evaluate.macro({c: evaluate.unit_metrics(yv, pv[c], thr)
                                   for c in schema.ANKLE_CONFIGS})

        d = out_root / row / f"seed{a.seed}"
        d.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(model, d / "model.joblib")
        pd.DataFrame(ttab).to_csv(d / "threshold_units.csv", index=False)
        pd.DataFrame([{"config": c, **v} for c, v in res.items()]).to_csv(
            d / "test_by_config.csv", index=False)
        meta = {"row": row, "seed": a.seed, "protocol": schema.PROTOCOL_HASH,
                "protocol_version": schema.PROTOCOL_VERSION,
                "n_features": int(X.shape[1]), "n_train_rows": int(len(X)),
                "nan_cells_in_training_matrix": nan_note,
                "finite_route": finite, "augmented": row in ("E01", "E11"),
                "threshold": thr, "units_used": int(sum(1 for t in ttab if t["two_class"])),
                "macro_test": macro_te, "macro_val": macro_va,
                "elapsed_s": round(time.time() - t1, 1)}
        (d / "metrics.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
        (d / "protocol.json").write_text(json.dumps(schema.PROTOCOL, indent=2,
                                                    ensure_ascii=False), encoding="utf-8")
        summary[row] = {"n_features": int(X.shape[1]), "n_train": int(len(X)),
                        "threshold": thr, **{k: macro_te[k] for k in ("auroc", "auprc", "f1")}}
        print(f"[{row}] feats={X.shape[1]} rows={len(X)} thr={thr:.3f} "
              f"macro test AUROC={macro_te['auroc']:.4f} AP={macro_te['auprc']:.4f} "
              f"F1={macro_te['f1']:.4f} ({round(time.time()-t1,1)}s)", flush=True)

    (out_root / f"summary_seed{a.seed}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {round(time.time()-t0,1)}s", flush=True)


if __name__ == "__main__":
    main()
