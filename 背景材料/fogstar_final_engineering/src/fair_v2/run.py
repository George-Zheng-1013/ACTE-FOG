"""Experiment runner for fair_ctrl_v2.

    # batch 1 style, one seed
    python -m src.fair_v2.run --exp M --seed 1 --device cuda \
        --out results/fair_ctrl_v2_tuned \
        --cfg-json results/fair_ctrl_v2_tuned/tuning/selected_config.json

    # batch 2, five seeds, all registered experiments
    python -m src.fair_v2.run --all --seeds 1,2,3,4,5 --device cuda \
        --out results/fair_ctrl_v2_batch2 \
        --cfg-json results/fair_ctrl_v2_tuned/tuning/selected_config.json

Layout: one directory per experiment (and per seed when several are requested),
each containing protocol.json, windows.csv, per-model models and predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from . import base, evaluate, features, models, schema, views

ENGINE_ROOT = Path(__file__).resolve().parents[2]

# family: which protocol table the row belongs to
# changed: the single factor this row changes relative to its reference
EXPERIMENTS = {
    # ---------------- reference and model comparison (M) ----------------
    "M":  dict(family="M", sensor="S0", proc={}, reps=["raw"], models=["cnn", "et"],
               changed="none (reference)"),
    # ---------------- processing ablation (P) ---------------------------
    "P1": dict(family="P", sensor="S0", proc={"filter.enabled": False},
               reps=["raw"], models=["cnn", "et"], changed="band-pass filter off"),
    "P2": dict(family="P", sensor="S0", proc={"demean.enabled": False},
               reps=["raw"], models=["cnn", "et"], changed="per-window demean off"),
    "P3": dict(family="P", sensor="S0", proc={"standardize.enabled": True},
               reps=["raw"], models=["cnn", "et"], changed="train-fitted z-score on"),
    "P4": dict(family="P", sensor="S0",
               proc={"fill.method": "linear", "fill.value": None},
               reps=["raw"], models=["cnn", "et"], changed="linear interpolation instead of zero fill"),
    "P5": dict(family="P", sensor="S0", proc={"missing_indicator.enabled": True},
               reps=["raw"], models=["cnn", "et"], view_mask=True,
               changed="raw observation mask appended for both models"),
    "P6": dict(family="P", sensor="S0", proc={}, reps=["raw"], models=["cnn", "et"],
               smoothing=True, reuse="M", changed="shared 7-window centered smoothing"),
    "XF": dict(family="P", sensor="S0",
               proc={"filter.enabled": False, "demean.enabled": False},
               reps=["raw"], models=["cnn", "et"],
               changed="filter x demean interaction cell (both off)"),
    "W0": dict(family="W", sensor="S0", proc={}, reps=["raw"], models=["cnn", "et"],
               weight_mode="uniform", changed="uniform sample weight 1 (both models)"),
    # ---------------- sensor inputs (S) --------------------------------
    "S1": dict(family="S", sensor="S1", proc={}, reps=["raw"], models=["cnn", "et"],
               changed="add right ankle gyro"),
    "S2": dict(family="S", sensor="S2", proc={}, reps=["raw"], models=["cnn", "et"],
               changed="add bilateral ankle accelerometer"),
    "S3": dict(family="S", sensor="S3", proc={}, reps=["raw"], models=["cnn", "et"],
               changed="add back acc/gyro"),
    "S4": dict(family="S", sensor="S4", proc={}, reps=["raw"], models=["cnn", "et"],
               changed="add wrist acc/gyro"),
    "S5": dict(family="S", sensor="S5", proc={}, reps=["raw"], models=["cnn", "et"],
               changed="four body parts (24 channels)"),
    # ---------------- hand-crafted features (F, ET only) --------------
    "F0": dict(family="F", sensor="S5", proc={}, reps=["raw"], models=["et"],
               changed="reference: lossless flatten of the S5 tensor"),
    "F1": dict(family="F", sensor="S5", proc={}, reps=["features_full"], models=["et"],
               changed="132 hand-crafted features instead of the flatten"),
    "F2": dict(family="F", sensor="S5", proc={}, reps=["features_no_stdrms"], models=["et"],
               changed="remove STD + RMS block"),
    "F3": dict(family="F", sensor="S5", proc={}, reps=["features_nomad"], models=["et"],
               changed="remove adjacent-difference block"),
    "F4": dict(family="F", sensor="S5", proc={}, reps=["features_nofreeze"], models=["et"],
               changed="remove freeze-index block"),
    "F5": dict(family="F", sensor="S5", proc={}, reps=["features_nomissing"], models=["et"],
               changed="remove missing-ratio block"),
    "F6": dict(family="F", sensor="S5", proc={}, reps=["features_nocoord"], models=["et"],
               changed="remove coordination block"),
    "F7": dict(family="F", sensor="S5", proc={}, reps=["features_acte72"], models=["et"],
               changed="ACTE 72-dim ankle-dominant selection"),
}

FEATURE_REPS = {
    "features_full": lambda: (list(range(132)), list(features.FEATURE_NAMES)),
    "features_nocoord": lambda: _feat(features.indices_without("coord")),
    "features_no_stdrms": lambda: _feat(features.indices_without("std", "rms")),
    "features_nomad": lambda: _feat(features.indices_without("mad")),
    "features_nofreeze": lambda: _feat(features.indices_without("freeze")),
    "features_nomissing": lambda: _feat(features.indices_without("missing")),
    "features_acte72": lambda: (features.acte72_indices(),
                                [features.FEATURE_NAMES[i] for i in features.acte72_indices()]),
}


def _feat(idx):
    return idx, [features.FEATURE_NAMES[i] for i in idx]


# --------------------------------------------------------------------------- #
def apply_overrides(p: dict, overrides: dict) -> dict:
    p = deepcopy(p)
    for k, v in overrides.items():
        a, b = k.split(".")
        p[a][b] = v
    return p


def file_hash(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def environment(device: str) -> dict:
    import scipy
    import sklearn
    import torch
    info = {
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "host": socket.gethostname(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()), "device_used": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": int(torch.get_num_threads()),
    }
    try:
        import subprocess
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ENGINE_ROOT, text=True).strip()
    except Exception:
        info["git_commit"] = None
    return info


def representation_matrices(ds, rep: str, sensor: str, with_mask: bool):
    """Return (X_for_et, feature_names, view_meta, ch_idx)."""
    ch_idx = schema.channel_indices(sensor)
    if rep == "raw":
        X = views.et_view(ds["T"], ds["M"], ch_idx, with_mask)
        rt = views.roundtrip_check(ds["T"], ds["M"], ch_idx, with_mask)
        chans = [schema.CHANNELS[i] for i in ch_idx]
        if with_mask:
            chans = chans + [f"{c}__observed" for c in chans]
        meta = {"kind": "flattened signal", "channels": chans,
                "cnn_shape": rt["cnn_shape"], "et_shape": rt["et_shape"], "roundtrip": rt}
        return X, [f"{c}@{t}" for t in range(schema.WINDOW) for c in chans], meta, ch_idx
    if rep not in FEATURE_REPS:
        raise ValueError(rep)
    F = features.build_features(ds["T"], ds["M"])
    idx, names = FEATURE_REPS[rep]()
    meta = {"kind": rep, "n_features": len(idx),
            "feature_blocks": dict(features.BLOCKS),
            "kept_blocks": sorted({b for b, (a, z) in features.BLOCKS.items()
                                   if any(a <= i < z for i in idx)})}
    return F[:, idx], names, meta, ch_idx


# --------------------------------------------------------------------------- #
def run_experiment(name: str, seed: int, csv: str, exp_dir: Path, device: str,
                   cache_dir: Path, cnn_cfg: dict | None = None,
                   et_cfg: dict | None = None, reuse_dir: Path | None = None) -> dict:
    spec = EXPERIMENTS[name]
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name} seed={seed}] family={spec['family']} sensor={spec['sensor']}", flush=True)
    t_start = time.time()

    p_cfg = apply_overrides(schema.P_REF, spec["proc"])
    if spec.get("smoothing"):
        p_cfg = apply_overrides(p_cfg, {"smoothing.enabled": True})
    ds = base.get_dataset(csv, p_cfg, cache_dir)
    win, T, M = ds["windows"], ds["T"], ds["M"]
    y = win.y.to_numpy(np.int64)
    tr = (win.split == "train").to_numpy()
    va = (win.split == "val").to_numpy()
    te = (win.split == "test").to_numpy()
    if spec.get("weight_mode") == "uniform":
        w_full = np.ones(len(y))
        w_note = "uniform weight 1 for every training window (W0)"
    else:
        w_full = np.ones(len(y))
        w_full[tr] = base.sample_weights(y[tr])
        w_note = "w_c = N_train / (2 * N_train,c) computed on train patients"
    with_mask = bool(spec.get("view_mask"))

    results = {}
    protocol = {
        "experiment_id": name, "family": spec["family"], "seed": seed,
        "changed_factor": spec.get("changed", "none"),
        "reference": "M" if name != "M" else None,
        "sensor_set": spec["sensor"],
        "channels": schema.SENSOR_SETS[spec["sensor"]],
        "missing_indicator_view": with_mask,
        "weight_mode": spec.get("weight_mode", "balanced"),
        "weight_note": w_note,
        "split": schema.SPLITS, "label_rule": schema.POS_TIE_RULE,
        "processing": p_cfg, "processing_config_hash": ds["processing_hash"],
        "cohort_hash": schema.short_hash({"windows": win.window_id.tolist()}),
        "cnn_cfg": {**models.CNN_CFG, **(cnn_cfg or {})},
        "et_cfg": {**models.ET_CFG, **(et_cfg or {})},
        "recipe_source": ("src.fair_v2.tune (equal-budget validation search)"
                          if (cnn_cfg or et_cfg) else "fixed defaults"),
        "threshold_rule": "0..1 step 0.001, minimise |sens-spec|, ties -> smallest",
        "metric_rule": "AUROC/AUPRC on probabilities, rest at the val threshold",
        "reuse_source": spec.get("reuse"),
    }
    (exp_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    win.to_csv(exp_dir / "windows.csv", index=False)
    exc = cache_dir / "exclusions.csv"
    if exc.exists():
        pd.read_csv(exc).to_csv(exp_dir / "exclusions.csv", index=False)
    pd.DataFrame({"window_id": win.window_id, "split": win.split, "y": y,
                  "weight": w_full}).to_csv(exp_dir / "weights.csv", index=False)

    for rep in spec["reps"]:
        X, feat_names, view_meta, ch_idx = representation_matrices(
            ds, rep, spec["sensor"], with_mask)
        cov = M[:, :, ch_idx].mean(axis=(1, 2))
        pd.DataFrame({"window_id": win.window_id, "subject": win.subject, "split": win.split,
                      "y": y, "coverage_selected": cov,
                      "n_valid_obs_all24": win.n_valid_obs}).to_csv(
            exp_dir / f"coverage_{rep}.csv", index=False)

        for mname in spec["models"]:
            tag = f"{rep}__{mname}"
            mdir = exp_dir / tag
            mdir.mkdir(parents=True, exist_ok=True)
            Xc = views.cnn_view(T, M, ch_idx, with_mask) if mname == "cnn" else X

            if spec.get("reuse"):
                src = (reuse_dir or (exp_dir.parent / spec["reuse"])) / tag
                p_raw = np.zeros(len(y))
                for split, mask in (("val", va), ("test", te)):
                    d = pd.read_csv(src / f"{split}_predictions.csv").set_index("window_id")
                    p_raw[mask] = d.loc[win.window_id[mask], "p_raw"].to_numpy()
                p_eval = evaluate.smooth_segments(p_raw, win, 7)
                thr = evaluate.eer_threshold(y[va], p_eval[va])
                met = {"val": evaluate.binary_metrics(y[va], p_eval[va], thr),
                       "test": evaluate.binary_metrics(y[te], p_eval[te], thr)}
                (mdir / "fit_info.json").write_text(json.dumps(
                    {"reused_from": str(src), "retrained": False}, indent=2), encoding="utf-8")
                (mdir / "view.json").write_text(
                    (src / "view.json").read_text(encoding="utf-8"), encoding="utf-8")
                train_seconds, n_params = 0.0, None
            else:
                if mname == "cnn" and rep != "raw":
                    raise ValueError("CNN receives only the time-series view")
                fn = models.train_cnn if mname == "cnn" else models.train_et
                r = fn(Xc, y, w_full, tr, va, te, seed=seed, device=device,
                       cfg=cnn_cfg if mname == "cnn" else et_cfg)
                train_seconds, n_params = r["train_seconds"], r.get("n_params")
                p_raw = np.zeros(len(y))
                p_raw[va] = r["p_val"]
                p_raw[te] = r["p_test"]
                p_eval = (evaluate.smooth_segments(p_raw, win, 7)
                          if p_cfg["smoothing"]["enabled"] else p_raw.copy())
                thr = evaluate.eer_threshold(y[va], p_eval[va])
                met = {"val": evaluate.binary_metrics(y[va], p_eval[va], thr),
                       "test": evaluate.binary_metrics(y[te], p_eval[te], thr)}

                if mname == "cnn":
                    import torch
                    torch.save(r["state_dict"], mdir / "model.pt")
                    pd.DataFrame(r["history"]).to_csv(
                        exp_dir / f"history_{rep}.csv", index=False)
                    (mdir / "fit_info.json").write_text(json.dumps(
                        {"n_params": r["n_params"], "best_epoch": r["best_epoch"],
                         "best_score": r["best_score"], "monitor": r["cfg"]["monitor"],
                         "epochs_run": r["epochs_run"], "device": r["device"],
                         "in_ch": r["in_ch"], "cfg": r["cfg"]}, indent=2), encoding="utf-8")
                else:
                    import joblib
                    joblib.dump(r["model"], mdir / "model.joblib")
                    (mdir / "fit_info.json").write_text(json.dumps(
                        {"fit_info": r["fit_info"], "n_features_in": r["n_features_in"],
                         "cfg": r["cfg"]}, indent=2), encoding="utf-8")

            meta = {"experiment_id": name, "model": mname, "representation": rep,
                    "seed": seed, "protocol_id": schema.short_hash(protocol)}
            for split, mask in (("val", va), ("test", te)):
                df = evaluate.prediction_frame(win, mask, p_raw[mask], p_eval[mask], thr, meta)
                df.to_csv(mdir / f"{split}_predictions.csv", index=False)
                np.savez(mdir / f"{split}_predictions.npz",
                         window_id=win.window_id[mask].to_numpy().astype(str),
                         y_true=win.y[mask].to_numpy(np.int64),
                         p_raw=p_raw[mask].astype(np.float64),
                         p_eval=p_eval[mask].astype(np.float64),
                         threshold=np.float64(thr))
            (mdir / "metrics.json").write_text(json.dumps(
                {"experiment": name, "model": mname, "representation": rep, "seed": seed,
                 "threshold": thr, "smoothing": p_cfg["smoothing"]["enabled"], **met},
                indent=2), encoding="utf-8")
            (mdir / "view.json").write_text(json.dumps(view_meta, indent=2), encoding="utf-8")

            results[tag] = {"metrics": met, "threshold": thr, "train_seconds": train_seconds,
                            "n_params": n_params, "reused": bool(spec.get("reuse")),
                            "input_shape": list(Xc.shape) if mname == "cnn" else list(X.shape)}
            print(f"[{name} s{seed}] {tag}: val={met['val']['auroc']:.4f} "
                  f"test={met['test']['auroc']:.4f} thr={thr:.3f} ({train_seconds}s)", flush=True)

    (exp_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (exp_dir / "environment.json").write_text(
        json.dumps(environment(device), indent=2), encoding="utf-8")
    (exp_dir / "run_config.json").write_text(json.dumps({
        "argv": sys.argv, "engine_root": str(ENGINE_ROOT), "data_csv": str(csv),
        "data_sha256_16": file_hash(Path(csv)), "out_dir": str(exp_dir),
        "seed": seed, "device": device,
        "torch_threads": __import__("torch").get_num_threads(),
        "elapsed_s": round(time.time() - t_start, 2),
        "windows": {"total": int(len(y)), "train": int(tr.sum()), "val": int(va.sum()),
                    "test": int(te.sum()), "pos_train": int(y[tr].sum()),
                    "pos_val": int(y[va].sum()), "pos_test": int(y[te].sum())},
        "T_hash": schema.array_hash(T), "M_hash": schema.array_hash(M),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if "raw__cnn" in results and "raw__et" in results:
        c, e = results["raw__cnn"]["metrics"], results["raw__et"]["metrics"]
        comp = {"experiment": name, "seed": seed, "pair": "CNN - ET",
                "val": {k: c["val"][k] - e["val"][k] for k in
                        ("auroc", "auprc", "f1", "sensitivity", "specificity")},
                "test": {k: c["test"][k] - e["test"][k] for k in
                         ("auroc", "auprc", "f1", "sensitivity", "specificity")},
                "cnn": c["test"], "et": e["test"]}
        (exp_dir / "comparison.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")

    print(f"[{name} s{seed}] done in {round(time.time() - t_start, 1)}s", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None, choices=sorted(EXPERIMENTS))
    ap.add_argument("--all", action="store_true", help="run every registered experiment")
    ap.add_argument("--exps", default=None, help="comma separated subset, e.g. P6,S1,S5,W0")
    ap.add_argument("--per-seed", action="store_true",
                    help="write <out>/<exp>/seed<k> even for a single seed")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--csv", default=str(ENGINE_ROOT / "data" / "sensor_data.csv"))
    ap.add_argument("--out", default=str(ENGINE_ROOT / "results" / "fair_ctrl_v2"))
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--cfg-json", default=None)
    ap.add_argument("--cache-dir", default=None,
                    help="share processed tensors across output roots (keyed by processing hash)")
    a = ap.parse_args()

    out_root = Path(a.out)
    cnn_cfg = et_cfg = None
    if a.cfg_json:
        sel = json.loads(Path(a.cfg_json).read_text(encoding="utf-8"))
        cnn_keys = ("lr", "batch_size", "weight_decay", "max_epochs", "patience",
                    "monitor", "min_delta", "l2_penalty", "dropout")
        et_keys = ("n_estimators", "min_samples_leaf", "max_features", "class_weight")
        cnn_cfg = {k: v for k, v in sel.get("cnn", {}).items() if k in cnn_keys} or None
        et_cfg = {k: v for k, v in sel.get("et", {}).items() if k in et_keys} or None

    seeds = [int(s) for s in a.seeds.split(",")]
    if a.exps:
        exps = [e.strip() for e in a.exps.split(",") if e.strip()]
    else:
        exps = sorted(EXPERIMENTS) if a.all else [a.exp]
    per_seed = a.per_seed or len(seeds) > 1
    for seed in seeds:
        for name in exps:
            exp_dir = (out_root / name / f"seed{seed}") if per_seed else (out_root / name)
            # P6-style reuse must read the reference of the same seed
            reuse_dir = (out_root / "M" / f"seed{seed}") if per_seed else (out_root / "M")
            run_experiment(name, seed, a.csv, exp_dir, a.device,
                           cache_dir=Path(a.cache_dir) if a.cache_dir else out_root / "_cache",
                           cnn_cfg=cnn_cfg, et_cfg=et_cfg, reuse_dir=reuse_dir)


if __name__ == "__main__":
    main()
