"""Experiment runner for fair_ctrl_v2.

Each invocation builds one experiment directory under results/fair_ctrl_v2/<EXP>/
and writes the artefacts required by protocol 2.10.2.

Examples
    python -m src.fair_v2.run --exp M   --seed 1
    python -m src.fair_v2.run --exp P1  --seed 1
    python -m src.fair_v2.run --exp S5  --seed 1
    python -m src.fair_v2.run --exp F1  --seed 1
    python -m src.fair_v2.run --exp F6  --seed 1
    python -m src.fair_v2.run --exp P6  --seed 1     # reuses M, no training
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
BATCH = "1"          # phase-1 acceptance batch, seed=1 only

EXPERIMENTS = {
    # name: family, sensor set, processing overrides, representations, reuse
    "M":  dict(family="M", sensor="S0", proc={}, reps=["raw"], models=["cnn", "et"]),
    "P1": dict(family="P", sensor="S0", proc={"filter.enabled": False},
               reps=["raw"], models=["cnn", "et"], changed_factor="band-pass filter off"),
    "S5": dict(family="S", sensor="S5", proc={}, reps=["raw"], models=["cnn", "et"],
               changed_factor="four-body-part channels"),
    "F1": dict(family="F", sensor="S5", proc={}, reps=["features_full"], models=["et"],
               changed_factor="hand-crafted features instead of flattened signal"),
    "F6": dict(family="F", sensor="S5", proc={}, reps=["features_nocoord"], models=["et"],
               changed_factor="coordination block removed"),
    "P6": dict(family="P6", sensor="S0", proc={}, reps=["raw"], models=["cnn", "et"],
               changed_factor="shared 7-window centered smoothing", reuse="M"),
}


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
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": device,
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


def representation_matrices(ds, rep: str, sensor: str):
    """Return (X, feature_names, view_meta) for one representation."""
    ch_idx = schema.channel_indices(sensor)
    if rep == "raw":
        X = views.et_view(ds["T"], ch_idx)
        rt = views.roundtrip_check(ds["T"], ch_idx)
        meta = {"kind": "flattened signal", "channels": [schema.CHANNELS[i] for i in ch_idx],
                "cnn_shape": rt["cnn_shape"], "et_shape": rt["et_shape"],
                "roundtrip": rt}
        return X, [f"{c}@{t}" for t in range(schema.WINDOW) for c in meta["channels"]], meta, ch_idx
    F = features.build_features(ds["T"], ds["M"])
    if rep == "features_full":
        idx, names = list(range(132)), list(features.FEATURE_NAMES)
    elif rep == "features_nocoord":
        idx = features.indices_without("coord")
        names = [features.FEATURE_NAMES[i] for i in idx]
    else:
        raise ValueError(rep)
    meta = {"kind": rep, "n_features": len(idx),
            "feature_blocks": {k: v for k, v in features.BLOCKS.items()},
            "dropped_blocks": [b for b in features.BLOCKS if rep == "features_nocoord" and b == "coord"]}
    return F[:, idx], names, meta, ch_idx


# --------------------------------------------------------------------------- #
def run_experiment(name: str, seed: int, csv: str, out_root: Path, device: str,
                   cache_dir: Path, reuse_dir: Path | None = None,
                   cnn_cfg: dict | None = None, et_cfg: dict | None = None) -> dict:
    spec = EXPERIMENTS[name]
    print(f"[{name}] family={spec['family']} sensor={spec['sensor']} seed={seed}", flush=True)
    t_start = time.time()

    p_cfg = apply_overrides(schema.P_REF, spec["proc"])
    if spec.get("smoothing") or name == "P6":
        p_cfg = apply_overrides(p_cfg, {"smoothing.enabled": True})
    ds = base.get_dataset(csv, p_cfg, cache_dir)
    win, T, M = ds["windows"], ds["T"], ds["M"]
    y = win.y.to_numpy(np.int64)
    tr = (win.split == "train").to_numpy()
    va = (win.split == "val").to_numpy()
    te = (win.split == "test").to_numpy()
    w = base.sample_weights(y[tr].astype(np.int64))     # train-only vector
    w_full = np.ones(len(y))
    w_full[tr] = w

    exp_dir = out_root / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    protocol = {
        "experiment_id": name, "family": spec["family"], "batch": BATCH, "seed": seed,
        "changed_factor": spec.get("changed_factor", "none (reference)"),
        "pair_id": spec.get("reuse") or "P0",
        "sensor_set": spec["sensor"],
        "channels": schema.SENSOR_SETS[spec["sensor"]],
        "split": schema.SPLITS,
        "label_rule": schema.POS_TIE_RULE,
        "processing": p_cfg,
        "processing_config_hash": ds["processing_hash"],
        "cohort_hash": schema.short_hash({"windows": win.window_id.tolist()}),
        "cnn_cfg": {**models.CNN_CFG, **(cnn_cfg or {})},
        "et_cfg": {**models.ET_CFG, **(et_cfg or {})},
        "recipe_source": ("src.fair_v2.tune (equal-budget validation search)"
                          if (cnn_cfg or et_cfg) else "fixed defaults (CNN_CFG / ET_CFG)"),
        "threshold_rule": "0..1 step 0.001, minimise |sens-spec|, ties -> smallest",
        "metric_rule": "AUROC/AUPRC on probabilities, rest at the val threshold",
        "seed_plan": {"batch1": [1], "batch2": [1, 2, 3, 4, 5]},
        "reuse_source": spec.get("reuse"),
    }
    (exp_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    win.to_csv(exp_dir / "windows.csv", index=False)
    if (cache_dir / "exclusions.csv").exists():
        pd.read_csv(cache_dir / "exclusions.csv").to_csv(exp_dir / "exclusions.csv", index=False)
    pd.DataFrame({"window_id": win.window_id, "split": win.split, "y": y,
                  "weight": np.where(tr, w_full, np.nan)}).to_csv(
        exp_dir / "weights.csv", index=False)

    for rep in spec["reps"]:
        X, feat_names, view_meta, ch_idx = representation_matrices(ds, rep, spec["sensor"])
        cov = M[:, :, ch_idx].mean(axis=(1, 2))
        pd.DataFrame({"window_id": win.window_id, "subject": win.subject, "split": win.split,
                      "y": y, "coverage_selected": cov,
                      "n_valid_obs_all24": win.n_valid_obs}).to_csv(
            exp_dir / f"coverage_{rep}.csv", index=False)

        for mname in spec["models"]:
            tag = f"{rep}__{mname}"
            mdir = exp_dir / tag
            mdir.mkdir(parents=True, exist_ok=True)

            if spec.get("reuse"):
                # P6 style: reuse the reference probabilities, never retrain
                src = out_root / spec["reuse"] / tag
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
                (mdir / "view.json").write_text((src / "view.json").read_text(encoding="utf-8"),
                                                encoding="utf-8")
                train_seconds = 0.0
            else:
                print(f"[{name}] {tag}: X={X.shape}", flush=True)
                if mname == "cnn" and rep != "raw":
                    raise ValueError("CNN receives only the time-series view in this protocol")
                Xc = views.cnn_view(T, ch_idx) if mname == "cnn" else X
                fn = models.train_cnn if mname == "cnn" else models.train_et
                cfg_used = cnn_cfg if mname == "cnn" else et_cfg
                r = fn(Xc, y, w_full, tr, va, te, seed=seed, device=device, cfg=cfg_used)
                train_seconds = r["train_seconds"]

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
                    pd.DataFrame(r["history"]).to_csv(exp_dir / f"history_{rep}.csv", index=False)
                    (mdir / "fit_info.json").write_text(json.dumps(
                        {"n_params": r["n_params"], "best_epoch": r["best_epoch"],
                         "best_score": r["best_score"], "monitor": r["cfg"]["monitor"],
                         "epochs_run": r["epochs_run"],
                         "device": r["device"], "in_ch": r["in_ch"], "cfg": r["cfg"]},
                        indent=2), encoding="utf-8")
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
                # exact float64 copy: CSV text round-trip is not bit-exact, and the
                # smoothed probabilities contain ties that AUROC is sensitive to
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

            results[tag] = {"metrics": met, "threshold": thr,
                            "train_seconds": train_seconds,
                            "n_params": None, "reused": bool(spec.get("reuse")),
                            "input_shape": list(X.shape)}
            print(f"[{name}] {tag}: val AUROC={met['val']['auroc']:.4f} "
                  f"test AUROC={met['test']['auroc']:.4f} thr={thr:.3f} "
                  f"({train_seconds}s)", flush=True)

    (exp_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (exp_dir / "environment.json").write_text(
        json.dumps(environment(device), indent=2), encoding="utf-8")
    (exp_dir / "run_config.json").write_text(json.dumps({
        "argv": sys.argv, "engine_root": str(ENGINE_ROOT),
        "data_csv": str(csv), "data_sha256_16": file_hash(Path(csv)),
        "out_dir": str(exp_dir), "seed": seed, "device": device,
        "torch_threads": __import__("torch").get_num_threads(),
        "elapsed_s": round(time.time() - t_start, 2),
        "windows": {"total": int(len(y)), "train": int(tr.sum()),
                    "val": int(va.sum()), "test": int(te.sum()),
                    "pos_train": int(y[tr].sum()), "pos_val": int(y[va].sum()),
                    "pos_test": int(y[te].sum())},
        "T_hash": schema.array_hash(T), "M_hash": schema.array_hash(M),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # CNN vs ET pairing table when both ran
    if "raw__cnn" in results and "raw__et" in results:
        c, e = results["raw__cnn"]["metrics"], results["raw__et"]["metrics"]
        comp = {"experiment": name, "seed": seed, "pair": "CNN - ET",
                "val": {k: (None if c["val"].get(k) is None else c["val"][k] - e["val"][k])
                        for k in ("auroc", "auprc", "f1", "sensitivity", "specificity")},
                "test": {k: (None if c["test"].get(k) is None else c["test"][k] - e["test"][k])
                         for k in ("auroc", "auprc", "f1", "sensitivity", "specificity")},
                "cnn": c["test"], "et": e["test"]}
        (exp_dir / "comparison.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
        rows = [{"experiment": name, "seed": seed, "model": m,
                 "split": s, **results[f"raw__{m}"]["metrics"][s]}
                for m in ("cnn", "et") for s in ("val", "test")]
        pd.DataFrame(rows).to_csv(exp_dir / "comparison_by_seed.csv", index=False)

    print(f"[{name}] done in {round(time.time() - t_start, 1)}s -> {exp_dir}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csv", default=str(ENGINE_ROOT / "data" / "sensor_data.csv"))
    ap.add_argument("--out", default=str(ENGINE_ROOT / "results" / "fair_ctrl_v2"))
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--cfg-json", default=None,
                    help="frozen recipe from src.fair_v2.tune (keys: cnn / et)")
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
    run_experiment(a.exp, a.seed, a.csv, out_root, a.device, cache_dir=out_root / "_cache",
                   cnn_cfg=cnn_cfg, et_cfg=et_cfg)


if __name__ == "__main__":
    main()
