"""Aggregate multi-seed batch results (protocol 2.11.1 / 2.11.2).

Per experiment and model: mean and sample sd across seeds.
Per contrast: per-seed paired difference, then mean and sd of those differences
(never a difference of independently averaged numbers).

    python -m src.fair_v2.report_batch2 --root results/fair_ctrl_v2_batch2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import PAIRS, model_dir

ENGINE_ROOT = Path(__file__).resolve().parents[2]
KEYS = ["auroc", "auprc", "f1", "sensitivity", "specificity", "precision", "accuracy"]


def collect(root: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("_cache",)):
        for s in seeds:
            for model in ("cnn", "et"):
                md = model_dir(exp_dir / f"seed{s}", model)
                f = md / "metrics.json" if md else None
                if f is None or not f.exists():
                    continue
                d = json.loads(f.read_text(encoding="utf-8"))
                for split in ("val", "test"):
                    rows.append({"experiment": exp_dir.name, "model": model,
                                 "representation": d["representation"], "seed": s,
                                 "split": split, "threshold": d["threshold"],
                                 **{k: d[split].get(k) for k in KEYS}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ENGINE_ROOT / "results" / "fair_ctrl_v2_batch2"))
    ap.add_argument("--seeds", default="1,2,3,4,5")
    a = ap.parse_args()
    root = Path(a.root)
    seeds = [int(s) for s in a.seeds.split(",")]
    df = collect(root, seeds)
    df.to_csv(root / "batch2_runs.csv", index=False)

    test = df[df.split == "test"]
    agg = (test.groupby(["experiment", "model", "representation"], as_index=False)
           .agg(n=("seed", "nunique"),
                auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                auprc=("auprc", "mean"), auprc_sd=("auprc", "std"),
                f1=("f1", "mean"), f1_sd=("f1", "std"),
                sensitivity=("sensitivity", "mean"), specificity=("specificity", "mean"),
                accuracy=("accuracy", "mean")))
    agg.to_csv(root / "batch2_aggregate.csv", index=False)

    # paired per-seed deltas
    piv = {(r.experiment, r.model, r.seed): r for r in test.itertuples()}
    deltas = []
    for fam, ref, var, label in PAIRS:
        for m in ("cnn", "et"):
            pairs = [(piv[(var, m, s)], piv[(ref, m, s)]) for s in seeds
                     if (var, m, s) in piv and (ref, m, s) in piv]
            if not pairs:
                continue
            row = {"family": fam, "model": m, "contrast": label, "n_seeds": len(pairs)}
            for k in ("auroc", "auprc", "f1"):
                d = np.array([b._asdict()[k] - a._asdict()[k] for b, a in pairs], float)
                row[f"d_{k}"] = float(d.mean())
                row[f"d_{k}_sd"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
            row["ref_auroc"] = float(np.mean([a.auroc for _, a in pairs]))
            row["var_auroc"] = float(np.mean([b.auroc for b, _ in pairs]))
            deltas.append(row)
    # CNN - ET model contrast per experiment
    for e in sorted(test.experiment.unique()):
        pairs = [(piv[(e, "cnn", s)], piv[(e, "et", s)]) for s in seeds
                 if (e, "cnn", s) in piv and (e, "et", s) in piv]
        if not pairs:
            continue
        row = {"family": "M" if e == "M" else "model", "model": "cnn-et",
               "contrast": f"{e}: CNN - ET", "n_seeds": len(pairs)}
        for k in ("auroc", "auprc", "f1"):
            d = np.array([b._asdict()[k] - a._asdict()[k] for b, a in pairs], float)
            row[f"d_{k}"] = float(d.mean())
            row[f"d_{k}_sd"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        row["ref_auroc"] = float(np.mean([a.auroc for _, a in pairs]))
        row["var_auroc"] = float(np.mean([b.auroc for b, _ in pairs]))
        deltas.append(row)
    dd = pd.DataFrame(deltas)
    dd.to_csv(root / "batch2_deltas.csv", index=False)

    lines = [f"# fair_ctrl_v2 第二批（seeds={seeds}）", "",
             "自动生成。指标为各种子分别计算后的均值 ± 样本标准差；差值为逐种子配对差值的均值 ± 标准差。",
             "", "## 测试集指标", "",
             "| 实验 | 模型 | 表示 | n | AUROC | AUPRC | F1 | 敏感度 | 特异度 |",
             "|---|---|---|---:|---|---|---|---|---|"]
    for r in agg.sort_values(["experiment", "model"]).itertuples():
        lines.append(f"| {r.experiment} | {r.model} | {r.representation} | {r.n} | "
                     f"{r.auroc:.4f} ± {r.auroc_sd:.4f} | {r.auprc:.4f} ± {r.auprc_sd:.4f} | "
                     f"{r.f1:.4f} ± {r.f1_sd:.4f} | {r.sensitivity:.4f} | {r.specificity:.4f} |")
    lines += ["", "## 成对差值", "",
              "| 实验族 | 模型 | 对照 | n | ΔAUROC | ΔAUPRC | ΔF1 |", "|---|---|---|---:|---|---|---|"]
    for r in dd.itertuples():
        lines.append(f"| {r.family} | {r.model} | {r.contrast} | {r.n_seeds} | "
                     f"{r.d_auroc:+.4f} ± {r.d_auroc_sd:.4f} | "
                     f"{r.d_auprc:+.4f} ± {r.d_auprc_sd:.4f} | "
                     f"{r.d_f1:+.4f} ± {r.d_f1_sd:.4f} |")
    (root / "batch2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220)
    print(agg.sort_values(["experiment", "model"]).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
