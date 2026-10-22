"""Build the batch-1 comparison tables from saved artefacts.

    python -m src.fair_v2.report

Writes batch1_comparison.csv, batch1_deltas.csv and batch1_summary.md into
results/fair_ctrl_v2/. Deltas follow protocol 2.11.1 exactly:
model effect = CNN - ET; module-on effect = on - off; group contribution =
full - without_group.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ENGINE_ROOT = Path(__file__).resolve().parents[2]
OUT = ENGINE_ROOT / "results" / "fair_ctrl_v2"
KEYS = ["auroc", "auprc", "f1", "sensitivity", "specificity", "precision", "accuracy"]

# (family, ref_exp, new_exp, factor, direction_of_reading)
PAIRS = [
    ("M", "M", "M", "模型家族（CNN vs ET，同输入 S0）", "model"),
    ("P1", "P1", "M", "带通滤波 开 - 关", "new_minus_ref"),
    ("S5", "M", "S5", "加入右踝/背部/腕部通道（S5 - S0）", "new_minus_ref"),
    ("F", "F6", "F1", "协调特征组（F1 全组 - F6 去掉）", "new_minus_ref"),
    ("P6", "M", "P6", "七窗居中平滑 开 - 关", "new_minus_ref"),
]


def load(out: Path):
    rows = []
    for m in sorted(out.glob("*/*/metrics.json")):
        d = json.loads(m.read_text(encoding="utf-8"))
        for split in ("val", "test"):
            rows.append({"experiment": d["experiment"], "model": d["model"],
                         "representation": d["representation"], "split": split,
                         **{k: d[split].get(k) for k in KEYS},
                         "threshold": d["threshold"]})
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    df = load(out)
    df.to_csv(out / "batch1_comparison.csv", index=False)
    piv = {(r.experiment, r.model, r.split): r for r in df.itertuples()}
    tag = "（等预算调参配方）" if out.name.endswith("tuned") else "（固定参数配方）"

    deltas = []
    for fam, ref, new, factor, mode in PAIRS:
        for split in ("val", "test"):
            if mode == "model":
                a = piv[(new, "cnn", split)]
                b = piv[(new, "et", split)]
                deltas.append({"family": fam, "split": split, "factor": factor,
                               "contrast": f"{new} CNN - {new} ET", "changed": "model family",
                               "cnn_or_new": None,
                               **{f"d_{k}": (None if getattr(a, k) is None or getattr(b, k) is None
                                             else getattr(a, k) - getattr(b, k)) for k in KEYS}})
                continue
            for model in ("cnn", "et"):
                ka, kb = (new, model, split), (ref, model, split)
                if ka not in piv or kb not in piv:
                    continue
                a, b = piv[ka], piv[kb]
                deltas.append({"family": fam, "split": split, "factor": factor,
                               "contrast": f"{new} - {ref} ({model})", "changed": factor,
                               **{f"d_{k}": (None if getattr(a, k) is None or getattr(b, k) is None
                                             else getattr(a, k) - getattr(b, k)) for k in KEYS}})
    dd = pd.DataFrame(deltas)
    dd.to_csv(out / "batch1_deltas.csv", index=False)

    acc = {}
    for e in sorted(q.name for q in out.iterdir() if (q / "acceptance.json").exists()):
        rows = json.loads((out / e / "acceptance.json").read_text(encoding="utf-8"))
        acc[e] = (sum(r["ok"] for r in rows), len(rows))

    lines = [f"# fair_ctrl_v2 第一批（seed=1）汇总{tag}", "",
             "自动生成，数据源为各实验目录下的 metrics.json / acceptance.json。", "",
             "## 验收", "", "| 实验 | 通过 |", "|---|---:|"]
    for e, (a, b) in acc.items():
        lines.append(f"| {e} | {a}/{b} |")

    lines += ["", "## 测试集指标", "",
              "| 实验 | 模型 | 表示 | 阈值 | AUROC | AUPRC | F1 | 敏感度 | 特异度 | accuracy |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in df[df.split == "test"].itertuples():
        lines.append(f"| {r.experiment} | {r.model} | {r.representation} | {r.threshold:.3f} | "
                     f"{r.auroc:.4f} | {r.auprc:.4f} | {r.f1:.4f} | {r.sensitivity:.4f} | "
                     f"{r.specificity:.4f} | {r.accuracy:.4f} |")

    lines += ["", "## 成对差值", "",
              "| 实验族 | 划分 | 对照 | 主动改变 | ΔAUROC | ΔAUPRC | ΔF1 |",
              "|---|---|---|---|---:|---:|---:|"]
    for r in dd.itertuples():
        lines.append(f"| {r.family} | {r.split} | {r.contrast} | {r.changed} | "
                     f"{r.d_auroc:+.4f} | {r.d_auprc:+.4f} | {r.d_f1:+.4f} |")
    (out / "batch1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
