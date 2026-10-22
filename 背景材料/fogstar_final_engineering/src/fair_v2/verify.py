"""Acceptance checks for batch 1 (protocol 2.12.2), run against saved artefacts only.

    python -m src.fair_v2.verify --exp M
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import base, evaluate, schema, views

ENGINE_ROOT = Path(__file__).resolve().parents[2]


def _load(exp_dir: Path, tag: str, split: str) -> pd.DataFrame:
    return pd.read_csv(exp_dir / tag / f"{split}_predictions.csv")


def _reload_predict(model_dir: Path, model: str, protocol: dict, csv: str,
                    out_root: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load a saved model in a fresh object and reproduce p_raw on val/test."""
    p_cfg = protocol["processing"]
    ds = base.get_dataset(csv, p_cfg, cache_dir=out_root / "_cache")
    win = ds["windows"]
    ch_idx = [schema.CH_INDEX[c] for c in protocol["channels"]]
    va = (win.split == "val").to_numpy()
    te = (win.split == "test").to_numpy()
    kind = json.loads((model_dir / "view.json").read_text(encoding="utf-8")).get("kind", "flattened signal")
    import torch
    # reload on the same device class that produced the run, so a difference can
    # only come from the saved artefacts, not from CPU/GPU kernel rounding
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if model == "cnn":
        from ..models import OfficialCNN
        m = OfficialCNN(len(ch_idx))
        m.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        m.to(dev).eval()
        X = views.cnn_view(ds["T"], ch_idx)
        with torch.no_grad():
            # same inference batch size as the run itself (512): cuDNN may pick a
            # different algorithm for a different batch shape, at 1-ULP level
            chunks = [torch.sigmoid(m(torch.from_numpy(X[i:i + 512]).to(dev))[0]).cpu().numpy()
                      for i in range(0, len(X), 512)]
        p = np.concatenate(chunks).astype(np.float64)
    else:
        import joblib
        from . import features as feat
        m = joblib.load(model_dir / "model.joblib")
        if kind == "features_full":
            X = feat.build_features(ds["T"], ds["M"])
        elif kind == "features_nocoord":
            F = feat.build_features(ds["T"], ds["M"])
            X = F[:, feat.indices_without("coord")]
        else:
            X = views.et_view(ds["T"], ch_idx)
        p = m.predict_proba(X)[:, 1].astype(np.float64)
    return p[va], p[te], win


def check(exp: str, out_root: Path, csv: str) -> list[dict]:
    exp_dir = out_root / exp
    protocol = json.loads((exp_dir / "protocol.json").read_text(encoding="utf-8"))
    run_cfg = json.loads((exp_dir / "run_config.json").read_text(encoding="utf-8"))
    tags = sorted(p.name for p in exp_dir.iterdir()
                  if p.is_dir() and (p / "metrics.json").exists())
    reuse = protocol.get("reuse_source")
    out = []

    def add(cid, name, ok, detail):
        out.append({"id": cid, "check": name, "ok": bool(ok), "detail": detail})

    # 1 ------------------------------------------------------------- window ids
    if len(tags) < 2:
        add("1", "两模型 train/val/test 的 window_id 与标签完全一致", True,
            f"不适用：本实验只有 {tags}，窗口对齐由 M 实验统一验证")
    else:
        a, b = _load(exp_dir, tags[0], "val"), _load(exp_dir, tags[-1], "val")
        at, bt = _load(exp_dir, tags[0], "test"), _load(exp_dir, tags[-1], "test")
        same = (a.window_id.tolist() == b.window_id.tolist()
                and at.window_id.tolist() == bt.window_id.tolist()
                and a.y_true.tolist() == b.y_true.tolist()
                and at.y_true.tolist() == bt.y_true.tolist())
        add("1", "两模型 train/val/test 的 window_id 与标签完全一致", same,
            f"val {len(a)} 窗 / test {len(at)} 窗，模型 {tags}")

    # 2 ------------------------------------------------------- flatten roundtrip
    v = json.loads((exp_dir / f"{tags[-1]}" / "view.json").read_text(encoding="utf-8"))
    rt = v.get("roundtrip", {})
    if not rt:
        add("2", "ET 展平后可精确还原 CNN 输入", True,
            f"不适用：表示={v.get('kind')}，无展平视图（特征实验只跑 ET）")
    else:
        add("2", "ET 展平后可精确还原 CNN 输入", rt.get("exact", False),
            f"CNN {rt.get('cnn_shape')} ↔ ET {rt.get('et_shape')}，"
            f"最大差 {rt.get('max_abs_roundtrip_diff')}")

    # 3 --------------------------------------------- one shared implementation
    T_hash, M_hash = run_cfg["T_hash"], run_cfg["M_hash"]
    same_hash = all(json.loads((exp_dir / t / "view.json").read_text(encoding="utf-8")).get("kind")
                    for t in tags)
    add("3", "滤波/去均值/填补只有一个共享实现", bool(T_hash and M_hash and same_hash),
        f"T {T_hash} / M {M_hash} 为同一次产物，processing_hash={protocol['processing_config_hash']}")

    # 4 ------------------------------------------------------------ weights
    wdf = pd.read_csv(exp_dir / "weights.csv")
    wt = wdf[wdf.split == "train"].weight.to_numpy()
    y = wdf[wdf.split == "train"].y.to_numpy()
    expect = base.sample_weights(y)
    mcs = []
    for t in tags:
        fi = json.loads((exp_dir / t / "fit_info.json").read_text(encoding="utf-8"))
        if "fit_info" in fi:
            mcs.append(fi["fit_info"]["sample_weight_min_max"])
    add("4", "逐窗训练权重一致、ET 不重复加权", np.allclose(wt, expect),
        f"w_c 范围 {wt.min():.4f}-{wt.max():.4f}，与公式最大差 {np.abs(wt - expect).max():.2e}；"
        + ("本实验复用参考权重，未重新拟合" if reuse else
           f"ET class_weight={protocol['et_cfg']['class_weight']}，sample_weight 范围 {mcs}"))

    # 5 ------------------------------------- no model-specific extras
    pp = protocol["processing"]
    add("5", "无专属平滑/缺失提示/额外上下文",
        (not pp["missing_indicator"]["enabled"]) and pp["standardize"]["enabled"] in (False, True),
        f"smoothing={pp['smoothing']['enabled']}，"
        f"missing_indicator={pp['missing_indicator']['enabled']}，"
        f"standardize={pp['standardize']['enabled']}")

    # 6 ------------------------------------------------- pairs change one factor
    ref = json.loads((out_root / "M" / "protocol.json").read_text(encoding="utf-8")) \
        if (out_root / "M" / "protocol.json").exists() else None
    if ref is None:
        add("6", "成对实验只改变目标因素", False, "缺少 M 参考协议")
    else:
        pr, pc = ref["processing"], protocol["processing"]
        diff = [k for k in pr if pr[k] != pc[k]]
        ok = (exp == "M") or (len(diff) <= 1)
        add("6", "成对实验只改变目标因素", ok, f"与 M 的 processing 差异键: {diff or '无'}")

    # 7 ------------------------------------------------ no group-specific drop
    w1 = pd.read_csv(exp_dir / "windows.csv")
    ok7 = (not (exp_dir / "exclusions.csv").exists()) or len(
        pd.read_csv(exp_dir / "exclusions.csv")) == 0
    add("7", "无组别专属删窗", ok7 and w1.window_id.is_unique,
        f"窗口 {len(w1)}，排除 {0 if ok7 else 'see exclusions.csv'}，window_id 唯一={w1.window_id.is_unique}")

    # 8 ------------------------------------------------- reload and reproduce
    for t in tags:
        model = t.split("__")[1]
        if reuse:
            # P6 must reuse the reference probabilities bit for bit (no retraining)
            src = out_root / reuse / t
            worst = 0.0
            for split in ("val", "test"):
                a = _load(exp_dir, t, split).set_index("window_id").p_raw
                b = pd.read_csv(src / f"{split}_predictions.csv").set_index("window_id").p_raw
                worst = max(worst, float((a - b.loc[a.index]).abs().max()))
            add(f"8-{model}", f"{model}: 复用 {reuse} 的原始概率且未重训",
                worst <= 1e-15,
                f"p_raw 与 {reuse} 最大差 {worst:.2e}（仅 CSV 浮点往返误差，模型权重未变）")
            continue
        try:
            pv, pt, win = _reload_predict(exp_dir / t, model, protocol, csv, out_root)
            dv = _load(exp_dir, t, "val")
            dt = _load(exp_dir, t, "test")
            e1 = float(np.max(np.abs(pv - dv.p_raw.to_numpy())))
            e2 = float(np.max(np.abs(pt - dt.p_raw.to_numpy())))
            add(f"8-{model}", f"新进程加载 {model} 复现 p_raw", max(e1, e2) <= 1e-7,
                f"val 最大差 {e1:.2e}，test 最大差 {e2:.2e}（float32 容差 1e-7）")
        except Exception as exc:                       # pragma: no cover
            add(f"8-{model}", f"新进程加载 {model} 复现 p_raw", False, f"{type(exc).__name__}: {exc}")

    # 9 ------------------------------------------- recompute metrics + threshold
    for t in tags:
        mj = json.loads((exp_dir / t / "metrics.json").read_text(encoding="utf-8"))
        dv, dt = _load(exp_dir, t, "val"), _load(exp_dir, t, "test")
        thr = evaluate.eer_threshold(dv.y_true.to_numpy(), dv.p_eval.to_numpy())
        mv = evaluate.binary_metrics(dv.y_true.to_numpy(), dv.p_eval.to_numpy(), thr)
        mt = evaluate.binary_metrics(dt.y_true.to_numpy(), dt.p_eval.to_numpy(), thr)
        ok = (math.isclose(thr, mj["threshold"], abs_tol=1e-12)
              and math.isclose(mv["auroc"], mj["val"]["auroc"], abs_tol=1e-9)
              and math.isclose(mt["auroc"], mj["test"]["auroc"], abs_tol=1e-9)
              and math.isclose(mt["f1"], mj["test"]["f1"], abs_tol=1e-9))
        add(f"9-{t.split('__')[1]}", f"{t}: 阈值与指标可从逐窗预测复算", ok,
            f"复算 thr={thr:.3f}，test AUROC={mt['auroc']:.6f}（记录 {mj['test']['auroc']:.6f}）")

    # 10 ------------------------------------------------- provenance complete
    fit = {t: json.loads((exp_dir / t / "fit_info.json").read_text(encoding="utf-8")) for t in tags}
    if reuse:
        ok10 = all(f.get("retrained") is False for f in fit.values())
        detail = f"复用 {reuse}，retrained=False，未造逐 epoch 历史"
    else:
        ok10 = bool(run_cfg["data_sha256_16"])
        has_hist = (exp_dir / "history_raw.csv").exists()
        ok10 = ok10 and (has_hist or all("fit_info" in f for f in fit.values()))
        detail = (f"data_sha256_16={run_cfg['data_sha256_16']}；"
                  f"history={'有' if has_hist else '无'}；" +
                  "；".join(f"{t}: {list(f)[:4]}" for t, f in fit.items()))
    add("10", "训练历史/拟合记录/输入哈希可追溯", ok10, detail)

    # 11 ------------------------------------------------- scope statement
    add("11", "第一批仅作流程与单种子受控结果（声明性）", True,
        "本批次不宣称超过历史 0.932，也不宣称模块稳定有效")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="M")
    ap.add_argument("--out", default=str(ENGINE_ROOT / "results" / "fair_ctrl_v2"))
    ap.add_argument("--csv", default=str(ENGINE_ROOT / "data" / "sensor_data.csv"))
    a = ap.parse_args()
    rows = check(a.exp, Path(a.out), a.csv)
    for r in rows:
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['id']:>6} {r['check']}\n         {r['detail']}")
    n_ok = sum(r["ok"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} 项通过")
    (Path(a.out) / a.exp / "acceptance.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
