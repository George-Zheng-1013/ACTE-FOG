"""The two model branches. Intrinsic differences allowed; protocol differences not.

CNN  : reference structure (OfficialCNN), val-patient early stopping, weighted BCE.
ET   : ExtraTrees, fixed integer max_features, sample_weight, class_weight=None.
Both receive the same per-window weight vector and the same split masks.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..models import OfficialCNN

CNN_CFG = {
    "arch": "OfficialCNN (120xC reference: conv9-16, conv7-16, pool4, conv5-24, pool3, conv5-24, gap, fc48, fc1)",
    "lr": 8e-3,
    "weight_decay": 4e-4,
    "optimizer": "AdamW",
    "batch_size": 64,
    "max_epochs": 120,
    "patience": 10,
    "min_delta": 1e-3,
    "l2_penalty": 1e-3,
    "l2_applies_to": "all Conv1d/Linear except fc2",
    "loss": "weighted BCEWithLogits (per-window w_c) + L2",
    "early_stop_metric": "val loss (weighted BCE + L2) on patients 12-16",
    "dropout": 0.4,
}

ET_CFG = {
    "n_estimators": 500,
    "min_samples_leaf": 3,
    "max_features": 16,
    "max_features_note": "fixed integer, not a fraction, so channel/dim changes do not alter candidate counts",
    "class_weight": None,
    "n_jobs": -1,
    "sample_weight": "same per-window w_c vector as CNN",
}


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _l2(model: nn.Module) -> torch.Tensor:
    s = torch.zeros((), device=next(model.parameters()).device)
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)) and name != "fc2":
            s = s + (m.weight * m.weight).sum()
    return s


@torch.no_grad()
def _predict(model: nn.Module, X: np.ndarray, dev: torch.device, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    ld = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch)
    for (x,) in ld:
        logit, _ = model(x.to(dev))
        out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _wloss(logit, yb, wb, bce):
    return (bce(logit, yb) * wb).mean()


def train_cnn(X, y, w, tr, va, te, seed=1, device="auto", cfg=None) -> dict:
    cfg = {**CNN_CFG, **(cfg or {})}
    dev = resolve_device(device)
    seed_all(seed)
    t0 = time.time()

    xtr = torch.from_numpy(np.ascontiguousarray(X[tr], np.float32))
    ytr = torch.from_numpy(y[tr].astype(np.float32))
    wtr = torch.from_numpy(w[tr].astype(np.float32))
    xva = torch.from_numpy(np.ascontiguousarray(X[va], np.float32))
    yva = torch.from_numpy(y[va].astype(np.float32))
    wva = torch.from_numpy(w[va].astype(np.float32))

    model = OfficialCNN(X.shape[2]).to(dev)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ld = DataLoader(TensorDataset(xtr, ytr, wtr), batch_size=cfg["batch_size"], shuffle=True)

    best, best_ep, bad = None, 0, 0
    hist = []
    for ep in range(1, cfg["max_epochs"] + 1):
        model.train()
        ls = []
        for xb, yb, wb in ld:
            xb, yb, wb = xb.to(dev), yb.to(dev), wb.to(dev)
            opt.zero_grad()
            logit, _ = model(xb)
            loss = _wloss(logit, yb, wb, bce) + cfg["l2_penalty"] * _l2(model)
            loss.backward()
            opt.step()
            ls.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            logit, _ = model(xva.to(dev))
            vl = float(_wloss(logit, yva.to(dev), wva.to(dev), bce).item()
                       + cfg["l2_penalty"] * _l2(model).item())
        hist.append({"epoch": ep, "train_loss": float(np.mean(ls)), "val_loss": vl})
        if vl < (best or 1e18) - cfg["min_delta"]:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break

    model.load_state_dict(best_state)
    p_va = _predict(model, np.ascontiguousarray(X[va], np.float32), dev)
    p_te = _predict(model, np.ascontiguousarray(X[te], np.float32), dev)
    return {
        "model": model, "state_dict": best_state, "n_params": n_params,
        "p_val": p_va, "p_test": p_te, "history": hist,
        "best_epoch": best_ep, "best_val_loss": best, "epochs_run": len(hist),
        "train_seconds": round(time.time() - t0, 2), "device": str(dev),
        "in_ch": int(X.shape[2]), "cfg": cfg,
    }


def train_et(X, y, w, tr, va, te, seed=1, cfg=None, device=None) -> dict:
    """ExtraTrees runs on CPU; `device` is accepted only for a uniform call signature."""
    from sklearn.ensemble import ExtraTreesClassifier

    cfg = {**ET_CFG, **(cfg or {})}
    seed_all(seed)
    t0 = time.time()
    model = ExtraTreesClassifier(
        n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"], class_weight=cfg["class_weight"],
        n_jobs=cfg["n_jobs"], random_state=seed)
    model.fit(X[tr], y[tr], sample_weight=w[tr])
    p_va = model.predict_proba(X[va])[:, 1]
    p_te = model.predict_proba(X[te])[:, 1]
    return {
        "model": model, "n_params": None,
        "n_features_in": int(X.shape[1]),
        "p_val": p_va.astype(np.float64), "p_test": p_te.astype(np.float64),
        "history": [], "train_seconds": round(time.time() - t0, 2),
        "device": "cpu", "cfg": cfg,
        "fit_info": {
            "n_train_windows": int(tr.sum()),
            "n_features": int(X.shape[1]),
            "class_counts": [int((y[tr] == 0).sum()), int((y[tr] == 1).sum())],
            "sample_weight_min_max": [float(w[tr].min()), float(w[tr].max())],
        },
    }
