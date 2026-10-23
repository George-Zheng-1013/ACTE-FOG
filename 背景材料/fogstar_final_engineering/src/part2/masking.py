"""空间屏蔽清单：每个训练原窗 1 个自然版本 + 3 个人工版本。

规则（Q11 / 12.1）：
  * 只从质量合格组中选择；
  * 至少保留一侧合格踝 gyro；
  * 随机种子由 seed 与 window_id 决定，与模型名无关，所有增强行共用同一清单；
  * 4 个版本平分原窗权重。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import schema


def _rng(seed: int, window_id: str) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}|{window_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _ankle_gyro_ok(keep: np.ndarray) -> bool:
    return bool(keep[schema.GROUP_INDEX[("ankleL", "gyro")]]
                or keep[schema.GROUP_INDEX[("ankleR", "gyro")]])


def build(train_windows: pd.DataFrame, G: np.ndarray, seed: int) -> dict:
    """返回每个原窗的 4 个 keep 版本与对应权重倍率。"""
    n = len(train_windows)
    keeps = np.zeros((n, 4, len(schema.GROUPS)), bool)
    counts = {"spatial": 0, "mixed": 0, "fallback_repeat": 0}

    for i, wid in enumerate(train_windows.window_id):
        g = G[i].copy()
        keeps[i, 0] = g                                  # 自然版本
        if not _ankle_gyro_ok(g):
            keeps[i, 1:] = g                             # 不可预测窗口不生成人工版本
            continue
        rng = _rng(seed, str(wid))
        nodes_ok = [ni for ni, node in enumerate(schema.NODES)
                    if g[ni * 2] or g[ni * 2 + 1]]
        ankle_nodes = [ni for ni in nodes_ok if schema.NODES[ni] in ("ankleL", "ankleR")]
        other_nodes = [ni for ni in nodes_ok if ni not in ankle_nodes]

        def node_keep(subset: list[int]) -> np.ndarray:
            k = np.zeros(len(schema.GROUPS), bool)
            for ni in subset:
                k[ni * 2] = g[ni * 2]
                k[ni * 2 + 1] = g[ni * 2 + 1]
            return k

        # 版本 1：2 个部位（必含一侧合格踝）
        pick = [ankle_nodes[rng.integers(len(ankle_nodes))]]
        rest = [x for x in nodes_ok if x not in pick]
        if rest:
            pick.append(rest[rng.integers(len(rest))])
        keeps[i, 1] = node_keep(pick)
        counts["spatial"] += 1

        # 版本 2：3 个部位（必含一侧合格踝）
        pick2 = list(pick)
        rest2 = [x for x in nodes_ok if x not in pick2]
        if rest2:
            pick2.append(rest2[rng.integers(len(rest2))])
        keeps[i, 2] = node_keep(pick2)
        counts["spatial"] += 1

        # 版本 3：部位 + 信号组混合，保证至少一侧踝 gyro
        k = np.zeros(len(schema.GROUPS), bool)
        chosen = pick2 if len(pick2) >= 2 else pick
        ank = [ni for ni in chosen if schema.NODES[ni] in ("ankleL", "ankleR")]
        ank_pick = ank[rng.integers(len(ank))]
        k[ank_pick * 2 + 1] = g[ank_pick * 2 + 1]        # 该踝 gyro 必留
        for ni in chosen:
            for si in (0, 1):
                if ni == ank_pick and si == 1:
                    continue
                if g[ni * 2 + si] and rng.random() < 0.6:
                    k[ni * 2 + si] = True
        if not _ankle_gyro_ok(k):
            k[ank_pick * 2 + 1] = g[ank_pick * 2 + 1]
            counts["fallback_repeat"] += 1
        keeps[i, 3] = k
        counts["mixed"] += 1

    return {"keeps": keeps, "counts": counts,
            "n_versions": 4, "weight_split": 0.25,
            "seed": seed, "rule": schema.MASKING["rules"]}


def expand(windows: pd.DataFrame, G: np.ndarray, weights: np.ndarray,
           manifest: dict) -> dict:
    """把训练窗展开成 4N 行（含自然版本），权重各为 1/4。"""
    keeps = manifest["keeps"]
    n, v, _ = keeps.shape
    parent = np.repeat(np.arange(n), v)
    version = np.tile(np.arange(v), n)
    return {
        "parent": parent,
        "version": version,
        "keep": keeps.reshape(n * v, -1),
        "weight": np.repeat(weights * manifest["weight_split"], v),
        "window_id": np.repeat(windows.window_id.to_numpy(), v),
        "y": np.repeat(windows.y.to_numpy(), v),
    }


def requested_keep(G: np.ndarray, request: str) -> np.ndarray:
    """评价用：把部位保留请求施加到自然可用组上（effective = G ∩ 请求）。"""
    nodes = schema.request_to_nodes(request)
    keep = np.zeros_like(G)
    for ni, node in enumerate(schema.NODES):
        if node in nodes:
            keep[:, ni * 2] = G[:, ni * 2]
            keep[:, ni * 2 + 1] = G[:, ni * 2 + 1]
    return keep
