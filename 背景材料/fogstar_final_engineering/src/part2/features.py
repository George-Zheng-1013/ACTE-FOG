"""ACTE72 的缺失语义版本（acte72_missing_v1）。

与 Part 1 的差别只在缺失处理，公式与列顺序不变：
    [STD 12][RMS 12][MAD 12][freeze 12][missing 12][coord 12] = 72
* 不可用组 / 被人工屏蔽组对应的运动统计与频谱特征记 NaN，不计算"零运动"统计；
* 缺失比例始终有定义：按 (M_raw ∩ keep_group) 计算；
* 协调项任一节点不可用则记 NaN，不用 0 伪装"不相关"。
"""
from __future__ import annotations

import numpy as np

from . import schema

LOCo = (0.5, 3.0)
FREEZE = (3.0, 8.0)

# 双踝 12 个通道（ankleL 6 + ankleR 6）在每个 24 维块里的相对位置
ANKLE_CH = list(range(0, 12))

BLOCKS = {"std": (0, 12), "rms": (12, 24), "mad": (24, 36),
          "freeze": (36, 48), "missing": (48, 60), "coord": (60, 72)}

FEATURE_NAMES = (
    [f"std_{schema.CHANNELS[i]}" for i in ANKLE_CH]
    + [f"rms_{schema.CHANNELS[i]}" for i in ANKLE_CH]
    + [f"mad_{schema.CHANNELS[i]}" for i in ANKLE_CH]
    + [f"freeze_{schema.CHANNELS[i]}" for i in ANKLE_CH]
    + [f"missing_{schema.CHANNELS[i]}" for i in ANKLE_CH]
    + [f"coord_{s}_{i}_{j}" for s in schema.SIGS for i in range(4) for j in range(i + 1, 4)]
)
assert len(FEATURE_NAMES) == 72

GROUP_EFFECT = [f"{n}_{s}" for n, s in schema.GROUPS]


def build(T: np.ndarray, M: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T: (N,120,24) 已去均值、不可用组为 NaN；M: 原始观测；keep: (N,8) 人工保留。

    返回 F (N,72) 可含 NaN，V (N,72) 特征有效性。
    """
    N, W, _ = T.shape
    F = np.full((N, 72), np.nan, np.float32)
    fs = schema.FS
    f = np.fft.rfftfreq(W, d=1.0 / fs)
    loco = (f >= LOCo[0]) & (f < LOCo[1])
    fre = (f >= FREEZE[0]) & (f < FREEZE[1])

    for i in range(N):
        z = T[i]
        m_eff = M[i] & np.repeat(keep[i], 3)          # (120,24) 有效观测 = 原始 ∩ 人工保留
        for k, ch in enumerate(ANKLE_CH):
            node, sig = schema.GROUPS[ch // 6]
            gi = ch // 6 * 2 + (1 if sig == "gyro" else 0)
            if not keep[i, gi] or not np.isfinite(z[:, ch]).all():
                continue                               # 四个块保持 NaN
            x = z[:, ch].astype(np.float64)
            F[i, 0 * 12 + k] = x.std()
            F[i, 1 * 12 + k] = np.sqrt((x ** 2).mean())
            F[i, 2 * 12 + k] = np.abs(np.diff(x)).mean()
            sp = np.abs(np.fft.rfft(x)) ** 2
            pl = sp[loco].sum(); pf = sp[fre].sum()
            F[i, 3 * 12 + k] = np.log1p(pf / (pl + 1e-6))
            # 缺失比例始终有定义
            F[i, 4 * 12 + k] = 1.0 - m_eff[:, ch].mean()

        # 协调：两两节点模长相关，四个节点的 acc（或 gyro）都要可用
        for si, sig in enumerate(schema.SIGS):
            mags = []
            for ni, node in enumerate(schema.NODES):
                sl = schema.group_slice(node, sig)
                gi = ni * 2 + si
                if not keep[i, gi] or not np.isfinite(z[:, sl]).all():
                    mags = None
                    break
                mags.append(np.sqrt((z[:, sl].astype(np.float64) ** 2).sum(1)))
            if mags is None:
                continue
            Mx = np.stack(mags)                        # (4,120)
            c = np.corrcoef(Mx)
            if not np.isfinite(c).all():
                continue
            iu = np.triu_indices(4, 1)
            F[i, 60 + si * 6: 60 + (si + 1) * 6] = c[iu[0], iu[1]]

    F = F.astype(np.float32)
    return F, np.isfinite(F)


def to_finite(F: np.ndarray, V: np.ndarray) -> np.ndarray:
    """占位 0 + 有效性标记路线的输入：144 列。"""
    return np.concatenate([np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0),
                           V.astype(np.float32)], axis=1)
