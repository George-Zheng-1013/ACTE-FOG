"""缺失感知 STMAE（FoG 适配版，非上游原样复现）。

与上游的差异（必须记录）：
  * 输入为「有限时序值 + 原始观测掩码 + 组可用性」，新增掩码投影，使"真实 0"与
    "不可用占位 0"在表示上不同；
  * 保留独立重建目标（上游对设备置零后取目标）；
  * 重建损失只计「原本有可靠观测 ∩ 本轮被人工隐藏」的位置，逐视图归一后按视图权重聚合；
  * 重建误差按训练患者每物理通道标准差归一；常值通道用安全尺度 1；
  * squeeze 只压指定维，batch=1 不丢批次维。
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class MaskedSTMAE(nn.Module):
    """输入 (B,120,24) 有限值 + (B,120,24) 观测掩码 + (B,8) 组可用性。"""

    def __init__(self, n_ch=24, n_groups=8, win=120, dim=512, layers=6, heads=4,
                 dec_dim=256, dec_layers=2, dec_heads=4, mlp_ratio=4.0, n_class=1):
        super().__init__()
        self.win, self.n_ch, self.n_groups = win, n_ch, n_groups
        # 每个时间步的输入：24 值 + 24 观测位 + 24 通道级组可用位
        self.proj = nn.Linear(n_ch * 3, dim)
        self.pos = nn.Parameter(torch.zeros(1, win, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.enc_blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(layers)])
        self.enc_norm = nn.LayerNorm(dim)
        self.dec_in = nn.Linear(dim, dec_dim)
        self.dec_pos = nn.Parameter(torch.zeros(1, win, dec_dim))
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads, mlp_ratio)
                                         for _ in range(dec_layers)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_out = nn.Linear(dec_dim, n_ch)
        self.head = nn.Linear(dim, n_class)

    def _tokens(self, x, mask, group_ok):
        B, T, C = x.shape
        gm = group_ok.repeat_interleave(3, dim=1).to(x.dtype).unsqueeze(1).expand(B, T, C)
        feat = torch.cat([x, mask, gm], dim=2)
        return self.proj(feat) + self.pos

    def forward(self, x, mask, group_ok, return_recon=False):
        h = torch.cat([self.cls.expand(len(x), -1, -1), self._tokens(x, mask, group_ok)], 1)
        for b in self.enc_blocks:
            h = b(h)
        h = self.enc_norm(h)
        cls, tok = h[:, 0], h[:, 1:]
        logit = self.head(cls).squeeze(-1)
        if not return_recon:
            return logit
        d = self.dec_in(tok) + self.dec_pos
        for b in self.dec_blocks:
            d = b(d)
        return logit, self.dec_out(self.dec_norm(d))

    @torch.no_grad()
    def encode(self, x, mask, group_ok):
        """冻结特征：末层归一化 -> 去 class token -> 120 个时间 token 均值。"""
        h = torch.cat([self.cls.expand(len(x), -1, -1), self._tokens(x, mask, group_ok)], 1)
        for b in self.enc_blocks:
            h = b(h)
        return self.enc_norm(h)[:, 1:].mean(1)


def channel_scale(x: np.ndarray, mask: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """训练患者每物理通道的有效标准差；常值或全缺失用 1。"""
    v = x[fit_rows]
    m = mask[fit_rows]
    out = np.ones(v.shape[2], np.float64)
    for c in range(v.shape[2]):
        vals = v[:, :, c][m[:, :, c]]
        if len(vals) > 1:
            s = float(np.std(vals))
            if s > 0 and math.isfinite(s):
                out[c] = s
    return out


def recon_loss(recon, target, valid_original, artificial_hidden, scale, view_w):
    """逐视图归一 -> 按视图权重聚合。只算"原本有效且被人工隐藏"的位置。"""
    err = (recon - target) ** 2 / torch.as_tensor(scale, dtype=recon.dtype,
                                                  device=recon.device) ** 2
    sel = valid_original * artificial_hidden
    denom = sel.sum(dim=(1, 2))
    num = (err * sel).sum(dim=(1, 2))
    ok = denom > 0
    if not ok.all():
        raise RuntimeError("a view has no reconstruction target; mask logic is wrong")
    per_view = num / denom
    w = torch.as_tensor(view_w, dtype=per_view.dtype, device=per_view.device)
    return (per_view * w).sum() / w.sum()


def time_mask(x, mask, group_ok, ratio, gen):
    """时间 token 遮蔽：仅用于 SSL 训练，返回值副本与隐藏位。"""
    B, T, _ = x.shape
    hidden = torch.zeros_like(mask)
    keep = torch.rand(B, T, generator=gen, device=x.device) >= ratio
    hidden[:] = ~keep.unsqueeze(-1)
    return torch.where(hidden, torch.zeros_like(x), x), hidden
