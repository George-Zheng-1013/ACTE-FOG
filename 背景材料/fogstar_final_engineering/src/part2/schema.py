"""Part 2 的固定词表与协议常量。

节点顺序与通道顺序是跨数据集契约的一部分：任何患者、任何数据集都映射到同一
24 列，缺列不压缩、不改序，只把该组标为不可用。
"""
from __future__ import annotations

import hashlib
import json

PROTOCOL_VERSION = "part2_v1_20261023"

NODES = ["ankleL", "ankleR", "back", "wrist_affected"]
SIGS = ["acc", "gyro"]
AXES = ["x", "y", "z"]

# node -> sig -> axis，24 列
CHANNELS = [f"{n}_{s}_{a}" for n in NODES for s in SIGS for a in AXES]
CH_INDEX = {c: i for i, c in enumerate(CHANNELS)}

# 8 个三轴组，顺序固定
GROUPS = [(n, s) for n in NODES for s in SIGS]
GROUP_INDEX = {g: i for i, g in enumerate(GROUPS)}


def group_columns(node: str, sig: str) -> list[str]:
    return [f"{node}_{sig}_{a}" for a in AXES]


def group_slice(node: str, sig: str) -> slice:
    base = CH_INDEX[f"{node}_{sig}_x"]
    return slice(base, base + 3)


# 12 个含至少一侧踝的部位保留请求（主配置集）
ANKLE_CONFIGS = ["L", "L+B", "L+W", "L+B+W",
                 "R", "R+B", "R+W", "R+B+W",
                 "LR", "LR+B", "LR+W", "LR+B+W"]

# 额外信号类型检查（不进主宏平均）
SIGNAL_CHECKS = ["gyro_only_all", "ankleL_gyro_only", "ankleR_gyro_only"]


def request_to_nodes(req: str) -> set[str]:
    """把 L+B+W 这类请求解析成节点集合。"""
    nodes = set()
    i = 0
    while i < len(req):
        if req.startswith("LR", i):
            nodes |= {"ankleL", "ankleR"}; i += 2
        elif req[i] == "L":
            nodes.add("ankleL"); i += 1
        elif req[i] == "R":
            nodes.add("ankleR"); i += 1
        elif req[i] == "B":
            nodes.add("back"); i += 1
        elif req[i] == "W":
            nodes.add("wrist_affected"); i += 1
        elif req[i] == "+":
            i += 1
        else:
            raise ValueError(req)
    return nodes


FS = 60.0
WINDOW = 120
STRIDE = 30

FILTER = {"enabled": True, "order": 4, "band_hz": [0.5, 20.0], "min_run": 31,
          "short_run_policy": "keep_raw"}

QUALITY = {
    "min_coverage": 0.90,      # 每个三轴组各轴原始覆盖率
    "max_internal_gap": 6,     # 点，允许插值的内部连续缺口上限
    "edge_policy": "no_extrapolation",
    "min_input": "at least one ankle gyro group qualified",
    "note": "工程起点，不是临床标准；不按测试分数调整",
}

SPLITS = {
    "train": list(range(1, 12)),
    "val": list(range(12, 17)),
    "test": list(range(17, 23)),
}

MASKING = {
    "versions_per_window": 4,        # 1 自然 + 3 人工
    "spatial_versions": 3,
    "rules": "整部位子集 ×2 + 部位/信号组混合 ×1；仅从质量合格组中选；至少保留一侧踝 gyro",
    "weight_split": "原窗权重等分给 4 个版本",
}

TRAIN = {
    "et": {"n_estimators": 500, "min_samples_leaf": 3, "max_features": 32,
           "class_weight": None},
    "stmae": {
        "enc_dim": 512, "enc_layers": 6, "enc_heads": 4,
        "dec_dim": 256, "dec_layers": 2, "dec_heads": 4, "mlp_ratio": 4,
        "pretrain_epochs": 100, "supervised_max_epochs": 100,
        "patience": 15, "min_delta": 1e-4,
        "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 0.05,
        "effective_batch": 32, "time_mask_ratio": 0.5,
        "frozen_vec": "last norm -> drop class token -> mean over 120 time tokens",
    },
}

STATS = {"bootstrap": 2000, "boot_seed": 20261023, "seeds": [1, 2, 3, 4, 5]}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def short_hash(obj, n: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "nodes": NODES, "sigs": SIGS, "channels": CHANNELS,
    "fs": FS, "window": WINDOW, "stride": STRIDE,
    "filter": FILTER, "quality": QUALITY,
    "splits": SPLITS, "masking": MASKING, "train": TRAIN, "stats": STATS,
    "label_rule": "positive points > 60 -> 1, == 60 -> 0, unknown never 0",
    "datasets_in_use": ["fogstar"],
    "datasets_pending": ["weargait_v1"],
}

PROTOCOL_HASH = short_hash(PROTOCOL)
