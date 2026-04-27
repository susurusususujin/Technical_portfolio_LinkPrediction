"""
config.py — 전역 설정, 하이퍼파라미터, 도메인 상수, 유틸리티
"""
import os
import re
import random

import numpy as np
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── 도메인 ID 상수 ─────────────────────────────────────────────
AI, MB, AE = 0, 1, 2

DOMAIN_MAP = {"ai": 0, "MB": 1, "mb": 1, "AE": 2, "ae": 2}
ROLE_MAP   = {"problem": 0, "solution": 1}
_SPLIT_RE  = re.compile(r"[;/|,_\s&:\-]+")


def parse_domain_role_from_labels(label):
    """문자열/리스트 어느 형태든 robust하게 도메인·역할 파싱."""
    dom = -1; role = -1
    if label is None:
        return dom, role

    raw_tokens = []
    if isinstance(label, str):
        raw_tokens = [label]
    elif isinstance(label, (list, tuple)):
        for x in label:
            if x is None: continue
            raw_tokens.append(x if isinstance(x, str) else str(x))
    else:
        raw_tokens = [str(label)]

    toks = []
    for s in raw_tokens:
        for t in _SPLIT_RE.split(s):
            t = t.strip().lower()
            if t: toks.append(t)
    seen = set(); toks_unique = []
    for t in toks:
        if t not in seen:
            toks_unique.append(t); seen.add(t)

    for t in toks_unique:
        if t in ROLE_MAP and role < 0:
            role = ROLE_MAP[t]

    # 도메인 우선순위: MB(3) > AE(2) > AI(1)
    prio = {"MB": 3, "mb": 3, "AE": 2, "ae": 2, "ai": 1}
    best = (-1, 0)
    for t in toks_unique:
        if t in DOMAIN_MAP:
            cand = (DOMAIN_MAP[t], prio.get(t, 0))
            if cand[1] > best[1]:
                best = cand
    dom = best[0]
    return dom, role


# ── 하이퍼파라미터 ─────────────────────────────────────────────
class Args:
    graph             = "01.12_total.json"
    epochs            = 200
    seed              = 42
    lr                = 3e-3
    encoder_dim       = 128
    init_dim          = 64
    gcn_dim           = 128
    dropout           = 0.1
    entropy_w         = 0.01
    grl_lambda_max    = 0.05
    print_every       = 1
    patience          = 20
    neg_ratio         = 8
    max_negs          = 150_000
    eval_sample_limit = 4096
    rank_sample_queries = 1000
    balance_ce        = False

    # 그리드 서치 탐색 범위
    search_neg_ratio    = [1, 5, 7, 8, 9]
    search_grl_lambda   = [0.01, 0.05, 0.10, 0.15, 0.20]
    search_entropy_w    = [0.01, 0.02, 0.03, 0.04, 0.05]
    gs_epochs_per_trial = 500

    # 하드 네거티브 마이닝
    hard_negative        = True
    hard_warmup          = 3
    hard_pool_multiplier = 5
    hard_pool_cap        = 200_000
    hard_topk_per_pos    = 2
    hard_strategy        = "topk"   # "topk" | "semi"
    hard_semi_low        = 0.4
    hard_semi_high       = 0.7
    false_neg_filter     = True


# ── 유틸리티 ──────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
