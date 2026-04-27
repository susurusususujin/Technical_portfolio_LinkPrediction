"""
eval.py — 평가 지표 모음
  - eval_link_metrics            : ACC / balACC / AUC / ACC@best
  - eval_ranking_metrics         : Hits@10 (필터드)
  - eval_ranking_metrics_mrr     : MRR + Hits@10 (메모리 효율)
  - eval_joint_reltype_with_nolink: ACC / AUC / MRR / F1 (No_Link 포함)
  - eval_joint_micro             : 임계값 탐색 F1 / ACC
  - get_H                        : 임베딩 계산 헬퍼
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import filter_edges_by_domain_pairs, sample_neg_pairs


# ── 임베딩 헬퍼 ───────────────────────────────────────────────
@torch.no_grad()
def get_H(encoder, graph, device=None):
    if device is None: device = next(encoder.parameters()).device
    ei = graph.edge_index.to(device=device, dtype=torch.long).contiguous()
    et = graph.edge_type.to(device=device, dtype=torch.long).contiguous()
    return encoder(ei, et)


# ── 링크 분류 지표 ────────────────────────────────────────────
@torch.no_grad()
def eval_link_metrics(H, edge_index, head: nn.Module,
                      restrict_pairs: set = None, node_domain: torch.Tensor = None,
                      sample_limit: int = 4096, nodes_by_domain: dict = None,
                      domain_matched_neg: bool = True, return_best: bool = True):
    """ACC(0.5) / balACC / AUC(순위 근사) + 최적 임계값 ACC."""
    device = H.device
    Eh = edge_index.size(1) // 2
    _zero = lambda v=0.0: torch.tensor(v, device=device)
    _empty = {"bal_acc": _zero(), "acc": _zero(), "auc": _zero(0.5)}
    if return_best: _empty.update({"acc_best": _zero(), "thr_best": 0.5})
    if Eh == 0: return _empty

    pos = edge_index[:, :Eh].to(device)
    if restrict_pairs is not None and node_domain is not None:
        m = filter_edges_by_domain_pairs(pos, node_domain, restrict_pairs)
        pos = pos[:, m]
        if pos.numel() == 0: return _empty

    P = pos.size(1)
    if P > sample_limit:
        pos = pos[:, torch.randperm(P, device=device)[:sample_limit]]
        P = pos.size(1)

    # 음성 샘플링
    if domain_matched_neg and nodes_by_domain is not None and node_domain is not None:
        ds = node_domain[pos[1]].to(device)
        neg_t = torch.empty_like(ds)
        for d in ds.unique().tolist():
            mask = (ds == d)
            pool = nodes_by_domain.get(int(d), None)
            if pool is None or pool.numel() == 0:
                neg_t[mask] = torch.randint(0, H.size(0), (int(mask.sum()),), device=device)
            else:
                pool = pool.to(device)
                neg_t[mask] = pool[torch.randint(0, pool.size(0), (int(mask.sum()),), device=device)]
        neg = torch.stack([pos[0], neg_t], dim=0)
    else:
        neg = torch.stack([pos[0], torch.randint(0, H.size(0), (P,), device=device)], dim=0)

    s_pos = F.softmax(head(H[pos[0]], H[pos[1]]), dim=-1)[:, 1]
    s_neg = F.softmax(head(H[neg[0]], H[neg[1]]), dim=-1)[:, 1]

    bal_acc = ((s_pos >= 0.5).float().mean() + (s_neg < 0.5).float().mean()) / 2
    acc = torch.cat([(s_pos >= 0.5).float(), (s_neg < 0.5).float()]).mean()

    # AUC (순위 기반 근사)
    scores = torch.cat([s_pos, s_neg])
    labels = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])
    _, order = torch.sort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=device, dtype=torch.float)
    pos_ranks = ranks[labels == 1]
    auc = (pos_ranks.sum() - s_pos.numel() * (s_pos.numel() + 1) / 2) / \
          (s_pos.numel() * s_neg.numel() + 1e-12)

    out = {"bal_acc": bal_acc, "acc": acc, "auc": auc}

    if return_best:
        thr_grid = torch.linspace(0.05, 0.95, steps=19, device=device)
        accs = torch.stack([((s_pos >= t).float().mean() + (s_neg < t).float().mean()) / 2
                            for t in thr_grid])
        best_idx = torch.argmax(accs)
        out["acc_best"] = accs[best_idx]
        out["thr_best"] = float(thr_grid[best_idx].item())
    return out


# ── 랭킹 지표 (Hits@10) ───────────────────────────────────────
@torch.no_grad()
def eval_ranking_metrics(H, edge_index, head: nn.Module, node_domain: torch.Tensor,
                          restrict_pairs: set, nodes_by_domain: dict, adj_sets: list,
                          sample_queries: int = 1000, filtered: bool = True,
                          batch_size: int = 4096):
    """Filtered Hits@10."""
    device = H.device
    Eh = edge_index.size(1) // 2
    _zero = torch.tensor(0.0, device=device)
    if Eh == 0: return {"hits@10": _zero}

    pos = edge_index[:, :Eh].to(device)
    m = filter_edges_by_domain_pairs(pos, node_domain, restrict_pairs)
    pos = pos[:, m]; P = pos.size(1)
    if P == 0: return {"hits@10": _zero}
    if P > sample_queries:
        pos = pos[:, torch.randperm(P, device=device)[:sample_queries]]
        P = pos.size(1)

    hit10 = 0
    for j in range(P):
        u, v_true = pos[0, j].item(), pos[1, j].item()
        dt = int(node_domain[v_true].item())
        cands = nodes_by_domain.get(dt, None)
        if cands is None or cands.numel() == 0: continue
        cands = cands.to(device)

        if filtered and len(adj_sets[u]) > 0:
            bad = torch.tensor([v for v in adj_sets[u] if v != v_true], device=device, dtype=torch.long)
            if bad.numel() > 0:
                keep = torch.ones(cands.size(0), dtype=torch.bool, device=device)
                keep[torch.isin(cands, bad)] = False
                keep = keep | (cands == v_true)
                cands = cands[keep]
                if cands.numel() == 0: continue

        hu = H[u].unsqueeze(0).expand(cands.size(0), -1)
        scores = torch.cat([
            F.softmax(head(hu[st:min(st+batch_size, cands.size(0))],
                          H[cands[st:min(st+batch_size, cands.size(0))]]), dim=-1)[:, 1]
            for st in range(0, cands.size(0), batch_size)
        ])
        s_true = scores[cands == v_true][0]
        if 1 + (scores > s_true).sum().item() <= 10:
            hit10 += 1

    return {"hits@10": torch.tensor(hit10 / float(P), device=device)}


# ── 랭킹 지표 (MRR, 메모리 효율) ─────────────────────────────
@torch.inference_mode()
def eval_ranking_metrics_mrr(H, edge_index, head, node_domain, restrict_pairs,
                              nodes_by_domain, adj=None, sample_queries=300,
                              filtered=True, batch_size=512, device=None):
    """스트리밍 방식으로 MRR + Hits@10 계산 (거대 행렬 미생성)."""
    device = device or H.device
    edge_index = edge_index.to(device=device, dtype=torch.long).contiguous()
    nd = node_domain.to(device=device, dtype=torch.long)
    Eh = edge_index.size(1) // 2
    _z = torch.tensor(0.0, device=device)
    if Eh == 0: return {"hits@10": _z, "mrr": _z, "queries": 0}

    uv = edge_index[:, :Eh]
    ds, dt = nd[uv[0]], nd[uv[1]]
    mask = torch.zeros(Eh, dtype=torch.bool, device=device)
    for a, b in restrict_pairs: mask |= ((ds == a) & (dt == b))
    uv = uv[:, mask]; P = uv.size(1)
    if P == 0: return {"hits@10": _z, "mrr": _z, "queries": 0}
    if sample_queries and P > sample_queries:
        uv = uv[:, torch.randperm(P, device=device)[:sample_queries]]
    Q = int(uv.size(1))

    # 필터드 인접 사전 (CPU set)
    py_adj = None
    if filtered:
        py_adj = adj if isinstance(adj, dict) else {}
        if not isinstance(adj, dict):
            for u_i, v_i in uv.detach().cpu().numpy().T:
                py_adj.setdefault(int(u_i), set()).add(int(v_i))

    head.eval()
    tot_rr = 0.0; tot_h10 = 0.0

    for i in range(Q):
        u, v_true = int(uv[0, i].item()), int(uv[1, i].item())
        t_dom = int(nd[v_true].item())
        pool_list = nodes_by_domain.get(t_dom,
                    torch.nonzero(nd == t_dom, as_tuple=False).view(-1)).detach().cpu().tolist()
        if filtered and py_adj:
            forbid = set(py_adj.get(u, set())) - {v_true}
            pool_list = [x for x in pool_list if x not in forbid]
        if v_true not in pool_list: pool_list.append(v_true)

        pool = torch.tensor(pool_list, dtype=torch.long, device=device)
        hu = H[u].unsqueeze(0)
        s_true_val = float((1.0 - torch.exp(
            F.log_softmax(head(hu, H[v_true].unsqueeze(0)), dim=-1)[:, 0])).item())

        num_greater = sum(
            int((1.0 - torch.exp(F.log_softmax(
                head(hu.expand(vs.size(0), -1), H[vs]), dim=-1)[:, 0]) > s_true_val).sum().item())
            for st in range(0, pool.numel(), batch_size)
            for vs in [pool[st:st + batch_size]]
        )
        rank = 1 + num_greater
        tot_rr += 1.0 / float(rank)
        if rank <= 10: tot_h10 += 1.0

    return {"hits@10": torch.tensor(tot_h10 / Q, device=device),
            "mrr":     torch.tensor(tot_rr  / Q, device=device),
            "queries": Q}


# ── 관계 타입 분류 지표 (No_Link 포함) ───────────────────────
@torch.no_grad()
def eval_joint_reltype_with_nolink(encoder, joint_head, graph, num_rel, restrict_pairs,
                                    neg_ratio_eval=10, sample_limit_pos=4096,
                                    sample_limit_neg=None, batch=2048, device=None):
    """ACC / AUC / MRR / Hits@10 / F1 (No_Link=0, rel=1..R)."""
    if device is None: device = next(encoder.parameters()).device
    encoder.eval(); joint_head.eval()
    ei = graph.edge_index.to(device=device, dtype=torch.long).contiguous()
    et = graph.edge_type.to(device=device, dtype=torch.long).contiguous()
    nd = graph.node_domain.to(device=device, dtype=torch.long)
    C = num_rel + 1
    _z = lambda: torch.tensor(0.0, device=device)
    _empty = {"acc": _z(), "auc_macro": _z(), "mrr": _z(), "hits10": _z(),
              "macro_f1": _z(), "weighted_f1": _z(),
              "per_class_f1": torch.zeros(C, device=device), "n_pos": 0, "n_neg": 0}

    Eh = ei.size(1) // 2
    if Eh == 0: return _empty

    uv = ei[:, :Eh]; r = et[:Eh]
    ds, dt = nd[uv[0]], nd[uv[1]]
    mask = torch.zeros(Eh, dtype=torch.bool, device=device)
    for a, b in restrict_pairs: mask |= ((ds == a) & (dt == b))
    uv, r = uv[:, mask], r[mask]; P = int(uv.size(1))
    if P == 0: return _empty

    if P > sample_limit_pos:
        idx = torch.randperm(P, device=device)[:sample_limit_pos]
        uv, r = uv[:, idx], r[idx]; P = int(uv.size(1))
    if sample_limit_neg is None: sample_limit_neg = P * neg_ratio_eval

    neg_uv = sample_neg_pairs(uv, graph.node_domain, graph.nodes_by_domain,
                              k=int(neg_ratio_eval), device=device)
    N = int(neg_uv.size(1)) if neg_uv is not None else 0
    if N > sample_limit_neg:
        neg_uv = neg_uv[:, torch.randperm(N, device=device)[:sample_limit_neg]]
        N = int(neg_uv.size(1))

    H = encoder(ei, graph.edge_type.to(device))

    def _batched_logits(pair_uv):
        return torch.cat([joint_head(H[pair_uv[0, s:min(s+batch, pair_uv.size(1))]],
                                     H[pair_uv[1, s:min(s+batch, pair_uv.size(1))]])
                         for s in range(0, pair_uv.size(1), batch)], dim=0) \
            if pair_uv.size(1) > 0 else torch.empty((0, C), device=device)

    logits = torch.cat([_batched_logits(uv), _batched_logits(neg_uv)], dim=0)
    labels = torch.cat([r + 1, torch.zeros(N, dtype=torch.long, device=device)], dim=0)
    T = int(labels.numel())
    if T == 0: return _empty

    probs = F.softmax(logits, dim=-1); pred = probs.argmax(dim=1)
    acc = (pred == labels).float().mean()

    # Per-class F1
    f1s = torch.zeros(C, device=device); supports = torch.zeros(C, device=device)
    for c in range(C):
        tp = ((pred == c) & (labels == c)).sum().item()
        fp = ((pred == c) & (labels != c)).sum().item()
        fn = ((pred != c) & (labels == c)).sum().item()
        supports[c] = float((labels == c).sum().item())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1s[c] = 2 * prec * rec / max(1e-12, prec + rec) if (prec + rec) > 0 else 0.0
    macro_f1   = f1s.mean()
    weighted_f1 = ((supports / float(T)) * f1s).sum()

    # MRR / Hits@10 (클래스 확률 랭크)
    s_true = probs.gather(1, labels.view(-1, 1)).squeeze(1)
    ranks  = 1 + (probs > s_true.unsqueeze(1)).sum(dim=1)
    hits10 = (ranks <= min(C, 10)).float().mean()
    mrr    = (1.0 / ranks.float()).mean()

    # OVR AUC macro
    auc_sum = auc_cnt = 0
    for c in range(C):
        y_bin = (labels == c); n_p = int(y_bin.sum()); n_n = T - n_p
        if n_p == 0 or n_n == 0: continue
        s = probs[:, c]
        _, order = torch.sort(s)
        rks = torch.empty_like(order, dtype=torch.float)
        rks[order] = torch.arange(1, T + 1, device=device, dtype=torch.float)
        auc_c = (rks[y_bin].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n + 1e-12)
        auc_sum += float(auc_c.item()); auc_cnt += 1
    auc_macro = torch.tensor(auc_sum / max(1, auc_cnt), device=device)

    return {"acc": acc, "auc_macro": auc_macro, "mrr": mrr, "hits10": hits10,
            "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "per_class_f1": f1s, "n_pos": P, "n_neg": N}


# ── 임계값 최적화 F1/ACC ──────────────────────────────────────
@torch.no_grad()
def eval_joint_micro(H, edge_index, edge_type, head, node_domain, nodes_by_domain,
                     restrict_pairs, neg_ratio_eval=7, sample_limit=8192,
                     device=None, batch=4096, n_thr=101):
    """p(link) 임계값 그리드 탐색으로 최적 F1/ACC와 임계값 반환."""
    if device is None: device = H.device
    edge_index = edge_index.to(device=device, dtype=torch.long).contiguous()
    nd = node_domain.to(device=device, dtype=torch.long)
    Eh = edge_index.size(1) // 2
    _empty = {"thr_best": 0.5, "f1_best": torch.tensor(0.0, device=device),
              "acc_best": torch.tensor(0.0, device=device), "n_pos": 0, "n_neg": 0}
    if Eh == 0: return _empty

    uv = edge_index[:, :Eh]
    mask = torch.zeros(Eh, dtype=torch.bool, device=device)
    for a, b in restrict_pairs: mask |= ((nd[uv[0]] == a) & (nd[uv[1]] == b))
    uv = uv[:, mask]; P = int(uv.size(1))
    if P == 0: return _empty
    if sample_limit and P > sample_limit:
        uv = uv[:, torch.randperm(P, device=device)[:sample_limit]]; P = int(uv.size(1))

    neg_uv = sample_neg_pairs(uv, node_domain, nodes_by_domain, k=int(neg_ratio_eval), device=device)
    N = int(neg_uv.size(1)) if neg_uv is not None else 0

    def _p_link(pair):
        if pair is None or pair.numel() == 0: return torch.empty(0, device=device)
        return torch.cat([
            1.0 - F.softmax(head(H[pair[0, s:min(s+batch, pair.size(1))]],
                                  H[pair[1, s:min(s+batch, pair.size(1))]]), dim=-1)[:, 0]
            for s in range(0, pair.size(1), batch)
        ])

    s_pos = _p_link(uv); s_neg = _p_link(neg_uv)
    if s_pos.numel() == 0 or s_neg.numel() == 0:
        return {**_empty, "n_pos": int(s_pos.numel()), "n_neg": int(s_neg.numel())}

    scores = torch.cat([s_pos, s_neg])
    labels = torch.cat([torch.ones(P, dtype=torch.long, device=device),
                        torch.zeros(N, dtype=torch.long, device=device)])

    best_thr, best_f1, best_acc = 0.5, -1.0, 0.0
    for t in torch.linspace(0.0, 1.0, steps=n_thr, device=device):
        pred = (scores >= t).long()
        tp = (pred.eq(1) & labels.eq(1)).sum().item()
        fp = (pred.eq(1) & labels.eq(0)).sum().item()
        fn = (pred.eq(0) & labels.eq(1)).sum().item()
        tn = (pred.eq(0) & labels.eq(0)).sum().item()
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1  = 2 * prec * rec / max(1e-12, prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        if f1 > best_f1:
            best_f1, best_thr, best_acc = f1, float(t.item()), acc

    return {"thr_best": best_thr, "f1_best": torch.tensor(best_f1, device=device),
            "acc_best": torch.tensor(best_acc, device=device), "n_pos": P, "n_neg": N}
