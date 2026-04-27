"""
data.py — 그래프 로딩, 서브그래프 구성, 음성(negative) 샘플링
"""
import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from config import DOMAIN_MAP, ROLE_MAP, _SPLIT_RE, parse_domain_role_from_labels


# ── 내부 헬퍼 ─────────────────────────────────────────────────
def _parse_identity_from_elementId(elementId: str):
    try:
        return int(str(elementId).split(":")[-1])
    except Exception:
        return None


# ── 그래프 로딩 ───────────────────────────────────────────────
def load_graph_json(path: str):
    """JSON 파일에서 지식 그래프를 로드하고 양방향 엣지 텐서를 반환."""
    with open(path, "r", encoding="utf-8-sig") as f:
        s = f.read().strip()
    if not s:
        raise ValueError(f"Empty JSON: {path}")
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        obj = [json.loads(ln) for ln in lines]

    def is_row(x):
        return isinstance(x, dict) and ("n" in x and "r" in x and "m" in x)

    nodes, edges = [], []

    if isinstance(obj, list) and len(obj) > 0 and is_row(obj[0]):
        # Triplet 스타일 (Neo4j export)
        node_map = {}

        def pick_name(nd):
            props = nd.get("properties", {}) or {}
            return (props.get("name") or props.get("title") or
                    props.get("label") or str(nd.get("identity", "")) or nd.get("elementId", ""))

        for row in obj:
            n, m, r = row.get("n", {}), row.get("m", {}), row.get("r", {})
            for nd in (n, m):
                nid = nd.get("identity", nd.get("elementId", None))
                if nid is None: continue
                nid_str = str(nid)
                raw_labels = nd.get("labels", []) or nd.get("label") or nd.get("type") or ""
                dom, role = parse_domain_role_from_labels(raw_labels)
                if nid_str not in node_map:
                    node_map[nid_str] = {"id": nid_str, "name": pick_name(nd),
                                         "labels": raw_labels, "domain": dom, "role": role}
                else:
                    if node_map[nid_str]["domain"] < 0 and dom >= 0:
                        node_map[nid_str]["domain"] = dom
                    if node_map[nid_str]["role"] < 0 and role >= 0:
                        node_map[nid_str]["role"] = role
            start, end, rtype = r.get("start"), r.get("end"), r.get("type", "rel")
            if start is not None and end is not None:
                edges.append((str(start), str(end), str(rtype)))
        nodes = list(node_map.values())

    else:
        # 자유형 JSON
        def is_edge_dict(d):
            return isinstance(d, dict) and \
                   any(k in d for k in ("source", "start", "src")) and \
                   any(k in d for k in ("target", "end", "dst"))

        def is_node_dict(d):
            return isinstance(d, dict) and \
                   any(k in d for k in ("id", "name", "label", "labels", "type", "domain", "role"))

        def walk(x):
            if isinstance(x, dict):
                if is_edge_dict(x): edges.append(x)
                elif is_node_dict(x): nodes.append(x)
                for v in x.values(): walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x: walk(v)
        walk(obj)

        if not nodes and edges:
            tmp = {}
            for e in edges:
                for nid in (e.get("source", e.get("start", e.get("src"))),
                            e.get("target", e.get("end", e.get("dst")))):
                    if nid is not None: tmp[str(nid)] = {"id": str(nid)}
            nodes = list(tmp.values())

        E2 = []
        for e in edges:
            src = e.get("source", e.get("start", e.get("src")))
            dst = e.get("target", e.get("end", e.get("dst")))
            rname = e.get("type", e.get("label", e.get("relation", "rel")))
            if src is not None and dst is not None:
                E2.append((str(src), str(dst), str(rname)))
        edges = E2

    # 노드 인덱싱
    node_ids = []
    for e in nodes:
        nid = e.get("id", e.get("name"))
        if nid is not None: node_ids.append(str(nid))
    for u, v, _ in edges:
        node_ids += [u, v]
    node_ids = list(dict.fromkeys(node_ids))
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    N = len(id2idx)

    idx2orig_id = []
    for nid in node_ids:
        try:
            idx2orig_id.append(int(nid))
        except Exception:
            idx2orig_id.append(_parse_identity_from_elementId(nid))

    node_domain = torch.full((N,), -1, dtype=torch.long)
    node_role   = torch.full((N,), -1, dtype=torch.long)
    names = [""] * N
    labels_raw_per_idx = [None] * N

    for e in nodes:
        nid = e.get("id", e.get("name"))
        if nid is None: continue
        i = id2idx[str(nid)]
        names[i] = e.get("name", str(nid))
        raw_labels = e.get("labels", []) or e.get("label") or e.get("type") or ""
        labels_raw_per_idx[i] = raw_labels
        dom = e.get("domain", -1); role = e.get("role", -1)
        if dom < 0 or role < 0:
            d2, r2 = parse_domain_role_from_labels(raw_labels)
            if dom  < 0 and d2 >= 0: dom  = d2
            if role < 0 and r2 >= 0: role = r2
        node_domain[i] = dom; node_role[i] = role

    rel2id = {}
    for _, _, r in edges:
        if r not in rel2id: rel2id[r] = len(rel2id)
    num_rel = len(rel2id)

    f_src, f_dst, f_type = [], [], []
    b_src, b_dst, b_type = [], [], []
    for u_id, v_id, rname in edges:
        if u_id not in id2idx or v_id not in id2idx: continue
        u, v, rid = id2idx[u_id], id2idx[v_id], rel2id[rname]
        f_src.append(u); f_dst.append(v); f_type.append(rid)
        b_src.append(v); b_dst.append(u); b_type.append(rid + num_rel)

    edge_index = torch.tensor([f_src + b_src, f_dst + b_dst], dtype=torch.long) \
        if (f_src or b_src) else torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.tensor(f_type + b_type, dtype=torch.long) \
        if (f_type or b_type) else torch.zeros((0,), dtype=torch.long)

    data = SimpleNamespace(
        num_nodes=N, names=names,
        edge_index=edge_index, edge_type=edge_type,
        node_domain=node_domain, node_role=node_role,
        num_rel=num_rel, labels_raw=labels_raw_per_idx, idx2orig_id=idx2orig_id
    )
    meta = SimpleNamespace(rel2id=rel2id)
    print(f"[Loader] nodes={N}, edges(undirected)={edge_index.size(1)//2}, #rel={num_rel}")
    return data, meta


# ── 서브그래프 구성 ───────────────────────────────────────────
def build_subgraph(data, keep_domains: set):
    """지정 도메인 노드만 남긴 서브그래프 반환."""
    keep = torch.tensor([(d.item() in keep_domains) for d in data.node_domain], dtype=torch.bool)
    old2new = -torch.ones(data.num_nodes, dtype=torch.long)
    kept_idx = torch.nonzero(keep, as_tuple=False).view(-1)
    for new_i, old_i in enumerate(kept_idx.tolist()):
        old2new[old_i] = new_i

    Eh = data.edge_index.size(1) // 2
    ei_f, et_f = data.edge_index[:, :Eh], data.edge_type[:Eh]
    ei_b, et_b = data.edge_index[:, Eh:], data.edge_type[Eh:]

    def filt(ei, et):
        s, t = ei[0], ei[1]
        m = keep[s] & keep[t]
        return torch.stack([old2new[s[m]], old2new[t[m]]], dim=0), et[m]

    ei_f2, et_f2 = filt(ei_f, et_f)
    ei_b2, et_b2 = filt(ei_b, et_b)
    idxs = kept_idx.tolist()

    return SimpleNamespace(
        num_nodes=kept_idx.numel(),
        edge_index=torch.cat([ei_f2, ei_b2], dim=1),
        edge_type=torch.cat([et_f2, et_b2], dim=0),
        node_domain=data.node_domain[keep],
        node_role=data.node_role[keep],
        num_rel=data.num_rel,
        labels_raw=[data.labels_raw[i] for i in idxs] if data.labels_raw else None,
        names=[data.names[i] for i in idxs] if data.names else None,
        idx2orig_id=[data.idx2orig_id[i] for i in idxs] if data.idx2orig_id else None,
    )


def build_nodes_by_domain(node_domain: torch.Tensor) -> dict:
    """도메인 → 노드 인덱스 리스트 딕셔너리."""
    return {int(d): torch.nonzero(node_domain == d, as_tuple=False).view(-1)
            for d in node_domain.unique().tolist() if d >= 0}


def filter_edges_by_domain_pairs(edge_index, node_domain: torch.Tensor, keep_pairs: set):
    """지정 도메인 쌍에 해당하는 엣지 마스크 반환."""
    s, t = edge_index[0], edge_index[1]
    if node_domain.device != s.device:
        node_domain = node_domain.to(s.device)
    ds, dt = node_domain[s], node_domain[t]
    m = torch.zeros(s.size(0), dtype=torch.bool, device=s.device)
    for a, b in keep_pairs:
        m |= ((ds == a) & (dt == b))
    return m


def build_adj_sets(edge_index, num_nodes: int) -> list:
    """필터드 랭킹 평가용 인접 집합."""
    Eh = edge_index.size(1) // 2
    pos = edge_index[:, :Eh].cpu()
    adj = [set() for _ in range(num_nodes)]
    for u, v in zip(pos[0].tolist(), pos[1].tolist()):
        adj[u].add(v)
    return adj


# ── 음성(Negative) 샘플링 ─────────────────────────────────────
def sample_pos_neg_edges(edge_index, num_nodes, num_neg=None, device=None):
    if device is None: device = edge_index.device
    Eh = edge_index.size(1) // 2
    if Eh == 0: return None, None
    pos = edge_index[:, :Eh]
    if num_neg is None: num_neg = Eh
    idx = torch.randint(0, Eh, (num_neg,), device=device)
    neg_src = pos[0, idx]
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    return pos, torch.stack([neg_src, neg_dst], dim=0)


def sample_neg_domain_matched(pos, node_domain: torch.Tensor, nodes_by_domain: dict,
                               k: int, max_negs: int, device):
    """tail 노드와 같은 도메인에서 음성 샘플링."""
    if pos is None or pos.size(1) == 0: return None
    P = pos.size(1)
    k_eff = max(1, min(int(P * max(1, k)), max_negs) // P)

    pos = pos.to(device)
    node_domain = node_domain.to(device)
    src_rep = pos[0].repeat_interleave(k_eff)
    desired_t_dom = node_domain[pos[1]].repeat_interleave(k_eff)
    neg_dst = torch.empty_like(src_rep)

    for d, pool in nodes_by_domain.items():
        pool = pool.to(device)
        mask = (desired_t_dom == d)
        m = int(mask.sum().item())
        if m > 0:
            neg_dst[mask] = pool[torch.randint(0, pool.size(0), (m,), device=device)]
    return torch.stack([src_rep, neg_dst], dim=0)


@torch.no_grad()
def sample_neg_pairs(pos_uv: torch.Tensor, node_domain: torch.Tensor,
                     nodes_by_domain: dict, k: int, device):
    """JointHead 학습용 도메인 매칭 음성 샘플링."""
    if pos_uv is None or pos_uv.numel() == 0 or k <= 0: return None
    pos_uv = pos_uv.to(device=device, dtype=torch.long).contiguous()
    nd = node_domain.to(device=device, dtype=torch.long)
    u_rep = pos_uv[0].repeat_interleave(k)
    v_dom = nd[pos_uv[1]].repeat_interleave(k)
    v_neg = torch.empty_like(u_rep, dtype=torch.long, device=device)

    for d in v_dom.unique().tolist():
        d_int = int(d)
        mask = (v_dom == d_int)
        count = int(mask.sum().item())
        pool = nodes_by_domain.get(d_int, None)
        if pool is None or pool.numel() == 0:
            v_neg[mask] = torch.randint(0, int(nd.size(0)), (count,), device=device)
        else:
            pool_dev = pool.to(device=device, dtype=torch.long)
            v_neg[mask] = pool_dev[torch.randint(0, pool_dev.size(0), (count,), device=device)]
    return torch.stack([u_rep, v_neg], dim=0)


@torch.no_grad()
def _false_neg_mask_by_adj(pos, neg, adj_sets, batch=65536):
    """neg 중 실제 양성(false negative) 마스킹."""
    if adj_sets is None:
        return torch.zeros(neg.size(1), dtype=torch.bool, device=neg.device)
    M = neg.size(1)
    mask = torch.zeros(M, dtype=torch.bool, device=neg.device)
    for st in range(0, M, batch):
        ed = min(st + batch, M)
        for i, (uu, vv) in enumerate(zip(neg[0, st:ed].tolist(), neg[1, st:ed].tolist())):
            if vv in adj_sets[uu]:
                mask[st + i] = True
    return mask


@torch.no_grad()
def select_hard_negatives(pos, neg_pool, H, head, adj_sets=None,
                           strategy="topk", topk_per_pos=1,
                           semi_low=0.4, semi_high=0.7):
    """하드 네거티브 선택 (topk 또는 semi-hard)."""
    device = H.device
    P = pos.size(1)
    if P == 0 or neg_pool is None or neg_pool.size(1) == 0:
        return neg_pool

    k_eff = max(1, neg_pool.size(1) // P)
    s_neg = torch.cat([
        F.softmax(head(H[neg_pool[0, st:min(st + 131072, neg_pool.size(1))]],
                       H[neg_pool[1, st:min(st + 131072, neg_pool.size(1))]]), dim=-1)[:, 1]
        for st in range(0, neg_pool.size(1), 131072)
    ], dim=0)

    if adj_sets is not None:
        s_neg = s_neg.masked_fill(_false_neg_mask_by_adj(pos, neg_pool, adj_sets), float("-inf"))

    s_neg = s_neg.view(P, k_eff)
    neg_u = neg_pool[0].view(P, k_eff)
    neg_v = neg_pool[1].view(P, k_eff)

    if strategy == "topk":
        k = min(topk_per_pos, k_eff)
        _, idx = torch.topk(s_neg, k=k, dim=1, largest=True, sorted=False)
        rows = torch.arange(P, device=device).unsqueeze(1).expand_as(idx)
        return torch.stack([neg_u[rows, idx].reshape(-1), neg_v[rows, idx].reshape(-1)], dim=0)

    elif strategy == "semi":
        keep_mask = (s_neg >= semi_low) & (s_neg <= semi_high)
        hard_u_list, hard_v_list = [], []
        for i in range(P):
            cu, cv = neg_u[i][keep_mask[i]], neg_v[i][keep_mask[i]]
            if cu.numel() == 0:
                top_idx = torch.argmax(s_neg[i])
                cu, cv = neg_u[i][top_idx:top_idx + 1], neg_v[i][top_idx:top_idx + 1]
            if cu.numel() > topk_per_pos:
                ridx = torch.randperm(cu.numel(), device=device)[:topk_per_pos]
                cu, cv = cu[ridx], cv[ridx]
            hard_u_list.append(cu); hard_v_list.append(cv)
        return torch.stack([torch.cat(hard_u_list), torch.cat(hard_v_list)], dim=0)

    return neg_pool
