"""
predict.py — 링크 예측 및 후처리(이름 복원, CSV 저장)
  - _score_pack              : 점수 계산 모드 (link / rel / margin)
  - predict_joint_flexible   : 방향별 링크 예측 → DataFrame + CSV
  - build_identity_maps      : JSON KG에서 identity→name/props 추출
  - attach_names_to_predictions : 예측 CSV에 노드 이름 부착
"""
import json
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


# ── 내부 헬퍼 ─────────────────────────────────────────────────
_NAME_KEYS = ("name", "Name", "label", "title")


def _extract_name(obj: dict, keys=_NAME_KEYS):
    """dict 또는 props 서브딕셔너리에서 이름 추출."""
    if not isinstance(obj, dict): return None
    props = obj.get("properties", obj) or {}
    for k in keys:
        v = props.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _parse_identity_from_elementId(elementId: str):
    try: return int(str(elementId).split(":")[-1])
    except Exception: return None


def _build_idx2identity(data) -> dict:
    """서브그래프 idx → 원본 identity 매핑."""
    if hasattr(data, "idx2orig_id") and data.idx2orig_id is not None:
        return {i: v for i, v in enumerate(data.idx2orig_id) if v is not None}
    return {}


# ── 점수 계산 ─────────────────────────────────────────────────
def _score_pack(logits, *, score_mode="link", temperature=1.0, alpha_no_link_bias=0.0):
    """logits [B, 1+R] → (p_link, p_best_rel, winner_rel, score)
    score_mode: "link" | "rel" | "margin"
    """
    if alpha_no_link_bias != 0.0:
        logits = logits.clone()
        logits[:, 0] = logits[:, 0] - float(alpha_no_link_bias)
    if temperature and temperature != 1.0:
        logits = logits / float(temperature)

    prob = F.softmax(logits, dim=-1)
    p_no = prob[:, 0]; p_rel = prob[:, 1:]
    p_best, winner_rel = torch.max(p_rel, dim=1)

    if score_mode == "link":
        score = 1.0 - p_no
    elif score_mode == "rel":
        score = p_best
    elif score_mode == "margin":
        max_rel_logit, _ = torch.max(logits[:, 1:], dim=1)
        score = max_rel_logit - logits[:, 0]
    else:
        raise ValueError(f"Unknown score_mode: '{score_mode}'. Choose 'link' | 'rel' | 'margin'.")

    return 1.0 - p_no, p_best, winner_rel, score


# ── 주 예측 함수 ──────────────────────────────────────────────
def predict_joint_flexible(H, data, head, meta,
                           directions=("AI->AE", "AE->AI"),
                           thr: float = 0.4,
                           score_mode: str = "link",
                           min_prob_rel=None,
                           temperature: float = 1.0,
                           alpha_no_link_bias: float = 0.0,
                           top_k_per_src=None,
                           batch: int = 4096,
                           exclude_existing: bool = True,
                           csv_path: str = "pred_edges.csv") -> pd.DataFrame:
    """
    directions 방향별 링크 예측.
    thr        : score 임계값 (낮출수록 더 많은 링크 예측)
    score_mode : "link"(기본) | "rel" | "margin"
    top_k_per_src: None이면 임계값만 사용, 정수이면 소스별 Top-K 보정 추가
    """
    device = H.device
    R = int(data.num_rel)
    N = int(data.num_nodes)
    if H.size(0) < N:
        raise RuntimeError(f"H.size(0)={H.size(0)} < data.num_nodes={N}")
    if H.size(0) != N:
        H = H.narrow(0, 0, N)

    Eh = int(data.edge_index.size(1) // 2)
    existing = set()
    if exclude_existing and Eh > 0:
        for u, v in data.edge_index[:, :Eh].detach().cpu().numpy().T.tolist():
            existing.add((u, v))

    dom_id = {"AI": 0, "MB": 1, "AE": 2}
    id2rel = [f"REL_{i}" for i in range(R)]
    for name, rid in meta.rel2id.items():
        if isinstance(rid, int) and 0 <= rid < R:
            id2rel[rid] = str(name)

    idx2identity = _build_idx2identity(data)
    # names dict: identity → name (활용 가능 시)
    id2name_nodes: dict = {}
    if hasattr(data, "names") and data.names:
        for i, nm in enumerate(data.names):
            orig = idx2identity.get(i)
            if orig is not None and nm:
                id2name_nodes[orig] = nm

    rows = []

    for direc in directions:
        s_str, t_str = direc.split("->")
        s_req, t_req = dom_id[s_str], dom_id[t_str]
        src_nodes = torch.nonzero(data.node_domain == s_req, as_tuple=False).view(-1).tolist()
        dst_nodes = torch.nonzero(data.node_domain == t_req, as_tuple=False).view(-1).tolist()
        if not src_nodes or not dst_nodes:
            print(f"[WARN] No nodes for direction {direc}.")
            continue

        vs_all = torch.tensor(dst_nodes, device=device, dtype=torch.long)

        for u in src_nodes:
            keep_by_v: dict = {}         # v → (score, p_link, p_best, winner_rel)
            scores_cpu: list = []        # Top-K 보정용 CPU 버퍼

            for st in range(0, vs_all.numel(), batch):
                vs = vs_all[st:min(st + batch, vs_all.numel())]
                if vs.numel() == 0: continue
                hu = H[u].unsqueeze(0).expand(vs.size(0), -1)
                logits = head(hu, H[vs])
                p_link, p_best, winner_rel, score = _score_pack(
                    logits, score_mode=score_mode,
                    temperature=temperature, alpha_no_link_bias=alpha_no_link_bias
                )
                # 임계값 필터
                mask = (score >= float(thr))
                if min_prob_rel is not None:
                    mask = mask & (p_best >= float(min_prob_rel))

                for j in torch.nonzero(mask, as_tuple=False).view(-1).tolist():
                    v = int(vs[j].item())
                    if exclude_existing and (u, v) in existing: continue
                    sc = float(score[j].item())
                    cur = keep_by_v.get(v)
                    if cur is None or sc > cur[0]:
                        keep_by_v[v] = (sc, float(p_link[j].item()),
                                        float(p_best[j].item()), int(winner_rel[j].item()))

                if top_k_per_src is not None:
                    scores_cpu.append((st, score.detach().cpu(), p_link.detach().cpu(),
                                       p_best.detach().cpu(), winner_rel.detach().cpu()))

            # Top-K 보정 (임계값과 합산)
            if top_k_per_src is not None and top_k_per_src > 0 and scores_cpu:
                sc_cat = torch.cat([x[1] for x in scores_cpu])
                pl_cat = torch.cat([x[2] for x in scores_cpu])
                pb_cat = torch.cat([x[3] for x in scores_cpu])
                wr_cat = torch.cat([x[4] for x in scores_cpu])
                k = min(int(top_k_per_src), sc_cat.numel())
                for li in torch.topk(sc_cat, k=k, largest=True, sorted=False).indices.tolist():
                    v = int(vs_all[li].item())
                    if exclude_existing and (u, v) in existing: continue
                    sc = float(sc_cat[li].item())
                    cur = keep_by_v.get(v)
                    if cur is None or sc > cur[0]:
                        keep_by_v[v] = (sc, float(pl_cat[li].item()),
                                        float(pb_cat[li].item()), int(wr_cat[li].item()))

            # 행 저장
            for v, (sc, pl, pb, wr) in keep_by_v.items():
                if int(data.node_domain[u].item()) != s_req: continue
                if int(data.node_domain[v].item()) != t_req: continue
                src_ident = idx2identity.get(int(u))
                dst_ident = idx2identity.get(int(v))
                rows.append((int(u), int(v), src_ident, dst_ident,
                             float(pl), int(wr), id2rel[int(wr)],
                             id2name_nodes.get(src_ident, str(int(u))),
                             id2name_nodes.get(dst_ident, str(int(v))),
                             int(data.node_domain[u].item()),
                             int(data.node_domain[v].item()),
                             float(sc), float(pb)))

    df = pd.DataFrame(rows, columns=[
        "src_id", "dst_id", "src_identity", "dst_identity",
        "link_prob", "rel_id", "rel_type", "src_name", "dst_name",
        "src_dom_true", "dst_dom_true", "score", "rel_prob_best"
    ])

    if not df.empty:
        df = df.sort_values(["score", "link_prob", "rel_id"],
                            ascending=[False, False, True]).reset_index(drop=True)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"💾 Saved {len(df)} predictions → {csv_path}")
        print(df.value_counts(["src_dom_true", "dst_dom_true"]).to_string())
    else:
        print("[INFO] No predictions. thr를 낮추거나 top_k_per_src를 설정해보세요.")

    return df


# ── 후처리: JSON KG에서 이름 복원 ────────────────────────────
def _load_json_bomsafe(path: str):
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"JSON not found: {path}")
    with open(p, "r", encoding="utf-8-sig") as f: txt = f.read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        items = [json.loads(ln) for ln in txt.splitlines() if ln.strip()]
        if items: return items
        raise e


def build_identity_maps(json_path: str) -> tuple[dict, dict]:
    """JSON KG에서 identity→name 및 identity→properties 매핑 생성."""
    data = _load_json_bomsafe(json_path)
    records = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        def walk(x):
            if isinstance(x, dict):
                if "n" in x and "m" in x: records.append(x)
                for v in x.values(): walk(v)
            elif isinstance(x, list):
                for v in x: walk(v)
        walk(data)

    id2name, id2props = {}, {}
    for rec in records:
        for k in ("n", "m"):
            node = rec.get(k, {})
            if not isinstance(node, dict): continue
            try: ident = int(node.get("identity"))
            except (TypeError, ValueError): continue
            props = node.get("properties", {}) or {}
            if ident not in id2props: id2props[ident] = props
            nm = _extract_name(node)
            if nm: id2name[ident] = nm

    print(f"• identity→name: {len(id2name)} | identity→props: {len(id2props)}")
    return id2name, id2props


def attach_names_to_predictions(pred_csv: str, json_kg: str, out_csv: str,
                                 idx2identity: dict) -> pd.DataFrame:
    """예측 CSV에 원본 노드 이름과 properties를 부착해 저장."""
    df = pd.read_csv(pred_csv)
    df["src_id"] = pd.to_numeric(df["src_id"], errors="coerce").astype("Int64")
    df["dst_id"] = pd.to_numeric(df["dst_id"], errors="coerce").astype("Int64")

    id2name, id2props = build_identity_maps(json_kg)

    if idx2identity:
        df["src_identity"] = df["src_id"].map(idx2identity)
        df["dst_identity"] = df["dst_id"].map(idx2identity)
    else:
        print("⚠️  idx→identity 매핑 없음. 이름 복원 생략.")
        df["src_identity"] = pd.NA
        df["dst_identity"] = pd.NA

    def _name(ident):
        if pd.isna(ident): return None
        return id2name.get(int(ident))

    def _props(ident):
        if pd.isna(ident): return {}
        return id2props.get(int(ident), {})

    if df["src_identity"].notna().any():
        df["src_name"] = df["src_identity"].apply(_name).fillna(df["src_id"].astype(str))
        df["dst_name"] = df["dst_identity"].apply(_name).fillna(df["dst_id"].astype(str))
        df["src_properties"] = df["src_identity"].apply(
            lambda x: json.dumps(_props(x), ensure_ascii=False))
        df["dst_properties"] = df["dst_identity"].apply(
            lambda x: json.dumps(_props(x), ensure_ascii=False))
    else:
        df["src_name"] = df["src_id"].astype(str)
        df["dst_name"] = df["dst_id"].astype(str)
        df["src_properties"] = "{}"; df["dst_properties"] = "{}"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"💾 Named predictions → {out_csv}")
    return df
