"""
train.py — 학습 루프, 그리드 서치, 2단계 파인튜닝
  - TrainState      : 학습 상태 컨테이너 (global 변수 대체)
  - train_for_epochs: Stage0 PairClassifier 학습
  - run_grid_search : 하이퍼파라미터 탐색
  - train_stage1_joint / train_stage2_joint : JointHead 학습
"""
import itertools
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Args, AI, MB, AE, set_seed
from data import (filter_edges_by_domain_pairs, sample_pos_neg_edges,
                   sample_neg_domain_matched, sample_neg_pairs)

# eval은 순환 import 방지를 위해 함수 내에서 지연 import
# (eval.py 도 train.py를 import하지 않으므로 안전)

# ── Stage2 기본값 ──────────────────────────────────────────────
STAGE2_EPOCHS          = 100
STAGE2_PATIENCE        = 60
JOINT_NEG_RATIO_STAGE2 = 7
JOINT_NEG_RATIO_EVAL   = 7
GRL_MAX_STAGE2         = 0.2


# ── 유틸 ──────────────────────────────────────────────────────
def _to_device_graph(g, device):
    """그래프 텐서를 한 번만 device로 올리고 캐시."""
    if getattr(g, "_on_device", None) == device:
        return g
    g.edge_index  = g.edge_index.to(device=device, dtype=torch.long).contiguous()
    g.edge_type   = g.edge_type.to(device=device, dtype=torch.long).contiguous()
    g.node_domain = g.node_domain.to(device=device, dtype=torch.long)
    g._on_device  = device
    return g


def joint_batch_loss(H, edge_index, edge_type, head,
                     node_domain, nodes_by_domain, restrict_pairs,
                     neg_ratio, device):
    """JointHead용 CE 손실: 양성(rel_id+1) vs 음성(0)."""
    edge_index = edge_index.to(device=device, dtype=torch.long).contiguous()
    edge_type  = edge_type.to(device=device, dtype=torch.long).contiguous()
    nd = node_domain.to(device=device, dtype=torch.long)
    Eh = edge_index.size(1) // 2
    pos_uv, pos_r = edge_index[:, :Eh], edge_type[:Eh]

    ds, dt = nd[pos_uv[0]], nd[pos_uv[1]]
    mask = torch.zeros(Eh, dtype=torch.bool, device=device)
    for a, b in restrict_pairs:
        mask |= ((ds == a) & (dt == b))
    pos_uv, pos_r = pos_uv[:, mask], pos_r[mask]
    if pos_uv.size(1) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    neg_uv = sample_neg_pairs(pos_uv, node_domain, nodes_by_domain,
                              k=int(neg_ratio), device=device)
    logits = torch.cat([head(H[pos_uv[0]], H[pos_uv[1]]),
                        head(H[neg_uv[0]], H[neg_uv[1]])], dim=0)
    labels = torch.cat([pos_r + 1,
                        torch.zeros(neg_uv.size(1), dtype=torch.long, device=device)], dim=0)
    return F.cross_entropy(logits, labels)


# ── 학습 상태 컨테이너 ────────────────────────────────────────
@dataclass
class TrainState:
    """global 변수 없이 학습 상태를 한 곳에서 관리."""
    encoder:      nn.Module
    pair_head:    nn.Module
    grl:          nn.Module
    domain_head:  nn.Module
    optim:        torch.optim.Optimizer
    current_epoch:    int   = 0
    best_tgt:         float = 0.0
    best_src:         float = 0.0
    best_ep:          int   = 0
    patience_counter: int   = 0
    stopped_early:    bool  = False


# ── Stage0: PairClassifier 학습 ───────────────────────────────
def train_for_epochs(state: TrainState, src, tgt, args: Args, device,
                     keep_src_pairs, keep_tgt_pairs_inter,
                     adj_src, adj_tgt, num_epochs: int = 5) -> TrainState:
    from eval import eval_link_metrics, eval_ranking_metrics  # 지연 import

    if state.stopped_early:
        print("⏹ Already early-stopped.")
        return state

    for it in range(1, num_epochs + 1):
        if state.current_epoch >= args.epochs:
            print(f"⏹ Reached max epochs ({args.epochs}).")
            state.stopped_early = True
            break

        state.current_epoch += 1
        ep = state.current_epoch
        prog = it / float(max(1, num_epochs))
        state.grl.set_lambda(min(prog, args.grl_lambda_max))
        entw_now = prog * args.entropy_w

        state.encoder.train(); state.pair_head.train(); state.domain_head.train()
        state.optim.zero_grad()

        Hs = state.encoder(src.edge_index.to(device), src.edge_type.to(device))
        Ht = state.encoder(tgt.edge_index.to(device), tgt.edge_type.to(device))

        # 소스 CE 손실
        pos_all, _ = sample_pos_neg_edges(src.edge_index.to(device), src.num_nodes, device=device)
        if pos_all is not None:
            pos = pos_all[:, filter_edges_by_domain_pairs(pos_all, src.node_domain, keep_src_pairs)]
        else:
            pos = None

        if pos is None or pos.size(1) == 0:
            cls_loss = torch.tensor(0.0, device=device)
            P = N = 0
        else:
            log_pos = state.pair_head(Hs[pos[0]], Hs[pos[1]])
            y_pos = torch.ones(log_pos.size(0), dtype=torch.long, device=device)
            P = y_pos.numel()
            neg = sample_neg_domain_matched(pos, src.node_domain, src.nodes_by_domain,
                                            k=args.neg_ratio, max_negs=args.max_negs, device=device)
            log_neg = state.pair_head(Hs[neg[0]], Hs[neg[1]])
            y_neg = torch.zeros(log_neg.size(0), dtype=torch.long, device=device)
            N = y_neg.numel()
            logits = torch.cat([log_pos, log_neg], dim=0)
            y = torch.cat([y_pos, y_neg], dim=0)

            if getattr(args, "balance_ce", True):
                total = float(P + N) + 1e-12
                weights = torch.tensor([total / (2 * N + 1e-12),
                                        total / (2 * P + 1e-12)], device=device)
                cls_loss = F.cross_entropy(logits, y, weight=weights)
            else:
                cls_loss = F.cross_entropy(logits, y)

        # GRL 도메인 적대 손실
        ds = state.domain_head(Hs); dt_d = state.domain_head(Ht)
        grl_loss = F.cross_entropy(
            torch.cat([ds, dt_d], dim=0),
            torch.cat([torch.zeros(ds.size(0), dtype=torch.long, device=device),
                       torch.ones(dt_d.size(0), dtype=torch.long, device=device)], dim=0)
        )

        # 타깃 엔트로피 (inter 도메인 엣지 대상)
        Eh_t = tgt.edge_index.size(1) // 2
        if Eh_t > 0:
            pos_t = tgt.edge_index[:, :Eh_t].to(device)
            mask_inter = filter_edges_by_domain_pairs(pos_t, tgt.node_domain, keep_tgt_pairs_inter)
            idx = torch.nonzero(mask_inter, as_tuple=False).view(-1) if mask_inter.any() \
                else torch.randint(0, Eh_t, (min(2048, Eh_t),), device=device)
            if idx.numel() > 2048:
                idx = idx[torch.randperm(idx.numel(), device=device)[:2048]]
            pt = F.softmax(state.pair_head(Ht[pos_t[0, idx]], Ht[pos_t[1, idx]]), dim=-1).clamp_min(1e-9)
            ent_loss = -(pt * pt.log()).sum(dim=-1).mean()
        else:
            ent_loss = torch.tensor(0.0, device=device)

        # L2 정규화
        reg = sum(1e-4 * (p ** 2).mean()
                  for m in [state.encoder, state.pair_head]
                  for p in m.parameters() if p.requires_grad and p.dim() > 1)

        loss = cls_loss + grl_loss + entw_now * ent_loss + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(state.encoder.parameters()) + list(state.pair_head.parameters()), 5.0)
        state.optim.step()

        # 평가
        state.encoder.eval(); state.pair_head.eval()
        with torch.no_grad():
            Hs_e = state.encoder(src.edge_index.to(device), src.edge_type.to(device))
            Ht_e = state.encoder(tgt.edge_index.to(device), tgt.edge_type.to(device))
            src_m = eval_link_metrics(Hs_e, src.edge_index.to(device), state.pair_head,
                                      restrict_pairs=keep_src_pairs,
                                      node_domain=src.node_domain.to(device),
                                      sample_limit=args.eval_sample_limit,
                                      nodes_by_domain=src.nodes_by_domain,
                                      domain_matched_neg=True, return_best=True)
            tgt_m = eval_link_metrics(Ht_e, tgt.edge_index.to(device), state.pair_head,
                                      restrict_pairs=keep_tgt_pairs_inter,
                                      node_domain=tgt.node_domain.to(device),
                                      sample_limit=args.eval_sample_limit,
                                      nodes_by_domain=tgt.nodes_by_domain,
                                      domain_matched_neg=True, return_best=True)
            q = min(args.rank_sample_queries, 256)
            rank_src = eval_ranking_metrics(Hs_e, src.edge_index.to(device), state.pair_head,
                                            src.node_domain.to(device), keep_src_pairs,
                                            src.nodes_by_domain, adj_src, sample_queries=q)
            rank_tgt = eval_ranking_metrics(Ht_e, tgt.edge_index.to(device), state.pair_head,
                                            tgt.node_domain.to(device), keep_tgt_pairs_inter,
                                            tgt.nodes_by_domain, adj_tgt, sample_queries=q)

        if ep % args.print_every == 0:
            print(f"[{ep:03d}] loss={loss.item():.4f} | "
                  f"SRC AUC={src_m['auc'].item():.4f} ACC*={src_m['acc_best'].item():.4f} "
                  f"H@10={rank_src['hits@10'].item():.4f} | "
                  f"TGT AUC={tgt_m['auc'].item():.4f} ACC*={tgt_m['acc_best'].item():.4f} "
                  f"H@10={rank_tgt['hits@10'].item():.4f} | "
                  f"cls={cls_loss.item():.4f} grl={grl_loss.item():.4f} "
                  f"ent={ent_loss.item():.4f} (P={P}, N={N})")

        # Early stopping (TGT AUC 기준)
        tgt_score = tgt_m["auc"].item()
        if tgt_score > state.best_tgt:
            state.best_tgt = tgt_score
            state.best_src = src_m["auc"].item()
            state.best_ep  = ep
            state.patience_counter = 0
        else:
            state.patience_counter += 1
            if state.patience_counter >= args.patience:
                print(f"⏹ Early stopping @ ep {ep}. BEST TGT={state.best_tgt:.4f} @ {state.best_ep}")
                state.stopped_early = True
                break

    return state


# ── 그리드 서치 ───────────────────────────────────────────────
def _init_trial_models(src, tgt, data, args, device):
    from net import CompRGCNEncoder, PairClassifier, GRL
    enc = CompRGCNEncoder(
        num_ent=max(src.num_nodes, tgt.num_nodes), num_rel=data.num_rel,
        init_dim=args.init_dim, gcn_dim=args.gcn_dim, embed_dim=args.encoder_dim,
        dropout=args.dropout, opn="corr", bias=True, gcn_layers=2
    ).to(device)
    ph  = PairClassifier(args.encoder_dim, hidden=128, num_classes=2).to(device)
    gr  = GRL().to(device)
    dh  = nn.Sequential(gr, nn.Linear(args.encoder_dim, 64), nn.ReLU(),
                        nn.Dropout(0.1), nn.Linear(64, 2)).to(device)
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(ph.parameters()) + list(dh.parameters()), lr=args.lr)
    return enc, ph, gr, dh, opt


def _train_one_trial(neg_ratio, grl_lambda_max, entropy_w, max_epochs,
                     keep_src, keep_tgt, src, tgt, data, args, device):
    from eval import eval_link_metrics  # 지연 import

    best_tgt_auc = best_tgt_acc = best_src_auc = best_src_acc = 0.0
    best_ep = patience_ctr = 0
    enc, ph, grl, dh, opt = _init_trial_models(src, tgt, data, args, device)

    for ep in range(1, max_epochs + 1):
        enc.train(); ph.train(); dh.train()
        opt.zero_grad()
        prog = ep / float(max_epochs)
        grl.set_lambda(min(prog, grl_lambda_max))
        entw_now = prog * entropy_w

        Hs = enc(src.edge_index.to(device), src.edge_type.to(device))
        Ht = enc(tgt.edge_index.to(device), tgt.edge_type.to(device))

        pos_all, _ = sample_pos_neg_edges(src.edge_index.to(device), src.num_nodes, device=device)
        if pos_all is not None:
            pos = pos_all[:, filter_edges_by_domain_pairs(pos_all, src.node_domain, keep_src)]
        else:
            pos = None

        if pos is None or pos.size(1) == 0:
            cls_loss = torch.tensor(0.0, device=device)
        else:
            log_pos = ph(Hs[pos[0]], Hs[pos[1]])
            y_pos = torch.ones(log_pos.size(0), dtype=torch.long, device=device)
            neg = sample_neg_domain_matched(pos, src.node_domain, src.nodes_by_domain,
                                            k=neg_ratio, max_negs=args.max_negs, device=device)
            log_neg = ph(Hs[neg[0]], Hs[neg[1]])
            y_neg = torch.zeros(log_neg.size(0), dtype=torch.long, device=device)
            logits = torch.cat([log_pos, log_neg], dim=0)
            y = torch.cat([y_pos, y_neg], dim=0)
            if getattr(args, "balance_ce", True):
                P, N = y_pos.numel(), y_neg.numel(); total = float(P + N) + 1e-12
                w = torch.tensor([total/(2*N+1e-12), total/(2*P+1e-12)], device=device)
                cls_loss = F.cross_entropy(logits, y, weight=w)
            else:
                cls_loss = F.cross_entropy(logits, y)

        ds = dh(Hs); dt_d = dh(Ht)
        grl_loss = F.cross_entropy(
            torch.cat([ds, dt_d], dim=0),
            torch.cat([torch.zeros(ds.size(0), dtype=torch.long, device=device),
                       torch.ones(dt_d.size(0), dtype=torch.long, device=device)]))

        Eh_t = tgt.edge_index.size(1) // 2
        if Eh_t > 0:
            pos_t = tgt.edge_index[:, :Eh_t].to(device)
            mask_inter = filter_edges_by_domain_pairs(pos_t, tgt.node_domain, keep_tgt)
            idx = torch.nonzero(mask_inter, as_tuple=False).view(-1) if mask_inter.any() \
                else torch.randint(0, Eh_t, (min(2048, Eh_t),), device=device)
            if idx.numel() > 2048:
                idx = idx[torch.randperm(idx.numel(), device=device)[:2048]]
            pt = F.softmax(ph(Ht[pos_t[0, idx]], Ht[pos_t[1, idx]]), dim=-1).clamp_min(1e-9)
            ent_loss = -(pt * pt.log()).sum(dim=-1).mean()
        else:
            ent_loss = torch.tensor(0.0, device=device)

        reg = sum(1e-4 * (p**2).mean() for m in [enc, ph]
                  for p in m.parameters() if p.requires_grad and p.dim() > 1)
        loss = cls_loss + grl_loss + entw_now * ent_loss + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(ph.parameters()), 5.0)
        opt.step()

        enc.eval(); ph.eval()
        with torch.no_grad():
            Hs_e = enc(src.edge_index.to(device), src.edge_type.to(device))
            Ht_e = enc(tgt.edge_index.to(device), tgt.edge_type.to(device))
            src_m = eval_link_metrics(Hs_e, src.edge_index.to(device), ph,
                                      restrict_pairs=keep_src, node_domain=src.node_domain.to(device),
                                      sample_limit=args.eval_sample_limit,
                                      nodes_by_domain=src.nodes_by_domain,
                                      domain_matched_neg=True, return_best=True)
            tgt_m = eval_link_metrics(Ht_e, tgt.edge_index.to(device), ph,
                                      restrict_pairs=keep_tgt, node_domain=tgt.node_domain.to(device),
                                      sample_limit=args.eval_sample_limit,
                                      nodes_by_domain=tgt.nodes_by_domain,
                                      domain_matched_neg=True, return_best=True)

        score = tgt_m["auc"].item()
        if score > best_tgt_auc:
            best_tgt_auc, best_tgt_acc = score, tgt_m["acc_best"].item()
            best_src_auc, best_src_acc = src_m["auc"].item(), src_m["acc_best"].item()
            best_ep = ep; patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience: break

    return {"best_ep": best_ep, "tgt_auc": best_tgt_auc, "tgt_acc": best_tgt_acc,
            "src_auc": best_src_auc, "src_acc": best_src_acc,
            "neg_ratio": neg_ratio, "grl_lambda_max": grl_lambda_max, "entropy_w": entropy_w}


def run_grid_search(src, tgt, data, args: Args, device) -> list:
    """(neg_ratio, grl_lambda, entropy_w) 조합 전수 탐색."""
    combos = list(itertools.product(args.search_neg_ratio, args.search_grl_lambda, args.search_entropy_w))
    keep_src = {(AI, AI), (AI, MB), (MB, MB), (MB, AI)}
    keep_tgt = {(AI, AE), (AE, AI)}

    results = []
    print(f"Grid search: {len(combos)} combos")
    for i, (nr, gl, ew) in enumerate(combos, 1):
        set_seed(args.seed + i * 1000)
        res = _train_one_trial(nr, gl, ew, args.gs_epochs_per_trial,
                               keep_src, keep_tgt, src, tgt, data, args, device)
        results.append(res)
        print(f"[{i:02d}/{len(combos)}] neg={nr} grl={gl} ent={ew} "
              f"→ BEST@{res['best_ep']:03d} TGT={res['tgt_auc']:.4f} SRC={res['src_auc']:.4f}")

    results.sort(key=lambda x: (x["tgt_auc"], x["tgt_acc"]), reverse=True)
    print("\n=== Top-5 ===")
    for k, r in enumerate(results[:5], 1):
        print(f"#{k}: neg={r['neg_ratio']} grl={r['grl_lambda_max']} ent={r['entropy_w']} "
              f"| TGT={r['tgt_auc']:.4f} SRC={r['src_auc']:.4f}")
    return results


# ── Stage1: JointHead 소스 학습 ───────────────────────────────
def train_stage1_joint(encoder, joint_head, optim, src, args: Args, device,
                       keep_src_pairs, neg_ratio=10, epochs=50):
    encoder.train(); joint_head.train()
    for ep in range(1, epochs + 1):
        optim.zero_grad()
        Hs = encoder(src.edge_index.to(device), src.edge_type.to(device))
        loss = joint_batch_loss(Hs, src.edge_index, src.edge_type, joint_head,
                                src.node_domain, src.nodes_by_domain,
                                restrict_pairs=keep_src_pairs, neg_ratio=neg_ratio, device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(joint_head.parameters()), 5.0)
        optim.step()
        if ep % 5 == 0:
            print(f"[S1:{ep:03d}] loss={loss.item():.4f}")


# ── Stage2: 도메인 적응 ───────────────────────────────────────
def train_stage2_joint(encoder, joint_head, optim, grl, src, tgt, adj_tgt,
                       args: Args, device, keep_src_pairs,
                       epochs=STAGE2_EPOCHS, patience=STAGE2_PATIENCE,
                       eval_every=5, eval_sample_queries=300, eval_batch_size=512,
                       accum_steps=1):
    from eval import eval_ranking_metrics_mrr  # 지연 import

    _to_device_graph(src, device); _to_device_graph(tgt, device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    encoder.train(); joint_head.train()
    optim.zero_grad(set_to_none=True)

    for ep in range(1, epochs + 1):
        grl.set_lambda(min(GRL_MAX_STAGE2, GRL_MAX_STAGE2 * ep / epochs))

        with torch.cuda.amp.autocast():
            Hs = encoder(src.edge_index, src.edge_type)
            _  = encoder(tgt.edge_index, tgt.edge_type)   # GRL 적용
            loss = joint_batch_loss(Hs, src.edge_index, src.edge_type, joint_head,
                                    src.node_domain, src.nodes_by_domain,
                                    restrict_pairs=keep_src_pairs,
                                    neg_ratio=JOINT_NEG_RATIO_STAGE2, device=device) / accum_steps

        scaler.scale(loss).backward()
        if ep % accum_steps == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(joint_head.parameters()), 5.0)
            scaler.step(optim); scaler.update()
            optim.zero_grad(set_to_none=True)

        del Hs, loss
        if ep % 2 == 0: torch.cuda.empty_cache()

        if eval_every and (ep % eval_every == 0):
            encoder.eval(); joint_head.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                Ht_e = encoder(tgt.edge_index, tgt.edge_type)
            rank_tgt = eval_ranking_metrics_mrr(
                Ht_e, tgt.edge_index, joint_head, tgt.node_domain,
                {(AI, AE), (AE, AI)}, tgt.nodes_by_domain, adj_tgt,
                sample_queries=eval_sample_queries, filtered=True,
                batch_size=eval_batch_size, device=device
            )
            print(f"[S2:{ep:03d}] MRR={rank_tgt['mrr']:.4f} H@10={rank_tgt['hits@10']:.4f}")
            del Ht_e
            encoder.train(); joint_head.train()
            torch.cuda.empty_cache()

    print("✅ Stage2 done.")
