"""
main.py — UDAGCN 파이프라인 진입점
단계: 데이터 로드 → 모델 초기화 → Stage0(PairClassifier)
    → 그리드 서치 → Stage1(JointHead) → Stage2(도메인 적응)
    → 최종 평가 → 예측 및 이름 복원
"""
import inspect as _inspect
from collections import namedtuple

import torch
import torch.nn as nn

from config import Args, AI, MB, AE, set_seed, get_device
from data import (load_graph_json, build_subgraph, build_nodes_by_domain, build_adj_sets)
from net import CompRGCNEncoder, PairClassifier, GRL, JointHead
from train import (TrainState, train_for_epochs, run_grid_search,
                   train_stage1_joint, train_stage2_joint, JOINT_NEG_RATIO_EVAL)
from eval import (eval_joint_reltype_with_nolink, eval_joint_micro, get_H)
from predict import (predict_joint_flexible, attach_names_to_predictions)

# Python 3.11+ inspect.getargspec 제거 대응 패치
if not hasattr(_inspect, "getargspec"):
    _ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
    def _getargspec(f):
        fs = _inspect.getfullargspec(f)
        return _ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)
    _inspect.getargspec = _getargspec


def _build_domain_head(encoder_dim: int, grl: GRL, device) -> nn.Module:
    return nn.Sequential(
        grl,
        nn.Linear(encoder_dim, 64), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, 2)
    ).to(device)


def main():
    args   = Args()
    device = get_device()
    set_seed(args.seed)
    print(f"device={device} | torch={torch.__version__} | cuda={torch.version.cuda}")

    # ── 1. 데이터 로드 ──────────────────────────────────────────
    data, meta = load_graph_json(args.graph)

    src = build_subgraph(data, keep_domains={AI, MB})
    tgt = build_subgraph(data, keep_domains={AI, AE})
    src.nodes_by_domain = build_nodes_by_domain(src.node_domain)
    tgt.nodes_by_domain = build_nodes_by_domain(tgt.node_domain)
    adj_src = build_adj_sets(src.edge_index, src.num_nodes)
    adj_tgt = build_adj_sets(tgt.edge_index, tgt.num_nodes)
    print(f"SRC(AI+MB) N={src.num_nodes}  TGT(AI+AE) N={tgt.num_nodes}")

    # ── 2. 모델 초기화 ──────────────────────────────────────────
    num_ent = max(src.num_nodes, tgt.num_nodes)
    encoder = CompRGCNEncoder(
        num_ent=num_ent, num_rel=data.num_rel,
        init_dim=args.init_dim, gcn_dim=args.gcn_dim, embed_dim=args.encoder_dim,
        dropout=args.dropout, opn="corr", bias=True, gcn_layers=2
    ).to(device)
    pair_head  = PairClassifier(args.encoder_dim, hidden=128, num_classes=2).to(device)
    joint_head = JointHead(dim=args.encoder_dim, hidden=128, num_rel=data.num_rel).to(device)
    grl        = GRL().to(device)
    domain_head = _build_domain_head(args.encoder_dim, grl, device)

    optim = torch.optim.Adam(
        list(encoder.parameters()) + list(pair_head.parameters()) +
        list(domain_head.parameters()) + list(joint_head.parameters()),
        lr=args.lr
    )
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_ph  = sum(p.numel() for p in pair_head.parameters())
    n_jh  = sum(p.numel() for p in joint_head.parameters())
    print(f"Params — Encoder: {n_enc:,} | PairHead: {n_ph:,} | JointHead: {n_jh:,}")

    # ── 3. 도메인 페어 정의 ─────────────────────────────────────
    keep_src_pairs       = {(AI, AI), (AI, MB), (MB, MB), (MB, AI)}
    keep_tgt_pairs_inter = {(AI, AE), (AE, AI)}

    # ── 4. Stage0: PairClassifier 사전학습 ──────────────────────
    print("\n=== Stage0: PairClassifier pre-training ===")
    state = TrainState(encoder=encoder, pair_head=pair_head,
                       grl=grl, domain_head=domain_head, optim=optim)
    state = train_for_epochs(state, src, tgt, args, device,
                             keep_src_pairs, keep_tgt_pairs_inter,
                             adj_src, adj_tgt, num_epochs=100)
    print(f"Stage0 done. BEST@{state.best_ep}: SRC={state.best_src:.4f} TGT={state.best_tgt:.4f}")

    # ── 5. 그리드 서치 (선택 실행) ─────────────────────────────
    print("\n=== Grid Search ===")
    gs_results = run_grid_search(src, tgt, data, args, device)
    best = gs_results[0]
    print(f"Best: neg={best['neg_ratio']} grl={best['grl_lambda_max']} ent={best['entropy_w']}")

    # ── 6. Stage1: JointHead 소스 학습 ─────────────────────────
    print("\n=== Stage1: JointHead source training ===")
    train_stage1_joint(encoder, joint_head, optim, src, args, device,
                       keep_src_pairs, neg_ratio=10, epochs=50)

    # ── 7. Stage2: 타깃 도메인 적응 ─────────────────────────────
    print("\n=== Stage2: Target domain adaptation ===")
    train_stage2_joint(encoder, joint_head, optim, grl, src, tgt, adj_tgt,
                       args, device, keep_src_pairs)

    # ── 8. 최종 평가 ─────────────────────────────────────────────
    print("\n=== Final Evaluation ===")
    metrics = eval_joint_reltype_with_nolink(
        encoder, joint_head, tgt, num_rel=data.num_rel,
        restrict_pairs=keep_tgt_pairs_inter,
        neg_ratio_eval=JOINT_NEG_RATIO_EVAL,
        sample_limit_pos=args.eval_sample_limit,
        batch=2048, device=device
    )
    print(f"ACC={metrics['acc']:.4f} AUC={metrics['auc_macro']:.4f} "
          f"MRR={metrics['mrr']:.4f} H@10={metrics['hits10']:.4f} "
          f"MacroF1={metrics['macro_f1']:.4f} WeightedF1={metrics['weighted_f1']:.4f}")

    H_all = get_H(encoder, tgt, device=device)
    met_micro = eval_joint_micro(
        H_all, tgt.edge_index, tgt.edge_type, joint_head,
        tgt.node_domain, tgt.nodes_by_domain,
        restrict_pairs=keep_tgt_pairs_inter,
        neg_ratio_eval=JOINT_NEG_RATIO_EVAL,
        sample_limit=args.eval_sample_limit, device=device
    )
    print(f"thr_best={met_micro['thr_best']:.3f}  F1@best={met_micro['f1_best'].item():.4f}")

    # ── 9. 예측 및 후처리 ────────────────────────────────────────
    print("\n=== Prediction ===")
    df_pred = predict_joint_flexible(
        H_all, tgt, joint_head, meta,
        directions=("AI->AE", "AE->AI"),
        thr=0.4, score_mode="link",
        alpha_no_link_bias=0.0, top_k_per_src=None,
        batch=4096, exclude_existing=True,
        csv_path="scopus_pred_edges_AE_Threshold.csv"
    )

    if not df_pred.empty:
        idx2identity = _build_idx2identity_from_subgraph(tgt)
        attach_names_to_predictions(
            pred_csv="scopus_pred_edges_AE_Threshold.csv",
            json_kg=args.graph,
            out_csv="01.20_pred_edges_named.csv",
            idx2identity=idx2identity
        )


def _build_idx2identity_from_subgraph(g) -> dict:
    if hasattr(g, "idx2orig_id") and g.idx2orig_id is not None:
        return {i: v for i, v in enumerate(g.idx2orig_id) if v is not None}
    return {}


if __name__ == "__main__":
    main()
