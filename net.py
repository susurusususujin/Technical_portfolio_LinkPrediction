"""
model.py — 신경망 모듈
  - CompRGCNEncoder : CompGCN 기반 관계형 그래프 인코더
  - GRL / GradReverse : Gradient Reversal Layer
  - PairClassifier  : 링크 존재 이진 분류기
  - JointHead       : 링크 존재 + 관계 타입 동시 예측
"""
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from model.compgcn_conv import CompGCNConv


class CompRGCNEncoder(nn.Module):
    """CompGCN 레이어 2개를 쌓은 관계형 그래프 인코더."""

    def __init__(self, num_ent, num_rel, init_dim=64, gcn_dim=128, embed_dim=64,
                 dropout=0.1, opn="corr", bias=True, gcn_layers=2):
        super().__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.init_emb = nn.Parameter(torch.randn(num_ent, init_dim))
        self.init_rel = nn.Parameter(torch.randn(num_rel * 2, init_dim))   # 양방향

        self.p = SimpleNamespace(
            dropout=dropout, opn=opn, bias=bias, num_bases=0, gcn_layer=gcn_layers,
            embed_dim=embed_dim, init_dim=init_dim, gcn_dim=gcn_dim,
            num_ent=num_ent, num_rel=num_rel
        )
        self.conv1 = CompGCNConv(init_dim, gcn_dim, num_rel, act=torch.tanh, params=self.p)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CompGCNConv(gcn_dim, embed_dim, num_rel, act=torch.tanh, params=self.p) \
            if gcn_layers == 2 else None
        self.drop2 = nn.Dropout(dropout)
        self.register_parameter("bias", Parameter(torch.zeros(num_ent)))

    def forward(self, edge_index, edge_type):
        x, r = self.conv1(self.init_emb, edge_index, edge_type, rel_embed=self.init_rel)
        x = self.drop1(x)
        if self.conv2 is not None:
            x, r = self.conv2(x, edge_index, edge_type, rel_embed=r)
            x = self.drop2(x)
        return x


class GradReverse(torch.autograd.Function):
    """역전파 시 그래디언트에 -λ 를 곱하는 커스텀 autograd 함수."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lambd, None


class GRL(nn.Module):
    """λ 스케줄러를 포함한 Gradient Reversal 래퍼."""

    def __init__(self):
        super().__init__()
        self._l = 0.0

    def set_lambda(self, l: float):
        self._l = float(l)

    def forward(self, x):
        return GradReverse.apply(x, self._l)


class PairClassifier(nn.Module):
    """두 노드 임베딩(hu, hv)으로 링크 존재 여부를 이진 분류."""

    def __init__(self, d, hidden=128, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d * 4, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, hu, hv):
        z = torch.cat([hu, hv, (hu - hv).abs(), hu * hv], dim=-1)
        return self.fc(z)


class JointHead(nn.Module):
    """링크 유무 + 관계 타입 동시 예측.
    출력 레이블: 0 = No_Link, 1..R = 관계 타입 ID."""

    def __init__(self, dim, hidden, num_rel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, num_rel + 1)
        )

    def forward(self, hu, hv):
        z = torch.cat([hu, hv, (hu - hv).abs(), hu * hv], dim=-1)
        return self.net(z)
