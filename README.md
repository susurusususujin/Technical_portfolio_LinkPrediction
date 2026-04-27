# Technical_portfolio_vision
서울과학기술대학교_데이터사이언스학과_24장수진_졸업심사_테크니컬포트폴리오
## Technical Portfolio — Link Prediction on Knowledge Graph

Knowledge Graph 기반 도메인 간 링크 예측(Link Prediction) 파이프라인으로 CompRGCN 인코더 + Gradient Reversal Layer(GRL)를 활용한 도메인 적응(Domain Adaptation) 방법론을 구현



### Overview

세 가지 도메인(MB / AI / AE)으로 구성된 지식 그래프에서,  **소스 도메인(AI+MB)** 에서 학습한 링크 예측 모델을 **타깃 도메인(AI+AE)** 에 적용하여 타깃 도메인에서의 연결 관계를 예측

- MB = Molecular Biology
- AI = Artificial Intelligence
- AE = Aerospace Engineering

**학습 파이프라인:**
```
Stage0: PairClassifier 사전학습
  → Grid Search: 최적 하이퍼파라미터 탐색
  → Stage1: JointHead 소스 도메인 학습
  → Stage2: GRL 기반 타깃 도메인 적응
  → 최종 평가 & 예측 CSV 출력
```



### 폴더 구조

```
├── main.py                  # 파이프라인 진입점
├── config.py                # 하이퍼파라미터 및 설정
├── data.py                  # 그래프 로딩, 서브그래프 구성, 음성 샘플링
├── net.py                   # CompRGCNEncoder, GRL, PairClassifier, JointHead
├── train.py                 # 학습 루프, 그리드 서치
├── eval.py                  # 평가 지표 (AUC, MRR, Hits@10, F1 등)
├── predict.py               # 링크 예측 및 후처리
├── helper.py                # CompGCN 유틸 함수 (get_param, ccorr 등)
├── model/                   # CompGCN 원본 레이어 모듈
└── 01.12_total.json         # 입력 지식 그래프 데이터
```



### 설치

```bash
python >= 3.12
torch >= 2.0
torch-geometric
torch-scatter
scikit-learn
numpy
```

### Run

```bash
source .lpvenv/bin/activate
python main.py
```

실행 완료 후 `scopus_pred_edges_AE_Threshold.csv`와 `01.20_pred_edges_named.csv` 파일이 생성됨



### Evaluation Metrics

- 아래의 네 가지 평가지표를 활용
  
| Metric | Description |
|---|---|
| AUC | 링크 존재 여부 이진 분류 |
| MRR | Mean Reciprocal Rank |
| Hits@10 | Top-10 내 정답 비율 |
| Macro / Weighted F1 | 관계 타입 분류 |



### Notes

- GPU가 없는 경우 CPU에서도 실행 가능하긴 하지만, 속도가 매우 느림
- 그리드 서치는 125개 조합을 탐색하며 시간이 많이 소요될 수 있음
