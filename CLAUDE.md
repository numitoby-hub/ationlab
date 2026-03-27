# PanCADNet v2 — Panoptic Segmentation for Architectural CAD Drawings

## 연구 주제

건축 CAD 도면에서 LINE primitive 단위의 panoptic segmentation.
픽셀이 아닌 CAD 엔티티(LINE) 단위로 31개 건축 요소 클래스를 인식하고, thing/stuff를 구분하여 인스턴스까지 분할한다.

## 파이프라인 구조

```
PNG 이미지 + JSON 어노테이션
        │
   ┌────┴────┐
   │ Vision  │  SegFormer-B2 → FPN fusion (256ch)
   │ Branch  │  → HybridLineSampling (line + region tokens)
   └────┬────┘
        │
   AdaptiveFusion (gating) ──→ Transformer Decoder (100 queries)
        │                         ├─ Panoptic head (cls + mask)
   ┌────┴────┐                    └─ Semantic head (per-primitive)
   │ Graph   │  8D 기하특징 → Progressive GATv2 (k=3,8,16)
   │ Branch  │  multi-scale kNN 그래프 + skip connection
   └─────────┘
```

- **Vision Branch**: SegFormer로 이미지 특징 추출 후, LINE의 start→end를 따라 grid_sample + 주변 region sampling
- **Graph Branch**: 기하 특징(길이, 각도, 중심좌표 등 8D) + multi-scale kNN 그래프 → GATv2Conv 3층
- **Fusion**: graph 특징과 visual 특징을 gating 메커니즘으로 결합
- **Decoder**: Mask2Former 스타일 query-based decoder → Hungarian matching + CE/BCE/Dice/Semantic loss

## 파일 구조

```
archcad/codes/0325/
├── config.py           # 경로, 하이퍼파라미터, 클래스 정의 (31cls, thing/stuff 분류)
├── utils.py            # get_valid_files() — JSON+PNG 매칭, drawing 단위 셔플 후 ID 리스트 반환
├── step1_dataset.py    # PanCADDataset — LINE 파싱, 기하특징 추출, kNN 그래프/GT mask 생성
├── precache.py         # 전체 데이터셋을 .pt 파일로 캐싱 (학습 속도 향상)
├── step2_model.py      # PanCADNetV2 모델 전체 (SegFormer, LineSampling, GATv2, Fusion, Decoder, Loss)
├── step3_h100.py       # 학습 루프 — 캐시 기반, AMP, CosineAnnealing, epoch당 20k 서브샘플링
└── step4_eval.py       # 테스트 평가 — PQ/SQ/RQ (log-weighted IoU) + F1/wF1, 클래스별 리포트
```

## 주요 설정 (config.py)

- 이미지: 512×512, scale 기준 980.0
- 그래프: kNN k=[3, 8, 16], GATv2 hidden=128, heads=4
- 디코더: 100 queries, hidden=256, heads=8, 3 layers
- 학습: lr=2e-4 (backbone 0.1x), batch=8, 30 epochs, AdamW
- Loss: λ_cls=2.0, λ_bce=5.0, λ_dice=5.0 + semantic CE

## 클래스 (31개)

Thing (21): 문류(4종), 엘리베이터, 계단, 위생기구(6종), 테이블, 의자, 침대, 소파, 기둥(2종), 주차, 파일, 소화전
Stuff (10): 축/그리드, 기타 기구류, 배수구, 유리, 벽, 보(2종), 기초, 철근, 기타

## 실행 환경

- Google Colab / H100 GPU 대상
- 데이터 경로: `/content/drive/MyDrive/archcad`, `/content/local_data/data`
- 캐시: `/content/graph_cache_v2`
- 출력: `/content/drive/MyDrive/archcad/pancadnet_v2_output`
