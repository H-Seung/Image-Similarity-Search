# 유사 이미지 검색 & 이상 탐지기

CLIP / ResNet-50 기반 이미지 유사도 검색과 AutoEncoder / PatchCore 기반 비지도 이상 탐지를 통합한 GUI 애플리케이션입니다.

<div style="display: flex; gap: 10px;">
    <img src="assets/result_similarity.png" height="300" style="object-fit: contain;"/>
    <img src="assets/result_anomaly.png" height="300" style="object-fit: contain;"/>
</div>

---

## 목차

1. [개요](#1-개요)
2. [시스템 설계](#2-시스템-설계)
3. [설치 및 실행](#3-설치-및-실행)
4. [이상 탐지 실험](#4-이상-탐지-실험)
5. [결론](#5-결론)
6. [한계 및 향후 과제](#6-한계-및-향후-과제)
- [부록 A. 설정 옵션](#부록-a-설정-옵션)
- [부록 B. 문제 해결](#부록-b-문제-해결)

---

## 1. 개요

### 주요 기능

**유사 이미지 검색**
- Drag & Drop 인터페이스로 쿼리 이미지 입력
- CLIP (ViT-B/32) / ResNet-50 모델 실시간 전환 가능
- 사전 계산된 임베딩 DB 기반 빠른 검색 — DB 크기와 무관하게 일정한 추론 시간
- L2 정규화 cosine similarity 기반 Top-K 유사 이미지 및 유사도 점수 표시

**이상 탐지**
- **비지도 학습** — 정상 이미지만으로 학습, 이상 레이블 불필요
- **두 가지 모델** — GUI RadioButton으로 실시간 전환 가능
  - **AutoEncoder**: 3-layer conv encoder-decoder, Patch MSE 기반 이상 스코어
  - **PatchCore**: WideResNet-50 pretrained feature 기반 메모리뱅크, KNN 거리로 이상 판단
- 카테고리별 threshold 자동 보정 (train/good 분포의 99th percentile, 비지도)
- 결함 위치 히트맵 시각화
- `config.py`로 모델 전환 및 threshold 파일 경로 통합 관리

### 실험 개요

MVTec-AD 7개 카테고리를 대상으로 **Global MSE → Patch MSE → PatchCore** 3단계 ablation을 수행했습니다. 각 단계에서 성능 변화의 원인을 분석해 두 가지를 확인했습니다.

- **Scoring 방식 변경만으로 성능 개선** (Step 1→2): 재학습 없이 평균 AUROC +0.110
- **재구성 오차 기반 접근의 구조적 한계** (Step 2→3): Scoring 개선으로 풀리지 않는 카테고리가 존재하며, PatchCore 전환 시 평균 +0.260 추가 향상

---

## 2. 시스템 설계

### 2-1. 유사 이미지 검색

**파이프라인**

매번 이미지 간 비교를 수행하는 대신, 준비된 DB 이미지 전체의 임베딩을 사전 계산해두고(DB pool) 쿼리 시점에 cosine similarity만 계산합니다.

```
[오프라인] DB 이미지 → Embedder → L2 정규화 → 임베딩 저장
[온라인]   쿼리 이미지 → Embedder → Cosine Similarity → Top-K 반환
```

두 모델 모두 임베딩을 L2 정규화하므로 Cosine Similarity = dot product로 계산됩니다.

**모델 비교**

| | CLIP (ViT-B/32) | ResNet-50 |
|---|---|---|
| 학습 방식 | 이미지-텍스트 contrastive learning | ImageNet 분류 학습 |
| 임베딩 차원 | 512-dim | 2048-dim (GAP layer) |
| 유사도 기준 | 의미론적(semantic) 유사도 | 시각적 feature 유사도 |
| 특성 | 카테고리가 같으면 외형이 달라도 유사하게 판단 | 외형·질감이 비슷한 이미지를 우선 반환 |

### 2-2. 이상 탐지

**핵심 문제**

비지도 이상 탐지(Unsupervised Anomaly Detection)는 이상 데이터 없이 정상 이미지만으로 학습한 뒤, 추론 시 정상 분포에서 벗어난 이미지를 탐지해야 합니다.

| 접근법 | 핵심 가정 | 구조적 한계 |
|--------|-----------|-------------|
| 재구성 오차 기반 (AE) | 정상은 잘 복원되고, 이상은 복원 오차가 높을 것 | (1) 이상 패턴까지 일반화하는 over-generalization 발생 가능 </br>(2) 정상 분포가 넓으면 (pose variation 등) 정상 샘플에서도 높은 오차 발생 |
| Feature Distance 기반 (PatchCore) | Pretrained feature space에서 정상은 응집되고 이상은 멀리 분포할 것 | (1) backbone feature 품질에 강하게 의존 </br>(2) patch 단위 비교로 global structure 이해 제한적 </br>(3) memory bank가 정상 분포를 완전히 대표 못하면 오탐 가능 |

---

## 3. 설치 및 실행

### 3-1. 요구사항 및 패키지 설치

- Python 3.10
- CUDA 12.1 (GPU 사용 시)

GPU 사용:
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CPU만 사용:
```bash
pip install torch torchvision
pip install -r requirements.txt
```

### 3-2. 프로젝트 구조

```
Image-Similarity-Search/
├── data/
│   ├── images/                       # 유사도 검색용 DB 이미지
│   └── {category}/
│       ├── train/good/               # 이상 탐지 학습 이미지 (정상만)
│       └── test/{defect_type}/       # 이상 탐지 테스트 이미지 (MVTec-AD 구조)
├── models/
│   ├── embedder.py                   # CLIP/ResNet 임베딩 추출
│   ├── used_categories.json          # 사용 카테고리 목록 (7개)
│   ├── autoencoder/
│   │   ├── model.py                  # AutoEncoder 아키텍처 
│   │   ├── inference.py              # run_anomaly_inference
│   │   ├── anomaly_processor.py      # 카테고리별 전처리
│   │   ├── __init__.py
│   │   └── weights/
│   │       ├── autoencoder_{category}.pth   # 카테고리별 가중치 (7개)
│   │       ├── thresholds_patch.json        # Patch MSE threshold
│   │       └── thresholds_globalMSE.json    # Global MSE threshold
│   └── patchcore/
│       ├── model.py                  # WideResNet-50 feature extractor
│       ├── inference.py
│       ├── __init__.py
│       └── memory_bank/
│           ├── patchcore_{category}.pkl     # 카테고리별 coreset 메모리뱅크 (7개)
│           └── thresholds.json              # KNN 거리 기반 threshold
├── utils/
│   ├── search.py                     # 유사도 검색 함수
│   ├── common.py                     # 공유 유틸리티
│   ├── train_autoencoder.py          # AE 학습 스크립트
│   ├── ae_calibrate.py               # AE threshold 보정
│   ├── ae_evaluate.py                # AE AUROC 평가 + KDE 시각화
│   ├── pc_calibrate.py               # PatchCore 메모리뱅크 생성 + threshold 보정
│   ├── pc_evaluate.py                # PatchCore AUROC 평가 + 결함별 분석
│   └── visualize_comparison.py       # 두 모델 비교 ROC + 표 생성
├── results/
│   ├── autoencoder/                  # AE 평가 결과 (분포 KDE, ROC, .npz)
│   ├── patchcore/                    # PatchCore 평가 결과 (분포 KDE, ROC, 결함별 CSV·PNG)
│   └── comparison/                   # 두 모델 비교 시각화 (ROC, 비교표)
├── assets/
├── gui_app.py                        # 통합 GUI 앱
├── config.py                         # 경로 및 모델 설정
└── requirements.txt
```

### 3-3. 데이터 준비

**유사 이미지 검색용**

`data/images/` 폴더에 이미지를 넣어주세요. 파일명은 `카테고리명_일련번호.png` 형태여야 합니다.

```
data/images/cable_001.png
data/images/grid_000.png
```

**이상 탐지용 (MVTec-AD 구조)**

```
data/{category}/train/good/    ← 정상 이미지만
data/{category}/test/good/     ← 정상 테스트
data/{category}/test/{defect}/ ← 이상 테스트
```

### 3-4. 모델 준비

**AutoEncoder (최초 1회)**

```bash
# 1. 학습
python utils/train_autoencoder.py
# → models/autoencoder/weights/autoencoder_{category}.pth

# 2. Threshold 보정 (train/good 스코어의 99th percentile)
python utils/ae_calibrate.py
# → models/autoencoder/weights/thresholds_patch.json

# 3. 정량 평가 (선택)
python utils/ae_evaluate.py
# → results/autoencoder/
```

**PatchCore (최초 1회)**

```bash
# 1. 메모리뱅크 생성 + Threshold 보정
python utils/pc_calibrate.py
# → models/patchcore/memory_bank/patchcore_{category}.pkl, thresholds.json

# 2. 정량 평가 (선택)
python utils/pc_evaluate.py                           # 전체 카테고리
python utils/pc_evaluate.py --categories grid carpet  # 특정 카테고리만
# → results/patchcore/  (AUROC, score 분포, 결함별 분석 CSV·PNG)
```

**두 모델 비교 시각화 (선택)**

```bash
python utils/visualize_comparison.py
# ae_evaluate.py, pc_evaluate.py 결과(.npz)를 재사용합니다 (재추론 없음)
# → results/comparison/comparison_table.png, comparison_roc.png
```

### 3-5. GUI 실행

```bash
python gui_app.py
```

이상 탐지 탭에서 **AutoEncoder / PatchCore** 중 모델을 선택한 뒤 이미지를 드래그 앤 드롭합니다.

---

## 4. 이상 탐지 실험

- **데이터셋**: MVTec-AD 7개 카테고리 (bottle, cable, carpet, grid, hazelnut, leather, metal_nut)
- **학습 조건**: 비지도 — 이상 레이블 및 test 이미지 일절 사용하지 않음
- **평가 지표**: Image-level AUROC (1.0 = 완벽, 0.5 = 랜덤)

---

### Step 1 (Baseline) — AutoEncoder + Global MSE

#### 설계

- **아키텍처**: 3-layer conv encoder-decoder, 입력 128×128
  - 128 / 256 / 512 해상도 예비 실험에서 정상/비정상 간 anomaly score 차이가 128×128에서 가장 명확하게 나타나 채택
- **이상 스코어**: 전체 이미지 픽셀의 평균 재구성 오차 (Global MSE)
- **학습 설정**: Adam (lr=1e-3), MSELoss, batch=16, epochs=50
- **Threshold**: train/good 스코어의 99th percentile (비지도)

#### 결과

<img src="results/autoencoder/score_distributions_global.png" width="90%"/>

<img src="results/autoencoder/roc_curves_global.png" width="50%"/>

| Category   | AE (Global MSE) |
|------------|:--------------:|
| bottle     | 0.7921         |
| cable      | 0.5260         |
| carpet     | 0.4282         |
| grid       | 0.7561         |
| hazelnut   | 0.9146         |
| leather    | 0.5880         |
| metal_nut  | 0.3578         |
| **Mean**   | **0.6232**     |

구조가 단순한 hazelnut·bottle에서는 준수한 성능을 보였으나, texture 기반(carpet)과 pose variation이 큰 데이터(metal_nut)에서 성능이 크게 저하됐다.

결함 영역은 이미지 전체의 1~5%를 차지하며, Global MSE는 이 국소 이상 신호를 전체 평균 과정에서 희석시킨다. → **Scoring 방식 개선 필요**

---

### Step 2 — AutoEncoder + Patch MSE

#### 설계

모델 재학습 없이 scoring 방식과 threshold 보정만 변경했다.

- 128×128 이미지를 16×16 패치 64개로 분할
- 각 패치별 MSE를 계산해 **최댓값**을 이상 스코어로 사용
  - 최댓값: 결함이 단 하나의 패치에만 집중되더라도 탐지 가능
  - 평균을 쓰면 Step 1과 동일한 희석 문제가 반복됨

#### 결과

<img src="results/autoencoder/score_distributions_patch.png" width="90%"/>

<img src="results/autoencoder/roc_curves_patch.png" width="50%"/>

| Category   | AE (Global MSE) | AE (Patch MSE) | Δ (Step 1→2) |
|------------|:--------------:|:--------------:|:------------:|
| bottle     | 0.7921         | 0.8206         | +0.029       |
| cable      | 0.5260         | 0.5697         | +0.044       |
| carpet     | 0.4282         | 0.5116         | +0.083       |
| grid       | 0.7561         | 0.9323         | **+0.176**   |
| hazelnut   | 0.9146         | 0.9439         | +0.029       |
| leather    | 0.5880         | 0.9209         | **+0.333**   |
| metal_nut  | 0.3578         | 0.4321         | +0.074       |
| **Mean**   | **0.6232**     | **0.7330**     | **+0.110**   |

재학습 없이 평균 **+0.110 AUROC** 향상. 국소 결함이 주된 leather(+0.333)·grid(+0.176)에서 개선 폭이 크게 나타났다. 이는 anomaly score 정의에 localization이 반영되어야 함을 보여준다.

cable·carpet·metal_nut는 scoring 방식 변경으로 개선되지 않았다.

- **cable**: pose variation으로 정상 이미지 자체의 재구성 오차가 높음 → 정상/이상 score 분포가 겹침
- **carpet**: 불규칙 texture를 AE가 blur 형태로 복원 → 결함 영역도 유사하게 복원되어 오차가 낮아짐 (over-generalization)
- **metal_nut**: 심한 회전 variation으로 정상 분포가 넓게 퍼짐 → 정상/이상 score 분포가 겹침

→ scoring 방식이 아닌 **reconstruction 기반 접근의 구조적 한계**. 접근법 전환이 필요함.

---

### Step 3 — PatchCore

#### 설계

ImageNet pretrained feature space에서 정상 패턴은 응집(clustered)되고 이상 패턴은 자연스럽게 멀리 분포한다. Pretrained feature는 pose·texture variation에 더 강건하며, 정상 분포를 메모리뱅크에 저장한 뒤 **거리**로 이상 여부를 판단한다.

- **백본**: WideResNet-50 (ImageNet pretrained, 파라미터 전체 동결)
  - Wide channel이 동일 depth의 ResNet-50 대비 더 풍부한 feature 표현을 제공
- **Feature 추출**: layer2 (512ch, 28×28) + layer3 (1024ch, 14×14) → concat → 1536ch
  - layer2: 국소 texture · structure 이상을 포착하는 세밀한 공간 feature
  - layer3: 구조적 semantic 이상을 포착하는 고수준 feature
- **Locally Aware Pooling**: 3×3 average pooling으로 인접 패치 문맥 통합
- **메모리뱅크**: K-Center Greedy coreset (10%) — 전체 대비 성능을 거의 유지하면서 연산 비용 절감
- **이상 스코어**: KNN (k=1) 최근접 거리 — k>1로 평균화하면 분포 경계 케이스에서 이상 신호가 희석될 수 있음

#### 결과

<img src="results/patchcore/score_distributions.png" width="90%"/>

<img src="results/patchcore/roc_curves.png" width="50%"/>

| Category   | AE (Patch MSE) | PatchCore  | Δ (Step 2→3) |
|------------|:--------------:|:----------:|:------------:|
| bottle     | 0.8206         | 1.0000     | +0.179       |
| cable      | 0.5697         | 0.9940     | **+0.424**   |
| carpet     | 0.5116         | 0.9984     | **+0.487**   |
| grid       | 0.9323         | 0.9599     | +0.028       |
| hazelnut   | 0.9439         | 1.0000     | +0.056       |
| leather    | 0.9209         | 1.0000     | +0.079       |
| metal_nut  | 0.4321         | 0.9961     | **+0.564**   |
| **Mean**   | **0.7330**     | **0.9926** | **+0.260**   |

모든 카테고리에서 AUROC 0.95 이상. Step 2에서 낮았던 cable·carpet·metal_nut의 개선 폭이 압도적으로 크다. Pretrained feature가 pose variation·복잡한 texture에 구조적으로 강건함을 확인했다.

---

### 3단계 전체 비교

<img src="results/comparison/comparison_roc.png" width="50%"/>

| Category   | AE (Global MSE) | AE (Patch MSE) | PatchCore  | Δ (Step 1→2) | Δ (Step 2→3) |
|------------|:--------------:|:--------------:|:----------:|:------------:|:------------:|
| bottle     | 0.7921         | 0.8206         | 1.0000     | +0.029       | **+0.179**   |
| cable      | 0.5260         | 0.5697         | 0.9940     | +0.044       | **+0.424**   |
| carpet     | 0.4282         | 0.5116         | 0.9984     | +0.083       | **+0.487**   |
| grid       | 0.7561         | 0.9323         | 0.9599     | **+0.176**   | +0.028       |
| hazelnut   | 0.9146         | 0.9439         | 1.0000     | +0.029       | +0.056       |
| leather    | 0.5880         | 0.9209         | 1.0000     | **+0.333**   | +0.079       |
| metal_nut  | 0.3578         | 0.4321         | 0.9961     | +0.074       | **+0.564**   |
| **Mean**   | **0.6232**     | **0.7330**     | **0.9926** | **+0.110**   | **+0.260**   |

**Step 1→2 (Global → Patch MSE): +0.110**  
- 재학습 없이 scoring 방식만 바꾸어 유의미한 성능 향상.  
- 국소 결함이 주된 leather(+0.333)·grid(+0.176)에서 개선 폭이 큼.  
- 반면 정상 분포 자체가 넓거나 texture가 복잡한 cable·carpet·metal_nut에서는 효과가 제한적  
→ scoring 방식 외적인 구조적 문제임을 확인.

**Step 2→3 (AE → PatchCore): +0.260**  
- AE에서 낮았던 카테고리(cable +0.424, carpet +0.487, metal_nut +0.564)에서 개선 폭이 압도적. 
- Pretrained feature가 pose variation, 복잡한 texture에 강건하다는 가설을 실험적으로 확인. 
- 반면 AE에서 이미 높은 성능을 보였던 hazelnut·grid는 Step 3 개선 폭이 작아, AE가 유리한 조건(균일·단순한 구조, 국소적 결함)이 존재함을 시사.

**PatchCore의 상대적 약점 — grid (0.9599)**  
7개 카테고리 중 유일하게 0.96 미만. 결함 유형별 세부 분석 결과, glue(0.8658)가 전체 AUROC를 끌어내리는 주요 원인으로 확인됨.   
→ 아래 세부 분석 참고

---

### 결함별 세부 분석 — grid · carpet

<img src="results/patchcore/defect_auroc.png" width="70%"/>

| Category | Defect Type         | AUROC  | n_samples |
|----------|---------------------|:------:|:---------:|
| grid     | ALL                 | 0.9599 | 57        |
| grid     | bent                | 0.9960 | 12        |
| grid     | broken              | 0.9960 | 12        |
| grid     | glue                | 0.8658 | 11        |
| grid     | metal_contamination | 0.9957 | 11        |
| grid     | thread              | 0.9394 | 11        |
| carpet   | ALL                 | 0.9984 | 89        |
| carpet   | color               | 1.0000 | 19        |
| carpet   | cut                 | 1.0000 | 17        |
| carpet   | hole                | 0.9979 | 17        |
| carpet   | metal_contamination | 0.9958 | 17        |
| carpet   | thread              | 0.9981 | 19        |

구조적 변형(bent·broken 0.996)은 격자 형태 자체를 변화시켜 feature embedding이 크게 이동하므로 안정적으로 탐지된다.  
반면 glue(0.8658)는 격자 구조를 유지한 채 표면 질감만 미세하게 바꾸어 정상 패치와의 feature distance가 작게 유지된다.

동일 결함(thread)이 있는 carpet과 grid 비교 시 carpet(0.9981) > grid(0.9394)로 성능의 차이가 나타난다.  
carpet의 굵은 텍스처 단위에서 결함이 feature에 더 명확히 반영되는 반면, grid에서는 규칙적인 패턴 내에 결함이 포함되어 희석된다.  
(단, 두 카테고리의 thread 결함 양상이 달라 직접 비교에는 한계가 있다.)

→ 결함 유형 자체보다 **해당 결함이 feature space에서 얼마나 큰 변화를 유도하는지**가 탐지 성능을 결정하는 핵심 요인이다.

---

## 5. 결론

이번 3단계 ablation은 비지도 이상 탐지에서 성능을 결정하는 요인을 단계적으로 드러낸다.

**Step 1→2: 어디서 오차를 보는가 (Scoring Granularity)**  
재학습 없이 Global MSE → Patch MSE 변경만으로 평균 +0.110 향상.  
결함은 이미지 전체가 아닌 국소 영역에 존재하므로, anomaly score 정의에 localization이 반영되어야 탐지 가능함을 확인했다.

**Step 2→3: 어떤 feature로 비교하는가 (Feature Representation)**  
Scoring 방식 개선으로도 풀리지 않던 cable·carpet·metal_nut이 PatchCore 전환 후 평균 +0.260 추가 향상.  
Reconstruction error보다 pretrained feature distance가 pose variation·complex texture에 본질적으로 강건함을 실험적으로 확인했다.

**결함별 분석: feature space 변화량이 탐지 가능성을 결정**  
grid 내 결함별 AUROC를 보면 구조적 변형(bent·broken 0.996)은 잘 탐지되고, 표면 질감만 바꾸는 glue(0.8658)는 탐지가 어렵다.  
결함 유형 자체보다 해당 결함이 정상 feature 분포에서 얼마나 벗어나는가가 탐지 성능의 핵심 요인이다.

**판단 기준**

| 조건 | 권장 모델 | 근거 |
|------|-----------|------|
| 균일한 구조, 국소적 결함 (grid·hazelnut류) | AE (Patch MSE) | 정상 분포가 좁아 재구성 기반으로도 분리 가능 |
| Pose variation·복잡한 texture (cable·carpet·metal_nut류) | PatchCore | Reconstruction이 정상 다양성을 커버하지 못함 |
| 표면 질감 미세 변화형 결함 (glue류) | 두 모델 모두 한계 | Feature 변화량 자체가 작아 탐지 난이도 높음 |

---

## 6. 한계 및 향후 과제

### 현재 한계

- **Localization 평가 미수행**: AUROC는 이미지 단위 분류 성능. 히트맵은 시각적으로 제공되지만, 결함 위치 정확도를 정량화하는 pixel-level AUROC·PRO score는 측정하지 않음
- **해상도 선택 근거 부족**: 128×128을 소수 샘플 기반 예비 실험으로 결정. 전체 테스트셋 기준 정량 비교 미수행
- **추론 속도 미측정**: AE·PatchCore 간 inference 속도 비교 없음. PatchCore는 메모리뱅크 크기에 따라 추론 비용이 달라짐
- **카테고리 범위 제한**: MVTec-AD 15개 카테고리 중 7개만 사용
- **유사 이미지 검색 정량 평가 없음**: Recall@K 등 retrieval 지표 미측정. ground truth 레이블 부재
- **PatchCore global structure 이해 제한**: patch 단위 비교로 전체 구조적 관계를 반영하지 못함

### 향후 과제

- Pixel-level Localization 평가 및 히트맵 정량화 (PRO score)
- 전체 15개 카테고리로 확장
- PatchCore 추론 속도 최적화 (FAISS 기반 ANN 탐색)
- 다양한 coreset 비율 실험 (1% / 5% / 10% / 25%)
- WideResNet-50 외 다른 백본 (EfficientNet, ViT 등) 비교
- 유사 이미지 검색 정량 평가 (Recall@K)

---

## 부록 A. 설정 옵션

```python
# config.py
MODEL_NAME = "clip"         # "clip" 또는 "resnet"  (유사도 검색 모델)
DEVICE     = "cuda"         # torch.cuda.is_available()로 자동 감지

AE_THRESHOLD_PATH = "models/autoencoder/weights/thresholds_patch.json"
PC_THRESHOLD_PATH = "models/patchcore/memory_bank/thresholds.json"

DEFAULT_TOP_K = 5           # 유사 이미지 검색 기본 개수
```

---

## 부록 B. 문제 해결

**`No module named 'models'`**  
`utils/` 하위 스크립트는 반드시 프로젝트 루트에서 실행하세요.
```bash
python utils/ae_evaluate.py     # ✅
cd utils && python ae_evaluate.py  # ❌
```

**AE 모델 파일 없음**  
`models/autoencoder/weights/autoencoder_{category}.pth` 파일이 필요합니다.  
`python utils/train_autoencoder.py`를 먼저 실행하세요.

**PatchCore 메모리뱅크 없음**  
`python utils/pc_calibrate.py`를 실행해 메모리뱅크를 생성하세요.  
WideResNet-50 다운로드가 최초 1회 자동으로 수행됩니다 (약 70MB).

**`visualize_comparison.py` 오류**  
`ae_evaluate.py`와 `pc_evaluate.py`를 먼저 실행해 `.npz` 파일을 생성해야 합니다.

**메모리 부족**  
`config.py`에서 `DEVICE = "cpu"`로 변경하세요.  
PatchCore의 경우 메모리뱅크 생성 단계에서만 대용량 메모리를 사용합니다.