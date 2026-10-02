# ESS 배터리 수명 예측

초기 100사이클의 열화 신호와 충전 조건을 사용해 배터리 셀의 총 Cycle Life를 예측한다.

## 프로젝트 개요

- 데이터셋: MIT–Stanford Battery Dataset (Severson et al., 2019)
- 학습 데이터: Batch 1 (`2017-05-12`)
- 평가 데이터: Batch 2 (`2018-02-20`)
- 추가 평가 데이터: Batch 3 (`2018-04-12`)
- 태스크: Regression — 총 Cycle Life 예측
- 예측 시점: Cycle 100 측정 완료 직후

## 파일 구조

```text
├── 01_EDA.ipynb
├── 02_Modeling.ipynb
├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
├── 2018-02-20_batchdata_updated_struct_errorcorrect.mat
├── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
├── results/
│   └── model_performance.csv
├── requirements.txt
└── README.md
```

`2018-04-03_varcharge` 데이터는 이번 분석에서 사용하지 않는다.

## 환경 설정 및 실행

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`.venv`의 Python 커널을 선택하고 `01_EDA.ipynb`, `02_Modeling.ipynb` 순서로 실행한다. DAY2 노트북은 원본 MAT 파일을 다시 읽기 때문에 단독 실행도 가능하다. 모든 무작위 단계에는 `random_state=42`를 사용한다.

## EDA

### Cycle Life 분포

- Batch 1: 중앙값 858.5 Cycle, 단수명(<500) 0.0%, 장수명(>1,000) 21.7%
- Batch 2: 중앙값 472.0 Cycle, 단수명 71.8%, 장수명 7.7%
- Batch 3: 중앙값 1,005.5 Cycle, 단수명 0.0%, 장수명 52.3%
- 핵심 발견: Batch 2는 단수명, Batch 3는 장수명 중심으로 배치별 수명 분포가 뚜렷하게 다르다.

### 열화 곡선

- 방전 용량은 일정하게 감소하지 않고 초기의 완만한 구간 이후 감소가 가속된다.
- Knee point 중앙값: Batch 1 585.5, Batch 2 345.0, Batch 3 779.0 Cycle
- 핵심 발견: 단수명 셀이 많은 Batch 2에서 급격한 열화가 가장 이르게 나타났다.

### ΔQ(V) 곡선

- `ΔQ(V) = Qdlin(Cycle 100) − Qdlin(Cycle 10)`으로 계산했다.
- 단수명 셀은 ΔQ의 음의 변화와 변동성이 더 컸다.
- `log10_delta_q_var`와 수명의 Spearman 상관은 Batch 1/2/3에서 각각 −0.871/−0.710/−0.800이었다.
- 핵심 발견: ΔQ 변동성은 수명과 가장 강하고 배치 간 방향이 일관된 초기 신호였다.

### 충전 조건

- 충전 프로토콜별 평균 수명 차이는 있었지만 첫 C-rate가 높다고 항상 수명이 짧지는 않았다.
- 동일한 수치 조건도 `newstructure` 여부에 따라 수명 차이가 나타났다.
- 핵심 발견: C-rate 하나보다 전환 SOC와 실험 구조를 함께 고려해야 한다.

## Modeling

### 피처 엔지니어링 전략

| 피처 | 선정 근거 |
|---|---|
| `log10_delta_q_var` | 수명과 강한 관계를 보인 ΔQ 변동성의 대표 피처 |
| `c_rate_1` | 첫 충전 단계의 충전 강도 |
| `switch_soc` | 두 충전 단계의 전환 조건 |
| `chargetime_mean` | 초기 100사이클의 평균 충전시간; 추가 효과 비교용 |

수명, 전체 기록 길이, Knee point, Cycle 100 이후 값은 입력에서 제외했다. 전처리는 각 훈련 폴드에서만 학습하도록 Pipeline으로 구성했다.

### 모델 선택 및 근거

- 후보 모델: LinearRegression, Ridge, RandomForestRegressor
- 내부 검증: Batch 1의 충전 정책 그룹을 분리한 Hold-out과 5-fold GroupKFold
- 최종 모델: Ridge Regression (`alpha=1.0`)
- 최종 피처: `log10_delta_q_var`, `c_rate_1`, `switch_soc`
- 선택 이유: Batch 1 개발 데이터의 평균 CV MAPE가 7.54%로 후보 조합 중 가장 낮았다.

## 성능 결과

| 구분 | n | MAPE | MAE (Cycle) | RMSE (Cycle) | R² |
|---|---:|---:|---:|---:|---:|
| Train (Batch 1 CV) | 35 | 7.54% | 63.59 | 82.94 | 0.646 |
| Valid (Batch 1 Hold-out) | 11 | 13.49% | 136.80 | 196.05 | −0.063 |
| Test (Batch 2) | 39 | 50.10% | 234.98 | 264.82 | −0.458 |
| Test (Batch 3) | 44 | 17.54% | 200.41 | 300.93 | 0.059 |

| 비교 | MAPE Gap |
|---|---:|
| Valid − CV | +5.95%p |
| Batch 2 − Valid | +36.61%p |
| Batch 2 − 원논문 목표 9.1% | +41.00%p |
| Batch 3 − Batch 2 | −32.57%p |
| Batch 3 − 원논문 목표 9.1% | +8.44%p |

## 오류 분석

- Batch 2는 39셀 중 36셀의 수명을 과대 예측했다.
- Batch 2의 30셀은 실제 수명이 Batch 1의 학습 수명 범위보다 짧았다.
- `3.6C(9%)-5C`의 두 셀은 실제 수명 393·396 Cycle을 약 1,004·998 Cycle로 예측해 오차가 가장 컸다.
- 작은 학습 표본, 학습에서 보지 못한 수명·충전 조건, 배치별 분포 차이가 성능 저하의 주요 원인 후보다.
- 개선을 위해서는 다양한 단수명 셀이 포함된 추가 학습 데이터, 배치·구조 차이를 설명하는 초기 피처, 별도의 미관측 테스트 데이터가 필요하다.

## ESS 도메인 해석

초기 사이클링 이후 셀 수명 스크리닝과 추가 시험 우선순위 결정에 활용할 가능성이 있다. 다만 현재 Batch 2 일반화 성능은 실제 교체·안전 의사결정에 사용하기에 부족하다.

이 모델은 실험 셀의 총 Cycle Life를 예측하며, 운영 중인 BESS의 RUL을 갱신하는 모델은 아니다. 실제 배포 전에는 새로운 운영 데이터 검증, 입력·성능 변화 감시, 안전 기준과 재학습 조건 정의가 필요하다.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.
- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)

## 수행 범위

- EDA, 피처 엔지니어링, 모델 개발, Batch 2·3 성능 평가
