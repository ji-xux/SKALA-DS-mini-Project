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

`2018-04-03_varcharge` 데이터는 이번 분석에서 사용하지 않는다. MAT 원본은 용량이 커서 Git에 포함하지 않는다.

## 환경 설정 및 실행

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`.venv`의 Python 커널을 선택하고 `01_EDA.ipynb`, `02_Modeling.ipynb` 순서로 실행한다. DAY2 노트북은 원본 MAT 파일을 다시 읽기 때문에 단독 실행도 가능하다. 모든 무작위 단계에는 `random_state=42`를 사용한다.

## EDA

### Question 1. Cycle Life 분포는 어떻게 생겼는가?

수명 라벨이 있는 129개 셀을 기준으로 분포와 장·단수명 비율을 비교했다. Batch 2의 8개, Batch 3의 2개 결측 수명은 분포 계산에서 제외했다.

| 배치 | 유효 셀 | 평균 / 중앙값 | 범위 | 단수명 <500 | 장수명 >1,000 |
|---|---:|---:|---:|---:|---:|
| Batch 1 | 46 | 844.7 / 858.5 | 534~1,227 | 0.0% | 21.7% |
| Batch 2 | 39 | 565.7 / 472.0 | 392~1,186 | 71.8% | 7.7% |
| Batch 3 | 44 | 1,059.7 / 1,005.5 | 541~1,935 | 0.0% | 52.3% |

- Batch 2는 단수명, Batch 3는 장수명 중심이며 Batch 1은 그 사이에 위치한다.
- 배치별 IQR 기준 하단 이상치는 없었다. 가장 짧은 셀은 Batch 1/2/3에서 각각 534/392/541 Cycle이었다.
- 짧은 셀과 특정 충전 정책은 확인했지만, 짧은 수명의 물리적 원인을 현재 EDA만으로 확정할 수는 없다.

### Question 2. 방전 용량은 어떻게 감소하며 Knee point는 어디인가?

Cycle 2~5의 QD 중앙값을 초기 용량으로 두고 용량 유지율을 계산했다. 열화 곡선은 초기의 완만한 구간 이후 감소가 빨라지는 비선형 형태를 보였다.

| 구분 | Batch 1 | Batch 2 | Batch 3 |
|---|---:|---:|---:|
| Cycle 100 유지율 중앙값 | 100.12% | 100.11% | 100.04% |
| Cycle 500 유지율 중앙값 | 97.77% | 97.26% | 98.33% |
| Knee point 중앙값 | 585.5 | 345.0 | 779.0 |

- Batch 2의 단수명 셀은 약 300~500 Cycle에서 급격한 감소가 두드러졌다.
- 장수명 셀일수록 Knee point가 뒤에 나타나는 경향이 있었다.
- 후반 Cycle에는 생존한 셀만 남으므로 시점별 평균을 같은 셀 집단의 연속 변화로 해석하지 않았다.
- Knee point는 전체 수명 곡선을 이용한 설명용 추정치이며 모델 입력으로 사용하지 않았다.

### Question 3. 초기 ΔQ(V) 곡선에서 수명 차이가 보이는가?

`ΔQ(V) = Qdlin(Cycle 100) − Qdlin(Cycle 10)`으로 계산하고, 셀마다 다른 전압축을 공통 2.0~3.6V 구간에 정렬했다.

- 단수명 셀은 약 2.4~3.2V 구간에서 더 큰 음의 변화와 변동성을 보였다.
- Batch 2의 2.8~3.2V 평균 ΔQ는 단수명 그룹 −0.0525, 장수명 그룹 −0.0116이었다.
- Batch 2의 `log10(ΔQ 분산)` 중앙값은 단수명 −3.4459, 장수명 −4.2714였다.
- `log10_delta_q_var`와 수명의 Spearman 상관은 Batch 1/2/3에서 −0.871/−0.710/−0.800이었다.

초기 ΔQ(V)만으로도 수명 그룹 차이가 확인됐으며, ΔQ 변동성은 수명과 가장 강하고 배치 간 방향도 일관된 초기 신호였다. 다만 이 관계만으로 개별 셀의 수명을 완벽하게 구분할 수 있다는 뜻은 아니다.

### Question 4. 충전 조건(C-rate)과 수명은 어떤 관계인가?

정책 문자열에서 첫 C-rate, 전환 SOC, 두 번째 C-rate를 추출해 수명과의 관계를 확인했다.

| 조건과 수명의 Spearman ρ | Batch 1 | Batch 2 | Batch 3 |
|---|---:|---:|---:|
| 첫 C-rate | −0.483 | +0.055 | −0.229 |
| 전환 SOC | +0.185 | +0.288 | +0.443 |
| 두 번째 C-rate | +0.047 | −0.119 | +0.016 |

- Batch 1의 `4C(80%)-4C`는 평균 1,226.5 Cycle, `5.4C(80%)-5.4C`는 546.5 Cycle이었다.
- 첫 C-rate가 높은 `8C(15%)-3.6C`도 평균 1,008.5 Cycle로, 고속 충전이 항상 단수명으로 이어지지는 않았다.
- Batch 2에서는 같은 수치 조건도 `newstructure` 여부에 따라 평균 수명이 크게 달랐다.

충전 프로토콜별 차이는 있지만 첫 C-rate 하나로 수명을 설명할 수 없다. C-rate, 전환 SOC, 셀 구조와 실험 조건을 함께 고려해야 하며 관찰된 관계를 인과관계로 단정하지 않았다.

### Question 5. 어떤 초기 신호가 수명과 관련되는가?

| 초기 피처와 수명의 Spearman ρ | Batch 1 | Batch 2 | Batch 3 |
|---|---:|---:|---:|
| `log10_delta_q_var` | −0.871 | −0.710 | −0.800 |
| `delta_q_abs_area` | −0.828 | −0.697 | −0.758 |
| `QD_change` | +0.706 | −0.193 | −0.041 |
| `chargetime_mean` | +0.613 | −0.387 | +0.196 |

- ΔQ 변동성이 수명과 가장 강하고 배치 간 방향도 일관됐다.
- `delta_q_std`와 `log10_delta_q_var`는 세 배치에서 Spearman ρ=1.000으로 같은 변동성 정보를 중복 표현했다.
- `delta_q_mean`과 `delta_q_abs_area`, 초기 `QD_mean`과 `QC_mean`도 높은 상관을 보여 정보 중복 가능성이 있었다.
- `QD_change`, 충전시간과 온도 변화량은 배치에 따라 상관 방향이 달랐다.

따라서 ΔQ 통계값을 모두 넣지 않고 `log10_delta_q_var` 하나를 대표 피처로 사용했다. 충전 조건과 충전시간의 추가 효과는 Batch 1 내부 검증에서 별도로 비교했다.

## Modeling

### 피처 엔지니어링 전략

| 피처 | 선정 근거 |
|---|---|
| `log10_delta_q_var` | 수명과 강한 관계를 보인 ΔQ 변동성의 대표 피처 |
| `c_rate_1` | 첫 충전 단계의 충전 강도 |
| `switch_soc` | 두 충전 단계의 전환 조건 |
| `chargetime_mean` | 초기 100사이클 평균 충전시간; 추가 효과 비교용 |

피처 조합은 A=ΔQ 핵심 피처, B=A+충전 조건, C=B+평균 충전시간으로 구성했다. 수명, 전체 기록 길이, Knee point, Cycle 100 이후 값은 입력에서 제외했다. 결측치 대체와 스케일링은 각 훈련 폴드에서만 학습하도록 Pipeline으로 구성했다.

### 모델 선택 및 근거

- 후보 모델: LinearRegression, Ridge, RandomForestRegressor
- 내부 검증: Batch 1에서 충전 정책 그룹을 분리한 Hold-out과 5-fold GroupKFold
- 선택 지표: 평균 CV MAPE
- 최종 모델: Ridge Regression (`alpha=1.0`)
- 최종 피처: `log10_delta_q_var`, `c_rate_1`, `switch_soc`
- 선택 이유: 평균 CV MAPE 7.54%로 후보 조합 중 가장 낮았다.

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
- 개선을 위해 다양한 단수명 셀이 포함된 추가 학습 데이터, 배치·구조 차이를 설명하는 초기 피처와 별도의 미관측 테스트 데이터가 필요하다.

## ESS 도메인 해석

초기 100사이클 이후 셀 수명 스크리닝과 추가 시험 우선순위 결정에 활용할 가능성이 있다. 그러나 현재 Batch 2 일반화 성능은 실제 교체·안전 의사결정에 사용하기에 부족하다.

이 모델은 실험 셀의 총 Cycle Life를 예측하며 운영 중인 BESS의 RUL을 갱신하는 모델은 아니다. 실제 배포 전에는 새로운 운영 데이터 검증, 입력·성능 변화 감시, 안전 기준과 재학습 조건 정의가 필요하다.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.
- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)

## 팀 구성

- `ji-xux` (개인 수행): EDA, 피처 엔지니어링, 모델 개발, Batch 2·3 성능 평가 및 결과 해석
