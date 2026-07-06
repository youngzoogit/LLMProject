# 프로젝트 설계서
## TCGA 공개 암 유전체 데이터를 활용한 AI 기반 암종 분류 및 설명 가능한 유전체 분석 지원 시스템

---

## 1. 프로젝트 개요

**목표**: 암종을 예측하는 것 자체가 아니라, "왜 그렇게 예측했는지"를 유전자 단위 근거와 함께 연구자에게 설명하는 시스템.

**핵심 차별점**: ML/DL 모델 비교는 방법론적 엄밀성을 담당하고, RAG+LLM 해석층이 프로젝트의 핵심 가치를 담당한다. 모델 정확도 자체를 극한까지 끌어올리는 데 시간을 쓰지 않는다.

**5일 성공 기준(Definition of Done)**:
- 샘플 하나를 선택하면 3개 모델의 예측 결과, 공통 Top gene, RAG 근거 문서, LLM 설명이 Streamlit에서 끝까지 나온다.
- 모델 3종 성능 비교표 + Confusion Matrix가 있다.
- RAG 근거 없이 LLM이 생물학적 주장을 하지 않는다(프롬프트로 강제).

---

## 2. 데이터 설계

**소스**: UCSC Xena — TCGA PANCAN 코호트
- Gene Expression RNA-seq (log2 변환된 정규화 값, 이미 배치효과 보정됨)
- Clinical/Phenotype 데이터 (샘플 바코드, 암종 라벨, 기타 임상변수)

**암종 선정 (5~8개, 확정 필요)**
난이도와 스토리를 위해 "구분이 쉬운 암종"과 "조직학적으로 유사해 헷갈리는 암종 쌍"을 섞는 것을 권장:

| 암종 코드 | 이름 | 선정 이유 |
|---|---|---|
| BRCA | 유방암 | 샘플 수 많음, 안정적 베이스라인 |
| LUAD | 폐선암 | LUSC와 혼동 쌍 → 모델이 진짜 어려운 걸 구분한다는 스토리 |
| LUSC | 폐편평세포암 | LUAD와 혼동 쌍 |
| KIRC | 신장암(투명세포) | 조직적으로 구분 뚜렷, 대조군 역할 |
| COAD | 대장암 | 샘플 수 충분 |
| THCA | 갑상선암 | 예후 좋은 암, 대조 스토리 |

(선택 사항: PRAD, LIHC 등 추가 가능. 최종 리스트는 다운로드 후 샘플 수 확인하고 확정)

**전처리 파이프라인**
1. Expression matrix 다운로드 (유전자 × 샘플 형태) → **전치(transpose)**하여 샘플 × 유전자로 변환
2. Clinical 데이터에서 샘플 바코드 기준으로 암종 라벨 매칭 (barcode join 시 정상조직 vs 종양조직 구분 코드 확인 — TCGA 바코드 14번째 자리가 조직 유형)
3. 선정한 5~8개 암종만 서브셋
4. 상위 분산(variance) 유전자 1,000~2,000개로 피처 축소 (차원 축소 근거를 문서화)
5. 결측치 처리 (임계치 이상 결측 유전자 제거, 나머지는 대체)
6. **Stratified train/test split** (암종별 비율 유지, 예: 80/20)

---

## 3. 모델 설계

### 3.1 모델 정의 및 세부 스펙
본 시스템은 다 클래스 암종 분류를 위해 3가지 종류의 AI/머신러닝 모델을 활용하며, 각 모델의 실제 하이퍼파라미터 스펙 및 설계 목적은 다음과 같습니다.

| 모델 (Algorithm) | 세부 파라미터 설정 (Parameters) | Top Gene 추출 및 설명 방식 (XAI Method) | 아키텍처적 선정 및 튜닝 이유 (Rationale) |
|---|---|---|---|
| **Logistic Regression** | <ul><li>`solver='saga'`</li><li>`l1_ratio=0.5` (ElasticNet)</li><li>`C=1.0`</li><li>`class_weight='balanced'`</li><li>`StandardScaler` 선행 적용</li></ul> | **계수(Coefficient) 절댓값**<br>- 각 클래스별 가중치 크기<br>- 전체 평균 가중치 크기 | <ul><li>SAGA 솔버를 사용하여 대규모 고차원 유전자 특징 공간에서 수렴 안정성 확보</li><li>L1/L2 규제를 반반씩 섞은 ElasticNet을 통해 불필요한 유전자 계수를 0으로 만들어 스파시티(Sparsity) 유도</li><li>StandardScaler를 전처리하여 계수 크기를 직접적으로 유전자 간 중요도로 비교 가능케 함</li></ul> |
| **Random Forest** | <ul><li>`n_estimators=400`</li><li>`max_depth=25`</li><li>`min_samples_leaf=2`</li><li>`max_features='sqrt'`</li><li>`class_weight='balanced_subsample'`</li></ul> | **Permutation Importance**<br>- 학습 데이터 상에서 타겟 유전자 컬럼 무작위 셔플 시의 Accuracy/Recall 감소량 | <ul><li>트리 기반 앙상블로 비선형 관계 포착</li><li>트리 기본 제공인 Impurity 기반 중요도는 상관관계가 높은 유전자 편향이 심하므로, 순열 중요도로 정밀하게 대체</li><li>`balanced_subsample`로 부트스트랩 샘플마다 가중치를 동적으로 조절하여 불균형 해결</li></ul> |
| **MLP (Deep Learning)** | <ul><li>`hidden_layer_sizes=(256, 64)`</li><li>`activation='relu'`</li><li>`alpha=1e-3` (L2 penalty)</li><li>`early_stopping=True`</li><li>`validation_fraction=0.1`</li><li>`n_iter_no_change=10`</li><li>`max_iter=300`</li><li>`StandardScaler` 선행 적용</li></ul> | **Permutation Importance**<br>- 훈련 세트 상에서의 글로벌 중요도 도출 후 상위 100개 대상 One-vs-Rest Recall 하락 평가 | <ul><li>scikit-learn MLP는 Dropout 레이어가 없으므로 강한 L2 규제(`alpha=1e-3`)와 Early Stopping을 적용하여 오버피팅 제어</li><li>유전체 데이터 특징인 '고차원 소표본(High-Dim, Low-Sample)' 특성에 맞춘 컴팩트한 레이어 설계</li></ul> |

### 3.2 핵심 설계 요소
*   **클래스 불균형(Class Imbalance) 해결**:
    *   BRCA(유방암) 등 특정 암종에 샘플 수가 편중되어 있어 모델이 다수 클래스에 유리하게 학습되는 것을 방지하기 위해 `class_weight='balanced'`(Logistic) 및 `class_weight='balanced_subsample'`(Random Forest)을 기본 적용합니다.
    *   학습 및 최종 평가는 단순 정확도(Accuracy)가 아닌, 각 클래스별 성능을 균등하게 가중 평균하는 **Macro F1을 최우선 평가지표**로 활용합니다.
*   **특성 스케일링(Feature Scaling)**:
    *   유전자 발현량 변수 간의 크기 편차가 학습에 악영향을 미치는 것을 막기 위해, Logistic Regression 및 MLP 전단에 `StandardScaler`를 결합한 Scikit-learn Pipeline 구조로 모델을 구성합니다.
*   **설명가능 AI(XAI) 연산 효율 최적화**:
    *   1,500개의 피처 전체에 대해 MLP/RF 모델의 순열 중요도(Permutation Importance)를 구하는 것은 극심한 속도 저하를 야기하므로, **2단계 구조**로 연산을 분할합니다.
    *   **1단계**: 전체 피처에 대해 Stratified Subsample 데이터셋으로 전체적인 글로벌 중요도를 1차 계산합니다.
    *   **2단계**: 각 클래스별(암종별) 세부 중요도를 산출할 때는 상위 100개 주요 유전자(`PERCLASS_CANDIDATE_GENES = 100`)로 타겟 피처를 사전에 필터링한 후 `One-vs-Rest Recall`의 감소 폭을 계산하여 성능 보틀넥을 해소합니다.

**평가지표**: Accuracy, Precision/Recall/F1(**Macro F1을 주지표**), Confusion Matrix
**주의**: 클래스 불균형이 존재하므로 Accuracy 단독 해석을 금지하며, Macro F1 기준으로 최종 우수 모델을 선정합니다.

---

## 4. 해석 설계 — 3-way Top Gene 비교

1. 각 모델에서 암종별 Top10 유전자 추출 (LR=계수, RF/MLP=permutation importance)
2. 세 모델의 Top5(또는 Top10)를 표로 나열하고, **몇 개 모델에 공통으로 등장하는지 색/마크로 표시**
   - 3모델 공통 → 최우선 RAG 검색 대상
   - 2모델 공통 → 차순위
   - 1모델 고유 → 참고용으로만 표시
3. 벤 다이어그램(3-set) 시각화로 Streamlit 또는 발표자료에 활용

이 "모델 간 합의(consensus)"가 RAG 검색 우선순위와 발표 스토리의 핵심 근거가 된다.

---

## 5. RAG 설계

**코퍼스 구성 (Top gene 20~50개 기준)**
각 유전자당 문서 하나, 포함 내용:
- 유전자 기능 요약
- 해당 암종과의 관계 (알려진 driver/suppressor 여부)
- 관련 pathway
- 치료 타겟 여부 (표적치료제 존재 시 명시)
- 출처(논문/데이터베이스) 명시 — LLM이 인용할 수 있도록

**소스 후보**: GeneCards, OncoKB, CIViC 등 공개 큐레이션 데이터베이스 (자동 크롤링보다는 선정된 유전자에 한해 수동/반자동 정리 권장 — 5일 내 신뢰도 확보를 위해)

**구현**
- 청킹: 유전자 단위 문서 → 섹션별(기능/암연관/치료타겟) 분리 청크
- 메타데이터: 유전자 심볼, 암종, pathway, 치료타겟 여부(필터링용)
- 임베딩: sentence-transformers 등 로컬 임베딩
- Vector DB: FAISS 또는 Chroma
- 검색 방식: **메타데이터 필터(유전자/암종) + 의미 검색 하이브리드**

---

## 6. LLM 설계

**입력 프롬프트 구성**
- 예측 결과: 암종, 확률, 사용 모델
- Top gene 목록 + 모델 간 합의 여부(3-way 비교 결과)
- RAG로 검색된 근거 문서(출처 포함)

**출력 요구사항**
- 왜 해당 암종으로 예측했는지
- 어떤 유전자가 핵심이며 몇 개 모델이 동의했는지
- 해당 유전자의 역할(RAG 근거 기반)
- **근거 문서에 없는 내용은 "근거 제한적"이라고 명시** — 확신에 찬 지어내기 방지

**Hallucination 방지 규칙**
- 시스템 프롬프트에 "반드시 제공된 검색 문서에 근거하여만 답하라. 문서에 없는 정보는 추측하지 말고 불확실하다고 밝혀라" 명시
- 출력에 인용 출처 포함 강제

---

## 7. Streamlit 구성

1. 샘플 선택 (드롭다운/검색)
2. 암종 예측 결과 (3개 모델 비교 표시)
3. 예측 확률 (막대/게이지)
4. 중요 유전자 Top10 — **모델 간 합의 표시 포함**
5. RAG 검색 결과 (근거 문서 카드, 출처 명시)
6. LLM 종합 설명 (근거 인용 포함)

---

## 8. 폴더 구조 (제안)

```
tcga-cancer-classifier/
├── data/
│   ├── raw/                  # Xena 원본 다운로드
│   └── processed/            # 전처리 완료본
├── src/
│   ├── data_loader.py        # 다운로드·전치·라벨매칭
│   ├── preprocess.py         # 분산필터링·split
│   ├── models/
│   │   ├── logistic.py
│   │   ├── random_forest.py
│   │   └── mlp.py
│   ├── interpret.py          # permutation importance, coefficient 추출, 3-way 비교
│   ├── rag/
│   │   ├── build_corpus.py
│   │   ├── index.py          # FAISS/Chroma 인덱싱
│   │   └── retrieve.py
│   └── llm/
│       └── explain.py        # 프롬프트·생성
├── app.py                    # Streamlit 진입점
├── notebooks/                # EDA·실험
└── README.md
```

---

## 9. 5일 개발 일정

| Day | 작업 | 완료 기준 |
|---|---|---|
| 1 | Xena 다운로드, 전치, 라벨매칭, 분산필터링, split, RF 베이스라인 | RF로 end-to-end 학습·예측 1회 성공 |
| 2 | LR·MLP 학습, 3모델 비교표·Confusion Matrix, Permutation Importance/계수로 Top Gene 추출, 3-way 비교 | 최소 대시보드에 예측+Top Gene 표시 |
| 3 | RAG 코퍼스 작성(20~50 유전자), 청킹·임베딩·인덱싱, 검색 품질 확인 | Top gene 쿼리 시 관련 문서 검색됨 |
| 4 | LLM 프롬프트 설계·연결, Streamlit 6개 섹션 전체 연결 | 샘플 선택→설명까지 전체 흐름 1회 완주 |
| 5 | 버퍼: 시각화 다듬기, SHAP(RF, 여유 시), 발표자료 정리 | 발표 가능 상태 |

---

## 10. 주요 리스크

| 리스크 | 완화 방안 |
|---|---|
| 33개 전체 암종은 5일에 과함 | 5~8개로 축소 (2번 참고) |
| RF impurity importance의 상관유전자 편향 | Permutation Importance로 통일 |
| MLP가 RF/LR보다 성능이 낮을 수 있음 | 실패가 아니라 "고차원 저샘플 데이터에서 DL이 항상 우세하지 않다"는 결과로 프레이밍 |
| LLM의 생물학적 hallucination | RAG 근거 강제 프롬프트 + 출처 표시 + 불확실성 명시 |
| 데이터 전치·바코드 매칭 실수 | Day 1에 형태 확인을 최우선으로 처리 |
| 임상적 오해 소지 | 대시보드·발표자료에 "교육·연구 목적, 임상 진단 아님" 명시 |
