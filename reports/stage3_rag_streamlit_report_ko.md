# 3단계 보고서 — RAG + LLM 프롬프트 + Streamlit MVP (한국어)

> 영어 원본: `reports/stage3_rag_streamlit_report.md` (삭제하지 않음).
> 이 문서는 발표 준비자가 빠르게 이해할 수 있도록 정리한 요약본입니다.

## 한 줄 요약
샘플 하나를 고르면 **3개 모델 예측 → 확률 → 합의 유전자 → 유전자 근거 문서(RAG)
→ LLM 설명 프롬프트 초안**까지 한 화면에서 확인할 수 있는 MVP를 완성했습니다.
실제 LLM API 호출은 아직 하지 않으며, 프롬프트 초안까지만 생성합니다.

## 전체 흐름 (발표 스토리)
1. **샘플 선택**: `sample_id` 드롭다운에서 하나 선택 (실제 라벨도 함께 표시).
2. **예측**: Logistic Regression / Random Forest / MLP 세 모델이 각각 암종을
   예측하고 확률 Top-3 를 보여줍니다.
3. **입력 품질 체크**: 외부 raw 벡터를 넣을 경우 누락 유전자 비율을 계산해
   10% 이상이면 경고합니다(발표에서는 내부 샘플이라 항상 정상).
4. **합의 유전자**: 2개 이상 모델이 공통으로 중요하다고 본 유전자(primary 18개)를
   표로 보여주고, 각 유전자의 **근거 문서 작성 여부(curated/draft)** 를 표시합니다.
5. **RAG 근거 문서**: 선택한 유전자의 근거 문서를 보여주되, 근거가 없거나 템플릿만
   있으면 "근거 문서 없음" / "근거 제한적" 으로 명확히 구분합니다.
6. **LLM 프롬프트 초안**: 위 정보를 모아 "근거 없는 주장 금지, 임상 아님" 규칙이
   포함된 프롬프트를 생성해 화면에 표시합니다(전송하지 않음).

## 구성 요소별 요약
| 구성 | 파일 | 핵심 |
|---|---|---|
| 예측 헬퍼 | `src/predict.py` | 입력 품질(`input_quality`) 필드 포함, 모델별 예측+확률 반환 |
| RAG 코퍼스 빌더 | `src/rag/build_corpus.py` | primary 18개 유전자 markdown 템플릿 생성(사실 날조 금지) |
| RAG 검색 | `src/rag/retrieve.py` | `retrieve_gene_evidence(gene, cancer_type)`, 없으면 "근거 문서 없음" |
| LLM 프롬프트 | `src/llm/explain.py` | `generate_explanation_prompt(...)`, 근거 강제 규칙 내장 |
| 코퍼스 문서 | `data/rag_corpus/*.md` | 18개 중 8개 curated, 10개 draft |
| 발표용 화면 | `app.py` | 한국어 UI, 임상 아님 경고, curated 상태 표시 |

## 이번 단계에서 새로 한 일
- **한국어 발표용 UI**: 제목/설명/섹션을 한국어로 바꾸고, "연구/교육 목적, 임상
  진단 아님" 문구를 붉은 경고 배너로 상단에 고정 노출.
- **근거 상태 시각화**: 유전자 표에 `evidence_status`(curated/draft) 컬럼 추가,
  유전자 선택 시 초록(✅ curated) / 노랑(📝 draft) 배지로 구분.
- **RAG 코퍼스 일부 실제 근거화**: 8개 유전자(SFTPB, LUM, GPRC5A, HOXC6, MT1H,
  RGN, PRODH, CYP2S1)를 공개 DB(GeneCards/NCBI Gene/UniProt) 기반으로 채우고
  `status: curated` 로 전환. 불확실한 부분은 `근거 제한적` 으로 명시.

## 검증 결과 (실제 실행)
- `python -m src.rag.retrieve SFTPB LUSC`
  → `found: True | curated: True`, `cancer_type match: True`,
    메시지 "근거 문서 있음 (curated evidence available)", 섹션 내용 정상 출력.
- `python -m src.llm.explain`
  → SYSTEM 규칙(근거만 사용/근거 없는 주장 금지/근거 제한적 명시/임상 아님) 포함
    프롬프트 정상 생성.
- `streamlit run app.py`
  → `streamlit.testing.v1.AppTest` 기준 **예외 0개**, 4개 섹션 모두 렌더링,
    임상 경고 배너 표시 확인 → 실행 가능.
- 코퍼스 상태: **curated 8개, draft 10개** (자동 집계로 확인).

## 발표 시 주의 (반드시 언급)
- 모델 정확도가 매우 높지만(Macro F1 약 0.98) 이는 6개 암종이 발현 프로파일로
  잘 구분되기 때문이며, 목표는 정확도가 아니라 **설명 가능성**입니다.
- RAG 근거는 8개 유전자만 실제 작성되어 있고, 나머지는 템플릿입니다. LLM 층은
  근거가 없으면 "근거 제한적" 이라고 답하도록 강제되어 있습니다.
- 임상 진단이 아니라 연구/교육용임을 화면과 발표 모두에서 명시합니다.

## 남은 작업 (다음 단계 후보)
1. draft 10개 유전자 근거 작성 및 primary-literature PMID 보강.
2. keyword 검색 → sentence-transformers + FAISS/Chroma 의미 검색으로 확장.
3. `src/llm/explain.py` 뒤에 실제 LLM API 연결(가드레일 유지).
4. `sample_id` 드롭다운 UX 개선(3,604개 → 암종별 필터/검색).
5. 코퍼스를 secondary 유전자까지 확장.
