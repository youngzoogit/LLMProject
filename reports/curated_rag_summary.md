# RAG 코퍼스 큐레이션 요약

_대상: `data/rag_corpus/` (primary 후보 18개 유전자). 발표용으로 8개를 우선
curated 처리했습니다. 모든 근거는 공개 DB(GeneCards / NCBI Gene / UniProt) 기반이며,
불확실한 내용은 각 문서 `evidence_limitations` 에 "근거 제한적" 으로 명시했습니다._

## 1. curated 완료 유전자 (8개)
| gene | 핵심 기능 | 코호트 내 연관 암종 | 비고 |
|---|---|---|---|
| SFTPB | 폐 계면활성단백 B (폐 II형 세포 마커) | LUAD, LUSC | 폐 계통 마커로 확립 |
| GPRC5A | 레티노산 유도 GPCR, 폐 종양억제자 | LUAD, LUSC | 폐 근거 강함 |
| LUM | Lumican, ECM 프로테오글리칸 | BRCA, COAD | 방향성은 맥락 의존 |
| HOXC6 | 호메오박스 전사인자 | (전립선이 대표, 코호트 밖) | 코호트 내 근거 제한적 |
| MT1H | 메탈로티오네인(금속 결합) | (특이 암종 미지정) | 이소형 특이성 주의 |
| RGN | Regucalcin(칼슘 결합/항산화) | (간/전립선이 대표, 코호트 밖) | 코호트 내 근거 제한적 |
| PRODH | Proline oxidase, p53 표적 | (맥락 의존) | pro/anti-tumor 양면성 |
| CYP2S1 | Cytochrome P450(이물질 대사) | (특이 암종 미지정) | 암 근거 상대적으로 약함 |

- 8개 모두 `status: curated`, `summary/cancer_relevance/pathway/therapeutic_relevance/sources`
  작성 완료. `therapeutic_relevance` 는 대부분 "승인 표적치료 없음 → 근거 제한적".
- 지어낸 PMID 는 넣지 않았습니다. sources 는 검증 가능한 DB URL 이며, primary-literature
  PMID 는 심화 큐레이션 시 추가 예정임을 각 문서에 명시했습니다.

## 2. 아직 draft 인 유전자 (10개)
ANKRD43, C1orf106, FAM150B, LIX1, LOC145837, NOVA1, PRR15L, RIC3, SCN4A, SYT1

- 모두 템플릿 상태(`status: draft`, 각 섹션 `TODO(evidence_needed)`).
- RAG 검색 시 "근거 제한적" 으로 반환되며, LLM 프롬프트도 이들에 대해 근거 없는
  주장을 하지 않도록 강제됩니다.

## 3. 발표용 추천 sample_id (2~3개)
3개 모델이 모두 정확히 맞추고, curated 유전자와 연결해 설명하기 좋은 샘플입니다.

| sample_id | 실제 암종 | 3모델 예측 | 추천 이유 |
|---|---|---|---|
| `TCGA-85-8070-01` | LUSC | 3모델 모두 LUSC | SFTPB/GPRC5A(폐, curated)와 스토리 연결 |
| `TCGA-69-7763-01` | LUAD | 3모델 모두 LUAD | 폐선암 vs 편평세포암 혼동쌍 스토리 |
| `TCGA-LL-A6FR-01` | BRCA | 3모델 모두 BRCA | LUM(BRCA, curated) 근거 연결 |

(대안: `TCGA-CW-6093-01`, KIRC — 대조군으로 확률 1.000, 뚜렷한 구분 사례.)

## 4. 남은 리스크
- **근거 커버리지**: primary 18개 중 8개만 curated. 나머지 10개와 secondary 유전자는
  근거 없음 → LLM 설명이 "근거 제한적" 위주가 될 수 있음(설계상 의도된 안전장치).
- **출처 깊이**: 현재 sources 는 DB 수준(GeneCards/NCBI/UniProt)만. 논문 PMID 보강 필요.
- **코호트-특이성**: HOXC6/RGN 등은 대표 근거가 코호트 밖 암종(전립선/간)이라 본
  6개 암종과의 직접 연관은 제한적. 문서에 명시했으나 발표 시 오해 주의.
- **검색 방식**: 아직 keyword(심볼) 매칭. 동의어/부분표현 검색은 의미 검색(FAISS/Chroma)
  도입 후 개선 필요.
- **LLM 미연결**: 실제 생성 품질은 API 연결 후에야 검증 가능. 현재는 프롬프트 초안까지만.
- **임상 오해**: 정확도가 높아 임상 도구로 오인될 수 있음 → 화면/발표에서 연구·교육용
  명시 유지 필수.
