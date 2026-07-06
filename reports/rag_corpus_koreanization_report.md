# RAG 근거 문서 한국어화 보고서

_목표: RAG 근거 문서와 Streamlit 근거 영역을 한국어 사용자용으로 변경. 내부 파싱을
위한 영어 섹션 헤더는 유지하되, 화면 표시는 한국어 라벨/본문으로 보여준다._

## 1. 섹션 라벨 한국어 매핑 (화면 표시용)
`app.py`의 `SECTION_KO` 를 다음과 같이 변경했습니다(파일의 `## summary` 등 영어
헤더는 파서를 위해 그대로 유지).

| 내부 섹션명(파일) | 화면 표시 라벨 |
|---|---|
| summary | 유전자 기능 요약 |
| cancer_relevance | 암종 관련 근거 |
| pathway | 관련 경로/생물학적 기능 |
| therapeutic_relevance | 치료 표적 관련성 |
| sources | 출처 |
| evidence_limitations | 근거 한계 |

## 2. 화면에서 영어 섹션명 비노출
- `app.py` 근거 영역은 `SECTION_KO` 라벨로만 렌더링합니다.
- 검증에서 화면 텍스트에 `## summary` / `## pathway` 등 영어 헤더가 나타나지
  않음을 확인했습니다.

## 3. 개발자용 문구(TODO) 비노출
- draft 유전자는 근거 섹션 본문을 표시하지 않고 다음 문구만 보여줍니다:
  - "📝 (유전자) — 근거 정리 중. 아직 근거가 정리되지 않았습니다. 현재 이 유전자에
    대한 생물학적 설명은 근거 제한적입니다."
- 화면 텍스트에 `TODO(evidence_needed)` 가 노출되지 않음을 검증했습니다.

## 4. curated 문서 본문 한국어화 (8개)
SFTPB, GPRC5A, LUM, HOXC6, MT1H, RGN, PRODH, CYP2S1 문서의 본문
(summary/cancer_relevance/pathway/therapeutic_relevance/evidence_limitations)을
한국어로 재작성했습니다.
- **출처(sources)의 이름과 URL(GeneCards / NCBI Gene / UniProt 링크)은 원문 그대로
  유지**했습니다(요구사항 4).
- 불확실하거나 코호트 밖 근거는 본문/근거 한계에 "근거 제한적" 으로 명시했습니다.
- frontmatter 와 영어 섹션 헤더(`## summary` 등)는 파싱 호환을 위해 유지했습니다.

## 5. 상태 표현 통일 (4가지 표현)
`app.py` 에서 다음 표현을 사용합니다.
| 상태 | 표시 |
|---|---|
| 문서 없음 | ❌ 근거 문서 없음 |
| draft | 📝 근거 정리 중 (+ "근거 제한적") |
| curated | ✅ 근거 정리 완료 |
| 근거 불충분 안내 | 근거 제한적 |

- 유전자 표의 "근거 상태" 컬럼도 `✅ 근거 정리 완료 / 📝 근거 정리 중 /
  ❌ 근거 문서 없음` 으로 표시합니다.
- RAG Chat/설명(`src/llm/chat.py`)의 근거 관련 답변도 "근거 제한적" 표현을 사용합니다.

## 6. 수정/신규 파일
- 수정: `data/rag_corpus/{SFTPB,GPRC5A,LUM,HOXC6,MT1H,RGN,PRODH,CYP2S1}.md`
  (본문 한국어화, 헤더/출처 URL 유지).
- 수정: `app.py` (`SECTION_KO`/`STATUS_KO` 한국어화, draft/근거 안내 문구 변경).

## 7. 검증 결과 (Streamlit AppTest)
- **예외 0개 (CLEAN)**.
- curated 유전자(SFTPB) 선택 시: 한국어 섹션 라벨(유전자 기능 요약/암종 관련 근거/
  관련 경로·생물학적 기능/치료 표적 관련성/출처/근거 한계) 및 한국어 본문 표시,
  상태 "근거 정리 완료" 표시 확인.
- draft 유전자(ANKRD43 등): "근거 정리 중 / 아직 근거가 정리되지 않았습니다 /
  근거 제한적" 표시, 근거 섹션 본문 미표시.
- **비노출 확인**: `TODO(evidence_needed)`, 영어 섹션 헤더(`## summary`,
  `## pathway`, `sources` 단독)가 사용자 화면에 나타나지 않음.
- 유전자 표 "근거 상태" 값: `['📝 근거 정리 중', '✅ 근거 정리 완료']`.

## 8. 남은 작업
- draft 10개 유전자도 curated 로 채우면 동일 방식으로 한국어 본문이 자동 표시됩니다.
- 출처에 개별 논문 PMID 보강(현재는 DB 집계원 URL 중심).
