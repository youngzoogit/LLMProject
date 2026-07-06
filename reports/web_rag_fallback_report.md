# Web RAG Fallback 보고서

_목표: 로컬 RAG 문서가 없거나 draft 상태일 때, 신뢰 가능한 외부 출처를 "임시 외부
근거(검토 필요)" 로 제시한다. 생물학적 사실을 지어내지 않고, 출처 URL을 반드시
붙이며, 자동으로 curated 로 확정하지 않는다._

## 1. 구현 개요
- **신규 `src/rag/web_retrieve.py`**
  - `TRUSTED_SOURCES`: 허용 출처 6종과 유전자 질의 URL 패턴
    (NCBI Gene / Human Protein Atlas / GeneCards / PubMed / CIViC / OncoKB).
  - `GENE_ALIASES`: 별칭 매핑(예: `LASS3 -> [CERS3]`, `FAM150B -> [ALKAL2]`,
    `C1ORF106 -> [INAVA]`). 확장 가능.
  - `resolve_aliases(gene)`: 유전자 + 별칭 목록 반환.
  - `web_search_gene(gene)`: 유전자(+별칭)에 대한 신뢰 출처 링크 목록을 만들어
    "임시 외부 근거(검토 필요)" 로 반환. `status = external_found | none`.
  - `fetch_ncbi_summary(gene)`: **실데이터** NCBI Gene 요약을 best-effort 로 조회
    (E-utilities). 기본 비활성이며 `TCGA_WEB_RAG_LIVE=1` 일 때만 네트워크 호출.
    실패해도 예외 없이 링크만 반환(오프라인/테스트 안전).
- **`src/rag/retrieve.py`**: `retrieve_gene_evidence(gene, cancer_type, web_fallback=False)`
  - 로컬 문서가 curated 이면 그대로 사용.
  - curated 가 아니면(문서 없음 또는 draft) `web_fallback=True` 일 때 외부 검색을
    호출하고 결과를 `web` 키로 첨부.
  - `evidence_state`: `local_curated` | `external_review` | `none` 로 상태를 명시.
- **`src/llm/chat.py`**: 유전자 질문 시 `web_fallback=True` 로 조회하여 세 상태를
  구분해 답변(로컬 근거 / 외부 출처(검토 필요) / 근거 제한적).
- **`app.py`**: 유전자 근거 패널을 세 상태로 표시하고, 외부 출처는 클릭 가능한
  링크와 "검토 필요" 문구로 로컬 curated 와 분명히 구분.

## 2. 상태 구분 (요구사항 8)
| 상태 | 조건 | 화면 표시 |
|---|---|---|
| 로컬 문서 있음 | curated 로컬 문서 존재 | ✅ 로컬 근거 문서 (섹션 본문) |
| 로컬 없음, 외부 있음 | 로컬 없음/draft + 외부 링크 확보 | 🌐 임시 외부 근거(검토 필요) + 출처 URL |
| 둘 다 없음 | 유효 심볼 아님/외부도 없음 | ❌ 근거 제한적 |

## 3. 안전장치 (요구사항 4·5·6·9)
- 외부 결과는 항상 `임시 외부 근거 (검토 필요)` 라벨과 "자동 확정된 근거가 아님,
  사람이 검토 필요" 문구를 포함.
- 모든 외부 항목에 **출처 이름 + URL** 을 부착.
- 생물학적 설명은 로컬 curated 이거나(로컬 문서), 실데이터 NCBI 요약이 있을 때만
  제시. 그 외에는 링크만 제공하며 지어내지 않음.
- 출처를 만들 수 없거나 심볼이 유효하지 않으면 설명 없이 "근거 제한적".
- 외부 검색 결과는 절대 자동으로 `status: curated` 로 승격하지 않음.

## 4. LASS3 예시 (검증됨)
`LASS3` 는 THCA per-class 후보에 있으나 로컬 문서가 없습니다. 별칭 `CERS3`
(ceramide synthase 3) 로도 함께 검색합니다.

- `resolve_aliases("LASS3")` -> `['LASS3', 'CERS3']`
- `retrieve_gene_evidence("LASS3", "THCA", web_fallback=True)`
  -> `found=False`, `evidence_state="external_review"`, `web.sources` 12개
     (LASS3·CERS3 각각의 6개 출처).
- RAG Chat 답변(발췌):
  > **LASS3**: 로컬 근거 문서는 없습니다. 아래는 신뢰 가능한 외부 출처(임시 외부
  > 근거 (검토 필요))입니다. 자동 확정된 근거가 아니라 검토가 필요합니다.
  > - 함께 검색한 별칭: LASS3, CERS3
  > - NCBI Gene: https://www.ncbi.nlm.nih.gov/gene/?term=LASS3%5Bsym%5D+AND+human%5Borgn%5D
  > - Human Protein Atlas: https://www.proteinatlas.org/search/LASS3
  > - GeneCards: https://www.genecards.org/cgi-bin/carddisp.pl?gene=LASS3
  > - PubMed: https://pubmed.ncbi.nlm.nih.gov/?term=LASS3+cancer ...

- 참고: `python -m src.rag.retrieve LASS3 THCA` 로도 동일 fallback 확인 가능.

## 5. 검증 결과
- `resolve_aliases`, `web_search_gene`(오프라인) 단위 확인 완료.
- `retrieve_gene_evidence` 상태: SFTPB=local_curated, ANKRD43(draft)=external_review,
  LASS3(로컬 없음)=external_review, 잘못된 심볼=none.
- Streamlit AppTest(THCA 샘플 + LASS3 선택): **예외 0개**, "임시 외부 근거", "검토",
  별칭 "CERS3", 출처 URL(ncbi/genecards) 표시 확인, TODO 등 개발자 문구 비노출.

## 6. 남은 작업 / 주의
- 라이브 NCBI 요약은 `TCGA_WEB_RAG_LIVE=1` 에서만 동작(외부망 필요). 기본은 링크만.
- CIViC/OncoKB 딥링크 패턴은 best-effort 이며, 심볼에 따라 검색 페이지로 연결될 수
  있음(검토 필요 대상이므로 허용).
- 별칭 맵은 확신 있는 항목만 수록. 추가 별칭은 HGNC 확인 후 `GENE_ALIASES` 에 보강.
- 외부 요약을 curated 로 승격하려면 사람이 검토 후 로컬 문서로 저장하는 절차 권장.
