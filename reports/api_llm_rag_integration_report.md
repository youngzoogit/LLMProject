# API LLM + 하이브리드 RAG 통합 보고서

_기본 답변 생성을 외부 API LLM(OpenAI)으로 전환하고, Ollama Gemma는 fallback/실험용,
규칙·문서 기반은 최종 fallback으로 유지. 어느 단계가 실패해도 앱은 죽지 않는다._

## 1. Provider 체인 (`src/llm/providers.py`)
우선순위: **1) API LLM → 2) Ollama Gemma → 3) 규칙/문서 기반 fallback**
- `LLM_PROVIDER=api`(기본) → API 시도 후 실패 시 Ollama, 그다음 규칙 기반.
- `LLM_PROVIDER=ollama` → Ollama만 시도 후 규칙 기반.
- `LLM_PROVIDER=fallback` → LLM 미사용, 규칙/문서 기반만.
- 환경변수: `OPENAI_API_KEY`, `API_MODEL`(기본 `gpt-4o-mini`), `API_TIMEOUT`(30s),
  `API_RETRIES`(2), `OLLAMA_MODEL`(기본 `gemma2:2b`), `OLLAMA_TIMEOUT`.
- `generate_answer(system, user, ollama_prompt)` 가 각 provider를 순서대로 시도하고
  `{text, provider, error, attempts}` 반환. 실패는 예외를 던지지 않고 `attempts`에
  사유를 기록.

## 2. API LLM 연결 (OpenAI)
- `openai` SDK(2.x) 직접 사용, `client.chat.completions.create` (timeout/retry 적용).
- 키 없음/패키지 없음 → `api_available()`=False → 조용히 다음 provider로.
- 호출 실패(401, timeout 등)는 `error_type: message` 로 로깅 + `attempts` 기록 후 폴백.
- **개인정보 미포함**: 프롬프트에는 데모 sample_id/익명 환자명/모델 예측 요약/유전자
  근거/공개 출처 URL만 넣음(원본 개인정보 없음).

## 3. Gemma 제한 (fallback/실험용 유지)
- 삭제하지 않음. `context_builder`가 **compact_text(≤1000자)** 를 별도 생성 →
  Gemma에는 이 짧은 프롬프트만 전달(유전자 근거 ≤3, 출처 ≤3).
- Gemma 실패 시 UI에 "**로컬 LLM 응답 실패, 문서 기반 답변으로 전환**" 표시.
- 크래시 원인·제한은 `reports/ollama_runtime_diagnosis.md`에 문서화.

## 4. RAG 컨텍스트 (`src/llm/context_builder.py`)
- 질문에서 gene(별칭 포함), 예측 암종, evidence intent를 추출.
- 현재 선택 sample의 예측 요약을 컨텍스트에 포함.
- 로컬 `rag_corpus` 우선 검색(BM25) → 부족한 gene은 외부 출처 링크/NCBI 요약 부착.
- LLM 컨텍스트 = **모델 예측 요약 / 유전자 근거 / 출처 / 근거 한계**로 정리.
- 어떤 경로(API/Gemma/fallback)로 답했는지 UI에 표시(숨기지 않음).

## 5. 안전 문구 / 표현 원칙
- 모든 답변 말미에 반드시: "이 결과는 TCGA 데이터 기반 연구/교육용 설명이며, 임상
  진단으로 사용할 수 없습니다." (`_ensure_safety`가 누락 시 자동 추가).
- 유전자를 암의 직접 원인으로 단정하지 않음 → "모델이 암종 구분에 중요하게 본 유전자
  신호", "문헌상 관련성이 보고된 후보" 로 제한적 표현(system prompt + 규칙 기반 공통).

## 6. Streamlit UI 재구성 (`app.py`, "💬 AI에게 질문하기" 탭)
- **좌우 배치**: 왼쪽=현재 샘플 예측 요약/주요 유전자, 오른쪽=RAG 대화.
- **고정 높이 스크롤**: `st.container(height=340)` 안에서 최근 대화만 스크롤.
- **대화 기록 관리**: 최근 N턴만 기본 표시, "🗑️ 초기화" 버튼, "전체 대화 보기" expander.
- **추천 질문 버튼** 6개(클릭 즉시 답변): 왜 예측/중요 유전자/원인 vs 신호/근거 충분/
  gene 근거/외부 자료.
- **답변 표시**: 본문 먼저 → 하단에 "답변 생성: {API LLM|Ollama Gemma|문서 기반 fallback}"
  + 근거 유형, 출처는 expander로 접음, 에러는 긴 traceback 대신 사용자용 요약.
- **API 키 안내**: 키 미설정 시 "OPENAI_API_KEY 설정 시 API LLM 사용" 안내 노출.

## 7. 검증 결과
| 시나리오 | 결과 |
|---|---|
| API 키 없음 | provider="문서 기반 fallback", 안전문구 포함 정상 답변, 앱 정상 (AppTest 예외 0) |
| API 키 있음(가짜 키) | provider 체인이 API 시도 → 401 AuthenticationError를 **정상 포착**하여 attempts 기록 후 폴백 → OpenAI SDK 배선 확인(유효 키면 provider="api"로 생성) |
| Ollama 실패(미연결) | `gemma_available()`=False → attempts에 기록, 앱 안 죽고 폴백 |
| compact_text | 742자(≤1000) 확인 |
| AppTest(재구성 UI) | 예외 0, 좌우 배치·고정높이 컨테이너·추천버튼(6)+초기화(1)·provider 표시·API키 안내 렌더 확인 |

### 질문 예시 테스트(폴백 경로, 규칙 기반)
- "왜 이 샘플은 갑상선암으로 예측됐어?" → 3/3 동의·99.6%·신뢰도 높음 + 안전문구.
- "TG 유전자는 어떤 근거가 있어?" → 로컬 curated 근거(티로글로불린) 요약.
- "LASS3는 근거가 부족하다는데 외부 자료도 찾아줘" → 로컬 없음 → 웹 fallback(별칭
  CERS3, NCBI/GeneCards 등 출처 URL, 검토 필요).
- "이 유전자가 원인이야, 아니면 관련 신호야?" → "구분에 유용한 관련 신호이며 원인
  증명 아님".
- "근거가 충분해?" → curated 개수 + 근거 상태(충분/제한적).
> 위 답변들은 유효한 `OPENAI_API_KEY` 설정 시 동일 컨텍스트로 API LLM이 더 자연스러운
> 한국어 문장으로 생성합니다(규칙 기반은 키·Ollama 모두 실패 시의 안전망).

## 8. 실행
```
# API LLM 사용(권장)
set OPENAI_API_KEY=sk-...        # PowerShell: $env:OPENAI_API_KEY="sk-..."
streamlit run app.py

# 로컬 Gemma 실험
set LLM_PROVIDER=ollama & set OLLAMA_MODEL=gemma2:2b & streamlit run app.py
```
