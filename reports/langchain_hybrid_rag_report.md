# LangChain 하이브리드 RAG Chat 보고서

_목표: 사용자가 자유롭게 질문하면 (1) 모델 예측 확인 → (2) 로컬 RAG 검색 →
(3) 부족하면 웹 검색 → (4) 로컬+웹 근거+출처를 합쳐 → (5) Ollama Gemma2가 한국어로
답변. Ollama/패키지/네트워크가 없어도 앱이 죽지 않고 규칙 기반으로 폴백._

## 1. 구조 (하이브리드 흐름)
```
질문 → context_builder.build_llm_context()
        · 예측 요약(모델 동의/확률/신뢰도)
        · 모델이 중요하게 본 유전자 후보
        · 질문 속 유전자 감지 → 로컬 RAG(curated) / 없으면 웹 fallback
        · BM25로 추가 관련 로컬 문서 검색
        → context_text + sources(각 source_url) + evidence_mode(local/web/mixed/none)
     → gemma_available() ?
          예: ChatOllama(gemma2, temperature=0) 로 한국어 답변
          아니오/실패/타임아웃: 규칙 기반 answer_question 폴백
     → app.py: 답변 + 근거유형(로컬/웹) + 생성엔진 + 사용한 출처 URL 표시
```

## 2. 신규/수정 파일
| 파일 | 역할 |
|---|---|
| `src/rag/langchain_retriever.py` (신규) | `data/rag_corpus/*.md` 위 BM25 retriever(langchain-community), 미설치 시 키워드 폴백. gene/암종코드/한국어명 검색 |
| `src/rag/web_retrieve.py` (확장) | 허용 도메인 6종 + Tavily(키 있을 때) + NCBI 공개 API + 신뢰 링크, `source_url` 필수, 별칭(LASS3→CERS3) |
| `src/llm/langchain_gemma.py` (신규) | `ChatOllama(model="gemma2", temperature=0, base_url=http://localhost:11434)`, `gemma_available()`, 타임아웃 폴백 |
| `src/llm/context_builder.py` (신규) | 예측+로컬+웹 근거를 합쳐 context/sources/evidence_mode 생성 |
| `src/llm/chat.py` (수정) | `hybrid_answer()` 추가(LLM 또는 규칙 기반), 기존 규칙 기반 유지 |
| `app.py` (수정) | "💬 AI에게 질문하기" 탭, `st.chat_input`, 대화기록, 출처 URL + 로컬/웹 근거 표시 |

## 3. 답변 규칙 (system prompt + 규칙기반 공통)
- 제공된 컨텍스트(예측/근거/출처)에 있는 내용만 사용, 출처 없는 생물학 주장 금지.
- 유전자를 "암의 원인 유전자" 라고 단정하지 않고 "모델이 암종 구분에 중요하게 본
  유전자 신호" 로 표현.
- 근거가 부족하면 "근거 제한적".
- 웹 근거는 "검토 필요" 임시 근거로 명시, 자동 curated 승격 금지.
- 한국어, 마지막에 "연구/교육 목적이며 임상 진단이 아닙니다." 명시.

## 4. Graceful degradation (앱이 죽지 않음)
| 상황 | 동작 |
|---|---|
| Ollama+langchain 정상 | Gemma2가 컨텍스트 근거로 한국어 답변 |
| Ollama 꺼짐 / langchain 미설치 | `gemma_available()`=False → 규칙 기반 폴백 |
| Gemma2 응답이 느림 | `OLLAMA_TIMEOUT`(기본 120s) 초과 시 폴백(UI 멈춤 방지) |
| Tavily 키 없음 | NCBI 공개 API / 신뢰 링크로 웹 fallback |
| 네트워크 없음 | 로컬 근거 + 신뢰 링크(주소만) 제공 |

## 5. 검증 결과
1. **Gemma2 연결**: `gemma_available()`=True 확인(Ollama `gemma2:latest`). 단, 본
   실행 환경의 gemma2(9B)는 CPU라 매우 느려(40토큰도 3분 내 미완료) 타임아웃 폴백이
   동작함 → GPU/충분한 자원 환경에서 `OLLAMA_TIMEOUT` 상향 시 LLM 답변 활성화.
2. **Ollama 꺼짐**: `OLLAMA_HOST` 무효로 AppTest 실행 → 예외 0, "규칙 기반 폴백"
   배너 + 답변 정상(앱 안 죽음).
3. **"LASS3는 뭐야?"**: 로컬 문서 없음 → 웹 fallback 수행. 별칭 `CERS3` 함께 검색,
   NCBI/HPA/GeneCards/PubMed/CIViC/OncoKB 출처 URL 표시(evidence_mode=web/mixed).
4. **"이 환자는 왜 갑상선암으로 예측됐어?"**: 예측 요약(3/3 동의, 99.6%, 신뢰도
   높음) + 관련 유전자 + RAG 근거를 합친 컨텍스트로 답변("환자" 단어가 임상 질문으로
   오분류되던 문제 수정).
5. BM25 백엔드 활성(`retriever_backend()`="bm25"), 로컬 검색 정상.
- AppTest: 탭 `['🔬 예측 & 유전자 근거', '💬 AI에게 질문하기']`, 예외 0, 답변 하단에
  근거유형·생성엔진·사용한 출처 URL 표시 확인.

## 6. 설정 / 실행
- 실행: `streamlit run app.py` → "AI에게 질문하기" 탭.
- 환경변수: `OLLAMA_HOST`(기본 http://localhost:11434), `OLLAMA_MODEL`(기본 gemma2:2b),
  `OLLAMA_TIMEOUT`(기본 120초), `TAVILY_API_KEY`(있으면 Tavily 웹검색),
  `TCGA_WEB_RAG_LIVE=1`(기본 켜짐, NCBI 실시간 요약).

## 6-1. 추가 진단: "짧은 대화는 되는데 왜 RAG 답변은 느려/실패하나" (중요)
사용자가 gemma2:2b로 직접 교체 후 재확인한 결과, **모델 자체는 정상**입니다.
- 단순 대화(`"대화 돼?"`)는 warm 상태에서 **4.2초**에 정상 응답.
- 그러나 예측요약+유전자근거를 담은 RAG 프롬프트는 **압축(200자/블록, 출처 6개,
  num_predict 256) 후에도 계속 실패**했습니다.
- 이진 탐색으로 원인을 확정: **SystemMessage 유무와 무관하게, 프롬프트가 약
  500자를 넘으면 이 환경의 Ollama 서버 자체가 `wsarecv`(TCP 연결 강제 종료)로
  죽습니다.** 단일 HumanMessage로 합쳐도 동일하게 실패 — 즉 LangChain 메시지
  포맷/프롬프트 엔지니어링 문제가 아니라 **로컬 Ollama 설치/드라이버 쪽 크래시**
  입니다. 코드에서 프롬프트를 더 줄이는 방향으로는 근본 해결이 안 됩니다(RAG
  컨텍스트를 500자 밑으로 줄이면 근거 대부분을 버려야 함).
- 그래서 앱이 계속 "규칙 기반 폴백"으로 떨어졌고, 그 폴백 문구가 맥락 없는 고정
  안내문("현재는 규칙 기반 안내만...")이라 사용자가 보기에 더 어색했습니다.

### 이번에 코드로 고친 것 (환경 문제와 별개로 실제 버그였던 부분)
1. **자유 질문도 맥락 기반으로 답하도록 수정** (`src/llm/chat.py`): 키워드 매칭에
   안 걸리는 질문(`"대화 돼?"` 등)도 이제 실제 예측 요약(모델 동의/확률/신뢰도)을
   먼저 말한 뒤 질문 예시를 안내합니다. 정적 안내문만 반환하던 이전 동작을 수정.
2. **LLM 실패 사유 노출** (`src/llm/langchain_gemma.generate_verbose`,
   `hybrid_answer`의 `llm_error` 키, `app.py` 캡션): gemma 시도가 실패하면 화면에
   "⚠️ LLM 시도 실패 사유: ResponseError: ..." 처럼 실제 예외를 보여줘, "왜
   폴백됐는지"가 더 이상 블랙박스가 아님.

### 사용자가 로컬에서 직접 확인/조치할 것 (코드로 고칠 수 없는 부분)
- `ollama serve` 로그 확인 (Windows 서비스로 실행 중이면 이벤트 뷰어/로그 파일).
- Ollama 버전 업데이트 (`ollama --version` 후 최신으로), 또는 재설치.
- GPU 드라이버 문제 가능성 있으면 CPU-only로 강제 실행해 재현되는지 확인.
- 다른 모델(`llama3.2:1b` 등)로도 500자 이상 프롬프트가 크래시되는지 교차 확인 —
  gemma2:2b 특정 문제인지 Ollama 전체 문제인지 구분 가능.
- 위 문제가 해결되면 코드 변경 없이 바로 실제 Gemma RAG 답변이 나옵니다
  (`hybrid_answer`는 이미 `generate_verbose`로 성공 시 그대로 사용).

## 7. 남은 작업 / 주의
- 로컬 gemma2 속도: CPU 환경에서는 응답이 느려 폴백될 수 있음. GPU 또는 더 작은
  모델(gemma2:2b 등) 사용 또는 `OLLAMA_TIMEOUT` 조정 권장.
- 웹 근거는 여전히 "검토 필요" 상태이며 사람이 확인 후 curated 로 승격하는 절차 필요.
- BM25는 심볼/영문 위주 검색에 강함. 한국어 자연어 의미검색은 임베딩(Chroma) 도입 시
  개선 가능.
- 대화 맥락(이전 질문) 기반 후속 답변은 향후 확장.
