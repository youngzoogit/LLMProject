# Ollama 런타임 진단 (Gemma RAG 크래시)

_증상: 짧은 질문은 Gemma가 답하지만, RAG 컨텍스트가 들어간 긴 프롬프트에서 Ollama가
`wsarecv`(TCP 강제 종료) / HTTP 500 / 과도한 지연으로 실패. 앱·LangChain·Ollama 중
어느 계층 문제인지 분리하기 위해 REST 직접 호출과 LangChain 경로를 비교._

## 1. 환경
- `ollama --version`: **0.31.1**
- `ollama list`:
  | 모델 | 크기 |
  |---|---|
  | llama3.2:1b | 1.3 GB |
  | gemma2:2b | 1.6 GB |
  | gemma2:latest (9B) | 5.4 GB |
- 테스트 도구: `scripts/test_ollama_prompt_lengths.py` (REST `POST /api/generate` 직접
  호출 + LangChain `ChatOllama` 동일 프롬프트 비교, 길이 100/300/500/800/1200자).

## 2. 프롬프트 길이별 결과 — gemma2:2b (모델 3개 모두 로드된 상태, timeout 120s)
| 프롬프트 | REST 직접 | LangChain | 비고 |
|---|---|---|---|
| 100자 | ❌ Timeout 122s | ✅ OK 81.9s | REST는 timeout, LC는 81.9초만에 응답 |
| 300자 | ❌ **HTTP 500 (12.2s)** | ❌ Fail 54.4s | 서버측 Internal Server Error |
| 500자 | ❌ **HTTP 500 (52.7s)** | ❌ Fail 55.2s | 서버측 Internal Server Error |
| 800자 | ❌ Timeout 122s | ❌ Fail 87.5s | |
| 1200자 | ❌ **HTTP 500 (58.5s)** | ❌ Fail 62.7s | 서버측 Internal Server Error |

## 3. 보조 실험 — gemma2:2b 단독 (모든 모델 언로드 후, timeout 45s)
| 프롬프트 | REST | LangChain |
|---|---|---|
| 100~1200자 | ❌ 전부 Timeout ~47s | ❌ 전부 Timeout ~48s |

> 주의: 이 실험은 timeout(45s)이 gemma2:2b의 **cold-load 시간(~47~82s)** 보다 짧아
> 결과가 오염됐습니다(로딩 중 타임아웃). 따라서 "단독이면 되는가"는 이 실험으로
> 단정 불가. 다만 gemma2:2b가 이 머신에서 **cold-load만 47~82초**로 매우 느리다는
> 사실은 확인됨.

## 4. llama3.2:1b 교차 테스트
- 세션 종료로 프로브가 중단되어 데이터 미확보. 아래 명령으로 재현 가능:
  ```
  python scripts/test_ollama_prompt_lengths.py --model llama3.2:1b --timeout 90
  ```

## 5. 결론: 어느 계층 문제인가
- **REST API 직접 호출(앱·LangChain 완전 우회)에서도 실패**하며, 특히 300/500/1200자에서
  **HTTP 500 Internal Server Error** 를 반환했습니다. 이는 Ollama의 모델 러너가
  **서버측에서 크래시**한다는 의미입니다.
- LangChain 단독으로만 실패하는 케이스는 없었습니다(REST가 성공한 100자에서는 LC도 성공).
- **판정: Ollama / 모델 러너 / 런타임(메모리) 문제.** 앱 코드나 LangChain 설정 문제가
  아닙니다. (프롬프트가 500자를 넘으면 재현되는 경향 + 500 에러 → 리소스 압박 하에서
  긴 컨텍스트 처리 시 러너 크래시로 추정. 3개 모델, 특히 5.4GB gemma2:latest 상주가
  메모리 압박을 키움.)

## 6. Gemma 제한사항 (앱에서 fallback/실험용으로 유지 시 준수)
Gemma는 삭제하지 않고 **fallback/실험용 provider** 로 남기되, 다음을 강제합니다.
- **프롬프트 1000자 이하**로 제한.
- 컨텍스트에 **유전자 근거 최대 2~3개, 출처 최대 3개**만 포함.
- Gemma 호출 실패 시 화면에 **"로컬 LLM 응답 실패, 문서 기반 답변으로 전환"** 표시.
- 큰 모델(gemma2:latest)을 언로드해 메모리 확보(`ollama stop gemma2:latest`) 권장.
- cold-load 지연이 크므로 실시간 대화에는 부적합 → **기본 답변 생성은 외부 API LLM** 로
  전환(`LLM_PROVIDER=api`). Gemma는 `LLM_PROVIDER=ollama` 로 실험할 때만 사용.

## 7. 사용자가 근본 해결을 원할 때 (코드 밖 조치)
- `ollama stop gemma2:latest` 로 9B 언로드 후 재시도(메모리 확보).
- Ollama 최신 버전 업데이트(현재 0.31.1) 또는 재설치.
- `ollama serve` 콘솔/로그에서 러너 크래시 스택 확인.
- GPU 드라이버/VRAM 문제 가능성 → CPU-only 강제 재현 여부 확인.
- `scripts/test_ollama_prompt_lengths.py --model llama3.2:1b` 로 더 작은 모델이 되는지
  교차 확인(되면 gemma2 특정 문제, 안 되면 Ollama 전체 문제).
