# Neo4j GraphRAG 적용 보고서

## 적용 요약

현재 프로젝트에 lightweight Neo4j GraphRAG를 연결했다.

- `.env`의 `NEO4J_URI`를 `neo4j+ssc://a2d21947.databases.neo4j.io`로 변경했다.
- `neo4j+s://`는 이 PC/Python 환경에서 TLS 인증서 검증 실패가 발생했다.
- `neo4j+ssc://`에서는 `graph_ping()`과 `RETURN 1` 쿼리가 성공했다.
- `src/rag/graph_store.py`의 `build_graph()`로 유전자-암종-모델-근거문서 그래프를 Neo4j에 적재했다.
- `src/rag/graph_retrieve.py`로 gene/cancer 이웃 관계를 조회한다.
- `src/llm/context_builder.py`에서 GraphRAG edge를 RAG context에 추가한다.
- `src/llm/chat.py`와 `app.py`에서 GraphRAG 상태와 edge 수를 응답 payload/UI에 전달한다.

## 적재 결과

실행 명령:

```powershell
.\.venv\Scripts\python.exe -B scripts\build_gene_graph_neo4j.py
```

결과:

```text
Neo4j ping: OK
Build result: {'ok': True, 'counts': {'genes': 21, 'docs': 21, 'sources': 33, 'assoc': 19}}
```

## 조회 예시

`graph_context(['TG', 'TPO', 'TSHR'], 'THCA')` 결과에서 다음 관계가 확인되었다.

```text
TG -[DOCUMENTED_IN]-> TG
TG -[ASSOCIATED_WITH]-> THCA
TG -[IMPORTANT_FOR_MODEL]-> logistic
TG -[RELATED_TO]-> 갑상선호르몬 생합성
TG -[CO_RELATED_WITH]-> TPO
```

## RAG 답변 연결 검증

`build_llm_context()`와 `hybrid_answer()`에서 GraphRAG 정보가 포함되는지 확인했다.

```text
graph_available=True
graph_edges=13
has_graph_block=True
```

LLM provider를 `fallback`으로 둔 상태에서도 payload에 GraphRAG 관계가 포함된다.

## 주의 사항

- `neo4j+ssc://`는 self-signed certificate chain을 허용하는 방식이라 `neo4j+s://`보다 인증서 검증이 느슨하다.
- 발표/개발 환경에서는 사용할 수 있지만, 운영 환경에서는 인증서 체인 문제를 해결하고 `neo4j+s://` 사용을 우선 검토한다.
- GraphRAG 관계는 임상적 인과관계가 아니라 모델 해석과 문헌 근거 탐색을 보조하는 연결 정보다.
- 유전자는 “암의 직접 원인”이 아니라 “모델이 암종 구분에 중요하게 본 유전자 신호/근거 후보”로 표현해야 한다.