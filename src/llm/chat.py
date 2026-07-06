"""Cancer-aware RAG chat and explanation helpers."""

from __future__ import annotations

import re
from collections import Counter

from src.llm.context_builder import build_llm_context
from src.rag.retrieve import available_genes, retrieve_gene_evidence

SAFETY_LINE = "이 결과는 TCGA 데이터 기반 연구/교육용 설명이며, 임상 진단으로 사용할 수 없습니다."

HYBRID_SYSTEM_PROMPT = f"""당신은 TCGA 유전자 발현 기반 암종 예측 결과를 설명하는 연구/교육용 도우미입니다.
규칙:
1. 제공된 예측 요약, 유전자 근거, GraphRAG 연결, 출처 안의 정보만 사용합니다.
2. 유전자를 암의 직접 원인이라고 단정하지 않습니다.
3. '모델이 암종 구분에 중요하게 본 유전자 신호', '문헌상 관련성이 보고된 후보'처럼 제한적으로 표현합니다.
4. 근거가 부족하면 '근거 제한적'이라고 말합니다.
5. 답변은 한국어로 간결하게 하고 마지막에 다음 문장을 포함합니다: {SAFETY_LINE}
"""

HIGH_PROB = 0.90
OK_PROB = 0.70

CONFIDENCE_NOTE = {
    "높음": "세 모델이 일관되게 같은 결론을 냈고 예측 확률도 높습니다.",
    "보통": "모델 다수가 동의하지만 확률이나 일치도는 추가 확인이 필요합니다.",
    "제한적": "모델 의견이 갈리거나 확률이 낮아 예측을 신중히 해석해야 합니다.",
}


def _ensure_safety(text: str) -> str:
    if "임상 진단" in text:
        return text
    return f"{text}\n\n{SAFETY_LINE}"


def summarize_prediction(predictions: dict) -> dict:
    labels = [out["predicted_label"] for out in predictions.values()]
    counts = Counter(labels)
    top_count = max(counts.values())
    tied = [lbl for lbl, c in counts.items() if c == top_count]
    if len(tied) == 1:
        final_label = tied[0]
    else:
        final_label = max(
            tied,
            key=lambda lbl: sum(o["probabilities"].get(lbl, 0.0) for o in predictions.values()),
        )
    n_models = len(predictions)
    n_agree = sum(1 for lbl in labels if lbl == final_label)
    avg_prob = sum(o["probabilities"].get(final_label, 0.0) for o in predictions.values()) / n_models
    if n_agree == n_models and avg_prob >= HIGH_PROB:
        confidence = "높음"
    elif n_agree >= 2 and avg_prob >= OK_PROB:
        confidence = "보통"
    else:
        confidence = "제한적"
    return {
        "final_label": final_label,
        "n_agree": n_agree,
        "n_models": n_models,
        "avg_prob": avg_prob,
        "confidence": confidence,
    }


def overall_evidence_status(cancer_genes: list[dict]) -> str:
    curated = sum(1 for g in cancer_genes if g.get("evidence_status") == "curated")
    if curated >= 3:
        return "충분"
    if curated >= 1:
        return "제한적"
    return "작성중"


def _curated_gene_names(cancer_genes: list[dict]) -> list[str]:
    return [g["gene"] for g in cancer_genes if g.get("evidence_status") == "curated"]


def generate_user_explanation(context: dict) -> str:
    s = context["summary"]
    cancer = context["cancer_name"]
    genes = context["cancer_genes"]
    parts = [
        f"이 샘플은 {s['n_models']}개 모델 중 {s['n_agree']}개가 **{cancer}**로 예측했습니다. "
        f"평균 예측 확률은 {s['avg_prob']:.1%}이고, 해석 신뢰도는 **{s['confidence']}**입니다. "
        f"{CONFIDENCE_NOTE[s['confidence']]}",
    ]
    if genes:
        names = ", ".join(g["gene"] for g in genes[:5])
        parts.append(
            f"모델이 {cancer} 구분에 중요하게 본 유전자 후보는 {names} 등입니다. "
            "이는 원인 유전자라는 뜻이 아니라, 암종 분류에 도움을 준 발현 신호 후보입니다."
        )
    curated = _curated_gene_names(genes)
    if curated:
        parts.append(
            f"이 중 {', '.join(curated)}는 근거 문서가 정리되어 있어 기능과 암 관련성을 함께 확인할 수 있습니다. "
            f"전체 근거 상태는 {context['evidence_status']}입니다."
        )
    else:
        parts.append("아직 정리된 근거 문서가 부족하여 생물학적 설명은 근거 제한적으로 해석해야 합니다.")
    parts.append(SAFETY_LINE)
    return "\n\n".join(parts)


def _known_genes(context: dict) -> set[str]:
    from src.rag.web_retrieve import GENE_ALIASES

    genes = {g["gene"].upper() for g in context.get("cancer_genes", [])}
    genes.update(g.upper() for g in available_genes())
    genes.update(GENE_ALIASES.keys())
    return genes


def _detect_genes(question: str, context: dict) -> list[str]:
    known = _known_genes(context)
    found: list[str] = []
    for token in re.split(r"[^A-Za-z0-9]+", question):
        up = token.upper()
        if up and up in known and up not in found:
            found.append(up)
    return found


def classify_question(question: str, context: dict) -> str:
    """Classify whether a question should use RAG or a lightweight guide answer."""
    q = (question or "").strip()
    q_lower = q.lower()
    compact = re.sub(r"[\s.!?~ㅋㅎㅠㅜ]+", "", q_lower)

    greeting_words = {
        "안녕",
        "안녕하세요",
        "하이",
        "hello",
        "hi",
        "hey",
        "반가워",
        "고마워",
        "감사",
        "대화돼",
        "대화되니",
        "말해줘",
    }
    if compact in greeting_words or (len(compact) <= 8 and any(w in compact for w in greeting_words)):
        return "greeting"

    if any(k in q for k in ("임상", "진단", "치료", "병원", "처방")):
        return "clinical"

    if _detect_genes(q, context):
        return "analysis"

    analysis_keywords = (
        "왜", "이유", "예측", "어떻게", "암종", "유전자",
        "gene", "중요", "feature", "근거", "충분", "부족",
        "한계", "원인", "신호", "cause", "signal", "관련",
        "graphrag", "graph", "연결",
    )
    if any(k in q_lower for k in analysis_keywords):
        return "analysis"

    return "general"


def _guide_answer(intent: str, context: dict) -> str:
    cancer = context["cancer_name"]
    genes = ", ".join(g["gene"] for g in context.get("cancer_genes", [])[:4]) or "주요 유전자"
    if intent == "clinical":
        return _ensure_safety(
            "이 화면은 연구/교육용 설명 시스템이라 임상 진단, 치료, 병원 선택 같은 결정을 안내할 수 없습니다. "
            "대신 현재 샘플의 모델 예측, 중요 유전자 후보, 근거 문서와 GraphRAG 연결 관계를 설명할 수 있습니다."
        )
    if intent == "greeting":
        return _ensure_safety(
            '안녕하세요. 이 화면에서는 선택된 샘플의 암종 예측 결과와 유전자 근거를 대화처럼 물어볼 수 있습니다.\n\n'
            f"지금 선택된 샘플은 {cancer} 예측 결과를 보고 있으며, 예를 들면 "
            f"'왜 {cancer}으로 예측됐어?', '{genes} 중 어떤 유전자가 중요해?', "
            "'GraphRAG 연결 관계를 보여줘'처럼 질문할 수 있습니다."
        )
    return _ensure_safety(
        "질문을 분석 질문으로 해석하기 어려웠습니다. "
        f"현재 샘플의 예측 암종은 {cancer}입니다. "
        "예측 이유, 중요 유전자, 특정 유전자 근거, 근거 충분성, GraphRAG 연결 관계를 물어볼 수 있습니다."
    )


def _gene_answer(genes: list[str], context: dict) -> str:
    cancer_code = context["cancer_code"]
    blocks = []
    for gene in genes:
        ev = retrieve_gene_evidence(gene, cancer_code, web_fallback=True)
        state = ev.get("evidence_state")
        if state == "local_curated":
            summ = ev["sections"].get("summary", "").strip()
            rel = ev["sections"].get("cancer_relevance", "").strip()
            blocks.append(f"**{gene}**: {summ}\n\n암종 관련 근거: {rel}")
        elif state == "external_review":
            web = ev["web"]
            lines = [f"**{gene}**: 로컬 근거가 부족해 외부 출처 검토가 필요합니다."]
            if len(web.get("aliases_searched", [])) > 1:
                lines.append(f"함께 검색한 별칭: {', '.join(web['aliases_searched'])}")
            if web.get("external_summary"):
                es = web["external_summary"]
                lines.append(f"외부 요약({es['source']}, 검증 필요): {es['text'][:200]}")
            for link in web.get("sources", [])[:6]:
                lines.append(f"- {link['source']}: {link['url']}")
            blocks.append("\n".join(lines))
        else:
            blocks.append(f"**{gene}**: 현재 근거가 제한적입니다.")
    blocks.append("참고: 유전자 후보는 직접 원인이 아니라 모델이 암종 구분에 중요하게 본 관련 신호입니다.")
    return "\n\n".join(blocks)


def answer_question(question: str, context: dict) -> str:
    q = (question or "").strip()
    s = context["summary"]
    cancer = context["cancer_name"]

    if any(k in q for k in ("임상", "진단", "치료", "병원")):
        return "이 시스템은 연구/교육 목적이며 임상 진단이나 치료 결정에 사용할 수 없습니다."

    genes = _detect_genes(q, context)
    if genes:
        return _gene_answer(genes, context)

    if any(k in q for k in ("원인", "신호", "cause", "signal", "관련")):
        return (
            "모델이 지목한 유전자는 암의 직접 원인이라고 단정할 수 없습니다. "
            "암종 구분에 중요하게 작용한 유전자 발현 신호 또는 문헌상 관련성이 보고된 후보로 해석해야 합니다."
        )

    if any(k in q for k in ("근거", "충분", "부족", "한계")):
        curated = _curated_gene_names(context["cancer_genes"])
        curated_txt = ", ".join(curated) if curated else "없음"
        return (
            f"{cancer} 관련 후보 {len(context['cancer_genes'])}개 중 근거 문서가 정리된 유전자는 {curated_txt}입니다. "
            f"전체 근거 상태는 {context['evidence_status']}이며, 문서가 없거나 draft인 유전자는 근거 제한적으로 표시합니다."
        )

    if any(k in q for k in ("유전자", "gene", "중요", "feature")):
        names = ", ".join(g["gene"] for g in context["cancer_genes"][:8])
        return f"모델이 {cancer} 구분에 중요하게 본 유전자 후보는 {names} 등입니다."

    if any(k in q for k in ("왜", "이유", "예측", "어떻게")):
        return (
            f"{s['n_models']}개 모델 중 {s['n_agree']}개가 {cancer}로 예측했고, "
            f"평균 예측 확률은 {s['avg_prob']:.1%}, 해석 신뢰도는 {s['confidence']}입니다. "
            f"{CONFIDENCE_NOTE.get(s['confidence'], '')}"
        )

    return (
        f"현재 샘플은 {s['n_models']}개 모델 중 {s['n_agree']}개가 {cancer}로 예측했습니다 "
        f"(평균 확률 {s['avg_prob']:.1%}, 신뢰도 {s['confidence']}). "
        "예측 이유, 중요 유전자, 근거 충분성, 특정 유전자 근거를 질문할 수 있습니다."
    )


def hybrid_answer(
    question: str,
    summary: dict,
    cancer_code: str,
    cancer_name: str,
    cancer_genes: list[dict],
) -> dict:
    from src.llm import providers

    fallback_context = {
        "summary": summary,
        "cancer_code": cancer_code,
        "cancer_name": cancer_name,
        "cancer_genes": cancer_genes,
        "evidence_status": overall_evidence_status(cancer_genes),
    }
    intent = classify_question(question, fallback_context)

    if intent in {"greeting", "general", "clinical"}:
        return {
            "answer": _guide_answer(intent, fallback_context),
            "provider": None,
            "provider_label": "질문 분류 안내",
            "fallback_reason": None,
            "attempts": [],
            "sources": [],
            "evidence_mode": "none",
            "used_llm": False,
            "detected_genes": [],
            "graph_available": False,
            "graph_edges": [],
            "intent": intent,
        }

    ctx = build_llm_context(question, summary, cancer_code, cancer_name, cancer_genes)

    full_prompt = f"[질문]\n{question}\n\n{ctx['context_text']}\n\n제공된 근거만 사용해 한국어로 답하세요."
    compact_prompt = f"[질문]\n{question}\n\n{ctx['compact_text']}\n\n근거만 사용해 한국어로 3~5문장 답하세요."
    result = providers.generate_answer(HYBRID_SYSTEM_PROMPT, full_prompt, ollama_prompt=compact_prompt)

    provider = result["provider"]
    if result["text"]:
        answer = _ensure_safety(result["text"])
        fallback_reason = None
    else:
        answer = _ensure_safety(answer_question(question, fallback_context))
        provider = None
        fallback_reason = result["error"]

    return {
        "answer": answer,
        "provider": provider,
        "provider_label": providers.provider_label(provider),
        "fallback_reason": fallback_reason,
        "attempts": result["attempts"],
        "sources": ctx["sources"],
        "evidence_mode": ctx["evidence_mode"],
        "used_llm": provider is not None,
        "detected_genes": ctx["detected_genes"],
        "graph_available": ctx.get("graph_available", False),
        "graph_edges": ctx.get("graph_edges", []),
        "intent": intent,
    }
