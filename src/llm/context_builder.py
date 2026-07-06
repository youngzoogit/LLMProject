"""Build grounded context for hybrid RAG answers.

Context combines prediction summary, local gene evidence, optional web fallback,
and optional Neo4j GraphRAG relationships.
"""

from __future__ import annotations

import re

from src.display import CANCER_KO
from src.rag.langchain_retriever import search_local
from src.rag.retrieve import available_genes, retrieve_gene_evidence

_URL_RE = re.compile(r"https?://[^\s)]+")
_MAX_TARGET_GENES = 4


def _detect_genes(question: str, cancer_genes: list[dict]) -> list[str]:
    from src.rag.web_retrieve import GENE_ALIASES

    known = {g["gene"].upper() for g in cancer_genes}
    known.update(g.upper() for g in available_genes())
    known.update(GENE_ALIASES.keys())
    found: list[str] = []
    for token in re.split(r"[^A-Za-z0-9]+", question):
        up = token.upper()
        if up and up in known and up not in found:
            found.append(up)
    return found


def _local_block(gene: str, ev: dict, sources: list[dict]) -> str:
    summary = ev["sections"].get("summary", "").strip() or "요약 없음"
    relevance = ev["sections"].get("cancer_relevance", "").strip() or "암종 관련 근거가 아직 부족합니다."
    for url in _URL_RE.findall(ev["sections"].get("sources", "")):
        sources.append({"type": "local", "label": f"{gene} 출처", "source_url": url})
    if ev.get("path"):
        sources.append({"type": "local", "label": f"{gene} 로컬 문서", "source_url": ev["path"]})
    return f"[{gene} 로컬 근거]\n기능: {summary}\n암종 관련성: {relevance}"


def _web_block(gene: str, ev: dict, sources: list[dict]) -> str:
    web = ev["web"]
    for link in web["sources"][:4]:
        sources.append({"type": "web", "label": link.get("source", "web"), "source_url": link["url"]})
    snippet = ""
    if web.get("external_summary"):
        snippet = web["external_summary"].get("text", "")
    alias = ""
    if len(web.get("aliases_searched", [])) > 1:
        alias = f" (함께 검색한 별칭: {', '.join(web['aliases_searched'])})"
    body = snippet if snippet else "외부 출처 링크만 제공됩니다. 검증이 필요합니다."
    return f"[{gene} 외부 근거 검토 필요{alias}] {body}"


def build_llm_context(
    question: str,
    summary: dict,
    cancer_code: str,
    cancer_name: str,
    cancer_genes: list[dict],
) -> dict:
    detected = _detect_genes(question, cancer_genes)
    target_genes = (detected or [g["gene"] for g in cancer_genes[:3]])[:_MAX_TARGET_GENES]

    sources: list[dict] = []
    blocks: list[str] = []
    modes: set[str] = set()

    for gene in target_genes:
        ev = retrieve_gene_evidence(gene, cancer_code, web_fallback=True)
        state = ev.get("evidence_state")
        if state == "local_curated":
            modes.add("local")
            blocks.append(_local_block(gene, ev, sources))
        elif state == "external_review":
            modes.add("web")
            blocks.append(_web_block(gene, ev, sources))
        else:
            modes.add("none")
            blocks.append(f"[{gene}] 근거 제한적: 로컬/외부 근거가 충분하지 않습니다.")

    query = f"{question} {cancer_code} {CANCER_KO.get(cancer_code, '')}"
    for hit in search_local(query, k=3):
        if hit["has_curated"] and hit["gene"] not in target_genes:
            modes.add("local")
            summ = hit["sections"].get("summary", "").strip()
            blocks.append(f"[{hit['gene']} 관련 로컬 근거] {summ}")
            sources.append({"type": "local", "label": f"{hit['gene']} 로컬 문서", "source_url": hit["path"]})

    graph_edges: list[str] = []
    graph_available = False
    try:
        from src.rag.graph_retrieve import graph_context

        graph = graph_context(target_genes, cancer_code)
        graph_available = bool(graph.get("available"))
        graph_edges = list(graph.get("edges") or [])
    except Exception:
        graph_edges = []

    if graph_edges:
        blocks.append("[GraphRAG 연결 근거]\n" + "\n".join(f"- {edge}" for edge in graph_edges[:8]))

    seen: set[str] = set()
    unique_sources = []
    for src in sources:
        if src["source_url"] not in seen:
            seen.add(src["source_url"])
            unique_sources.append(src)

    if "local" in modes and "web" in modes:
        evidence_mode = "mixed"
    elif "local" in modes:
        evidence_mode = "local"
    elif "web" in modes:
        evidence_mode = "web"
    else:
        evidence_mode = "none"

    gene_names = ", ".join(g["gene"] for g in cancer_genes[:8]) or "-"
    pred_line = (
        f"예측 암종: {cancer_name} / 모델 동의: {summary['n_agree']}/{summary['n_models']} / "
        f"평균 확률: {summary['avg_prob']:.1%} / 해석 신뢰도: {summary['confidence']}"
    )
    src_lines = [f"- ({s['type']}) {s['label']}: {s['source_url']}" for s in unique_sources]
    graph_lines = [f"- {edge}" for edge in graph_edges[:8]]

    context_text = (
        f"[예측 요약]\n{pred_line}\n\n"
        f"[모델이 암종 구분에 중요하게 본 유전자 후보]\n{gene_names}\n\n"
        f"[유전자 근거]\n" + "\n\n".join(blocks) + "\n\n"
        f"[GraphRAG 연결]\n" + ("\n".join(graph_lines) or "- 없음") + "\n\n"
        f"[출처]\n" + ("\n".join(src_lines[:6]) or "- 없음")
    )

    compact = (
        f"[예측] {pred_line}\n"
        f"[근거]\n" + "\n".join(b[:160] for b in blocks[:3]) + "\n"
        f"[GraphRAG]\n" + ("\n".join(graph_lines[:4]) or "- 없음") + "\n"
        f"[출처]\n" + ("\n".join(src_lines[:3]) or "- 없음") + "\n"
        "[주의] 근거가 부족한 유전자는 근거 제한적으로 답해야 합니다."
    )

    return {
        "context_text": context_text,
        "compact_text": compact[:1000],
        "sources": unique_sources,
        "evidence_mode": evidence_mode,
        "detected_genes": detected,
        "target_genes": target_genes,
        "graph_available": graph_available,
        "graph_edges": graph_edges,
    }