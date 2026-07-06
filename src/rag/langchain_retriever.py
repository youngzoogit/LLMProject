"""Local RAG retriever over data/rag_corpus/*.md (BM25, with keyword fallback).

Prefers a LangChain ``BM25Retriever`` when ``langchain-community`` + ``rank_bm25``
are installed; if not, falls back to a simple keyword match over the parsed
corpus. Either way ``search_local`` returns plain dicts so callers never depend
on LangChain being present.

Genes can be found by symbol, cancer code (in metadata), or Korean cancer name
(the caller adds the cancer name to the query text).
"""

from __future__ import annotations

from functools import lru_cache

from src.rag.retrieve import load_corpus

# Sections concatenated into the searchable text of each document.
_TEXT_SECTIONS = ("summary", "cancer_relevance", "pathway", "therapeutic_relevance")


@lru_cache(maxsize=1)
def _corpus_records() -> tuple[dict, ...]:
    """Parsed corpus as searchable records (gene, text, metadata, sections)."""
    records = []
    for doc in load_corpus().values():
        sections = doc["sections"]
        cancer_types = doc["frontmatter"].get("associated_cancer_types", [])
        text = doc["gene"] + " " + " ".join(sections.get(s, "") for s in _TEXT_SECTIONS)
        records.append(
            {
                "gene": doc["gene"],
                "text": text,
                "cancer_types": cancer_types,
                "status": doc["frontmatter"].get("status", "draft"),
                "has_curated": doc["has_curated_evidence"],
                "path": doc["path"],
                "sections": sections,
            }
        )
    return tuple(records)


@lru_cache(maxsize=1)
def _bm25_retriever():
    """Build a BM25Retriever, or None if LangChain/rank_bm25 is unavailable."""
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document

        docs = [
            Document(page_content=r["text"], metadata={"gene": r["gene"], "idx": i})
            for i, r in enumerate(_corpus_records())
        ]
        if not docs:
            return None
        return BM25Retriever.from_documents(docs)
    except Exception:
        return None


def _keyword_search(query: str, k: int) -> list[dict]:
    """Fallback: match gene symbol or cancer code appearing in the query."""
    upper = query.upper()
    hits = []
    for record in _corpus_records():
        if record["gene"].upper() in upper or any(
            c.upper() in upper for c in record["cancer_types"]
        ):
            hits.append(record)
    return hits[:k]


def search_local(query: str, k: int = 4) -> list[dict]:
    """Return up to ``k`` corpus records relevant to the query (dicts)."""
    retriever = _bm25_retriever()
    if retriever is not None:
        try:
            retriever.k = k
            records = _corpus_records()
            hits = retriever.invoke(query)
            out = []
            for hit in hits:
                idx = hit.metadata.get("idx")
                if idx is not None and 0 <= idx < len(records):
                    out.append(records[idx])
            if out:
                return out[:k]
        except Exception:
            pass
    return _keyword_search(query, k)


def retriever_backend() -> str:
    """Which backend is active: 'bm25' or 'keyword'."""
    return "bm25" if _bm25_retriever() is not None else "keyword"
