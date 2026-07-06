"""Keyword-based retrieval over the RAG corpus (stage 3, MVP).

Deliberately simple: it parses the markdown files under ``data/rag_corpus/`` and
looks a gene up by symbol. FAISS/Chroma semantic search is deferred. The key
guarantee is honesty about coverage: if a gene has no document, or the document
is still an unfilled template, that is reported explicitly so the LLM layer can
say "evidence limited" instead of inventing facts.

CLI::

    python -m src.rag.retrieve SFTPB
    python -m src.rag.retrieve SFTPB LUSC
"""

from __future__ import annotations

import sys
from functools import lru_cache

from src.data_loader import PROJECT_ROOT
from src.rag.build_corpus import EVIDENCE_NEEDED, SECTIONS

CORPUS_DIR = PROJECT_ROOT / "data" / "rag_corpus"

# Sections that must hold real content for a document to count as curated.
CURATABLE_SECTIONS = (
    "summary",
    "cancer_relevance",
    "pathway",
    "therapeutic_relevance",
    "sources",
)

NO_EVIDENCE_MESSAGE = "근거 문서 없음 (no evidence document found)"
TEMPLATE_ONLY_MESSAGE = (
    "근거 제한적 (document exists but is an unfilled template; no curated evidence)"
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_frontmatter(block: str) -> dict:
    """Parse the simple ``key: value`` / ``key: [a, b]`` frontmatter block."""
    meta: dict = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.split("#", 1)[0].strip()  # drop inline comments
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            meta[key] = [v.strip() for v in inner.split(",") if v.strip()] if inner else []
        else:
            meta[key] = value
    return meta


def _parse_document(text: str) -> dict:
    """Split a corpus markdown file into frontmatter + section text."""
    frontmatter: dict = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            frontmatter = _parse_frontmatter(parts[1])
            body = parts[2]

    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current = line[3:].strip()
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()

    return {"frontmatter": frontmatter, "sections": sections}


def _has_curated_evidence(parsed: dict) -> bool:
    """True if any curatable section holds content beyond the placeholder."""
    if parsed["frontmatter"].get("status") == "curated":
        return True
    for name in CURATABLE_SECTIONS:
        content = parsed["sections"].get(name, "")
        if content and EVIDENCE_NEEDED not in content:
            return True
    return False


@lru_cache(maxsize=1)
def load_corpus() -> dict[str, dict]:
    """Parse every corpus file, keyed by upper-cased gene symbol."""
    corpus: dict[str, dict] = {}
    if not CORPUS_DIR.exists():
        return corpus
    for path in sorted(CORPUS_DIR.glob("*.md")):
        parsed = _parse_document(path.read_text(encoding="utf-8"))
        gene = parsed["frontmatter"].get("gene", path.stem)
        parsed["gene"] = gene
        parsed["path"] = str(path.relative_to(PROJECT_ROOT))
        parsed["has_curated_evidence"] = _has_curated_evidence(parsed)
        corpus[gene.upper()] = parsed
    return corpus


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def available_genes() -> list[str]:
    """Gene symbols that have a corpus document."""
    return sorted(doc["gene"] for doc in load_corpus().values())


def search_corpus(query: str) -> list[str]:
    """Keyword search: gene symbols containing ``query`` (case-insensitive)."""
    q = query.strip().upper()
    return sorted(g for g in (d["gene"] for d in load_corpus().values()) if q in g.upper())


def _local_gene_evidence(gene: str, cancer_type: str | None = None) -> dict:
    """Return curated evidence for ``gene``, honestly flagging missing coverage.

    Result keys: gene, found, has_curated_evidence, cancer_type,
    cancer_type_match (bool | None), sections, frontmatter, path, message.
    """
    corpus = load_corpus()
    doc = corpus.get(gene.strip().upper())

    if doc is None:
        return {
            "gene": gene,
            "found": False,
            "has_curated_evidence": False,
            "cancer_type": cancer_type,
            "cancer_type_match": None,
            "sections": {},
            "frontmatter": {},
            "path": None,
            "message": NO_EVIDENCE_MESSAGE,
        }

    associated = [c.upper() for c in doc["frontmatter"].get("associated_cancer_types", [])]
    cancer_type_match = (
        None if cancer_type is None else cancer_type.strip().upper() in associated
    )

    if doc["has_curated_evidence"]:
        message = "근거 문서 있음 (curated evidence available)"
    else:
        message = TEMPLATE_ONLY_MESSAGE

    return {
        "gene": doc["gene"],
        "found": True,
        "has_curated_evidence": doc["has_curated_evidence"],
        "cancer_type": cancer_type,
        "cancer_type_match": cancer_type_match,
        "sections": {name: doc["sections"].get(name, "") for name in SECTIONS},
        "frontmatter": doc["frontmatter"],
        "path": doc["path"],
        "message": message,
    }


def retrieve_gene_evidence(
    gene: str, cancer_type: str | None = None, web_fallback: bool = False
) -> dict:
    """Local RAG evidence, with an optional trusted-web fallback.

    Adds two keys to the local result:
      - ``evidence_state``: "local_curated" | "external_review" | "none".
      - ``web``: the :func:`src.rag.web_retrieve.web_search_gene` result when a
        fallback was performed, else ``None``.

    The web fallback runs only when there is no curated local document (missing or
    draft) AND ``web_fallback=True``. Web results are review-required and are
    never treated as curated evidence.
    """
    result = _local_gene_evidence(gene, cancer_type)
    result["web"] = None

    if result["has_curated_evidence"]:
        result["evidence_state"] = "local_curated"
        return result

    if web_fallback:
        from src.rag.web_retrieve import web_search_gene

        web = web_search_gene(gene)
        result["web"] = web
        result["evidence_state"] = (
            "external_review" if web["status"] == "external_found" else "none"
        )
    else:
        result["evidence_state"] = "local_insufficient"
    return result


def _main(argv: list[str]) -> None:
    if not argv:
        genes = available_genes()
        print("Usage: python -m src.rag.retrieve <gene> [cancer_type]")
        print(f"{len(genes)} genes in corpus: {', '.join(genes)}")
        return
    gene = argv[0]
    cancer_type = argv[1] if len(argv) > 1 else None
    result = retrieve_gene_evidence(gene, cancer_type, web_fallback=True)
    print(f"gene: {result['gene']}")
    print(f"found: {result['found']} | curated: {result['has_curated_evidence']}")
    print(f"evidence_state: {result['evidence_state']}")
    print(f"message: {result['message']}")
    if result["has_curated_evidence"]:
        print(f"path: {result['path']}")
        for name, content in result["sections"].items():
            preview = content.replace("\n", " ")[:100]
            print(f"  [{name}] {preview}")
    elif result["web"] and result["web"]["status"] == "external_found":
        web = result["web"]
        print(f"web fallback ({web['label']}), aliases: {web['aliases_searched']}")
        for link in web["sources"]:
            print(f"  - {link['source']} ({link['queried_symbol']}): {link['url']}")
        if web["external_summary"]:
            print(f"  NCBI summary: {web['external_summary']['text'][:120]}")


if __name__ == "__main__":
    _main(sys.argv[1:])
