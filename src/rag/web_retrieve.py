"""Web RAG fallback: trusted external references for genes with no local doc.

Used only when the local RAG corpus has no curated document for a gene. It does
NOT fabricate biology: it returns links to a fixed allow-list of trusted sources
(and, optionally, a real NCBI Gene summary fetched live). Results are always
labelled "임시 외부 근거 (검토 필요)" and are never auto-promoted to curated.

Live network fetch (NCBI E-utilities) is OFF by default and only enabled when the
environment variable ``TCGA_WEB_RAG_LIVE=1`` is set, so offline/test runs stay
fast and deterministic while still returning the trusted links.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

# Allow-list of trusted sources and their gene-query URL patterns ({q} = symbol).
TRUSTED_SOURCES: dict[str, str] = {
    "NCBI Gene": "https://www.ncbi.nlm.nih.gov/gene/?term={q}%5Bsym%5D+AND+human%5Borgn%5D",
    "Human Protein Atlas": "https://www.proteinatlas.org/search/{q}",
    "GeneCards": "https://www.genecards.org/cgi-bin/carddisp.pl?gene={q}",
    "PubMed": "https://pubmed.ncbi.nlm.nih.gov/?term={q}+cancer",
    "CIViC": "https://civicdb.org/genes?name={q}",
    "OncoKB": "https://www.oncokb.org/gene/{q}",
}

# Well-known symbol aliases (old TCGA-era symbol -> current HGNC symbol).
# Extend as needed; only include aliases you are confident are correct.
GENE_ALIASES: dict[str, list[str]] = {
    "LASS3": ["CERS3"],
    "LASS2": ["CERS2"],
    "FAM150B": ["ALKAL2"],
    "C1ORF106": ["INAVA"],
}

REVIEW_LABEL = "임시 외부 근거 (검토 필요)"
REVIEW_NOTE = (
    "로컬 curated 문서가 아니며, 자동으로 확정된 근거가 아닙니다. 아래 외부 출처는 "
    "참고용이며 사람이 검토해야 합니다."
)

_NCBI_TIMEOUT = 5
_NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Domains a real web search (Tavily) is restricted to.
ALLOWED_DOMAINS = [
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "proteinatlas.org",
    "civicdb.org",
    "oncokb.org",
]


def tavily_available() -> bool:
    """True if a Tavily API key and the tavily package are both present."""
    if not os.environ.get("TAVILY_API_KEY"):
        return False
    try:
        from tavily import TavilyClient  # noqa: F401

        return True
    except Exception:
        return False


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Real web search restricted to ALLOWED_DOMAINS. [] on any failure."""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        resp = client.search(
            query=query, include_domains=ALLOWED_DOMAINS, max_results=max_results
        )
        results = []
        for item in resp.get("results", []):
            if not item.get("url"):
                continue
            results.append(
                {
                    "source": "Tavily",
                    "url": item["url"],
                    "title": item.get("title", ""),
                    "snippet": (item.get("content") or "")[:300],
                    "queried_symbol": query,
                }
            )
        return results
    except Exception:
        return []


def resolve_aliases(gene: str) -> list[str]:
    """Return the gene plus any known aliases (e.g. LASS3 -> [LASS3, CERS3])."""
    key = gene.strip().upper()
    names = [key]
    for alias in GENE_ALIASES.get(key, []):
        if alias.upper() not in (n.upper() for n in names):
            names.append(alias)
    return names


def build_source_links(name: str) -> list[dict]:
    """Trusted-source links for a single gene symbol."""
    q = urllib.parse.quote(name)
    return [
        {"source": src, "url": tpl.format(q=q)} for src, tpl in TRUSTED_SOURCES.items()
    ]


def _live_enabled() -> bool:
    # Live NCBI Gene summary is ON by default (public API, no key). Set
    # TCGA_WEB_RAG_LIVE=0 to disable network calls (e.g. fully offline).
    return os.environ.get("TCGA_WEB_RAG_LIVE", "1") == "1"


def fetch_ncbi_summary(gene: str, timeout: int = _NCBI_TIMEOUT) -> dict | None:
    """Best-effort live NCBI Gene summary (real data). None on any failure.

    Returns ``{"text", "source", "url", "entrez_id"}`` or ``None``. Never raises.
    """
    try:
        term = f"{gene}[sym] AND human[orgn]"
        params = urllib.parse.urlencode({"db": "gene", "term": term, "retmode": "json"})
        with urllib.request.urlopen(f"{_NCBI_ESEARCH}?{params}", timeout=timeout) as resp:
            ids = json.loads(resp.read()).get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None
        gid = ids[0]
        params = urllib.parse.urlencode({"db": "gene", "id": gid, "retmode": "json"})
        with urllib.request.urlopen(f"{_NCBI_ESUMMARY}?{params}", timeout=timeout) as resp:
            doc = json.loads(resp.read()).get("result", {}).get(gid, {})
        text = (doc.get("summary") or doc.get("description") or "").strip()
        if not text:
            return None
        return {
            "text": text,
            "source": "NCBI Gene",
            "url": f"https://www.ncbi.nlm.nih.gov/gene/{gid}",
            "entrez_id": gid,
        }
    except Exception:
        return None


def web_search_gene(gene: str, allow_network: bool | None = None) -> dict:
    """Return trusted external references for a gene (marked review-required).

    Args:
        gene: gene symbol.
        allow_network: override the ``TCGA_WEB_RAG_LIVE`` env flag for live NCBI
            fetch. When None, uses the env flag.

    Result keys: query_gene, aliases_searched, status ("external_found"|"none"),
    review_required, label, sources, external_summary, note.
    """
    symbol = (gene or "").strip()
    if not symbol or not symbol.replace("-", "").isalnum():
        return {
            "query_gene": gene,
            "aliases_searched": [],
            "status": "none",
            "review_required": False,
            "label": REVIEW_LABEL,
            "sources": [],
            "external_summary": None,
            "note": "유효한 유전자 심볼이 아니어서 외부 근거를 찾지 못했습니다.",
        }

    names = resolve_aliases(symbol)
    sources: list[dict] = []
    seen_urls: set[str] = set()
    for name in names:
        for link in build_source_links(name):
            if link["url"] not in seen_urls:
                seen_urls.add(link["url"])
                link = {**link, "queried_symbol": name}
                sources.append(link)

    use_net = _live_enabled() if allow_network is None else allow_network
    external_summary = None
    provider = "links"

    # Real web search via Tavily (when a key is configured), restricted to
    # ALLOWED_DOMAINS. Results carry source_url + snippet and go first.
    if tavily_available():
        tav = tavily_search(f"{symbol} gene cancer")
        if tav:
            provider = "tavily"
            for item in reversed(tav):
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    sources.insert(0, item)
            top = tav[0]
            if top.get("snippet"):
                external_summary = {
                    "text": top["snippet"],
                    "source": "Tavily",
                    "url": top["url"],
                }

    # Otherwise (or additionally) a real NCBI Gene summary when live is enabled.
    if external_summary is None and use_net:
        for name in names:
            external_summary = fetch_ncbi_summary(name)
            if external_summary:
                provider = "ncbi"
                break

    return {
        "query_gene": symbol,
        "aliases_searched": names,
        "status": "external_found",
        "review_required": True,
        "label": REVIEW_LABEL,
        "provider": provider,
        "sources": sources,
        "external_summary": external_summary,
        "note": REVIEW_NOTE,
    }
