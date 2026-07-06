"""Neo4j graph lookup helpers for the lightweight GraphRAG layer.

This module is deliberately optional. If Neo4j is not configured or the driver
cannot connect, callers receive an empty graph context and can continue with the
local document RAG path.
"""

from __future__ import annotations

import logging

from src.rag.graph_store import _cfg, get_driver, graph_enabled

logger = logging.getLogger(__name__)

_MAX_EDGES = 20


def graph_available() -> bool:
    """Return True when Neo4j credentials appear to be configured."""
    return graph_enabled()


def _run(cypher: str, params: dict) -> list[dict]:
    """Run a Cypher query and return plain dictionaries. Never raises."""
    driver = get_driver()
    if driver is None:
        return []
    try:
        with driver.session(database=_cfg()["database"]) as session:
            return [dict(record) for record in session.run(cypher, **params)]
    except Exception as exc:  # noqa: BLE001 - graph fallback must be robust
        logger.warning("Neo4j graph query failed: %s", exc)
        return []
    finally:
        try:
            driver.close()
        except Exception:
            pass


def gene_neighbors(gene: str, limit: int = _MAX_EDGES) -> list[dict]:
    """Return outgoing graph edges from a Gene node."""
    cypher = (
        "MATCH (g:Gene {name:$gene})-[r]->(target) "
        "RETURN type(r) AS rel, labels(target)[0] AS target_type, "
        "coalesce(target.name, target.code, target.gene, target.url, target.name_ko) "
        "AS target "
        "LIMIT $limit"
    )
    return _run(cypher, {"gene": gene.upper(), "limit": limit})


def cancer_neighbors(cancer_code: str, limit: int = _MAX_EDGES) -> list[dict]:
    """Return genes connected to a CancerType node."""
    cypher = (
        "MATCH (g:Gene)-[r:ASSOCIATED_WITH]->(c:CancerType {code:$code}) "
        "RETURN g.name AS gene, type(r) AS rel, c.code AS target "
        "LIMIT $limit"
    )
    return _run(cypher, {"code": cancer_code.upper(), "limit": limit})


def graph_context(genes: list[str], cancer_code: str | None = None) -> dict:
    """Return graph edges for genes and an optional cancer code.

    Shape:
        {
            "available": bool,
            "edges": ["TG -[ASSOCIATED_WITH]-> THCA", ...],
            "raw": [{"source": "TG", "rel": "...", "target": "..."}]
        }
    """
    if not graph_available():
        return {"available": False, "edges": [], "raw": []}

    edges: list[str] = []
    raw: list[dict] = []
    seen: set[str] = set()

    for gene in genes[:5]:
        gene_name = gene.upper()
        for row in gene_neighbors(gene_name):
            target = row.get("target")
            if not target:
                continue
            edge = f"{gene_name} -[{row['rel']}]-> {target}"
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)
            raw.append({"source": gene_name, **row})

    if cancer_code:
        code = cancer_code.upper()
        for row in cancer_neighbors(code, limit=8):
            edge = f"{row['gene']} -[{row['rel']}]-> {row['target']}"
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)
            raw.append(
                {"source": row["gene"], "rel": row["rel"], "target": row["target"]}
            )

    return {"available": True, "edges": edges[:_MAX_EDGES], "raw": raw}


def _main(argv: list[str]) -> None:
    if not graph_available():
        print("Neo4j is not configured. Graph search is disabled; local RAG can continue.")
        return
    arg = argv[0].upper() if argv else "TG"
    cancer = arg if len(arg) <= 5 and arg.isupper() else None
    ctx = graph_context([arg], cancer)
    print(f"graph edges for {arg}: {len(ctx['edges'])}")
    for edge in ctx["edges"]:
        print(" -", edge)


if __name__ == "__main__":
    import sys

    _main(sys.argv[1:])
