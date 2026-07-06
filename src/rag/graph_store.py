"""Lightweight Neo4j (Aura) graph store for the TCGA GraphRAG layer.

Rule-based, not LLM-based: the graph is built deterministically from the local
RAG corpus + the model importance CSVs, so it is safe and reproducible. Neo4j is
entirely optional -- if credentials are missing or the driver/connection fails,
every function degrades to a no-op and the app keeps working on local RAG.

Node labels:      Gene, CancerType, Model, EvidenceDocument, ExternalSource,
                  BiologicalConcept
Relationships:    ASSOCIATED_WITH, IMPORTANT_FOR_MODEL, DOCUMENTED_IN,
                  HAS_SOURCE, RELATED_TO, CO_RELATED_WITH
"""

from __future__ import annotations

import logging
import os
import re

from src.data_loader import PROJECT_ROOT
from src.display import CANCER_KO
from src.rag.retrieve import load_corpus

logger = logging.getLogger(__name__)

# Load .env explicitly so this module works when run directly
# (python -m src.rag.graph_store), not only through app.py. Optional dependency.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:  # python-dotenv missing -> env vars still work if already set
    pass

TOP_GENES_PATH = PROJECT_ROOT / "reports" / "top_genes_by_model.csv"
_URL_RE = re.compile(r"https?://[^\s)]+")

# Explicit domain rules (requirement 7): clear, textbook thyroid biology.
BIO_CONCEPTS = {
    "갑상선호르몬 생합성": ["TG", "TPO", "TSHR"],
}


def _cfg() -> dict:
    return {
        "uri": os.getenv("NEO4J_URI"),
        "user": os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
        "database": os.getenv("NEO4J_DATABASE") or "neo4j",
    }


def graph_enabled() -> bool:
    """True only if the minimum Neo4j connection settings are present."""
    cfg = _cfg()
    return bool(cfg["uri"] and cfg["password"])


def get_driver():
    """Return a Neo4j driver, or None if unavailable (never raises)."""
    if not graph_enabled():
        return None
    try:
        from neo4j import GraphDatabase

        cfg = _cfg()
        return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neo4j driver init failed: %s", exc)
        return None


def graph_diagnose() -> list[str]:
    """Actionable checklist for a failed Neo4j (Aura) connection."""
    cfg = _cfg()
    uri = cfg["uri"] or ""
    checks = [f"NEO4J_URI 설정됨: {bool(uri)}" + (f" ({uri.split('://')[0]}://...)" if uri else "")]
    if uri and not uri.startswith(("neo4j+s://", "neo4j+ssc://", "bolt+s://", "neo4j://", "bolt://")):
        checks.append("URI scheme 이상: Aura는 'neo4j+s://<id>.databases.neo4j.io' 형식이어야 함.")
    checks.append(f"NEO4J_USERNAME 설정됨: {bool(cfg['user'])} (Aura 기본 'neo4j')")
    checks.append(f"NEO4J_PASSWORD 설정됨: {bool(cfg['password'])}")
    checks.append(f"NEO4J_DATABASE: {cfg['database']} (Aura 기본 'neo4j')")
    checks.append("체크: Aura 콘솔에서 인스턴스가 'Running' 인지 확인 (Free는 미사용 시 Paused -> Resume).")
    checks.append("체크: 방화벽/네트워크가 7687(bolt) 아웃바운드를 허용하는지 확인.")
    checks.append("체크: 인스턴스 생성 시 받은 자격증명 파일 값과 .env가 일치하는지 확인.")
    return checks


def graph_ping() -> tuple[bool, str | None]:
    """Check connectivity. Returns (ok, error_message with a hint on failure)."""
    driver = get_driver()
    if driver is None:
        return False, "NEO4J_URI/PASSWORD 미설정 또는 neo4j 드라이버 없음"
    try:
        driver.verify_connectivity()
        return True, None
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        hint = ""
        if "ServiceUnavailable" in name or "routing" in str(exc).lower():
            hint = " | 힌트: Aura 인스턴스가 Paused/중지 상태이거나 URI가 잘못됐을 수 있습니다."
        return False, f"{name}: {str(exc)[:140]}{hint}"
    finally:
        try:
            driver.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Build (rule-based)
# --------------------------------------------------------------------------- #
def _read_per_class_genes() -> list[dict]:
    """(gene, cancer_type, model) rows from the per-class top-gene CSV."""
    import pandas as pd

    if not TOP_GENES_PATH.exists():
        return []
    frame = pd.read_csv(TOP_GENES_PATH)
    pc = frame[frame["scope"] == "per_class"]
    return pc[["gene", "cancer_type", "model"]].to_dict("records")


def build_graph() -> dict:
    """Populate Neo4j from the local corpus + CSVs. Returns a summary dict.

    Idempotent (MERGE-based): safe to re-run. On any failure returns
    ``{"ok": False, "error": ...}`` without raising.
    """
    driver = get_driver()
    if driver is None:
        return {"ok": False, "error": "Neo4j 미설정/드라이버 없음 (그래프 비활성)"}

    corpus = load_corpus()  # gene -> parsed doc
    per_class = _read_per_class_genes()
    database = _cfg()["database"]

    counts = {"genes": 0, "docs": 0, "sources": 0, "assoc": 0}
    try:
        with driver.session(database=database) as session:
            for code, name_ko in CANCER_KO.items():
                session.run(
                    "MERGE (c:CancerType {code:$code}) SET c.name_ko=$ko",
                    code=code, ko=name_ko,
                )

            for doc in corpus.values():
                gene = doc["gene"]
                fm = doc["frontmatter"]
                session.run("MERGE (g:Gene {name:$g})", g=gene)
                counts["genes"] += 1
                session.run(
                    "MERGE (d:EvidenceDocument {gene:$g}) "
                    "SET d.status=$s, d.path=$p "
                    "WITH d MATCH (g:Gene {name:$g}) MERGE (g)-[:DOCUMENTED_IN]->(d)",
                    g=gene, s=fm.get("status", "draft"), p=doc.get("path", ""),
                )
                counts["docs"] += 1
                for code in fm.get("associated_cancer_types", []):
                    session.run(
                        "MATCH (g:Gene {name:$g}) MERGE (c:CancerType {code:$c}) "
                        "MERGE (g)-[:ASSOCIATED_WITH]->(c)",
                        g=gene, c=code,
                    )
                    counts["assoc"] += 1
                for m in fm.get("flagged_by_models", []):
                    session.run(
                        "MATCH (g:Gene {name:$g}) MERGE (m:Model {name:$m}) "
                        "MERGE (g)-[:IMPORTANT_FOR_MODEL]->(m)",
                        g=gene, m=m,
                    )
                if doc.get("has_curated_evidence"):
                    for url in _URL_RE.findall(doc["sections"].get("sources", "")):
                        session.run(
                            "MATCH (d:EvidenceDocument {gene:$g}) "
                            "MERGE (s:ExternalSource {url:$u}) "
                            "MERGE (d)-[:HAS_SOURCE]->(s)",
                            g=gene, u=url,
                        )
                        counts["sources"] += 1

            for row in per_class:
                session.run(
                    "MERGE (g:Gene {name:$g}) "
                    "MERGE (c:CancerType {code:$c}) "
                    "MERGE (m:Model {name:$m}) "
                    "MERGE (g)-[:ASSOCIATED_WITH]->(c) "
                    "MERGE (g)-[:IMPORTANT_FOR_MODEL]->(m)",
                    g=row["gene"], c=row["cancer_type"], m=row["model"],
                )

            for concept, genes in BIO_CONCEPTS.items():
                session.run("MERGE (b:BiologicalConcept {name:$n})", n=concept)
                for g in genes:
                    session.run(
                        "MATCH (b:BiologicalConcept {name:$n}) MERGE (g:Gene {name:$g}) "
                        "MERGE (g)-[:RELATED_TO]->(b)",
                        n=concept, g=g,
                    )
                for i, g1 in enumerate(genes):
                    for g2 in genes[i + 1:]:
                        session.run(
                            "MATCH (a:Gene {name:$a}),(b:Gene {name:$b}) "
                            "MERGE (a)-[:CO_RELATED_WITH]->(b)",
                            a=g1, b=g2,
                        )
        return {"ok": True, "counts": counts}
    except Exception as exc:  # noqa: BLE001
        logger.error("build_graph failed: %s", exc)
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    finally:
        try:
            driver.close()
        except Exception:
            pass


def _main() -> None:
    ok, err = graph_ping()
    print(f"graph_enabled: {graph_enabled()} | ping ok: {ok} | {err or ''}")
    if not ok:
        print("Neo4j 연결 불가 -> 그래프 구축 생략(앱은 로컬 RAG로 동작).")
        return
    print("build result:", build_graph())


if __name__ == "__main__":
    _main()
