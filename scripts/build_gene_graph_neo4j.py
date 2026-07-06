"""Build the lightweight TCGA Gene-Cancer GraphRAG graph in Neo4j Aura.

Usage:
    python scripts/build_gene_graph_neo4j.py

The script is safe to re-run. It uses MERGE queries and exits gracefully when
Neo4j is not configured or unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.graph_store import build_graph, graph_diagnose, graph_ping  # noqa: E402


def main() -> int:
    ok, error = graph_ping()
    print(f"Neo4j ping: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(error or "unknown error")
        print("\nChecklist:")
        for line in graph_diagnose():
            print(f"- {line}")
        return 1

    result = build_graph()
    print("Build result:", result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
