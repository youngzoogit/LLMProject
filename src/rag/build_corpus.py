"""Build a manually-authorable RAG corpus for the primary candidate genes.

This does NOT crawl or fabricate biology. For each primary RAG-candidate gene
(agreed on by >= 2 models in ``reports/rag_candidate_genes.csv``) it writes a
markdown TEMPLATE under ``data/rag_corpus/`` with the required sections and
``TODO(evidence_needed)`` placeholders that a human curator fills from vetted
sources (GeneCards, OncoKB, CIViC, PubMed).

The only auto-filled content is *data-derived signal* (which models flagged the
gene and in which per-class Top-gene lists it appears) -- clearly labelled as a
model signal, never as a biological claim.

Existing files are NOT overwritten by default, so manual curation is preserved.
Run ``python -m src.rag.build_corpus`` (add ``--force`` to regenerate all).
"""

from __future__ import annotations

import sys

import pandas as pd

from src.data_loader import PROJECT_ROOT, is_numeric_gene_id

RAG_CANDIDATES_PATH = PROJECT_ROOT / "reports" / "rag_candidate_genes.csv"
TOP_GENES_PATH = PROJECT_ROOT / "reports" / "top_genes_by_model.csv"
CORPUS_DIR = PROJECT_ROOT / "data" / "rag_corpus"

# Token that marks a section as not yet curated (parsed by retrieve.py).
EVIDENCE_NEEDED = "TODO(evidence_needed)"

# Required section order (per the stage-3 spec).
SECTIONS = (
    "summary",
    "cancer_relevance",
    "pathway",
    "therapeutic_relevance",
    "sources",
    "evidence_limitations",
)


def load_primary_genes() -> pd.DataFrame:
    """Rows of the RAG-candidate table with ``rag_priority == 'primary'``."""
    candidates = pd.read_csv(RAG_CANDIDATES_PATH)
    return candidates[candidates["rag_priority"] == "primary"].reset_index(drop=True)


def gene_cancer_associations() -> dict[str, list[str]]:
    """Map each gene to the cancer types whose per-class Top list it appears in."""
    top = pd.read_csv(TOP_GENES_PATH)
    per_class = top[top["scope"] == "per_class"]
    assoc: dict[str, list[str]] = {}
    for gene, group in per_class.groupby("gene"):
        assoc[gene] = sorted(group["cancer_type"].unique())
    return assoc


def models_flagging(row: pd.Series, model_names: list[str]) -> list[str]:
    """Which models included this gene in their top-N overall list."""
    return [m for m in model_names if int(row.get(m, 0)) == 1]


def render_template(
    gene: str,
    n_models: int,
    flagged_by: list[str],
    cancer_types: list[str],
    numeric_flag: bool,
) -> str:
    """Render one gene's markdown template (no biological claims invented)."""
    cancer_list = ", ".join(cancer_types) if cancer_types else "none in per-class Top lists"
    models_list = ", ".join(flagged_by) if flagged_by else "none"
    numeric_note = (
        "  # WARNING: numeric-only id, no HGNC symbol for lookup"
        if numeric_flag
        else ""
    )

    return f"""---
gene: {gene}
n_models: {n_models}
flagged_by_models: [{models_list}]
associated_cancer_types: [{cancer_list}]{numeric_note}
numeric_gene_flag: {str(numeric_flag).lower()}
status: draft
---

# {gene}

<!--
  Data signal (NOT biological evidence): this gene was flagged by permutation
  importance in [{models_list}] and appears in the per-class Top-gene lists for
  [{cancer_list}]. Use this only to prioritise curation; it is not a claim about
  the gene's biology. Fill each section below from vetted sources, then change
  `status:` to `curated`. Do NOT invent facts.
-->

## summary
{EVIDENCE_NEEDED}: one to two sentence functional summary of {gene}.

## cancer_relevance
{EVIDENCE_NEEDED}: known relationship of {gene} to the associated cancer types
({cancer_list}) - driver / tumor suppressor / diagnostic marker, with direction.

## pathway
{EVIDENCE_NEEDED}: pathways / biological processes {gene} participates in.

## therapeutic_relevance
{EVIDENCE_NEEDED}: approved or investigational therapies targeting {gene}, if any.

## sources
{EVIDENCE_NEEDED}: curated citations (GeneCards / OncoKB / CIViC / PubMed IDs).
Required before `status: curated`.

## evidence_limitations
This document is an unfilled template; no curated biological evidence has been
added yet. Downstream LLM output must state that evidence is limited for {gene}
until the sections above are completed and sourced.
"""


def build_corpus(force: bool = False) -> list[tuple[str, str]]:
    """Write one markdown template per primary gene. Returns (gene, action)."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    primary = load_primary_genes()
    model_names = [c for c in ("logistic", "random_forest", "mlp") if c in primary.columns]
    associations = gene_cancer_associations()

    results: list[tuple[str, str]] = []
    for row in primary.itertuples(index=False):
        gene = row.gene
        path = CORPUS_DIR / f"{gene}.md"
        if path.exists() and not force:
            results.append((gene, "skipped (exists)"))
            continue
        text = render_template(
            gene=gene,
            n_models=int(row.n_models),
            flagged_by=models_flagging(pd.Series(row._asdict()), model_names),
            cancer_types=associations.get(gene, []),
            numeric_flag=is_numeric_gene_id(gene),
        )
        path.write_text(text, encoding="utf-8")
        results.append((gene, "created"))
    return results


def _main(argv: list[str]) -> None:
    force = "--force" in argv
    results = build_corpus(force=force)
    created = sum(1 for _, a in results if a == "created")
    skipped = sum(1 for _, a in results if a.startswith("skipped"))
    print(f"Corpus dir: {CORPUS_DIR}")
    for gene, action in results:
        print(f"  {gene:12s} {action}")
    print(f"Done: {created} created, {skipped} skipped ({len(results)} primary genes).")


if __name__ == "__main__":
    _main(sys.argv[1:])
