"""Build the LLM explanation prompt (stage 3). No API call is made here.

:func:`generate_explanation_prompt` assembles a grounded prompt from the model
predictions, the consensus/top genes, and the RAG evidence documents. The system
rules force the model to stay within the retrieved evidence, to flag anything not
covered as "근거 제한적", and to frame output as research/education only.
"""

from __future__ import annotations

SYSTEM_RULES = """You are a genomics research assistant that explains why a set of
machine-learning models predicted a cancer type from gene expression. Follow these
rules strictly:

1. Use ONLY the RAG evidence provided in this prompt. Do not use outside knowledge.
2. Make no biological claim (gene function, driver/suppressor role, pathway,
   therapy) unless it is stated in the provided evidence documents.
3. If a gene's evidence is missing or the document is an unfilled template, say
   explicitly that evidence is limited ("근거 제한적") for that gene. Never invent
   or guess facts to fill the gap.
4. Always cite the source document/section when you state a fact from evidence.
5. This output is for RESEARCH and EDUCATION only. It is NOT a clinical diagnosis
   and must not be used for medical decisions. State this in the answer.
6. Distinguish the model signal (which models agreed, prediction probability)
   from biological evidence; the model agreeing is not itself biological proof."""


def _format_predictions(predictions: dict) -> str:
    lines = []
    for model, out in predictions.items():
        probs = out.get("probabilities", {})
        top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        prob_str = ", ".join(f"{cls}={p:.3f}" for cls, p in top3)
        lines.append(f"- {model}: predicted {out.get('predicted_label')} ({prob_str})")
    return "\n".join(lines) if lines else "- (no predictions provided)"


def _format_top_genes(top_genes: list[dict]) -> str:
    if not top_genes:
        return "- (no top genes provided)"
    lines = []
    for g in top_genes:
        gene = g.get("gene", "?")
        n_models = g.get("n_models")
        priority = g.get("rag_priority")
        bits = [f"{gene}"]
        if n_models is not None:
            bits.append(f"agreed by {n_models} model(s)")
        if priority:
            bits.append(f"RAG priority: {priority}")
        lines.append("- " + ", ".join(bits))
    return "\n".join(lines)


def _format_evidence(evidence_docs: list[dict]) -> str:
    if not evidence_docs:
        return "(no evidence documents were retrieved)"
    blocks = []
    for doc in evidence_docs:
        gene = doc.get("gene", "?")
        if not doc.get("found"):
            blocks.append(f"### {gene}\nSTATUS: 근거 문서 없음 (no evidence document).")
            continue
        if not doc.get("has_curated_evidence"):
            blocks.append(
                f"### {gene}\nSTATUS: 근거 제한적 (template only, no curated evidence). "
                f"Treat all sections below as unfilled placeholders."
            )
            continue
        sections = doc.get("sections", {})
        section_text = "\n".join(
            f"- {name}: {content}" for name, content in sections.items() if content
        )
        blocks.append(f"### {gene}\nSTATUS: curated\n{section_text}")
    return "\n\n".join(blocks)


def generate_explanation_prompt(
    sample_id: str,
    predictions: dict,
    top_genes: list[dict],
    evidence_docs: list[dict],
) -> str:
    """Assemble the grounded explanation prompt (SYSTEM + USER sections).

    Args:
        sample_id: the sample being explained.
        predictions: ``{model: {"predicted_label": str, "probabilities": {...}}}``
            (the ``predictions`` sub-dict returned by :func:`src.predict.predict`).
        top_genes: list of dicts with at least ``gene`` (optionally ``n_models``,
            ``rag_priority``).
        evidence_docs: list of :func:`src.rag.retrieve.retrieve_gene_evidence`
            results.

    Returns:
        The full prompt as a single string (no API call is made).
    """
    return f"""[SYSTEM]
{SYSTEM_RULES}

[USER]
Sample: {sample_id}

## Model predictions
{_format_predictions(predictions)}

## Consensus / top genes (model signal, NOT biological proof)
{_format_top_genes(top_genes)}

## RAG evidence documents
{_format_evidence(evidence_docs)}

## Task
Explain, for a researcher, why these models predicted this cancer type:
1. Summarise the model agreement and prediction confidence.
2. For each key gene, state its role ONLY if the evidence above supports it;
   otherwise write "근거 제한적" for that gene.
3. Cite the evidence document/section for every biological statement.
4. End with a one-line reminder that this is research/education only, not a
   clinical diagnosis.
"""


if __name__ == "__main__":
    # Tiny smoke demo with synthetic inputs (no models/files loaded).
    demo = generate_explanation_prompt(
        sample_id="TCGA-XX-XXXX-01",
        predictions={
            "logistic": {"predicted_label": "LUSC", "probabilities": {"LUSC": 0.99, "LUAD": 0.01}},
        },
        top_genes=[{"gene": "SFTPB", "n_models": 2, "rag_priority": "primary"}],
        evidence_docs=[{"gene": "SFTPB", "found": True, "has_curated_evidence": False}],
    )
    print(demo)
