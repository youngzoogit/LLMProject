# Stage 3 - RAG + LLM prompt + Streamlit MVP report

_Scope: predict.py hardening, a manually-authorable RAG corpus, keyword
retrieval, a grounded LLM prompt builder, and a one-screen Streamlit MVP. No LLM
API is called yet._

## 1. predict.py input-quality hardening
`src/predict.py` now reports how well a raw expression vector covers the model's
1,500 genes, so an under-specified external vector is never silently scored.

- `assess_input_quality(expression)` returns `missing_gene_count`,
  `missing_gene_fraction`, `extra_gene_count`, and a `warning` (set when
  `missing_fraction >= 0.10`; also logged via `logging`).
- `predict_from_vector()` / `predict()` now return
  `{"input_quality": {...}, "predictions": {model: {...}}}`.
- `sample_id`-based prediction pulls the vector from the processed matrix, so it
  always reports `missing=0` and `warning=None` (verified).
- The CLI prints the input-quality line before the per-model predictions.

## 2. RAG corpus (manual-authoring templates)
- `src/rag/build_corpus.py` writes one markdown template per **primary**
  RAG-candidate gene (>= 2 models) from `reports/rag_candidate_genes.csv`.
- Output: `data/rag_corpus/<GENE>.md` (18 files created).
- Sections: `summary`, `cancer_relevance`, `pathway`, `therapeutic_relevance`,
  `sources`, `evidence_limitations`, plus frontmatter (gene, n_models,
  flagged_by_models, associated_cancer_types, numeric_gene_flag, status).
- **No biology is fabricated.** Every curatable section is filled with a
  `TODO(evidence_needed)` placeholder. The only auto-filled content is
  data-derived model signal (which models flagged the gene, which per-class Top
  lists it appears in), clearly labelled as *not* biological evidence.
- Existing files are not overwritten unless `--force` is passed, protecting
  manual curation.

## 3. Retrieval
- `src/rag/retrieve.py` provides `retrieve_gene_evidence(gene, cancer_type=None)`
  over a simple keyword (symbol) lookup. FAISS/Chroma is deferred.
- Honest coverage reporting:
  - no document -> `found=False`, message "근거 문서 없음 (no evidence document found)".
  - template only -> `has_curated_evidence=False`, message "근거 제한적 ...".
  - `cancer_type_match` compares the query cancer type against the document's
    `associated_cancer_types`.

## 4. LLM prompt builder (no API call)
- `src/llm/explain.py` -> `generate_explanation_prompt(sample_id, predictions,
  top_genes, evidence_docs)` returns a `[SYSTEM]`/`[USER]` prompt string.
- Enforced rules: use only provided RAG evidence; no unsupported biological
  claims; mark uncovered genes "근거 제한적"; cite sources; research/education
  only (not clinical); separate model signal from biological proof.

## 5. Streamlit MVP
- `app.py` -> `streamlit run app.py`. One screen:
  1. sample_id dropdown (+ ground-truth label).
  2. 3-model predicted labels and Top-3 probabilities, with an input-quality note.
  3. Primary RAG-candidate genes table.
  4. Selected gene's RAG evidence document (with curated/template status).
  5. The drafted LLM prompt (displayed, never sent).

## 6. Validation (commands + actual results)

```
$ python -m src.predict TCGA-KS-A4I5-01
Predictions for TCGA-KS-A4I5-01:
  input_quality: missing=0 (0.0%), extra=0, warning=none
  logistic       -> THCA   (THCA=1.000, ...)
  random_forest  -> THCA   (THCA=0.988, ...)
  mlp            -> THCA   (THCA=1.000, ...)

$ python -m src.rag.retrieve SFTPB LUSC
gene: SFTPB | found: True | curated: False | cancer_type match: True
message: 근거 제한적 (document exists but is an unfilled template ...)

$ python -m src.rag.retrieve TP53
gene: TP53 | found: False | message: 근거 문서 없음 (no evidence document found)
```

- Streamlit app verified via `streamlit.testing.v1.AppTest`: **0 exceptions**,
  all 4 sections render, predictions + ground-truth shown (e.g. all-THCA agreement
  on a THCA sample). `streamlit run app.py` is therefore runnable.
- `predict.py` unit checks: 30%-missing vector triggers the warning; extra genes
  are counted; sample_id path stays warning-free.

## 7. Remaining TODOs
- **Fill the 18 corpus templates** with curated evidence + sources, then set
  `status: curated`. Until then the LLM layer will correctly say "근거 제한적".
- **Semantic retrieval**: add sentence-transformers embeddings + FAISS/Chroma and
  metadata filters (gene/cancer/pathway), replacing keyword-only lookup.
- **Wire a real LLM call** behind `explain.py` (guarded by an API key; keep the
  no-evidence guardrails).
- **Corpus coverage**: optionally extend beyond the 18 primary genes to the
  secondary set and per-class Top genes.
- **Numeric gene ids**: none reached the primary set now, but keep the
  `numeric_gene_flag` filter when future numeric ids appear (skip from RAG).
- **Sample picker UX**: 3,604 ids in one dropdown is heavy; add cancer-type
  filtering or search-by-prefix.
- **Deprecation**: monitor Streamlit `width=` API changes across versions.
