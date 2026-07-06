"""Model interpretation: Top-gene extraction and cross-model consensus (stage 2).

Two importance sources, per the design doc:
- Logistic Regression -> absolute standardized coefficients (per class + overall).
- Random Forest / MLP  -> permutation importance (global + per-class one-vs-rest).

Permutation importance is computed on the TRAINING set only (the test set is
reserved for final metrics). To bound runtime on 1,500 features:
- Global importance uses a stratified subsample and a few repeats.
- Per-class importance is restricted to the top-N globally important genes and
  scored with a one-vs-rest recall scorer (a manual single-column shuffle so we
  can test just the candidate subset).

Every Top-gene record carries a ``numeric_gene_flag`` so that purely numeric
Entrez ids (which the RAG stage cannot look up by symbol) are visible downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from src.data_loader import is_numeric_gene_id

# Defaults (tunable from the caller).
GLOBAL_N_REPEATS = 5
GLOBAL_SAMPLE_SIZE = 1200
PERCLASS_N_REPEATS = 3
PERCLASS_SAMPLE_SIZE = 1000
PERCLASS_CANDIDATE_GENES = 100


# --------------------------------------------------------------------------- #
# Logistic Regression: coefficient importance
# --------------------------------------------------------------------------- #
def _extract_classifier(model):
    """Return the final estimator (unwraps a sklearn Pipeline if present)."""
    if hasattr(model, "named_steps"):
        return model.named_steps["clf"]
    return model


def coefficient_importance(
    model, feature_names: list[str], class_labels: list[str]
) -> tuple[dict[str, pd.Series], pd.Series]:
    """Per-class and overall importance from |coefficients|.

    Coefficients come from a scaled pipeline, so magnitudes are comparable
    across genes. Overall importance = mean |coef| across classes.
    """
    clf = _extract_classifier(model)
    coef = np.asarray(clf.coef_)  # shape (n_classes, n_features)
    model_classes = list(clf.classes_)

    per_class: dict[str, pd.Series] = {}
    for label in class_labels:
        row = coef[model_classes.index(label)]
        per_class[label] = pd.Series(
            np.abs(row), index=feature_names
        ).sort_values(ascending=False)

    overall = pd.Series(
        np.abs(coef).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)
    return per_class, overall


# --------------------------------------------------------------------------- #
# Random Forest / MLP: permutation importance
# --------------------------------------------------------------------------- #
def _subsample(
    x: pd.DataFrame, y: pd.Series, size: int, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample to bound permutation-importance runtime."""
    if size is None or size >= len(x):
        return x, y
    x_s, _, y_s, _ = train_test_split(
        x, y, train_size=size, stratify=y, random_state=random_state
    )
    return x_s, y_s


def permutation_global(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    n_repeats: int = GLOBAL_N_REPEATS,
    sample_size: int = GLOBAL_SAMPLE_SIZE,
    random_state: int = 42,
) -> pd.Series:
    """Global permutation importance (accuracy drop) over all features."""
    x_s, y_s = _subsample(x_train, y_train, sample_size, random_state)
    result = permutation_importance(
        model,
        x_s,
        y_s,
        scoring="accuracy",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.Series(
        result.importances_mean, index=feature_names
    ).sort_values(ascending=False)


def _class_recall(model, x: pd.DataFrame, y_arr: np.ndarray, cls: str) -> float:
    """One-vs-rest recall for a single class (0 when the class is never right)."""
    pred = model.predict(x)
    return recall_score(
        (y_arr == cls).astype(int),
        (pred == cls).astype(int),
        pos_label=1,
        zero_division=0,
    )


def permutation_per_class(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    class_labels: list[str],
    candidate_genes: list[str],
    n_repeats: int = PERCLASS_N_REPEATS,
    sample_size: int = PERCLASS_SAMPLE_SIZE,
    random_state: int = 42,
) -> dict[str, pd.Series]:
    """Per-class permutation importance over a candidate gene subset.

    Importance = drop in one-vs-rest recall when a gene's column is shuffled.
    Restricting to ``candidate_genes`` (top global genes) keeps this tractable.
    """
    x_s, y_s = _subsample(x_train, y_train, sample_size, random_state)
    x_s = x_s.copy()
    y_arr = y_s.to_numpy()
    rng = np.random.default_rng(random_state)

    per_class: dict[str, pd.Series] = {}
    for cls in class_labels:
        baseline = _class_recall(model, x_s, y_arr, cls)
        scores: dict[str, float] = {}
        for gene in candidate_genes:
            original = x_s[gene].to_numpy().copy()
            drops = []
            for _ in range(n_repeats):
                x_s[gene] = rng.permutation(original)
                drops.append(baseline - _class_recall(model, x_s, y_arr, cls))
            x_s[gene] = original  # restore
            scores[gene] = float(np.mean(drops))
        per_class[cls] = pd.Series(scores).sort_values(ascending=False)
    return per_class


# --------------------------------------------------------------------------- #
# Records + consensus
# --------------------------------------------------------------------------- #
def top_gene_records(
    model_name: str,
    importance_kind: str,
    per_class: dict[str, pd.Series],
    overall: pd.Series,
    top_n_class: int = 10,
    top_n_overall: int = 30,
) -> list[dict]:
    """Flatten per-class and overall Top genes into CSV-ready rows.

    Columns: model, importance_kind, scope, cancer_type, rank, gene, importance,
    numeric_gene_flag.
    """
    records: list[dict] = []

    for cls, series in per_class.items():
        for rank, (gene, value) in enumerate(series.head(top_n_class).items(), 1):
            records.append(
                {
                    "model": model_name,
                    "importance_kind": importance_kind,
                    "scope": "per_class",
                    "cancer_type": cls,
                    "rank": rank,
                    "gene": gene,
                    "importance": round(float(value), 6),
                    "numeric_gene_flag": is_numeric_gene_id(gene),
                }
            )

    for rank, (gene, value) in enumerate(overall.head(top_n_overall).items(), 1):
        records.append(
            {
                "model": model_name,
                "importance_kind": importance_kind,
                "scope": "overall",
                "cancer_type": "ALL",
                "rank": rank,
                "gene": gene,
                "importance": round(float(value), 6),
                "numeric_gene_flag": is_numeric_gene_id(gene),
            }
        )
    return records


def build_consensus(
    overall_by_model: dict[str, pd.Series], top_n: int = 20
) -> pd.DataFrame:
    """Cross-model consensus over each model's top-N overall genes.

    Returns one row per gene appearing in any model's top-N, with a 0/1 flag per
    model, the number of agreeing models, and the numeric-gene flag. Sorted by
    agreement (desc) then gene name.
    """
    model_names = list(overall_by_model)
    top_sets = {
        name: list(series.head(top_n).index)
        for name, series in overall_by_model.items()
    }
    all_genes = sorted({g for genes in top_sets.values() for g in genes})

    rows: list[dict] = []
    for gene in all_genes:
        row: dict = {"gene": gene}
        for name in model_names:
            row[name] = int(gene in top_sets[name])
        row["n_models"] = sum(row[name] for name in model_names)
        row["numeric_gene_flag"] = is_numeric_gene_id(gene)
        rows.append(row)

    consensus = pd.DataFrame(rows)
    return consensus.sort_values(
        ["n_models", "gene"], ascending=[False, True]
    ).reset_index(drop=True)


# Number of agreeing models required to be a primary RAG candidate.
RAG_PRIMARY_MIN_MODELS = 2


def rag_priority(n_models: int) -> str:
    """Map cross-model agreement to a RAG search priority.

    Since no gene reaches full 3-model consensus in this run, genes agreed on by
    >= 2 models are treated as the primary RAG search targets; single-model
    genes are secondary.
    """
    return "primary" if n_models >= RAG_PRIMARY_MIN_MODELS else "secondary"


def build_rag_candidates(consensus: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """Derive a RAG candidate-gene table from the consensus table.

    Columns: gene, n_models, <one per model>, numeric_gene_flag, rag_priority.
    Sorted by agreement (desc), then gene name.
    """
    out = consensus.copy()
    out["rag_priority"] = out["n_models"].apply(rag_priority)
    columns = ["gene", "n_models", *model_names, "numeric_gene_flag", "rag_priority"]
    out = out[columns].sort_values(
        ["n_models", "gene"], ascending=[False, True]
    ).reset_index(drop=True)
    return out
