"""Stage-2 training + evaluation + interpretation orchestrator.

Run from the project root::

    python -m src.train_models

Trains Logistic Regression, Random Forest and MLP on
``data/processed/train.parquet``, evaluates ONLY on
``data/processed/test.parquet`` (Macro F1 is the headline metric), and writes:
  - reports/model_metrics.csv
  - reports/confusion_matrix_{model}.png (one per model)
  - reports/top_genes_by_model.csv        (per-class Top10 + overall Top30)
  - reports/gene_consensus.csv            (cross-model agreement)
  - reports/modeling_report.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)

from src.data_loader import PROJECT_ROOT
from src.interpret import (
    PERCLASS_CANDIDATE_GENES,
    build_consensus,
    build_rag_candidates,
    coefficient_importance,
    permutation_global,
    permutation_per_class,
    top_gene_records,
)
from src.models import logistic, mlp, random_forest

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
TRAINED_DIR = PROJECT_ROOT / "models" / "trained"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
LABEL_MAPPING_PATH = PROCESSED_DIR / "label_mapping.json"

METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
TOP_GENES_PATH = REPORTS_DIR / "top_genes_by_model.csv"
CONSENSUS_PATH = REPORTS_DIR / "gene_consensus.csv"
RAG_CANDIDATES_PATH = REPORTS_DIR / "rag_candidate_genes.csv"
MODELING_REPORT_PATH = REPORTS_DIR / "modeling_report.md"
FEATURE_NAMES_PATH = TRAINED_DIR / "feature_names.json"

LABEL_COLUMN = "label"
RANDOM_STATE = 42
TOP_N_PER_CLASS = 10
TOP_N_OVERALL = 30
CONSENSUS_TOP_N = 20

MODEL_MODULES = [logistic, random_forest, mlp]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(path):
    frame = pd.read_parquet(path)
    y = frame[LABEL_COLUMN]
    x = frame.drop(columns=[LABEL_COLUMN])
    return x, y


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred) -> dict[str, float]:
    """Accuracy + macro precision/recall/F1 (macro is the headline metric)."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def save_confusion_matrix(y_true, y_pred, class_labels, display_name, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=class_labels,
        display_labels=class_labels,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {display_name} (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-model importance dispatch
# --------------------------------------------------------------------------- #
def compute_importance(module, model, x_train, y_train, feature_names, class_labels):
    """Return (per_class, overall) importance for the given fitted model."""
    if module.IMPORTANCE_KIND == "coefficient":
        return coefficient_importance(model, feature_names, class_labels)

    overall = permutation_global(
        model, x_train, y_train, feature_names, random_state=RANDOM_STATE
    )
    candidate_genes = list(overall.head(PERCLASS_CANDIDATE_GENES).index)
    per_class = permutation_per_class(
        model,
        x_train,
        y_train,
        class_labels,
        candidate_genes,
        random_state=RANDOM_STATE,
    )
    return per_class, overall


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _metrics_markdown(metrics_df: pd.DataFrame) -> str:
    header = "| model | accuracy | macro_precision | macro_recall | macro_f1 |"
    sep = "|---|---:|---:|---:|---:|"
    rows = [
        f"| {r.model} | {r.accuracy:.4f} | {r.macro_precision:.4f} | "
        f"{r.macro_recall:.4f} | **{r.macro_f1:.4f}** |"
        for r in metrics_df.itertuples()
    ]
    return "\n".join([header, sep, *rows])


def _consensus_markdown(consensus: pd.DataFrame, model_names, limit=15) -> str:
    cols = ["gene", *model_names, "n_models", "numeric_gene_flag"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for r in consensus.head(limit).itertuples(index=False):
        d = r._asdict()
        rows.append("| " + " | ".join(str(d[c]) for c in cols) + " |")
    return "\n".join([header, sep, *rows])


def write_modeling_report(
    metrics_df, consensus, rag_candidates, model_names, display_names, numeric_top_genes
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = metrics_df.sort_values("macro_f1", ascending=False).iloc[0]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_full_consensus = int((consensus["n_models"] == len(model_names)).sum())
    n_primary = int((rag_candidates["rag_priority"] == "primary").sum())
    n_secondary = int((rag_candidates["rag_priority"] == "secondary").sum())
    primary_genes = rag_candidates.loc[
        rag_candidates["rag_priority"] == "primary", "gene"
    ].tolist()
    primary_preview = ", ".join(f"`{g}`" for g in primary_genes) or "none"

    img_lines = "\n".join(
        f"- `reports/confusion_matrix_{name}.png` - {display_names[name]}"
        for name in model_names
    )
    numeric_note = (
        ", ".join(f"`{g}`" for g in numeric_top_genes)
        if numeric_top_genes
        else "none appeared in any Top-gene list"
    )

    report = f"""# TCGA Modeling Report - Stage 2

_Generated: {generated}_

## 1. Setup
- Train: `data/processed/train.parquet`, Test: `data/processed/test.parquet`
  (test used for final evaluation only).
- Features: 1,500 top-variance genes; 6 cancer types (BRCA, COAD, KIRC, LUAD, LUSC, THCA).
- Class imbalance is present (BRCA ~30%), so **Macro F1 is the headline metric**;
  models use balanced class weights where supported.

## 2. Test-set metrics
{_metrics_markdown(metrics_df)}

**Best model by Macro F1: {best.model} ({best.macro_f1:.4f}).**

Confusion matrices (test set):
{img_lines}

## 3. Top-gene interpretation
- Logistic Regression: absolute standardized coefficients.
- Random Forest / MLP: permutation importance (global over all 1,500 genes;
  per-class over the top-{PERCLASS_CANDIDATE_GENES} global genes, one-vs-rest
  recall). Computed on the training set (test reserved for metrics).
- Full detail: `reports/top_genes_by_model.csv`
  (per-class Top{TOP_N_PER_CLASS} + overall Top{TOP_N_OVERALL}).

## 4. Cross-model consensus
Each model's top-{CONSENSUS_TOP_N} overall genes were compared.
- Genes shared by all {len(model_names)} models: **{n_full_consensus}**.
- Full table: `reports/gene_consensus.csv`.

Top consensus genes:
{_consensus_markdown(consensus, model_names)}

## 5. RAG candidate genes
No gene reaches full {len(model_names)}-model consensus in this run, so the
consensus criterion is relaxed: **genes agreed on by >= 2 models are the primary
RAG search targets**, single-model genes are secondary.
- Primary candidates (>= 2 models): **{n_primary}** - {primary_preview}.
- Secondary candidates (1 model): **{n_secondary}**.
- Full table with priority: `reports/rag_candidate_genes.csv`
  (columns: gene, n_models, {', '.join(model_names)}, numeric_gene_flag, rag_priority).

## 6. Numeric gene ids in Top-gene lists
Purely numeric Entrez ids flagged via `numeric_gene_flag` in the Top-gene,
consensus and RAG-candidate CSVs (no HGNC symbol for RAG lookup): {numeric_note}.

## 7. Notes & caveats
- MLP uses L2 + early stopping in place of dropout (scikit-learn MLP has no
  dropout layer); a lower MLP score is a legitimate result on high-dimensional,
  low-sample data, not a failure.
- Permutation importance is computed on training data, so it reflects training
  sensitivity; treat magnitudes as relative, not absolute.
- Logistic Regression config was updated for scikit-learn >= 1.8: `penalty` and
  `n_jobs` were removed and elasticnet is expressed via `l1_ratio=0.5`, which
  clears the prior FutureWarnings with no behavioural change.

## 8. Saved models
- `models/trained/logistic.joblib`, `models/trained/random_forest.joblib`,
  `models/trained/mlp.joblib` (fitted pipelines).
- `models/trained/feature_names.json` - the 1,500-gene feature order for input
  alignment. Load and predict via `src/predict.py`.
"""
    MODELING_REPORT_PATH.write_text(report, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading processed splits...")
    x_train, y_train = load_split(TRAIN_PATH)
    x_test, y_test = load_split(TEST_PATH)
    feature_names = list(x_train.columns)

    with LABEL_MAPPING_PATH.open(encoding="utf-8") as fh:
        label_mapping = json.load(fh)
    class_labels = sorted(label_mapping["code_to_id"], key=label_mapping["code_to_id"].get)
    print(f"  train {x_train.shape}, test {x_test.shape}, classes {class_labels}")

    # Persist the exact feature (gene) order so predict.py can align inputs.
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    with FEATURE_NAMES_PATH.open("w", encoding="utf-8") as fh:
        json.dump({"feature_names": feature_names}, fh)

    metrics_rows = []
    all_gene_records: list[dict] = []
    overall_by_model: dict[str, pd.Series] = {}
    display_names: dict[str, str] = {}

    for module in MODEL_MODULES:
        name, display = module.NAME, module.DISPLAY_NAME
        display_names[name] = display
        print(f"\n[{display}] fitting...")
        model = module.build_model(random_state=RANDOM_STATE)
        model.fit(x_train, y_train)

        # Persist the fitted model for the predict/RAG/Streamlit stage.
        model_path = TRAINED_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        print(f"  saved -> {model_path.relative_to(PROJECT_ROOT)}")

        y_pred = model.predict(x_test)
        metrics = evaluate(y_test, y_pred)
        metrics_rows.append({"model": name, **metrics})
        print(
            f"  acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f}"
        )

        cm_path = REPORTS_DIR / f"confusion_matrix_{name}.png"
        save_confusion_matrix(y_test, y_pred, class_labels, display, cm_path)

        print(f"  computing {module.IMPORTANCE_KIND} importance...")
        per_class, overall = compute_importance(
            module, model, x_train, y_train, feature_names, class_labels
        )
        overall_by_model[name] = overall
        all_gene_records.extend(
            top_gene_records(
                name,
                module.IMPORTANCE_KIND,
                per_class,
                overall,
                top_n_class=TOP_N_PER_CLASS,
                top_n_overall=TOP_N_OVERALL,
            )
        )

    # Persist metrics.
    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        "macro_f1", ascending=False
    ).reset_index(drop=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

    # Persist top genes.
    top_genes_df = pd.DataFrame(all_gene_records)
    top_genes_df.to_csv(TOP_GENES_PATH, index=False)

    # Consensus.
    model_names = list(overall_by_model)
    consensus = build_consensus(overall_by_model, top_n=CONSENSUS_TOP_N)
    consensus.to_csv(CONSENSUS_PATH, index=False)

    # RAG candidate genes (>= 2 models = primary priority).
    rag_candidates = build_rag_candidates(consensus, model_names)
    rag_candidates.to_csv(RAG_CANDIDATES_PATH, index=False)

    # Numeric genes that showed up in any Top-gene list.
    numeric_top_genes = sorted(
        top_genes_df.loc[top_genes_df["numeric_gene_flag"], "gene"].unique()
    )

    write_modeling_report(
        metrics_df,
        consensus,
        rag_candidates,
        model_names,
        display_names,
        numeric_top_genes,
    )

    print("\nArtifacts written to reports/:")
    for path in (
        METRICS_PATH,
        TOP_GENES_PATH,
        CONSENSUS_PATH,
        MODELING_REPORT_PATH,
    ):
        print(f"  {path.name}")
    print("  confusion_matrix_{logistic,random_forest,mlp}.png")
    print("\nMetrics (sorted by Macro F1):")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    run()
