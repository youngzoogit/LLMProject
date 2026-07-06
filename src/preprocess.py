"""Stage-1 preprocessing pipeline for TCGA cancer-type classification.

Pipeline (run as ``python -m src.preprocess`` from the project root):

    1. Build the labeled sample x gene matrix (via :mod:`src.data_loader`).
    2. Drop genes whose missing-value fraction exceeds ``MAX_GENE_MISSING_FRAC``.
    3. Stratified 80/20 train/test split (label proportions preserved).
    4. Rank genes by variance ON THE TRAINING SET and keep the top ``TOP_K_GENES``.
    5. Median-impute remaining missing values using TRAINING-SET medians only.
    6. Persist train/test parquet files + a label mapping, and write a data audit.

Leakage note
------------
Variance ranking and median imputation are fit on the training split only and
then applied to the test split. This deviates from the literal step order in
``document/project_design.md`` (which lists variance filtering before the split)
on purpose: fitting feature selection/imputation on test data would leak
information. The design doc's intent - reduce to ~1-2k high-variance genes with
a stratified split - is preserved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import (
    DISEASE_TO_CODE,
    PROJECT_ROOT,
    LabeledMatrix,
    build_labeled_matrix,
    is_numeric_gene_id,
)

# --------------------------------------------------------------------------- #
# Tunable configuration
# --------------------------------------------------------------------------- #
# A gene is dropped if more than this fraction of samples are missing.
MAX_GENE_MISSING_FRAC = 0.20
# Number of top-variance genes to retain (design doc suggests 1,000-2,000).
TOP_K_GENES = 1500
# Fraction of samples held out for the test split.
TEST_SIZE = 0.20
# Reproducibility.
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
LABEL_MAPPING_PATH = PROCESSED_DIR / "label_mapping.json"
AUDIT_PATH = REPORTS_DIR / "data_audit.md"

# Column name used to store the label inside the saved parquet files.
LABEL_COLUMN = "label"


@dataclass
class SplitResult:
    """Train/test matrices with aligned labels and the retained gene list."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    selected_genes: list[str]


# --------------------------------------------------------------------------- #
# Missing-value handling
# --------------------------------------------------------------------------- #
def gene_missing_fraction(features: pd.DataFrame) -> pd.Series:
    """Fraction of missing values per gene (column)."""
    return features.isna().mean(axis=0)


def drop_high_missing_genes(
    features: pd.DataFrame, max_missing_frac: float = MAX_GENE_MISSING_FRAC
) -> tuple[pd.DataFrame, list[str]]:
    """Drop genes whose missing fraction exceeds the threshold.

    Returns the filtered matrix and the list of dropped gene names.
    """
    frac = gene_missing_fraction(features)
    keep_mask = frac <= max_missing_frac
    dropped = frac.index[~keep_mask].tolist()
    return features.loc[:, keep_mask], dropped


def impute_median(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fill missing values with per-gene medians computed on the training set."""
    medians = x_train.median(axis=0, numeric_only=True)
    # Any gene that is all-NaN on train (median NaN) gets 0.0 as a safe fallback.
    medians = medians.fillna(0.0)
    return x_train.fillna(medians), x_test.fillna(medians), medians


# --------------------------------------------------------------------------- #
# Variance-based feature selection
# --------------------------------------------------------------------------- #
def select_top_variance_genes(
    features: pd.DataFrame, top_k: int = TOP_K_GENES
) -> list[str]:
    """Return the names of the ``top_k`` highest-variance genes.

    Variance is computed column-wise. If fewer than ``top_k`` genes exist, all
    are returned. Sorting is stable for deterministic tie-breaking.
    """
    variances = features.var(axis=0, numeric_only=True)
    ranked = variances.sort_values(ascending=False, kind="stable")
    return ranked.head(top_k).index.tolist()


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def stratified_split(
    features: pd.DataFrame,
    labels: pd.Series,
    top_k: int = TOP_K_GENES,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> SplitResult:
    """Stratified split, then fit variance selection + imputation on train.

    Order matters for leakage safety:
      1. Split first (stratified by label).
      2. Rank variance on train, keep top-k genes, subset both splits.
      3. Median-impute using train medians on both splits.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    selected_genes = select_top_variance_genes(x_train, top_k)
    x_train = x_train.loc[:, selected_genes]
    x_test = x_test.loc[:, selected_genes]

    x_train, x_test, _ = impute_median(x_train, x_test)

    return SplitResult(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        selected_genes=selected_genes,
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def build_label_mapping(labels: pd.Series) -> dict[str, int]:
    """Deterministic code -> integer id mapping (alphabetical by code)."""
    codes = sorted(labels.unique())
    return {code: idx for idx, code in enumerate(codes)}


def save_processed(split: SplitResult, label_mapping: dict[str, int]) -> None:
    """Write train/test parquet files and the label mapping JSON."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train = split.x_train.copy()
    train[LABEL_COLUMN] = split.y_train
    test = split.x_test.copy()
    test[LABEL_COLUMN] = split.y_test

    train.to_parquet(TRAIN_PATH, index=True)
    test.to_parquet(TEST_PATH, index=True)

    with LABEL_MAPPING_PATH.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "code_to_id": label_mapping,
                "id_to_code": {str(v): k for k, v in label_mapping.items()},
                "disease_to_code": DISEASE_TO_CODE,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )


# --------------------------------------------------------------------------- #
# Audit report
# --------------------------------------------------------------------------- #
def _distribution_table(series: pd.Series, code_col: str = "code") -> str:
    counts = series.value_counts()
    total = int(counts.sum())
    lines = [f"| {code_col} | count | pct |", "|---|---:|---:|"]
    for name, n in counts.items():
        lines.append(f"| {name} | {int(n)} | {100 * n / total:.1f}% |")
    lines.append(f"| **total** | **{total}** | **100.0%** |")
    return "\n".join(lines)


def write_audit_report(
    labeled: LabeledMatrix,
    filtered_after_missing: pd.DataFrame,
    dropped_missing_genes: list[str],
    split: SplitResult,
) -> None:
    """Write ``reports/data_audit.md`` summarizing every pipeline stage."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    n_samples, n_genes_raw = labeled.features.shape
    overall_missing = float(labeled.features.isna().mean().mean())
    n_genes_after_missing = filtered_after_missing.shape[1]

    # Numeric-only gene ids: mostly HGNC symbols, but a handful of un-symbolized
    # Entrez ids remain (e.g. "100130426"). Downstream RAG keys on symbols.
    numeric_all = [g for g in labeled.features.columns if is_numeric_gene_id(g)]
    numeric_selected = [g for g in split.selected_genes if is_numeric_gene_id(g)]
    numeric_selected_preview = ", ".join(f"`{g}`" for g in numeric_selected[:10])
    if len(numeric_selected) > 10:
        numeric_selected_preview += ", ..."
    numeric_selected_preview = numeric_selected_preview or "none"

    disease_code_rows = "\n".join(
        f"| {disease} | {code} |" for disease, code in DISEASE_TO_CODE.items()
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report = f"""# TCGA Data Audit - Stage 1 (Validation & Preprocessing)

_Generated: {generated}_

## 1. Sources
- Expression: `dataset/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena`
  (gene x sample, 20,531 genes x 11,069 sample columns, log2-normalized, batch-corrected).
- Phenotype: `dataset/TCGA_phenotype_denseDataOnlyDownload.tsv` (12,804 rows).

## 2. Sample selection
- **Primary Tumor filter**: `sample_type_id == "01"` **OR** barcode suffix `-01`.
  Normal tissue (`-11`), metastatic (`-06`), recurrent (`-02`) and blood-derived
  (`-03`) samples are excluded, per the MVP scope.
- **Target cancer types**: the 6 candidates from `project_design.md`, mapped from
  the *actual* `_primary_disease` strings (not assumed codes):

| _primary_disease (actual) | code |
|---|---|
{disease_code_rows}

- **Inner join** on sample barcode with the expression matrix yields
  **{n_samples} usable samples** (samples present in phenotype but missing from
  the expression matrix are dropped by the join).

## 3. Label distribution (after inner join)

{_distribution_table(labeled.labels, "code")}

## 4. Feature (gene) summary
- Genes after transpose + de-duplication: **{n_genes_raw:,}**
  (duplicate gene id(s) collapsed to first occurrence: {labeled.dropped_duplicate_genes or "none"}).
- Overall missing-value rate across the matrix: **{overall_missing * 100:.4f}%**.
- Genes dropped for exceeding {MAX_GENE_MISSING_FRAC:.0%} missing:
  **{len(dropped_missing_genes)}** -> **{n_genes_after_missing:,}** genes retained.
- Top-variance selection: kept the **{len(split.selected_genes)}** highest-variance
  genes (ranked on the training split only, to avoid leakage).

### Gene identifier format
- Most gene ids are **HGNC symbols** (e.g. `A1BG`, `TP53`). However, some ids are
  **purely numeric Entrez ids** left un-symbolized in this Xena release
  (e.g. `100130426`).
- Numeric-only ids in the full gene set: **{len(numeric_all)}** of {n_genes_raw:,}.
- Numeric-only ids that survived into the top-{len(split.selected_genes)} feature
  set: **{len(numeric_selected)}** ({numeric_selected_preview}).
- These numeric ids are kept as features for modeling (they carry expression
  signal) but need special handling before the RAG stage - see the options below.

### Numeric gene id handling options for the RAG stage
The RAG corpus keys on gene symbols, so purely numeric Entrez ids cannot be
looked up directly. Before Stage 3 (RAG), pick one of:

1. **Drop numeric genes** - remove numeric-only ids from the feature set entirely.
   Simplest; loses a small amount of signal. Safe because only
   {len(numeric_selected)} of {len(split.selected_genes)} selected genes are numeric.
2. **Map to symbols via a lookup table** - convert Entrez id -> HGNC symbol using a
   reference (e.g. `mygene`, HGNC/NCBI table). Preserves all signal; adds a
   dependency and a small mapping-coverage risk (some ids have no current symbol).
3. **Keep for modeling, exclude from RAG** (recommended default) - retain numeric
   genes as model features, but skip them when they appear in Top-gene lists sent
   to RAG/LLM (flag them so the UI can show "no curated evidence available").

Stage 2 flags any numeric gene id that appears in a Top-gene list via a
`numeric_gene_flag` column, so option 3 is already actionable downstream.

## 5. Train/test split (stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)})
- Train samples: **{len(split.y_train)}**, Test samples: **{len(split.y_test)}**.
- Final feature width: **{split.x_train.shape[1]}** genes.
- Missing values after median imputation - train: **{int(split.x_train.isna().sum().sum())}**,
  test: **{int(split.x_test.isna().sum().sum())}**.

### Train label distribution
{_distribution_table(split.y_train, "code")}

### Test label distribution
{_distribution_table(split.y_test, "code")}

## 6. Artifacts
- `data/processed/train.parquet` - {len(split.y_train)} x {split.x_train.shape[1]} (+ `{LABEL_COLUMN}` column).
- `data/processed/test.parquet` - {len(split.y_test)} x {split.x_test.shape[1]} (+ `{LABEL_COLUMN}` column).
- `data/processed/label_mapping.json` - code<->id and disease->code mappings.

## 7. Methodology notes & caveats
- Gene identifiers in this release are **gene symbols** (e.g. `A1BG`, `TP53`),
  with a handful of Entrez-id fallbacks - convenient for downstream RAG lookups.
- Variance ranking and median imputation are fit on **train only**; the same
  gene list and medians are applied to test. This makes the split leakage-safe
  and reorders the design doc's step 4 vs 6 accordingly.
- Values are already log2-normalized and batch-corrected upstream (UCSC Xena),
  so no additional scaling is applied at this stage.
"""
    AUDIT_PATH.write_text(report, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> None:
    """Execute the full stage-1 pipeline and report progress to stdout."""
    print("[1/5] Building labeled matrix (reading expression subset)...")
    labeled = build_labeled_matrix()
    print(
        f"      -> {labeled.features.shape[0]} samples x "
        f"{labeled.features.shape[1]} genes"
    )

    print(f"[2/5] Dropping genes with >{MAX_GENE_MISSING_FRAC:.0%} missing...")
    filtered, dropped_missing = drop_high_missing_genes(labeled.features)
    print(
        f"      -> dropped {len(dropped_missing)} genes; "
        f"{filtered.shape[1]} retained"
    )

    print(f"[3/5] Stratified split + top-{TOP_K_GENES} variance selection...")
    split = stratified_split(filtered, labeled.labels)
    print(
        f"      -> train {len(split.y_train)}, test {len(split.y_test)}, "
        f"genes {split.x_train.shape[1]}"
    )

    print("[4/5] Saving processed parquet + label mapping...")
    label_mapping = build_label_mapping(labeled.labels)
    save_processed(split, label_mapping)
    print(f"      -> {TRAIN_PATH.name}, {TEST_PATH.name}, {LABEL_MAPPING_PATH.name}")

    print("[5/5] Writing data audit report...")
    write_audit_report(labeled, filtered, dropped_missing, split)
    print(f"      -> {AUDIT_PATH.relative_to(PROJECT_ROOT)}")

    print("\nDone. Label mapping:", label_mapping)


if __name__ == "__main__":
    run()
