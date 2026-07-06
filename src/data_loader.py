"""Data loading and labeling for TCGA cancer-type classification (stage 1).

Responsibilities:
- Load the phenotype table and select Primary Tumor samples only.
- Map the raw ``_primary_disease`` names to short cancer-type codes (BRCA, LUAD, ...).
- Read ONLY the needed sample columns from the ~1.2 GB expression matrix
  (gene x sample) and transpose it to sample x gene.
- Inner-join expression samples with phenotype labels.

Memory note
-----------
The expression file ``EB++AdjustPANCAN_...geneExp.xena`` is ~1.2 GB of text
(20,531 genes x 11,069 samples). Loading every column is wasteful because the
MVP only uses a few thousand Primary-Tumor samples from 6 cancer types.
:func:`read_expression_subset` therefore streams the file with ``usecols`` so
only the target sample columns are materialised (~1/3 of the width), and casts
values to ``float32``. This keeps peak memory well under ~1 GB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# A gene id that is all digits is an Entrez id kept because no HGNC symbol was
# available (e.g. "100130426"). Symbol ids (e.g. "A1BG", "TP53") are the norm.
_NUMERIC_GENE_ID_RE = re.compile(r"^\d+$")


def is_numeric_gene_id(gene_id: object) -> bool:
    """True if the gene id is purely numeric (an un-symbolized Entrez id)."""
    return bool(_NUMERIC_GENE_ID_RE.match(str(gene_id)))

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPRESSION_PATH = PROJECT_ROOT / "dataset" / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena"
PHENOTYPE_PATH = PROJECT_ROOT / "dataset" / "TCGA_phenotype_denseDataOnlyDownload.tsv"

# Column names in the phenotype TSV.
COL_SAMPLE = "sample"
COL_SAMPLE_TYPE_ID = "sample_type_id"
COL_SAMPLE_TYPE = "sample_type"
COL_DISEASE = "_primary_disease"

# TCGA sample-type code for a Primary Solid Tumor (barcode suffix / sample_type_id).
PRIMARY_TUMOR_CODE = "01"

# --------------------------------------------------------------------------- #
# Disease -> cancer-code mapping.
#
# IMPORTANT: keys are the EXACT ``_primary_disease`` strings observed in
# TCGA_phenotype_denseDataOnlyDownload.tsv (verified against the real file, not
# assumed). These are the 6 MVP candidates from document/project_design.md.
# To add a cancer type, append its exact disease string here.
# --------------------------------------------------------------------------- #
DISEASE_TO_CODE: dict[str, str] = {
    "breast invasive carcinoma": "BRCA",
    "lung adenocarcinoma": "LUAD",
    "lung squamous cell carcinoma": "LUSC",
    "kidney clear cell carcinoma": "KIRC",
    "colon adenocarcinoma": "COAD",
    "thyroid carcinoma": "THCA",
}


@dataclass
class LabeledMatrix:
    """Result of joining expression with phenotype labels.

    Attributes:
        features: sample x gene expression matrix (index = sample barcode).
        labels: cancer-type code per sample (index aligned with ``features``).
        diseases: raw ``_primary_disease`` string per sample (for auditing).
        dropped_duplicate_genes: gene ids collapsed during de-duplication.
    """

    features: pd.DataFrame
    labels: pd.Series
    diseases: pd.Series
    dropped_duplicate_genes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
def load_phenotype(path: Path = PHENOTYPE_PATH) -> pd.DataFrame:
    """Load the phenotype table as strings (barcodes/codes are categorical)."""
    if not path.exists():
        raise FileNotFoundError(f"Phenotype file not found: {path}")
    return pd.read_csv(path, sep="\t", dtype=str)


def select_primary_tumor(phenotype: pd.DataFrame) -> pd.DataFrame:
    """Keep Primary Tumor samples only.

    A sample is a Primary Tumor when ``sample_type_id == '01'`` OR its barcode
    ends with ``-01``. The OR is deliberate: some rows have a missing
    ``sample_type_id`` but a valid ``-01`` barcode. Normal tissue (``-11``),
    metastatic (``-06``), recurrent (``-02``) and blood-derived (``-03``)
    samples are excluded, as required for the MVP.
    """
    barcode_suffix = phenotype[COL_SAMPLE].str[-2:]
    is_primary = (phenotype[COL_SAMPLE_TYPE_ID] == PRIMARY_TUMOR_CODE) | (
        barcode_suffix == PRIMARY_TUMOR_CODE
    )
    return phenotype.loc[is_primary].copy()


def select_target_diseases(
    primary: pd.DataFrame, disease_to_code: dict[str, str] = DISEASE_TO_CODE
) -> pd.DataFrame:
    """Subset to the target cancer types and attach a ``code`` column."""
    subset = primary[primary[COL_DISEASE].isin(disease_to_code)].copy()
    subset["code"] = subset[COL_DISEASE].map(disease_to_code)
    return subset


# --------------------------------------------------------------------------- #
# Expression
# --------------------------------------------------------------------------- #
def read_expression_header(path: Path = EXPRESSION_PATH) -> list[str]:
    """Return the expression matrix column names (gene-id col + sample barcodes)."""
    if not path.exists():
        raise FileNotFoundError(f"Expression file not found: {path}")
    header = pd.read_csv(path, sep="\t", nrows=0)
    return list(header.columns)


def read_expression_subset(
    samples: list[str], path: Path = EXPRESSION_PATH
) -> tuple[pd.DataFrame, list[str]]:
    """Read only the requested sample columns and transpose to sample x gene.

    Args:
        samples: sample barcodes to keep. Barcodes absent from the matrix are
            silently skipped (handled by the caller's inner join).

    Returns:
        A ``(matrix, dropped_ids)`` tuple where ``matrix`` is indexed by sample
        barcode with gene-symbol columns (``float32``). Duplicate gene ids
        (e.g. ``SLC35E2`` appears twice in this release) are collapsed to their
        first occurrence before transposing, and ``dropped_ids`` lists them.
    """
    header = read_expression_header(path)
    gene_col = header[0]  # first column holds gene ids/symbols
    available = set(header[1:])
    wanted = [s for s in samples if s in available]

    usecols = [gene_col] + wanted
    # Read expression values as float32 to halve memory vs float64.
    dtypes: dict[str, str] = {c: "float32" for c in wanted}
    dtypes[gene_col] = "string"
    frame = pd.read_csv(path, sep="\t", usecols=usecols, dtype=dtypes)

    # Collapse duplicate gene symbols (keep first) so the id is a clean index.
    dup_mask = frame[gene_col].duplicated(keep="first")
    dropped = frame.loc[dup_mask, gene_col].tolist()
    frame = frame.loc[~dup_mask]

    frame = frame.set_index(gene_col)
    frame.index.name = "gene"
    # Transpose: gene x sample -> sample x gene.
    transposed = frame.T
    transposed.index.name = COL_SAMPLE
    return transposed, dropped


# --------------------------------------------------------------------------- #
# Join
# --------------------------------------------------------------------------- #
def build_labeled_matrix(
    expression_path: Path = EXPRESSION_PATH,
    phenotype_path: Path = PHENOTYPE_PATH,
    disease_to_code: dict[str, str] = DISEASE_TO_CODE,
) -> LabeledMatrix:
    """End-to-end: select primary-tumor target samples and join with expression.

    The join is an inner join on sample barcode, so only samples present in both
    the phenotype table and the expression matrix survive.
    """
    phenotype = load_phenotype(phenotype_path)
    primary = select_primary_tumor(phenotype)
    target = select_target_diseases(primary, disease_to_code)

    target_samples = target[COL_SAMPLE].tolist()
    expr, dropped_genes = read_expression_subset(target_samples, expression_path)

    # Inner join: intersection of expression samples and labeled samples.
    common = expr.index.intersection(pd.Index(target_samples))
    features = expr.loc[common]

    label_lookup = target.set_index(COL_SAMPLE)
    labels = label_lookup.loc[common, "code"]
    diseases = label_lookup.loc[common, COL_DISEASE]

    return LabeledMatrix(
        features=features,
        labels=labels,
        diseases=diseases,
        dropped_duplicate_genes=list(dropped_genes),
    )
