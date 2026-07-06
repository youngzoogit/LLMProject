"""Prediction helper for the Streamlit / RAG stage.

Loads the three fitted models saved by :mod:`src.train_models` and predicts a
cancer type from either a ``sample_id`` (looked up in the processed splits) or a
raw expression vector. Returns, per model, the predicted label and the full
class-probability distribution.

Example::

    from src.predict import predict

    result = predict("TCGA-OR-A5J1-01")          # by sample id
    result = predict(expression_series)          # by expression vector
    # result == {
    #   "input_quality": {"missing_gene_count": 0, "missing_gene_fraction": 0.0,
    #                     "extra_gene_count": 0, "warning": None},
    #   "predictions": {
    #     "logistic":      {"predicted_label": "KIRC",
    #                       "probabilities": {"BRCA": 0.001, ..., "KIRC": 0.98}},
    #     "random_forest": {...},
    #     "mlp":           {...},
    #   },
    # }

CLI::

    python -m src.predict TCGA-OR-A5J1-01
"""

from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAINED_DIR = PROJECT_ROOT / "models" / "trained"
FEATURE_NAMES_PATH = TRAINED_DIR / "feature_names.json"
LABEL_MAPPING_PATH = PROCESSED_DIR / "label_mapping.json"
LABEL_COLUMN = "label"

MODEL_NAMES = ("logistic", "random_forest", "mlp")

# Warn when this fraction (or more) of the model's genes are absent from a raw
# expression vector; below this, silent 0.0-fill is acceptable.
MISSING_WARN_THRESHOLD = 0.10


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_feature_names() -> tuple[str, ...]:
    """The exact gene order the models were trained on."""
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURE_NAMES_PATH} not found. Run `python -m src.train_models` first."
        )
    with FEATURE_NAMES_PATH.open(encoding="utf-8") as fh:
        return tuple(json.load(fh)["feature_names"])


@lru_cache(maxsize=1)
def load_models() -> dict:
    """Load the fitted model pipelines keyed by model name."""
    models = {}
    for name in MODEL_NAMES:
        path = TRAINED_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.train_models` first."
            )
        models[name] = joblib.load(path)
    return models


@lru_cache(maxsize=1)
def load_sample_matrix() -> pd.DataFrame:
    """All known samples (train + test) as a feature matrix for id lookup."""
    frames = []
    for split in ("train.parquet", "test.parquet"):
        path = PROCESSED_DIR / split
        if path.exists():
            frames.append(pd.read_parquet(path).drop(columns=[LABEL_COLUMN]))
    if not frames:
        raise FileNotFoundError("No processed parquet splits found for id lookup.")
    return pd.concat(frames, axis=0)


@lru_cache(maxsize=1)
def available_sample_ids() -> tuple[str, ...]:
    """Sample ids that can be looked up by :func:`predict`."""
    return tuple(load_sample_matrix().index)


# --------------------------------------------------------------------------- #
# Input alignment
# --------------------------------------------------------------------------- #
def get_sample_vector(sample_id: str) -> pd.Series:
    """Return the expression vector for a known ``sample_id``."""
    matrix = load_sample_matrix()
    if sample_id not in matrix.index:
        raise KeyError(
            f"Unknown sample_id {sample_id!r}. "
            f"{len(matrix)} samples are available (e.g. {matrix.index[0]})."
        )
    return matrix.loc[sample_id]


def _input_gene_names(expression) -> list[str] | None:
    """Gene names present in the input, or ``None`` for a positional array."""
    if isinstance(expression, pd.DataFrame):
        return [str(c) for c in expression.columns]
    if isinstance(expression, pd.Series):
        return [str(i) for i in expression.index]
    if isinstance(expression, dict):
        return [str(k) for k in expression]
    return None  # list/tuple/ndarray: positional, already in feature order


def assess_input_quality(expression) -> dict:
    """Report how well a raw input covers the model's expected genes.

    Returns ``missing_gene_count``, ``missing_gene_fraction`` (of the model's
    feature set), ``extra_gene_count`` and a ``warning`` string (or ``None``).
    A positional array/list is assumed pre-aligned, so it reports zero missing.
    """
    feature_names = load_feature_names()
    n_features = len(feature_names)
    gene_names = _input_gene_names(expression)

    if gene_names is None:
        return {
            "missing_gene_count": 0,
            "missing_gene_fraction": 0.0,
            "extra_gene_count": 0,
            "warning": None,
        }

    provided = set(gene_names)
    feature_set = set(feature_names)
    missing = feature_set - provided
    extra = provided - feature_set
    fraction = len(missing) / n_features if n_features else 0.0

    warning = None
    if fraction >= MISSING_WARN_THRESHOLD:
        warning = (
            f"{len(missing)} of {n_features} model genes "
            f"({fraction:.1%}) are missing from the input and were filled with "
            f"0.0; predictions may be unreliable."
        )
    return {
        "missing_gene_count": len(missing),
        "missing_gene_fraction": round(fraction, 4),
        "extra_gene_count": len(extra),
        "warning": warning,
    }


def align_expression(expression) -> pd.DataFrame:
    """Coerce any supported input into a 1-row DataFrame in feature order.

    Accepts a ``pd.Series``/dict keyed by gene, a 1-row ``pd.DataFrame``, or a
    list/array already in feature order. Genes not present are filled with 0.0;
    extra genes are ignored. Use :func:`assess_input_quality` to see how much was
    filled.
    """
    feature_names = list(load_feature_names())

    if isinstance(expression, pd.DataFrame):
        if len(expression) != 1:
            raise ValueError("DataFrame input must have exactly one row.")
        series = expression.iloc[0]
    elif isinstance(expression, pd.Series):
        series = expression
    elif isinstance(expression, dict):
        series = pd.Series(expression)
    elif isinstance(expression, (list, tuple, np.ndarray)):
        values = np.asarray(expression, dtype="float64").ravel()
        if len(values) != len(feature_names):
            raise ValueError(
                f"Array input has {len(values)} values but "
                f"{len(feature_names)} features are expected."
            )
        series = pd.Series(values, index=feature_names)
    else:
        raise TypeError(f"Unsupported expression type: {type(expression)!r}")

    aligned = series.reindex(feature_names).fillna(0.0).astype("float64")
    return aligned.to_frame().T[feature_names]


# --------------------------------------------------------------------------- #
# Prediction
def _predict_one(model, row: pd.DataFrame) -> dict:
    """Predicted label + full probability distribution for a single model."""
    # Windows sandboxed runs can fail when joblib opens worker pipes for
    # n_jobs=-1. Single-sample inference is cheap, so force serial prediction.
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except Exception:
            pass
    classes = [str(c) for c in model.classes_]
    proba = model.predict_proba(row)[0]
    probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
    predicted_label = classes[int(np.argmax(proba))]
    return {"predicted_label": predicted_label, "probabilities": probabilities}
    return {"predicted_label": predicted_label, "probabilities": probabilities}


def predict_from_vector(expression) -> dict:
    """Predict from an expression vector (see :func:`align_expression`).

    Returns ``{"input_quality": {...}, "predictions": {model: {...}}}``. When a
    raw vector is missing many model genes, ``input_quality["warning"]`` is set
    (and logged) so callers do not act on a silently degraded prediction.
    """
    quality = assess_input_quality(expression)
    if quality["warning"]:
        logger.warning(quality["warning"])

    row = align_expression(expression)
    models = load_models()
    predictions = {
        name: _predict_one(model, row) for name, model in models.items()
    }
    return {"input_quality": quality, "predictions": predictions}


def predict_from_sample_id(sample_id: str) -> dict:
    """Predict from a known ``sample_id``."""
    return predict_from_vector(get_sample_vector(sample_id))


def predict(expression_or_sample_id) -> dict:
    """Dispatch: a ``str`` is treated as a sample id, else an expression vector."""
    if isinstance(expression_or_sample_id, str):
        return predict_from_sample_id(expression_or_sample_id)
    return predict_from_vector(expression_or_sample_id)


def _main(argv: list[str]) -> None:
    if len(argv) != 1:
        ids = available_sample_ids()
        print("Usage: python -m src.predict <sample_id>")
        print(f"{len(ids)} samples available, e.g. {', '.join(ids[:3])}")
        return
    sample_id = argv[0]
    result = predict(sample_id)
    quality = result["input_quality"]
    print(f"Predictions for {sample_id}:")
    print(
        f"  input_quality: missing={quality['missing_gene_count']} "
        f"({quality['missing_gene_fraction']:.1%}), "
        f"extra={quality['extra_gene_count']}, "
        f"warning={quality['warning'] or 'none'}"
    )
    for name, out in result["predictions"].items():
        top = sorted(out["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
        top3 = ", ".join(f"{cls}={p:.3f}" for cls, p in top[:3])
        print(f"  {name:14s} -> {out['predicted_label']:5s}  ({top3})")


if __name__ == "__main__":
    _main(sys.argv[1:])
