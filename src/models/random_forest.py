"""Random Forest model.

Top-gene importance uses PERMUTATION importance (see :mod:`src.interpret`), not
impurity-based importance, because impurity importance is biased toward
correlated genes (design-doc risk note).
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

NAME = "random_forest"
DISPLAY_NAME = "Random Forest"
IMPORTANCE_KIND = "permutation"


def build_model(random_state: int = 42) -> RandomForestClassifier:
    """Random Forest with a depth cap and balanced class weights."""
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
