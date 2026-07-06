"""Logistic Regression model (multinomial, elasticnet, saga).

Top-gene importance is coefficient-based (handled in :mod:`src.interpret`).
A StandardScaler precedes the classifier so coefficient magnitudes are
comparable across genes and saga converges reliably.

sklearn >= 1.8 note: ``penalty`` and ``n_jobs`` are deprecated on
LogisticRegression. Elasticnet is now expressed purely via ``l1_ratio`` (0 = L2,
1 = L1, in-between = elasticnet), so we set ``l1_ratio=0.5`` and drop ``penalty``
and ``n_jobs`` to keep the same elasticnet behaviour without FutureWarnings.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "logistic"
DISPLAY_NAME = "Logistic Regression"
IMPORTANCE_KIND = "coefficient"


def build_model(random_state: int = 42) -> Pipeline:
    """StandardScaler + multinomial elasticnet Logistic Regression."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    l1_ratio=0.5,  # elasticnet (0=L2, 1=L1); replaces penalty=
                    C=1.0,
                    solver="saga",
                    max_iter=5000,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
