"""Multi-layer perceptron model (hidden layers 256 -> 64).

Note on "dropout": scikit-learn's MLPClassifier has no dropout layer. We
approximate the design-doc intent with L2 regularization (``alpha``) plus early
stopping, which serve the same overfitting-control purpose. Features are scaled
first. Top-gene importance uses permutation importance (see :mod:`src.interpret`).
"""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "mlp"
DISPLAY_NAME = "MLP"
IMPORTANCE_KIND = "permutation"


def build_model(random_state: int = 42) -> Pipeline:
    """StandardScaler + MLP (256, 64) with L2 + early stopping."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(256, 64),
                    activation="relu",
                    alpha=1e-3,
                    batch_size=64,
                    early_stopping=True,
                    n_iter_no_change=10,
                    validation_fraction=0.1,
                    max_iter=300,
                    random_state=random_state,
                ),
            ),
        ]
    )
