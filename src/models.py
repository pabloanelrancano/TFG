# src/models.py
# Pablo Anel Rancano - TFG HAR
"""
Central registry of the 4 models used in the project.
Each entry maps a name to a factory function, a display tag, and a file prefix.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def _random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )


def _logistic_regression(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=random_state,
        n_jobs=-1,
    )


def _linear_svm(random_state: int = 42) -> LinearSVC:
    return LinearSVC(
        random_state=random_state,
        max_iter=5000,
        dual="auto",
    )


def _knn(random_state: int = 42) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1,
    )


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "factory": _random_forest,
        "tag": "Random Forest",
        "prefix": "01_rf",
    },
    "logistic_regression": {
        "factory": _logistic_regression,
        "tag": "Logistic Regression",
        "prefix": "02_lr",
    },
    "linear_svm": {
        "factory": _linear_svm,
        "tag": "Linear SVM",
        "prefix": "03_lsvm",
    },
    "knn": {
        "factory": _knn,
        "tag": "k-NN (k=5)",
        "prefix": "04_knn",
    },
}


def get_model(name: str, random_state: int = 42):
    """Return a fresh model instance by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    entry = MODEL_REGISTRY[name]
    return entry["factory"](random_state=random_state)


def get_model_tag(name: str) -> str:
    return MODEL_REGISTRY[name]["tag"]


def get_model_prefix(name: str) -> str:
    return MODEL_REGISTRY[name]["prefix"]


def list_models():
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())
