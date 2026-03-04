# src/evaluation.py
# Pablo Anel Rancaño – TFG HAR
"""
Evaluation framework that works with any feature matrix.
Does two things for each model:
  A) Train on train, predict on official test, save metrics + confusion matrix.
  B) GroupKFold CV by subject on the training set.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


ACTIVITY_NAMES: Dict[int, str] = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
CLASS_IDS: List[int] = [1, 2, 3, 4, 5, 6]
CLASS_LABELS: List[str] = [ACTIVITY_NAMES[i] for i in CLASS_IDS]


def save_confusion_matrix(cm: np.ndarray, output_file: Path, title: str) -> None:
    """Save a confusion matrix as a PNG."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=13)
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("True", fontsize=11)
    plt.colorbar()

    plt.xticks(range(len(CLASS_LABELS)), CLASS_LABELS, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_LABELS)), CLASS_LABELS)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=9,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def write_text(out_path: Path, text: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_tag: str,
    results_dir: Path,
    results_prefix: str,
    n_splits: int = 5,
    normalize: bool = False,
) -> Dict[str, Any]:
    """Full evaluation: official test + GroupKFold CV by subject.

    If normalize=True, fits a StandardScaler on train and transforms both splits.
    Returns a dict with all the summary metrics (used for comparison tables).
    """
    t0 = time.time()

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # A) Official test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc_test = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, digits=4,
                                    target_names=CLASS_LABELS)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_IDS)

    cm_path = results_dir / f"{results_prefix}_confusion_test.png"
    save_confusion_matrix(cm, cm_path, f"{model_tag} – Test")

    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=CLASS_IDS)
    per_class_dict = {CLASS_LABELS[i]: round(float(per_class_f1[i]), 4)
                      for i in range(len(CLASS_IDS))}

    # B) GroupKFold CV on train (by subject, no leakage)
    from sklearn.base import clone
    model_cv = clone(model)

    gkf = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(
        model_cv, X_train, y_train,
        cv=gkf, groups=subjects_train,
        scoring="accuracy", n_jobs=-1,
    )

    elapsed = time.time() - t0

    # Save metrics to text file
    lines = [
        f"MODEL: {model_tag}",
        f"Pipeline prefix: {results_prefix}",
        f"Features: {X_train.shape[1]}",
        f"Train samples: {X_train.shape[0]}",
        f"Test samples: {X_test.shape[0]}",
        "",
        f"Test accuracy: {acc_test:.6f}",
        f"Test F1 (macro): {f1_macro:.6f}",
        f"Test F1 (weighted): {f1_weighted:.6f}",
        "",
        "Classification report (test):",
        report,
        "",
        f"CV by subject (GroupKFold, k={n_splits}):",
        f"  Fold accuracies: {scores}",
        f"  Mean: {scores.mean():.6f}",
        f"  Std:  {scores.std():.6f}",
        "",
        "Per-class F1 (test):",
    ]
    for cls, f1 in per_class_dict.items():
        lines.append(f"  {cls}: {f1}")
    lines.append("")
    lines.append(f"Confusion matrix: {cm_path.as_posix()}")
    lines.append(f"Elapsed: {elapsed:.1f}s")

    metrics_path = results_dir / f"{results_prefix}_metrics.txt"
    write_text(metrics_path, "\n".join(lines))

    print(f"  ✓ {model_tag} — test_acc={acc_test:.4f}, "
          f"cv_mean={scores.mean():.4f}±{scores.std():.4f} "
          f"({elapsed:.1f}s)")

    return {
        "model": model_tag,
        "n_features": X_train.shape[1],
        "test_accuracy": round(acc_test, 6),
        "test_f1_macro": round(f1_macro, 6),
        "test_f1_weighted": round(f1_weighted, 6),
        "cv_mean": round(float(scores.mean()), 6),
        "cv_std": round(float(scores.std()), 6),
        "per_class_f1": per_class_dict,
        "elapsed_s": round(elapsed, 1),
    }


def write_summary_table(
    results: List[Dict[str, Any]],
    output_dir: Path,
    pipeline_tag: str,
) -> Tuple[Path, Path]:
    """Write a per-pipeline summary table as CSV and Markdown. Returns (csv_path, md_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        row = {
            "Model": r["model"],
            "Features": r["n_features"],
            "Test Accuracy": r["test_accuracy"],
            "Test F1 (macro)": r["test_f1_macro"],
            "Test F1 (weighted)": r["test_f1_weighted"],
            "CV Mean": r["cv_mean"],
            "CV Std": r["cv_std"],
            "Time (s)": r["elapsed_s"],
        }
        for cls, f1 in r["per_class_f1"].items():
            row[f"F1_{cls}"] = f1
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = output_dir / f"summary_{pipeline_tag}.csv"
    df.to_csv(csv_path, index=False)

    md_path = output_dir / f"summary_{pipeline_tag}.md"
    _write_markdown_table(df, md_path, pipeline_tag)

    return csv_path, md_path


def _write_markdown_table(df: pd.DataFrame, path: Path, title: str) -> None:
    """Write a DataFrame as a Markdown table."""
    lines = [f"# Summary: {title}", ""]

    core_cols = ["Model", "Features", "Test Accuracy", "Test F1 (macro)",
                 "CV Mean", "CV Std", "Time (s)"]
    core_cols = [c for c in core_cols if c in df.columns]
    df_core = df[core_cols]

    header = "| " + " | ".join(str(c) for c in df_core.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df_core.columns) + " |"
    lines.append(header)
    lines.append(sep)

    for _, row in df_core.iterrows():
        vals = []
        for c in df_core.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")

    f1_cols = [c for c in df.columns if c.startswith("F1_")]
    if f1_cols:
        lines.append("## Per-class F1 (test)")
        lines.append("")
        header2 = "| Model | " + " | ".join(c.replace("F1_", "") for c in f1_cols) + " |"
        sep2 = "| --- | " + " | ".join("---" for _ in f1_cols) + " |"
        lines.append(header2)
        lines.append(sep2)
        for _, row in df.iterrows():
            vals = [str(row["Model"])]
            for c in f1_cols:
                vals.append(f"{row[c]:.4f}")
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
