# src/feature_selection_interpretable_sfs.py
# Pablo Anel Rancano - TFG HAR
"""Sequential Forward Selection (Linear SVM) on cached interpretable features only.

Writes under ``results/interpretable_sfs/`` without touching the frozen core study
(``results/interpretable/``, ``comparison_all_pipelines.*``).

Methodology (short): one greedy forward pass up to 200 features using the same
Grouped CV splits as inside sklearn's SequentialFeatureSelector; checkpoints at
50, 75, 100, 150, 200 match running SFS separately for each K (prefix property).
Post-selection CV mean/std on train uses the same precomputed GroupKFold splits.
The official test set is used only for reporting per-row test metrics; the
preferred subset size is chosen by CV mean on train (ties → fewer features).

This first SFS phase uses **cached interpretable matrices as-is** (no scaling).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_score

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (
    add_common_args,
    apply_cli_overrides,
    dataset_path,
    get_config,
    processed_dir,
    results_dir,
)
from dataset_loader import assert_dataset_exists, load_split
from evaluation import CLASS_IDS, save_confusion_matrix
from feature_extraction_interpretable import load_features
from models import get_model, get_model_tag

# Subset sizes requested for this phase (225 = full interpretable baseline).
SUBSET_GRID: List[int] = [50, 75, 100, 150, 200, 225]
SFS_MAX_STEPS = 200
CHECKPOINTS = {k for k in SUBSET_GRID if k < 225}


def _grouped_cv_splits(
    n_samples: int,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((n_samples, 1))
    return list(gkf.split(X_dummy, y, groups=groups))


def _forward_sfs_support_at_checkpoints(
    X: np.ndarray,
    y: np.ndarray,
    base_estimator,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    max_steps: int,
    checkpoints: set[int],
) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """Greedy forward SFS mirroring sklearn's tie rule; returns mask per checkpoint + added index path."""
    n_features = X.shape[1]
    current_mask = np.zeros(n_features, dtype=bool)
    saved: Dict[int, np.ndarray] = {}
    path_indices: List[int] = []

    for _step in range(max_steps):
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores: Dict[int, float] = {}
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            X_new = X[:, candidate_mask]
            scores[feature_idx] = float(
                cross_val_score(
                    clone(base_estimator),
                    X_new,
                    y,
                    cv=cv_splits,
                    scoring="accuracy",
                    n_jobs=-1,
                ).mean()
            )
        # Same rule as sklearn.feature_selection.SequentialFeatureSelector
        new_feature_idx = max(scores, key=lambda j: scores[j])
        current_mask[new_feature_idx] = True
        n_sel = int(current_mask.sum())
        path_indices.append(int(new_feature_idx))

        if n_sel in checkpoints:
            saved[n_sel] = current_mask.copy()

    return saved, path_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SFS with Linear SVM on cached interpretable features; "
            "outputs under results/interpretable_sfs/."
        )
    )
    add_common_args(parser)
    args = parser.parse_args()
    cfg = apply_cli_overrides(get_config(), args)

    ds = dataset_path(cfg)
    assert_dataset_exists(ds)
    proc = processed_dir(cfg)
    out_dir = results_dir(cfg) / "interpretable_sfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    random_state = int(cfg["evaluation"]["random_state"])
    n_splits = int(cfg["evaluation"]["n_splits"])

    X_train_df = load_features("train", proc, tag="interpretable")
    X_test_df = load_features("test", proc, tag="interpretable")
    feature_names = list(X_train_df.columns)
    if list(X_test_df.columns) != feature_names:
        raise ValueError("Train and test interpretable columns do not match.")

    _, y_train, sub_train = load_split("train", ds)
    _, y_test, _ = load_split("test", ds)

    if X_train_df.shape[0] != len(y_train) or X_test_df.shape[0] != len(y_test):
        raise ValueError("Row count mismatch between feature cache and official split.")

    X_train = np.ascontiguousarray(X_train_df.to_numpy(dtype=np.float64, copy=True))
    X_test = np.ascontiguousarray(X_test_df.to_numpy(dtype=np.float64, copy=True))

    n_feat_total = X_train.shape[1]
    if n_feat_total < 225 or SFS_MAX_STEPS >= n_feat_total:
        raise ValueError(
            f"Expected at least 225 interpretable columns and SFS_MAX_STEPS < n_features; "
            f"got n_features={n_feat_total}, SFS_MAX_STEPS={SFS_MAX_STEPS}."
        )

    base_estimator = get_model("linear_svm", random_state=random_state)
    model_tag = get_model_tag("linear_svm")

    cv_splits = _grouped_cv_splits(len(y_train), y_train, sub_train, n_splits)

    print(f"\n[interpretable_sfs] Linear SVM, GroupKFold n_splits={n_splits}, "
          f"checkpoints={sorted(CHECKPOINTS)} + baseline 225")
    print(f"[interpretable_sfs] Output directory: {out_dir.resolve()}\n")

    t_path = time.time()
    support_by_k, path_indices = _forward_sfs_support_at_checkpoints(
        X_train,
        y_train,
        base_estimator,
        cv_splits,
        max_steps=SFS_MAX_STEPS,
        checkpoints=CHECKPOINTS,
    )
    path_elapsed = time.time() - t_path
    print(f"[interpretable_sfs] Forward path (1..{SFS_MAX_STEPS}) built in {path_elapsed:.1f}s")

    path_rows = []
    for step, j in enumerate(path_indices, start=1):
        path_rows.append(
            {
                "step": step,
                "added_feature_index": j,
                "added_feature_name": feature_names[j],
            }
        )
    pd.DataFrame(path_rows).to_csv(out_dir / "sfs_forward_path_first_200.csv", index=False)

    selected_by_k: Dict[str, List[str]] = {}
    rows: List[Dict[str, Any]] = []

    full_mask = np.ones(n_feat_total, dtype=bool)

    for K in SUBSET_GRID:
        t0 = time.time()
        if K == n_feat_total:
            mask = full_mask.copy()
            selection_mode = "all_features_baseline_no_sfs"
        else:
            if K not in support_by_k:
                raise RuntimeError(f"Missing SFS mask for K={K} (checkpoints={sorted(support_by_k)})")
            mask = support_by_k[K].copy()
            selection_mode = "sequential_forward_selection_grouped_cv"

        idx = np.flatnonzero(mask)
        names_k = [feature_names[i] for i in idx]
        selected_by_k[str(K)] = names_k

        X_tr_k = X_train[:, idx]
        X_te_k = X_test[:, idx]

        cv_scores = cross_val_score(
            clone(base_estimator),
            X_tr_k,
            y_train,
            cv=cv_splits,
            scoring="accuracy",
            n_jobs=-1,
        )
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        final_model = clone(base_estimator)
        final_model.fit(X_tr_k, y_train)
        y_pred = final_model.predict(X_te_k)
        test_acc = float(accuracy_score(y_test, y_pred))
        test_f1_macro = float(
            f1_score(y_test, y_pred, average="macro", labels=CLASS_IDS, zero_division=0)
        )

        rows.append(
            {
                "n_features": K,
                "cv_mean_accuracy": cv_mean,
                "cv_std_accuracy": cv_std,
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1_macro,
                "selection_mode": selection_mode,
                "wall_clock_s": round(time.time() - t0, 2),
            }
        )
        print(
            f"  K={K:3d}: cv_mean={cv_mean:.4f} ± {cv_std:.4f}, "
            f"test_acc={test_acc:.4f}, test_f1_macro={test_f1_macro:.4f}"
        )

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(
        by=["cv_mean_accuracy", "n_features"],
        ascending=[False, True],
    )
    preferred_k = int(df_sorted.iloc[0]["n_features"])
    df["preferred_by_cv_train_only"] = (df["n_features"] == preferred_k).astype(int)

    csv_path = out_dir / "sfs_cv_and_test_by_k.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "model": model_tag,
        "estimator_registry": "linear_svm -> models.get_model (sklearn.svm.LinearSVC)",
        "grouped_cv": "GroupKFold on training subjects",
        "n_splits": n_splits,
        "random_state": random_state,
        "feature_scaling": "none (cached interpretable values used as stored)",
        "subset_sizes": SUBSET_GRID,
        "preferred_n_features_by_cv_mean_train": preferred_k,
        "tie_break_cv": "Max cv_mean_accuracy; on ties prefer smaller n_features (parsimony).",
        "sfs_implementation": (
            "Single greedy forward pass (same scoring/tie rule as sklearn "
            "SequentialFeatureSelector) up to 200 steps with GroupKFold splits "
            "passed as an iterable of (train_idx, test_idx). Subsets at "
            "50/75/100/150/200 are prefixes of that path (identical to separate "
            "SFS runs per K). K=225 uses all columns without selection."
        ),
        "nested_cv_note": (
            "This is NOT fully nested outer-inner CV. Selection uses grouped CV on "
            "the full training matrix; reported cv_mean/std per row refit "
            "LinearSVM on the selected columns with the same GroupKFold splits. "
            "Estimates are optimistic vs strict nested CV but comparable across K; "
            "the official test set was not used to choose K."
        ),
        "forward_path_wall_clock_s": round(path_elapsed, 2),
    }
    json_path = out_dir / "sfs_selected_features_per_k.json"
    json_path.write_text(
        json.dumps({"meta": meta, "selected_features_by_n": selected_by_k}, indent=2),
        encoding="utf-8",
    )

    # Confusion matrix on test for the CV-preferred subset only (K chosen without test).
    pref_idx = np.array(
        [feature_names.index(n) for n in selected_by_k[str(preferred_k)]],
        dtype=int,
    )
    X_tr_p = X_train[:, pref_idx]
    X_te_p = X_test[:, pref_idx]
    model_p = clone(base_estimator)
    model_p.fit(X_tr_p, y_train)
    y_pred_p = model_p.predict(X_te_p)
    cm = confusion_matrix(y_test, y_pred_p, labels=CLASS_IDS)
    cm_path = out_dir / "confusion_test_linear_svm_preferred_by_cv.png"
    save_confusion_matrix(
        cm,
        cm_path,
        f"{model_tag} (interpretable SFS, K={preferred_k} by CV) – Test",
    )

    md_path = out_dir / "summary_interpretable_sfs.md"
    baseline_row = df[df["n_features"] == 225].iloc[0]
    pref_row = df[df["n_features"] == preferred_k].iloc[0]

    md_cols = [
        "n_features",
        "cv_mean_accuracy",
        "cv_std_accuracy",
        "test_accuracy",
        "test_f1_macro",
        "preferred_by_cv_train_only",
    ]
    md_header = "| " + " | ".join(md_cols) + " |"
    md_sep = "| " + " | ".join("---" for _ in md_cols) + " |"
    md_body_lines = []
    for _, r in df.iterrows():
        vals = []
        for c in md_cols:
            v = r[c]
            if c in ("n_features", "preferred_by_cv_train_only"):
                vals.append(str(int(v)))
            else:
                vals.append(f"{float(v):.6f}")
        md_body_lines.append("| " + " | ".join(vals) + " |")

    lines = [
        "# Interpretable pipeline — Sequential Forward Selection (Linear SVM)",
        "",
        "## Design",
        "",
        "- **Input:** cached `X_*_interpretable.parquet` (no re-extraction).",
        "- **Scaling:** none — matrices are used as stored in Parquet.",
        "- **Model:** Linear SVM from `models.get_model(\"linear_svm\")` (sklearn `LinearSVC`).",
        "- **Selection:** greedy forward SFS with **GroupKFold** splits on **train subjects** "
        f"(`n_splits={n_splits}`), same tie-breaking as sklearn `SequentialFeatureSelector`.",
        "- **Subset sizes:** " + ", ".join(str(k) for k in SUBSET_GRID) + ".",
        "- **Preferred K:** highest `cv_mean_accuracy` on train; ties → smaller K.",
        "- **Test:** used only for reporting in this table — **not** used to pick K.",
        "",
        "### Nested CV?",
        "",
        meta["nested_cv_note"],
        "",
        "### SFS path efficiency",
        "",
        meta["sfs_implementation"],
        "",
        "## Results",
        "",
        md_header,
        md_sep,
        *md_body_lines,
        "",
        f"- **Preferred by CV (train):** K = **{preferred_k}** "
        f"(cv_mean = {pref_row['cv_mean_accuracy']:.4f}).",
        f"- **Full baseline (225 features):** test_acc = {baseline_row['test_accuracy']:.4f}, "
        f"cv_mean = {baseline_row['cv_mean_accuracy']:.4f}.",
        f"- **Preferred K test accuracy:** {pref_row['test_accuracy']:.4f} "
        f"(vs 225-feature test {baseline_row['test_accuracy']:.4f}).",
        "",
        f"Confusion matrix (test, preferred K): `{cm_path.name}`",
        "",
        f"Feature lists per K: `{json_path.name}`",
        f"Forward path (steps 1–{SFS_MAX_STEPS}): `sfs_forward_path_first_200.csv`",
        f"Table CSV: `{csv_path.name}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[interpretable_sfs] Wrote:\n  {csv_path}\n  {json_path}\n  {md_path}\n  {cm_path}")
    print(f"[interpretable_sfs] Preferred K by CV (train only): {preferred_k}")


if __name__ == "__main__":
    main()
