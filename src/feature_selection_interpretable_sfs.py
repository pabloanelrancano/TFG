# src/feature_selection_interpretable_sfs.py
# Pablo Anel Rancano - TFG HAR
"""Sequential Forward Selection (Linear SVM) on cached interpretable features only.

Writes under ``results/interpretable_sfs/`` without touching the frozen core study
(``results/interpretable/``, ``comparison_all_pipelines.*``).

Constrained design (laptop-friendly):
- Greedy forward SFS up to ``SFS_MAX_STEPS = 100`` selected features.
- Reported subsets: ``SUBSET_GRID = [20, 40, 60, 80, 100, 225]``; the five SFS
  rows share the same forward path (prefix property), and 225 is a baseline
  row that uses **all** features without selection.
- ``GroupKFold`` by training subjects (``n_splits`` from ``config.yaml``).
- Candidate-level parallelization at each forward step (joblib).
- Incremental disk writes: ``sfs_path_log.csv`` flushed after every step,
  ``sfs_selected_features_per_k.json`` rewritten after each checkpoint.

Scaling note (practical, not methodological): a ``StandardScaler`` fitted on
**train only** is applied before SFS. This is **not** introduced as a neutral
preprocessing choice; without scaling, the project's ``LinearSVC``
(``max_iter=5000``) repeatedly fails to converge on the raw interpretable
features in our tests, which makes a constrained-laptop SFS run effectively
unusable. Scaling here is a convergence/stability decision tied to this
specific budget; future thesis discussion can revisit it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

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

# Constrained budget for laptop execution.
SFS_MAX_STEPS: int = 100
SUBSET_GRID: List[int] = [20, 40, 60, 80, 100, 225]
CHECKPOINTS: set[int] = {20, 40, 60, 80, 100}


def _grouped_cv_splits(
    n_samples: int,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((n_samples, 1))
    return list(gkf.split(X_dummy, y, groups=groups))


def _atomic_write_text(path: Path, text: str) -> None:
    """Write file via tmp + os.replace so partial writes never appear on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        finally:
            raise


def _score_one_candidate(
    base_estimator,
    X: np.ndarray,
    y: np.ndarray,
    candidate_mask: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    return float(
        cross_val_score(
            clone(base_estimator),
            X[:, candidate_mask],
            y,
            cv=cv_splits,
            scoring="accuracy",
            n_jobs=1,
        ).mean()
    )


def _run_forward_sfs_with_logging(
    X: np.ndarray,
    y: np.ndarray,
    base_estimator,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    feature_names: List[str],
    max_steps: int,
    checkpoints: set[int],
    path_log_csv: Path,
    json_checkpoints_path: Path,
    json_meta_factory,
) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """Greedy forward SFS with candidate-level parallelism and incremental writes.

    - Tie rule mirrors sklearn ``SequentialFeatureSelector``:
      ``max(scores, key=lambda j: scores[j])`` over the dict built in ascending
      candidate-index order, so on ties the smallest candidate index wins.
    - Writes one row per step to ``path_log_csv`` and flushes immediately.
    - After each step whose new size hits ``checkpoints``, rewrites
      ``json_checkpoints_path`` atomically with the current per-K feature lists.
    """
    n_features = X.shape[1]
    current_mask = np.zeros(n_features, dtype=bool)
    saved: Dict[int, np.ndarray] = {}
    path_indices: List[int] = []
    selected_by_k: Dict[str, List[str]] = {}

    path_log_csv.parent.mkdir(parents=True, exist_ok=True)
    log_fh = path_log_csv.open("w", encoding="utf-8", newline="")
    writer = csv.writer(log_fh)
    writer.writerow(
        [
            "step",
            "n_features_after_step",
            "added_feature_index",
            "added_feature_name",
            "best_cv_mean_accuracy",
            "n_candidates_evaluated",
            "elapsed_s_step",
        ]
    )
    log_fh.flush()

    try:
        for step in range(1, max_steps + 1):
            t_step = time.time()
            candidate_feature_indices = np.flatnonzero(~current_mask).tolist()

            def _job(j: int) -> Tuple[int, float]:
                cand = current_mask.copy()
                cand[j] = True
                return j, _score_one_candidate(base_estimator, X, y, cand, cv_splits)

            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(_job)(j) for j in candidate_feature_indices
            )
            scores: Dict[int, float] = {j: s for j, s in results}

            new_feature_idx = max(scores, key=lambda j: scores[j])
            best_score = scores[new_feature_idx]
            current_mask[new_feature_idx] = True
            n_sel = int(current_mask.sum())
            path_indices.append(int(new_feature_idx))
            elapsed = time.time() - t_step

            writer.writerow(
                [
                    step,
                    n_sel,
                    int(new_feature_idx),
                    feature_names[new_feature_idx],
                    f"{best_score:.6f}",
                    len(candidate_feature_indices),
                    f"{elapsed:.2f}",
                ]
            )
            log_fh.flush()

            print(
                f"  [SFS step {step:2d}/{max_steps}] "
                f"+{feature_names[new_feature_idx]} "
                f"(idx={new_feature_idx}, candidates={len(candidate_feature_indices)}, "
                f"best_cv={best_score:.4f}, {elapsed:.1f}s)"
            )

            if n_sel in checkpoints:
                saved[n_sel] = current_mask.copy()
                idx = np.flatnonzero(current_mask)
                selected_by_k[str(n_sel)] = [feature_names[i] for i in idx]
                payload = {
                    "meta": json_meta_factory(progress=f"checkpoint_{n_sel}"),
                    "selected_features_by_n": dict(selected_by_k),
                }
                _atomic_write_text(
                    json_checkpoints_path,
                    json.dumps(payload, indent=2),
                )
    finally:
        log_fh.close()

    return saved, path_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SFS with Linear SVM on cached interpretable features (constrained budget); "
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

    X_train_raw = np.ascontiguousarray(X_train_df.to_numpy(dtype=np.float64, copy=True))
    X_test_raw = np.ascontiguousarray(X_test_df.to_numpy(dtype=np.float64, copy=True))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    n_feat_total = X_train.shape[1]
    if n_feat_total < 225:
        raise ValueError(
            f"Expected at least 225 interpretable columns; got {n_feat_total}."
        )
    if SFS_MAX_STEPS >= n_feat_total:
        raise ValueError(
            f"SFS_MAX_STEPS ({SFS_MAX_STEPS}) must be < n_features ({n_feat_total})."
        )

    base_estimator = get_model("linear_svm", random_state=random_state)
    model_tag = get_model_tag("linear_svm")

    cv_splits = _grouped_cv_splits(len(y_train), y_train, sub_train, n_splits)

    print(f"\n[interpretable_sfs] Linear SVM, GroupKFold n_splits={n_splits}, "
          f"checkpoints={sorted(CHECKPOINTS)} + baseline 225")
    print(f"[interpretable_sfs] Output directory: {out_dir.resolve()}")
    print(f"[interpretable_sfs] StandardScaler fit on train only "
          f"(see docstring: convergence/stability decision for LinearSVC).\n")

    path_log_csv = out_dir / "sfs_path_log.csv"
    json_path = out_dir / "sfs_selected_features_per_k.json"

    def _meta_factory(progress: str = "") -> Dict[str, Any]:
        return {
            "model": model_tag,
            "estimator_registry": "linear_svm -> models.get_model (sklearn.svm.LinearSVC)",
            "grouped_cv": "GroupKFold on training subjects",
            "n_splits": n_splits,
            "random_state": random_state,
            "feature_scaling": (
                "StandardScaler fit on train only. NOT a neutral preprocessing "
                "choice: enabled here as a practical convergence/stability "
                "decision for LinearSVC under the laptop-constrained SFS budget."
            ),
            "subset_sizes": SUBSET_GRID,
            "sfs_max_steps": SFS_MAX_STEPS,
            "tie_break_cv": "Max cv_mean_accuracy; on ties prefer smaller n_features (parsimony).",
            "sfs_implementation": (
                "Single greedy forward pass up to SFS_MAX_STEPS=100 with "
                "GroupKFold splits passed as an iterable of (train_idx, test_idx). "
                "Candidate-level parallelization (joblib) at each step. "
                "Subsets at 20/40/60/80/100 are prefixes of that path "
                "(identical to running SFS separately for each K). K=225 uses "
                "all columns without selection as a baseline reference."
            ),
            "nested_cv_note": (
                "This is NOT fully nested outer-inner CV. Selection uses grouped "
                "CV on the full training matrix; reported cv_mean/std per row "
                "refit LinearSVM on the selected columns with the same "
                "GroupKFold splits. Estimates are optimistic vs strict nested "
                "CV but comparable across K; the official test set was not used "
                "to choose K."
            ),
            "constrained_budget_note": (
                "An initial deeper design (up to ~200 selected features) was "
                "considered but proved computationally infeasible on the TFG "
                "laptop. After a first 40-step run, the budget was expanded to "
                "SFS_MAX_STEPS=100 with K grid {20, 40, 60, 80, 100} plus a "
                "225-feature baseline; this remains a constrained budget vs "
                "the originally considered design."
            ),
            "progress": progress,
        }

    t_path = time.time()
    support_by_k, path_indices = _run_forward_sfs_with_logging(
        X=X_train,
        y=y_train,
        base_estimator=base_estimator,
        cv_splits=cv_splits,
        feature_names=feature_names,
        max_steps=SFS_MAX_STEPS,
        checkpoints=CHECKPOINTS,
        path_log_csv=path_log_csv,
        json_checkpoints_path=json_path,
        json_meta_factory=_meta_factory,
    )
    path_elapsed = time.time() - t_path
    print(f"\n[interpretable_sfs] Forward path (1..{SFS_MAX_STEPS}) built "
          f"in {path_elapsed:.1f}s")

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
                raise RuntimeError(
                    f"Missing SFS mask for K={K} (checkpoints={sorted(support_by_k)})"
                )
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

    final_meta = _meta_factory(progress="finished")
    final_meta["preferred_n_features_by_cv_mean_train"] = preferred_k
    final_meta["forward_path_wall_clock_s"] = round(path_elapsed, 2)

    _atomic_write_text(
        json_path,
        json.dumps(
            {"meta": final_meta, "selected_features_by_n": selected_by_k},
            indent=2,
        ),
    )

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
        "# Interpretable pipeline — Sequential Forward Selection (Linear SVM, constrained budget)",
        "",
        "## Design",
        "",
        "- **Input:** cached `X_*_interpretable.parquet` (no re-extraction).",
        "- **Scaling (practical decision, not methodological):** `StandardScaler` "
        "is fitted on **train only** and applied to both train and test "
        "**before** SFS. This is **not** introduced as a neutral preprocessing "
        "step: without scaling, the project's `LinearSVC` (`max_iter=5000`) "
        "fails to converge consistently on the raw interpretable matrix, which "
        "makes the constrained-laptop SFS budget unusable. It is a "
        "convergence/stability choice tied to this specific budget and is "
        "discussed as such in the TFG.",
        "- **Model:** Linear SVM from `models.get_model(\"linear_svm\")` "
        "(sklearn `LinearSVC`).",
        "- **Selection:** greedy forward SFS with **GroupKFold** splits on "
        f"**train subjects** (`n_splits={n_splits}`), candidate-level "
        "parallelization (joblib), same tie-breaking as sklearn "
        "`SequentialFeatureSelector`.",
        f"- **SFS budget:** `SFS_MAX_STEPS = {SFS_MAX_STEPS}` (constrained for "
        "laptop execution; a larger design was considered but proved "
        "computationally infeasible).",
        "- **Subset sizes reported:** " + ", ".join(str(k) for k in SUBSET_GRID) +
        f" (the SFS rows are prefixes of the same {SFS_MAX_STEPS}-step forward "
        "path; 225 is the full interpretable baseline without selection).",
        "- **Preferred K:** highest `cv_mean_accuracy` on train; ties → smaller K.",
        "- **Test:** used only for reporting in this table — **not** used to pick K.",
        "",
        "### Nested CV?",
        "",
        final_meta["nested_cv_note"],
        "",
        "### Constrained budget",
        "",
        final_meta["constrained_budget_note"],
        "",
        "### SFS path efficiency",
        "",
        final_meta["sfs_implementation"],
        "",
        "## Results",
        "",
        md_header,
        md_sep,
        *md_body_lines,
        "",
        f"- **Preferred by CV (train):** K = **{preferred_k}** "
        f"(cv_mean = {pref_row['cv_mean_accuracy']:.4f}).",
        f"- **Full baseline (225 features):** test_acc = "
        f"{baseline_row['test_accuracy']:.4f}, "
        f"cv_mean = {baseline_row['cv_mean_accuracy']:.4f}.",
        f"- **Preferred K test accuracy:** {pref_row['test_accuracy']:.4f} "
        f"(vs 225-feature test {baseline_row['test_accuracy']:.4f}).",
        "",
        f"Confusion matrix (test, preferred K): `{cm_path.name}`",
        "",
        f"Feature lists per K: `{json_path.name}`",
        f"Per-step forward path log: `{path_log_csv.name}`",
        f"Table CSV: `{csv_path.name}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        f"\n[interpretable_sfs] Wrote:\n  {csv_path}\n  {json_path}\n"
        f"  {md_path}\n  {cm_path}\n  {path_log_csv}"
    )
    print(f"[interpretable_sfs] Preferred K by CV (train only): {preferred_k}")


if __name__ == "__main__":
    main()
