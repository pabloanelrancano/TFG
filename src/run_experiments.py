# src/run_experiments.py
# Pablo Anel Rancano - TFG HAR
"""
Main entry point to run all experiments. Supports three feature pipelines
(baseline_561, interpretable, tsfresh) and four models (RF, LR, SVM, KNN).

Usage:
  python src/run_experiments.py                              # all pipelines, all models
  python src/run_experiments.py --pipelines baseline_561     # just baseline
  python src/run_experiments.py --pipelines tsfresh --normalize
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config, add_common_args, apply_cli_overrides, dataset_path, processed_dir, results_dir
from dataset_loader import load_split, assert_dataset_exists
from models import get_model, get_model_tag, get_model_prefix, list_models, MODEL_REGISTRY
from evaluation import evaluate_model, write_summary_table


def _load_baseline_561(cfg: Dict[str, Any]):
    """Load the original 561-feature matrices straight from the dataset."""
    ds = dataset_path(cfg)
    X_train, y_train, sub_train = load_split("train", ds)
    X_test, y_test, _ = load_split("test", ds)
    return X_train, y_train, sub_train, X_test, y_test


def _load_interpretable(cfg: Dict[str, Any]):
    """Load interpretable features from cached parquets (must be extracted first)."""
    proc = processed_dir(cfg)
    ds = dataset_path(cfg)

    try:
        from feature_extraction_interpretable import load_features
        X_train_df = load_features("train", proc, tag="interpretable")
        X_test_df = load_features("test", proc, tag="interpretable")
    except FileNotFoundError as e:
        print(f"\nERROR: Interpretable features not found. Run extraction first:")
        print(f"   python src/feature_extraction_interpretable.py")
        raise SystemExit(1) from e

    _, y_train, sub_train = load_split("train", ds)
    _, y_test, _ = load_split("test", ds)

    return X_train_df.values, y_train, sub_train, X_test_df.values, y_test


def _load_tsfresh(cfg: Dict[str, Any]):
    """Load tsfresh features from cached parquets (must be extracted first)."""
    proc = processed_dir(cfg)
    ds = dataset_path(cfg)

    try:
        from feature_extraction_tsfresh import load_features
        X_train_df = load_features("train", proc, tag="tsfresh")
        X_test_df = load_features("test", proc, tag="tsfresh")
    except FileNotFoundError as e:
        print(f"\nERROR: tsfresh features not found. Run extraction first:")
        print(f"   python src/feature_extraction_tsfresh.py")
        raise SystemExit(1) from e

    _, y_train, sub_train = load_split("train", ds)
    _, y_test, _ = load_split("test", ds)

    # Keep only columns present in both train and test
    train_cols = set(X_train_df.columns)
    test_cols = set(X_test_df.columns)
    common_cols = sorted(train_cols & test_cols)
    if len(common_cols) < len(train_cols):
        dropped = len(train_cols) - len(common_cols)
        print(f"  [tsfresh] Aligning columns: keeping {len(common_cols)} "
              f"(dropped {dropped} train-only or test-only)")
    X_train_df = X_train_df[common_cols]
    X_test_df = X_test_df[common_cols]

    return X_train_df.values, y_train, sub_train, X_test_df.values, y_test


PIPELINE_LOADERS = {
    "baseline_561": _load_baseline_561,
    "interpretable": _load_interpretable,
    "tsfresh": _load_tsfresh,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TFG HAR - Unified Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument(
        "--pipelines", nargs="+", default=None,
        choices=list(PIPELINE_LOADERS.keys()),
        help="Feature pipelines to evaluate (default: all configured).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=list_models(),
        help="Models to evaluate (default: all configured).",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Apply StandardScaler to features before fitting.",
    )

    args = parser.parse_args()
    cfg = apply_cli_overrides(get_config(), args)

    ds = dataset_path(cfg)
    assert_dataset_exists(ds)

    pipelines = args.pipelines or cfg.get("pipelines", list(PIPELINE_LOADERS.keys()))
    model_names = args.models or cfg.get("models", list_models())
    n_splits = cfg["evaluation"]["n_splits"]
    random_state = cfg["evaluation"]["random_state"]
    base_results = results_dir(cfg)

    all_pipeline_results: Dict[str, List[Dict[str, Any]]] = {}

    total_t0 = time.time()

    for pipeline in pipelines:
        print(f"\n{'#'*70}")
        print(f"#  Pipeline: {pipeline}")
        print(f"{'#'*70}")

        if pipeline not in PIPELINE_LOADERS:
            print(f"  WARN: Unknown pipeline '{pipeline}', skipping.")
            continue

        try:
            X_train, y_train, sub_train, X_test, y_test = PIPELINE_LOADERS[pipeline](cfg)
        except SystemExit:
            continue

        print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

        for label, arr in [("X_train", X_train), ("X_test", X_test)]:
            n_nan = np.isnan(arr).sum()
            n_inf = np.isinf(arr).sum()
            if n_nan > 0 or n_inf > 0:
                print(f"  WARN: {label} has {n_nan} NaN and {n_inf} Inf values!")

        pipe_results_dir = base_results / pipeline
        pipe_results: List[Dict[str, Any]] = []

        for model_name in model_names:
            model = get_model(model_name, random_state=random_state)
            tag = get_model_tag(model_name)
            prefix = get_model_prefix(model_name)

            result = evaluate_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                subjects_train=sub_train,
                X_test=X_test,
                y_test=y_test,
                model_tag=tag,
                results_dir=pipe_results_dir,
                results_prefix=prefix,
                n_splits=n_splits,
                normalize=args.normalize,
            )
            result["pipeline"] = pipeline
            pipe_results.append(result)

        csv_path, md_path = write_summary_table(
            pipe_results, pipe_results_dir, pipeline
        )
        print(f"\n  Summary -> {csv_path}")
        print(f"  Summary -> {md_path}")

        all_pipeline_results[pipeline] = pipe_results

    # If we ran multiple pipelines, also write a combined comparison
    if len(all_pipeline_results) > 1:
        print(f"\n{'#'*70}")
        print(f"#  Global Comparison")
        print(f"{'#'*70}")

        all_rows = []
        for pipeline, results in all_pipeline_results.items():
            for r in results:
                row = dict(r)
                row["pipeline"] = pipeline
                all_rows.append(row)

        _write_global_comparison(all_rows, base_results)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"  All experiments complete in {total_elapsed:.1f}s")
    print(f"  Results in: {base_results.resolve()}")
    print(f"{'='*70}")


def _write_global_comparison(
    all_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write a cross-pipeline comparison table (CSV + Markdown)."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in all_rows:
        rows.append({
            "Pipeline": r["pipeline"],
            "Model": r["model"],
            "Features": r["n_features"],
            "Test Accuracy": r["test_accuracy"],
            "Test F1 (macro)": r["test_f1_macro"],
            "CV Mean": r["cv_mean"],
            "CV Std": r["cv_std"],
            "Time (s)": r["elapsed_s"],
        })

    df = pd.DataFrame(rows)

    csv_path = output_dir / "comparison_all_pipelines.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Global CSV -> {csv_path}")

    md_path = output_dir / "comparison_all_pipelines.md"
    lines = [
        "# Cross-Pipeline Comparison",
        "",
        "| Pipeline | Model | Features | Test Acc | F1 macro | CV Mean | CV Std |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['Pipeline']} | {row['Model']} | {row['Features']} "
            f"| {row['Test Accuracy']:.4f} | {row['Test F1 (macro)']:.4f} "
            f"| {row['CV Mean']:.4f} | {row['CV Std']:.4f} |"
        )

    best_idx = df["Test Accuracy"].idxmax()
    best = df.loc[best_idx]
    lines.append("")
    lines.append(f"**Best test accuracy:** {best['Model']} on {best['Pipeline']} "
                 f"({best['Test Accuracy']:.4f})")

    best_cv_idx = df["CV Mean"].idxmax()
    best_cv = df.loc[best_cv_idx]
    lines.append(f"**Best CV mean:** {best_cv['Model']} on {best_cv['Pipeline']} "
                 f"({best_cv['CV Mean']:.4f} ± {best_cv['CV Std']:.4f})")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Global MD  -> {md_path}")


if __name__ == "__main__":
    main()
