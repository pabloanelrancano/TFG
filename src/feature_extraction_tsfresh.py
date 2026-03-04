# src/feature_extraction_tsfresh.py
# Pablo Anel Rancaño – TFG HAR
"""
Brute-force feature extraction using tsfresh.

Converts the raw inertial signals to tsfresh's long format, extracts features
channel by channel (to stay within 16 GB RAM), cleans NaN/Inf values, and
saves the result as parquet with a cleaning report in the metadata JSON.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from inertial_loader import CHANNEL_NAMES, WINDOW_SIZE, load_inertial_signals


def _signals_to_long_format(
    signals: Dict[str, np.ndarray],
    channels: List[str] | None = None,
) -> pd.DataFrame:
    """Convert windowed signals to tsfresh's long format (id, time, values)."""
    channels = channels or CHANNEL_NAMES
    n_windows = signals[channels[0]].shape[0]
    n_samples = signals[channels[0]].shape[1]

    ids = np.repeat(np.arange(n_windows), n_samples)
    times = np.tile(np.arange(n_samples), n_windows)

    data = {"id": ids, "time": times}
    for ch in channels:
        data[ch] = signals[ch].ravel()

    df = pd.DataFrame(data)
    return df


def extract_tsfresh_features(
    signals: Dict[str, np.ndarray],
    settings_name: str = "efficient",
    n_jobs: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract features with tsfresh, one channel at a time to save memory.

    Returns a DataFrame with one row per window and all extracted features as columns.
    """
    import gc

    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import (
            MinimalFCParameters,
            EfficientFCParameters,
            ComprehensiveFCParameters,
        )
    except ImportError as e:
        raise ImportError(
            "tsfresh is required for brute-force feature extraction.\n"
            "Install it with: pip install tsfresh\n"
            f"Original error: {e}"
        )

    settings_map = {
        "minimal": MinimalFCParameters,
        "efficient": EfficientFCParameters,
        "comprehensive": ComprehensiveFCParameters,
    }
    if settings_name not in settings_map:
        raise ValueError(f"Unknown tsfresh settings: {settings_name}. "
                         f"Choose from {list(settings_map.keys())}")

    fc_params = settings_map[settings_name]()
    channels = CHANNEL_NAMES
    n_windows = signals[channels[0]].shape[0]
    n_samples = signals[channels[0]].shape[1]

    ids = np.repeat(np.arange(n_windows), n_samples)
    times = np.tile(np.arange(n_samples), n_windows)

    if verbose:
        print(f"  [tsfresh] Extracting features (settings={settings_name}), "
              f"channel-by-channel for {len(channels)} channels ...")

    t0 = time.time()
    channel_dfs: List[pd.DataFrame] = []

    for i, ch in enumerate(channels):
        if verbose:
            print(f"  [tsfresh]   channel {i+1}/{len(channels)}: {ch} ...", end=" ", flush=True)

        df_long = pd.DataFrame({"id": ids, "time": times, ch: signals[ch].ravel()})

        ch_t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_ch = extract_features(
                df_long,
                column_id="id",
                column_sort="time",
                default_fc_parameters=fc_params,
                n_jobs=n_jobs,
                disable_progressbar=True,
            )

        df_ch = df_ch.sort_index().reset_index(drop=True)
        channel_dfs.append(df_ch)

        if verbose:
            print(f"{df_ch.shape[1]} features in {time.time() - ch_t0:.1f}s")

        del df_long
        gc.collect()

    df_features = pd.concat(channel_dfs, axis=1)

    elapsed = time.time() - t0
    if verbose:
        print(f"  [tsfresh] Total: {df_features.shape[1]} features "
              f"for {df_features.shape[0]} windows in {elapsed:.1f}s")

    return df_features


def report_and_clean_features(
    df: pd.DataFrame,
    nan_threshold: float = 0.5,
    impute_remaining: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Clean NaN/Inf from tsfresh output. Steps:
    1) Replace Inf with NaN
    2) Drop columns with >nan_threshold fraction of NaN
    3) Impute remaining NaNs with column median
    4) Verify nothing is left
    Returns (clean_df, report_dict).
    """
    report: Dict = {}

    n_inf = np.isinf(df.values).sum()
    df = df.replace([np.inf, -np.inf], np.nan)
    report["n_inf_replaced"] = int(n_inf)

    n_total_cells = df.shape[0] * df.shape[1]
    nan_counts = df.isna().sum()
    n_nan_total = int(nan_counts.sum())
    n_cols_with_nan = int((nan_counts > 0).sum())

    report["n_total_cells"] = n_total_cells
    report["n_nan_total_before_cleaning"] = n_nan_total
    report["n_cols_with_nan"] = n_cols_with_nan
    report["n_features_before"] = df.shape[1]

    if verbose:
        print(f"  [clean] Inf cells replaced: {n_inf}")
        print(f"  [clean] NaN cells: {n_nan_total} / {n_total_cells} "
              f"({100*n_nan_total/max(n_total_cells,1):.2f}%)")
        print(f"  [clean] Columns with ≥1 NaN: {n_cols_with_nan}")

    # Drop columns that are mostly NaN
    nan_fractions = nan_counts / df.shape[0]
    cols_to_drop = nan_fractions[nan_fractions > nan_threshold].index.tolist()
    report["n_cols_dropped"] = len(cols_to_drop)
    report["nan_threshold"] = nan_threshold

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        if verbose:
            print(f"  [clean] Dropped {len(cols_to_drop)} columns "
                  f"(>{nan_threshold*100:.0f}% NaN)")

    remaining_nans = int(df.isna().sum().sum())
    report["n_nan_after_column_drop"] = remaining_nans

    if impute_remaining and remaining_nans > 0:
        medians = df.median()
        df = df.fillna(medians)
        df = df.fillna(0.0)  # safety net for all-NaN columns
        if verbose:
            print(f"  [clean] Imputed {remaining_nans} remaining NaN cells (median)")

    final_nans = int(df.isna().sum().sum())
    report["n_nan_final"] = final_nans
    report["n_features_after"] = df.shape[1]

    if final_nans > 0:
        raise ValueError(f"Cleaning failed: {final_nans} NaN cells remain!")

    if verbose:
        print(f"  [clean] Final: {df.shape[1]} features, 0 NaN/Inf ✓")

    return df, report


def save_features(
    df: pd.DataFrame,
    split: str,
    output_dir: Path,
    cleaning_report: Dict | None = None,
    tag: str = "tsfresh",
) -> Path:
    """Save feature DataFrame as parquet + metadata JSON (includes cleaning stats)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"X_{split}_{tag}.parquet"
    df.to_parquet(parquet_path, index=False)

    meta = {
        "tag": tag,
        "split": split,
        "n_windows": len(df),
        "n_features": len(df.columns),
        "feature_names": list(df.columns),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if cleaning_report:
        meta["cleaning_report"] = cleaning_report

    meta_path = output_dir / f"X_{split}_{tag}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return parquet_path


def load_features(
    split: str,
    output_dir: Path,
    tag: str = "tsfresh",
) -> pd.DataFrame:
    """Load previously saved tsfresh feature parquet."""
    parquet_path = output_dir / f"X_{split}_{tag}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Feature file not found: {parquet_path.resolve()}\n"
            f"Run the tsfresh extraction pipeline first."
        )
    return pd.read_parquet(parquet_path)


def main() -> None:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from config import get_config, add_common_args, apply_cli_overrides, dataset_path, processed_dir
    from dataset_loader import load_split, assert_dataset_exists
    from inertial_loader import load_inertial_signals, validate_alignment

    parser = argparse.ArgumentParser(
        description="Extract tsfresh features from UCI HAR Inertial Signals."
    )
    add_common_args(parser)
    parser.add_argument(
        "--tsfresh-settings", type=str, default=None,
        choices=["minimal", "efficient", "comprehensive"],
        help="tsfresh extraction level.",
    )
    args = parser.parse_args()
    cfg = apply_cli_overrides(get_config(), args)

    ds_path = dataset_path(cfg)
    proc = processed_dir(cfg)
    ts_cfg = cfg["tsfresh"]
    settings = args.tsfresh_settings or ts_cfg["extraction_settings"]

    assert_dataset_exists(ds_path)

    for split in ("train", "test"):
        print(f"\n{'='*60}")
        print(f"  Extracting tsfresh features – {split}")
        print(f"{'='*60}")

        signals = load_inertial_signals(split, ds_path)
        _, y, subjects = load_split(split, ds_path)
        validate_alignment(signals, y, subjects, split)

        df_raw = extract_tsfresh_features(signals, settings_name=settings)

        df_clean, report = report_and_clean_features(
            df_raw,
            nan_threshold=ts_cfg["nan_threshold"],
            impute_remaining=ts_cfg["impute_remaining"],
        )

        out = save_features(df_clean, split, proc, cleaning_report=report)
        print(f"  Saved → {out}")

    print("\n✓ tsfresh feature extraction complete.")


if __name__ == "__main__":
    main()
