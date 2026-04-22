# src/feature_extraction_tsfeatures_r.py
# Pablo Anel Rancano - TFG HAR
"""R tsfeatures (default only), per channel — additive pipeline ``tsfeatures_r``."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from inertial_loader import CHANNEL_NAMES, load_inertial_signals


def _r_driver_path() -> Path:
    return Path(__file__).resolve().parent / "r" / "tsfeatures_extract.R"


def _resolve_rscript_executable(rscript: str) -> str:
    path = Path(rscript).expanduser()
    if path.is_file():
        return str(path.resolve())
    which = shutil.which(rscript)
    if which:
        return which
    raise FileNotFoundError(
        f"Rscript not found: {rscript!r}\n"
        f"Install R (e.g. apt install r-base-core) or set tsfeatures_r.rscript_path "
        f"in config.yaml to the full path to Rscript."
    )


def _run_tsfeatures_r(
    signals: Dict[str, np.ndarray],
    rscript: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Call R once per split: per channel, list of ts (one per window); default tsfeatures(), cbind."""
    driver = _r_driver_path()
    if not driver.is_file():
        raise FileNotFoundError(f"R driver script not found: {driver}")

    rscript_bin = _resolve_rscript_executable(rscript)
    n_windows = next(iter(signals.values())).shape[0]
    for ch in CHANNEL_NAMES:
        if signals[ch].shape != (n_windows, 128):
            raise ValueError(f"Channel {ch}: expected shape ({n_windows}, 128), got {signals[ch].shape}")

    channels_arg = ",".join(CHANNEL_NAMES)

    with tempfile.TemporaryDirectory(prefix="har_tsfeatures_r_") as tmp:
        tmp_path = Path(tmp)
        for ch in CHANNEL_NAMES:
            np.savetxt(tmp_path / f"{ch}.csv", signals[ch], delimiter=",", fmt="%.17g")

        out_csv = tmp_path / "features_out.csv"
        cmd = [rscript_bin, str(driver), str(tmp_path), str(out_csv), channels_arg]
        if verbose:
            print(f"  [tsfeatures_r] Rscript + driver (channels={len(CHANNEL_NAMES)}) ...")

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            msg = (
                f"R tsfeatures extraction failed (exit {proc.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
            )
            if proc.stdout:
                msg += f"--- stdout ---\n{proc.stdout}\n"
            if proc.stderr:
                msg += f"--- stderr ---\n{proc.stderr}\n"
            msg += "Install CRAN package: install.packages('tsfeatures')\n"
            raise RuntimeError(msg)

        df = pd.read_csv(out_csv)
        if verbose:
            print(f"  [tsfeatures_r] R done in {time.time() - t0:.1f}s "
                  f"-> {df.shape[1]} columns, {len(df)} rows")

    if len(df) != n_windows:
        raise ValueError(
            f"R returned {len(df)} rows but expected {n_windows} (one per window)."
        )

    return df


def _clean_numeric_features(
    df: pd.DataFrame,
    nan_threshold: float = 0.5,
    impute_remaining: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from feature_extraction_tsfresh import report_and_clean_features

    return report_and_clean_features(
        df,
        nan_threshold=nan_threshold,
        impute_remaining=impute_remaining,
        verbose=verbose,
    )


def extract_tsfeatures_r_features(
    signals: Dict[str, np.ndarray],
    rscript: str,
    nan_threshold: float = 0.5,
    impute_remaining: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df_raw = _run_tsfeatures_r(signals, rscript=rscript, verbose=verbose)
    return _clean_numeric_features(
        df_raw,
        nan_threshold=nan_threshold,
        impute_remaining=impute_remaining,
        verbose=verbose,
    )


def save_features(
    df: pd.DataFrame,
    split: str,
    output_dir: Path,
    cleaning_report: Dict[str, Any] | None = None,
    tag: str = "tsfeatures_r",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"X_{split}_{tag}.parquet"
    df.to_parquet(parquet_path, index=False)

    meta: Dict[str, Any] = {
        "tag": tag,
        "split": split,
        "n_windows": len(df),
        "n_features": len(df.columns),
        "feature_names": list(df.columns),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Default tsfeatures::tsfeatures(M) per channel; M rows=windows, cols=128.",
    }
    if cleaning_report is not None:
        meta["cleaning_report"] = cleaning_report

    meta_path = output_dir / f"X_{split}_{tag}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return parquet_path


def load_features(
    split: str,
    output_dir: Path,
    tag: str = "tsfeatures_r",
) -> pd.DataFrame:
    parquet_path = output_dir / f"X_{split}_{tag}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Feature file not found: {parquet_path.resolve()}\n"
            f"Run: python src/feature_extraction_tsfeatures_r.py"
        )
    return pd.read_parquet(parquet_path)


def main() -> None:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from config import get_config, add_common_args, apply_cli_overrides, dataset_path, processed_dir
    from dataset_loader import load_split, assert_dataset_exists
    from inertial_loader import validate_alignment

    parser = argparse.ArgumentParser(
        description="Extract default R tsfeatures per channel (pipeline tsfeatures_r)."
    )
    add_common_args(parser)
    parser.add_argument(
        "--rscript",
        type=str,
        default=None,
        help="Rscript executable (overrides config tsfeatures_r.rscript_path).",
    )
    args = parser.parse_args()
    cfg = apply_cli_overrides(get_config(), args)

    ds_path = dataset_path(cfg)
    proc_dir = processed_dir(cfg)
    r_cfg = cfg.get("tsfeatures_r", {})
    rscript = args.rscript or r_cfg.get("rscript_path", "Rscript")
    ts_cfg = cfg.get("tsfresh", {})
    nan_thr = float(ts_cfg.get("nan_threshold", 0.5))
    impute = bool(ts_cfg.get("impute_remaining", True))

    assert_dataset_exists(ds_path)

    for split in ("train", "test"):
        print(f"\n{'='*60}")
        print(f"  R tsfeatures (default, per channel) — {split}")
        print(f"{'='*60}")

        signals = load_inertial_signals(split, ds_path)
        _, y, subjects = load_split(split, ds_path)
        validate_alignment(signals, y, subjects, split)

        df_clean, report = extract_tsfeatures_r_features(
            signals,
            rscript=rscript,
            nan_threshold=nan_thr,
            impute_remaining=impute,
        )
        out = save_features(df_clean, split, proc_dir, cleaning_report=report, tag="tsfeatures_r")
        print(f"  Saved -> {out}")

    print("\ntsfeatures_r extraction complete.")


if __name__ == "__main__":
    main()
