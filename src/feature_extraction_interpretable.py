# src/feature_extraction_interpretable.py
# Pablo Anel Rancano - TFG HAR
"""
Extracts a compact set of interpretable features from raw inertial signals.

Per channel (9 channels): 13 time-domain + 5 frequency-domain features.
Plus 3 magnitude vectors with the same features, and 9 axis-pair correlations.
Total: ~225 features per window.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from inertial_loader import CHANNEL_NAMES, WINDOW_SIZE, load_inertial_signals

FS = 50  # sampling rate in Hz


def _time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """Compute 13 time-domain features for a single 128-sample window."""
    n = len(signal)
    features: Dict[str, float] = {}

    features["mean"] = float(np.mean(signal))
    features["std"] = float(np.std(signal, ddof=0))
    features["min"] = float(np.min(signal))
    features["max"] = float(np.max(signal))
    features["median"] = float(np.median(signal))
    features["range"] = features["max"] - features["min"]

    q25, q75 = np.percentile(signal, [25, 75])
    features["iqr"] = float(q75 - q25)

    features["skewness"] = float(sp_stats.skew(signal, bias=True))
    features["kurtosis"] = float(sp_stats.kurtosis(signal, bias=True))

    features["rms"] = float(np.sqrt(np.mean(signal ** 2)))
    features["mad"] = float(np.mean(np.abs(signal - np.mean(signal))))

    zero_crossings = np.sum(np.diff(np.sign(signal - np.mean(signal))) != 0)
    features["zcr"] = float(zero_crossings / (n - 1))

    features["energy"] = float(np.sum(signal ** 2) / n)

    return features


def _frequency_domain_features(signal: np.ndarray, fs: int = FS) -> Dict[str, float]:
    """Compute 5 frequency-domain features via FFT."""
    n = len(signal)
    features: Dict[str, float] = {}

    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Skip DC component
    fft_mag_nodc = fft_mag[1:]
    freqs_nodc = freqs[1:]

    idx_peak = np.argmax(fft_mag_nodc)
    features["dominant_freq"] = float(freqs_nodc[idx_peak])

    features["spectral_energy"] = float(np.sum(fft_mag_nodc ** 2))

    psd = fft_mag_nodc ** 2
    psd_norm = psd / (psd.sum() + 1e-12)
    features["spectral_entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    features["mean_freq"] = float(np.sum(freqs_nodc * fft_mag_nodc) / (fft_mag_nodc.sum() + 1e-12))

    features["spectral_bandwidth"] = float(
        np.sqrt(np.sum(fft_mag_nodc * (freqs_nodc - features["mean_freq"]) ** 2) /
                (fft_mag_nodc.sum() + 1e-12))
    )

    return features


def _channel_features(
    signal: np.ndarray,
    channel_name: str,
    include_frequency: bool = True,
    fs: int = FS,
) -> Dict[str, float]:
    """All features for one channel, prefixed with channel name."""
    feats = _time_domain_features(signal)
    if include_frequency:
        feats.update(_frequency_domain_features(signal, fs))

    return {f"{channel_name}__{k}": v for k, v in feats.items()}


def _magnitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Euclidean magnitude of a 3-axis signal."""
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two signals."""
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def _cross_channel_features(
    signals: Dict[str, np.ndarray],
    window_idx: int,
    include_frequency: bool = True,
    fs: int = FS,
) -> Dict[str, float]:
    """Magnitude vectors + axis-pair correlations for one window."""
    feats: Dict[str, float] = {}

    mag_groups = {
        "total_acc_mag": ("total_acc_x", "total_acc_y", "total_acc_z"),
        "body_acc_mag": ("body_acc_x", "body_acc_y", "body_acc_z"),
        "body_gyro_mag": ("body_gyro_x", "body_gyro_y", "body_gyro_z"),
    }
    for mag_name, (cx, cy, cz) in mag_groups.items():
        mag_signal = _magnitude(
            signals[cx][window_idx],
            signals[cy][window_idx],
            signals[cz][window_idx],
        )
        mag_feats = _time_domain_features(mag_signal)
        if include_frequency:
            mag_feats.update(_frequency_domain_features(mag_signal, fs))
        feats.update({f"{mag_name}__{k}": v for k, v in mag_feats.items()})

    axis_pairs = [
        ("total_acc_x", "total_acc_y"),
        ("total_acc_x", "total_acc_z"),
        ("total_acc_y", "total_acc_z"),
        ("body_acc_x", "body_acc_y"),
        ("body_acc_x", "body_acc_z"),
        ("body_acc_y", "body_acc_z"),
        ("body_gyro_x", "body_gyro_y"),
        ("body_gyro_x", "body_gyro_z"),
        ("body_gyro_y", "body_gyro_z"),
    ]
    for ch_a, ch_b in axis_pairs:
        feats[f"corr__{ch_a}__{ch_b}"] = _correlation(
            signals[ch_a][window_idx], signals[ch_b][window_idx]
        )

    return feats


def extract_interpretable_features(
    signals: Dict[str, np.ndarray],
    include_frequency: bool = True,
    fs: int = FS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract interpretable features for all windows. Returns a DataFrame (one row per window)."""
    n_windows = next(iter(signals.values())).shape[0]
    all_rows: List[Dict[str, float]] = []

    t0 = time.time()
    for i in range(n_windows):
        row: Dict[str, float] = {}

        for ch in CHANNEL_NAMES:
            row.update(_channel_features(signals[ch][i], ch, include_frequency, fs))

        row.update(_cross_channel_features(signals, i, include_frequency, fs))

        all_rows.append(row)

        if verbose and (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  [interpretable] {i + 1}/{n_windows} windows "
                  f"({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    if verbose:
        print(f"  [interpretable] Done: {n_windows} windows, "
              f"{len(all_rows[0])} features, {elapsed:.1f}s")

    df = pd.DataFrame(all_rows)
    return df


def save_features(
    df: pd.DataFrame,
    split: str,
    output_dir: Path,
    tag: str = "interpretable",
) -> Path:
    """Save feature DataFrame as parquet + metadata JSON."""
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
    meta_path = output_dir / f"X_{split}_{tag}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return parquet_path


def load_features(
    split: str,
    output_dir: Path,
    tag: str = "interpretable",
) -> pd.DataFrame:
    """Load previously saved feature parquet."""
    parquet_path = output_dir / f"X_{split}_{tag}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Feature file not found: {parquet_path.resolve()}\n"
            f"Run the extraction pipeline first."
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
        description="Extract interpretable features from UCI HAR Inertial Signals."
    )
    add_common_args(parser)
    args = parser.parse_args()
    cfg = apply_cli_overrides(get_config(), args)

    ds_path = dataset_path(cfg)
    proc_dir = processed_dir(cfg)
    include_freq = cfg["interpretable"]["include_frequency"]
    fs = cfg["interpretable"]["sampling_rate"]

    assert_dataset_exists(ds_path)

    for split in ("train", "test"):
        print(f"\n{'='*60}")
        print(f"  Extracting interpretable features - {split}")
        print(f"{'='*60}")

        signals = load_inertial_signals(split, ds_path)

        _, y, subjects = load_split(split, ds_path)
        validate_alignment(signals, y, subjects, split)

        df = extract_interpretable_features(
            signals, include_frequency=include_freq, fs=fs
        )

        out = save_features(df, split, proc_dir, tag="interpretable")
        print(f"  Saved -> {out}")

    print("\nInterpretable feature extraction complete.")


if __name__ == "__main__":
    main()
